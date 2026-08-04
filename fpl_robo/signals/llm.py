"""Local Ollama helper for interpreting football news into FPL signals."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from fpl_robo.config import SIGNAL_INGEST

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You extract Fantasy Premier League player news signals from football headlines.
Return ONLY valid JSON with this shape:
{"signals":[{"web_name":"Player","team_short":"LIV","kind":"injury|rotation|delayed_return|suspension|positive|transfer|form|ignore","severity":0.0,"exclude":false,"evidence":"short quote from text","note":"short reason"}]}
Rules:
- Only include players from the provided KG candidate list.
- Prefer signals that are explicit in the headline or first sentence.
- If a headline clearly pairs a candidate player with injury/suspension/doubt language, you MUST include that signal.
- Confirmed club moves (signed/joins/completes deal) → kind=transfer (not ignore). Omit vague rumours ("linked with", "interested in").
- Confirmed exits out of the Premier League (sold to / joins Saudi / MLS / La Liga / leaves for non-PL club) → kind=transfer with exclude=true and high severity.
- Strong preseason/form claims (hat-trick, brace, scored in friendly, impressive debut) → kind=form.
- Fitness returns / cleared to play → kind=positive.
- The player must be named in the article text (headline counts).
- When surnames collide (e.g. two Gomez), use first_name + team_short from the KG line.
- team_short must match the player's club from the KG candidate list.
- evidence must be a short phrase from the article (headline is allowed).
- severity is 0..1 (higher = worse availability). For positive/inbound-transfer/form use 0..0.3; for outbound exits use >=0.85.
- exclude=true for very likely unavailability (severe injury/suspension / status already out / confirmed PL exit).
- Omit listicles / rankings with no concrete player claim.
- Maximum 6 signals per article.
"""


def ollama_available(base_url: str | None = None, timeout: float = 3.0) -> bool:
    url = (base_url or SIGNAL_INGEST["ollama_url"]).rstrip("/") + "/api/tags"
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost
            raw = json.loads(resp.read().decode("utf-8"))
        return isinstance(raw, dict) and "models" in raw
    except Exception:
        return False


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to salvage the first JSON object/array in the blob.
        start_candidates = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        snippet = text[start:]
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(snippet)
        return obj


def chat_json(
    *,
    user_prompt: str,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    system: str = SYSTEM_PROMPT,
    num_predict: int | None = None,
) -> Any:
    """Call Ollama /api/chat and parse a JSON response."""
    base = (base_url or SIGNAL_INGEST["ollama_url"]).rstrip("/")
    model = model or SIGNAL_INGEST["ollama_model"]
    if timeout is None:
        timeout = float(SIGNAL_INGEST.get("ollama_timeout_seconds") or 90)
    else:
        timeout = float(timeout)
    timeout = max(15.0, timeout)
    if num_predict is None:
        num_predict = int(SIGNAL_INGEST.get("ollama_num_predict") or 256)
    else:
        num_predict = int(num_predict)
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": max(64, num_predict)},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = Request(
        base + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost
            raw = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    content = ((raw.get("message") or {}).get("content")) or raw.get("response") or ""
    return _extract_json(content)


def interpret_article(
    *,
    title: str,
    summary: str,
    source: str,
    candidate_names: list[str] | None = None,
    candidate_lines: list[str] | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    num_predict: int | None = None,
) -> list[dict[str, Any]]:
    """Interpret one article into raw news signal dicts (unresolved IDs)."""
    lines = candidate_lines or []
    names = candidate_names or []
    if lines:
        hint = "KG candidates (only choose from these):\n- " + "\n- ".join(lines[:40]) + "\n"
    elif names:
        hint = "Known FPL web_names that may appear: " + ", ".join(names[:80]) + ".\n"
    else:
        hint = ""
    prompt = (
        f"Source: {source}\n"
        f"Headline: {title}\n"
        f"Summary: {summary}\n"
        f"{hint}"
        "Extract high-certainty FPL signals supported by the text "
        "(injuries, suspensions, confirmed transfers, strong preseason form).\n"
        "If the headline itself states injury/suspension/doubt/transfer/form for a KG candidate, include it.\n"
    )
    parsed = chat_json(
        user_prompt=prompt,
        model=model,
        base_url=base_url,
        timeout=timeout,
        num_predict=num_predict,
    )
    if isinstance(parsed, dict):
        signals = parsed.get("signals") or parsed.get("news") or []
    elif isinstance(parsed, list):
        signals = parsed
    else:
        signals = []

    out: list[dict[str, Any]] = []
    for item in signals:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or item.get("type") or "ignore").lower().strip()
        if kind in {"ignore", "none", "n/a", ""}:
            continue
        if kind not in {
            "injury",
            "rotation",
            "delayed_return",
            "suspension",
            "positive",
            "transfer",
            "form",
        }:
            # Map common synonyms.
            if kind in {"boost", "return", "fit", "starting"}:
                kind = "positive"
            elif kind in {"doubt", "knock"}:
                kind = "injury"
            elif kind in {"ban"}:
                kind = "suspension"
            elif kind in {"signing", "signed", "joined", "move"}:
                kind = "transfer"
            elif kind in {"hattrick", "hat-trick", "goals", "friendly", "preseason"}:
                kind = "form"
            else:
                continue
        web_name = str(item.get("web_name") or item.get("player") or "").strip()
        if not web_name:
            continue
        try:
            severity = float(item.get("severity") or 0.0)
        except (TypeError, ValueError):
            severity = 0.0
        severity = max(0.0, min(1.0, severity))
        exclude = bool(item.get("exclude") or False)
        if kind in {"injury", "suspension"} and severity >= 0.85:
            exclude = True
        evidence = str(item.get("evidence") or "").strip()[:220]
        note = str(item.get("note") or evidence or title)[:180]
        team_short = str(item.get("team_short") or "").strip().upper() or None
        out.append(
            {
                "web_name": web_name,
                "team_short": team_short,
                "type": kind,
                "severity": round(severity, 3),
                "exclude": exclude,
                "evidence": evidence,
                "note": f"[{source}] {note}",
                "source": "ollama",
            }
        )
    return out


LINEUP_SYSTEM_PROMPT = """You extract predicted Premier League starting XIs and expected minutes from team-news headlines.
Return ONLY valid JSON with this shape:
{"lineups":[{"team_short":"LIV","starters":["Salah",{"name":"Gakpo","predicted_minutes":65}],"bench":[{"name":"Gomez","predicted_minutes":20}],"doubts":["Jota"],"minutes":{"Salah":90},"confidence":0.6,"evidence":"short quote"}]}
Rules:
- Only include players from the provided KG candidate list for that club.
- Prefer explicit predicted/likely XIs or clear "to start"/"on the bench" claims.
- team_short must be a Premier League short code from the candidates.
- starters max 11; bench optional; doubts optional.
- Player entries may be a string name OR {"name":"...","predicted_minutes":0-90}.
- Optional top-level "minutes" map of web_name → expected minutes (0-90) when the article implies playtime.
- Only set predicted_minutes when the text supports it (starter ~75-90, rotation ~45-70, bench cameo ~10-30); omit if unknown.
- confidence 0..1 (higher if the article is a predicted XI / press conference team news).
- Ignore fantasy dream teams, transfer rumours, and historical lineups.
- If no predicted XI is present, return {"lineups":[]}.
"""


def _lineup_player_entry(raw: Any) -> tuple[str, int | None]:
    """Parse a starter/bench/doubt entry into (web_name, predicted_minutes)."""
    if isinstance(raw, dict):
        web_name = str(raw.get("name") or raw.get("web_name") or "").strip()
        mins = raw.get("predicted_minutes")
        if mins is None:
            mins = raw.get("minutes")
        try:
            predicted = int(mins) if mins is not None and mins != "" else None
        except (TypeError, ValueError):
            predicted = None
        if predicted is not None:
            predicted = max(0, min(90, predicted))
        return web_name, predicted
    web_name = str(raw or "").strip()
    return web_name, None


def interpret_lineup_article(
    *,
    title: str,
    summary: str,
    source: str,
    candidate_lines: list[str],
    gameweek: int | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    num_predict: int | None = None,
) -> list[dict[str, Any]]:
    """Interpret one article into unresolved predicted lineup player rows."""
    if not candidate_lines:
        return []
    gw_hint = f"Target gameweek: GW{gameweek}.\n" if gameweek else ""
    prompt = (
        f"Source: {source}\n"
        f"Headline: {title}\n"
        f"Summary: {summary}\n"
        f"{gw_hint}"
        "KG candidates (only choose from these):\n- "
        + "\n- ".join(candidate_lines[:60])
        + "\nExtract predicted starting XI / bench / doubts and minutes if present.\n"
    )
    parsed = chat_json(
        user_prompt=prompt,
        model=model,
        base_url=base_url,
        system=LINEUP_SYSTEM_PROMPT,
        timeout=(
            float(SIGNAL_INGEST.get("lineup_ollama_timeout_seconds") or 45)
            if timeout is None
            else float(timeout)
        ),
        num_predict=(
            int(SIGNAL_INGEST.get("ollama_num_predict") or 256)
            if num_predict is None
            else int(num_predict)
        ),
    )
    lineups = []
    if isinstance(parsed, dict):
        lineups = parsed.get("lineups") or []
    elif isinstance(parsed, list):
        lineups = parsed

    out: list[dict[str, Any]] = []
    for block in lineups:
        if not isinstance(block, dict):
            continue
        team_short = str(block.get("team_short") or "").strip().upper() or None
        try:
            confidence = float(block.get("confidence") or 0.55)
        except (TypeError, ValueError):
            confidence = 0.55
        confidence = max(0.0, min(1.0, confidence))
        evidence = str(block.get("evidence") or title)[:220]
        minutes_map: dict[str, int] = {}
        raw_minutes = block.get("minutes")
        if isinstance(raw_minutes, dict):
            for k, v in raw_minutes.items():
                name = str(k or "").strip()
                if not name:
                    continue
                try:
                    minutes_map[name.lower()] = max(0, min(90, int(v)))
                except (TypeError, ValueError):
                    continue
        for role, key in (("start", "starters"), ("bench", "bench"), ("doubt", "doubts")):
            names = block.get(key) or []
            if not isinstance(names, list):
                continue
            for entry in names[:15]:
                web_name, predicted_minutes = _lineup_player_entry(entry)
                if not web_name:
                    continue
                if predicted_minutes is None:
                    predicted_minutes = minutes_map.get(web_name.lower())
                row: dict[str, Any] = {
                    "web_name": web_name,
                    "team_short": team_short,
                    "role": role,
                    "confidence": round(confidence, 3),
                    "gameweek": gameweek,
                    "evidence": evidence,
                    "note": f"[{source}] predicted {role}: {web_name}",
                    "source": "ollama_lineup",
                    "article_title": title,
                }
                if predicted_minutes is not None:
                    row["predicted_minutes"] = predicted_minutes
                    row["note"] = (
                        f"[{source}] predicted {role}: {web_name} (~{predicted_minutes}m)"
                    )
                out.append(row)
    return out
