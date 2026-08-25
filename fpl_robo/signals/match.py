"""Match free-text player names onto FPL bootstrap players."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _NON_ALNUM.sub("", text.lower())


def alias_match_spans(text: str, alias: str) -> list[tuple[int, int]]:
    """Word-boundary spans for `alias` in `text` (case-insensitive).

    Prevents substring false positives such as "Raya" inside "Rayan Cherki".
    Boundaries are alphanumeric (Unicode letters/digits), so punctuation and
    whitespace still split tokens.
    """
    raw = (alias or "").strip()
    blob = text or ""
    if not raw or not blob:
        return []
    pattern = re.compile(
        rf"(?<![0-9A-Za-zÀ-ÖØ-öø-ÿ]){re.escape(raw)}(?![0-9A-Za-zÀ-ÖØ-öø-ÿ])",
        flags=re.IGNORECASE,
    )
    return [(m.start(), m.end()) for m in pattern.finditer(blob)]


def alias_in_text(text: str, alias: str) -> bool:
    """True if `alias` appears as a whole token in `text`."""
    return bool(alias_match_spans(text, alias))


def build_player_indexes(players: list[dict[str, Any]]) -> dict[str, Any]:
    """Build lookup maps for web_name / surname / full name."""
    by_id: dict[int, dict[str, Any]] = {}
    by_web: dict[str, list[int]] = {}
    by_norm: dict[str, list[int]] = {}

    for p in players:
        pid = int(p["id"])
        by_id[pid] = p
        web = str(p.get("web_name") or "").strip()
        first = str(p.get("first_name") or "").strip()
        second = str(p.get("second_name") or "").strip()
        full = f"{first} {second}".strip()
        if web:
            by_web.setdefault(web.lower(), []).append(pid)
            by_norm.setdefault(normalize_name(web), []).append(pid)
            # Multi-token web names ("João Pedro") → also index trailing token.
            if " " in web:
                by_norm.setdefault(normalize_name(web.split()[-1]), []).append(pid)
        if second:
            by_norm.setdefault(normalize_name(second), []).append(pid)
        if full:
            by_norm.setdefault(normalize_name(full), []).append(pid)
        if first and second:
            by_norm.setdefault(normalize_name(f"{first[0]}{second}"), []).append(pid)

    return {"by_id": by_id, "by_web": by_web, "by_norm": by_norm}


def resolve_player_name(
    name: str,
    indexes: dict[str, Any],
    *,
    team_short: str | None = None,
    candidate_ids: set[int] | None = None,
) -> dict[str, Any] | None:
    """Resolve a free-text name to a single FPL player when unambiguous.

    Optional `team_short` / `candidate_ids` (usually KG players mentioned in the
    article) are used to break surname ties.
    """
    raw = (name or "").strip()
    if not raw:
        return None
    by_web: dict[str, list[int]] = indexes["by_web"]
    by_norm: dict[str, list[int]] = indexes["by_norm"]
    by_id: dict[int, dict[str, Any]] = indexes["by_id"]

    def _filter(pids: list[int]) -> list[int]:
        out = list(dict.fromkeys(pids))
        if candidate_ids is not None:
            out = [p for p in out if p in candidate_ids]
        if team_short:
            want = team_short.strip().upper()
            team_hits = [
                p
                for p in out
                if str(by_id[p].get("team_short") or "").strip().upper() == want
            ]
            if team_hits:
                out = team_hits
        return out

    web_hits = _filter(list(by_web.get(raw.lower()) or []))
    if len(web_hits) == 1:
        return by_id[web_hits[0]]

    key = normalize_name(raw)
    candidates = _filter(list(by_norm.get(key) or []))
    if len(candidates) == 1:
        return by_id[candidates[0]]

    # Surname fallback: last token.
    parts = [p for p in re.split(r"\s+", raw) if p]
    if parts:
        surname_key = normalize_name(parts[-1])
        surname_hits = _filter(list(by_norm.get(surname_key) or []))
        if len(surname_hits) == 1:
            return by_id[surname_hits[0]]
    return None


def candidate_web_names_for_text(text: str, indexes: dict[str, Any], *, limit: int = 40) -> list[str]:
    """Cheap prefilter: web_names that appear as whole tokens in the article text."""
    blob = text or ""
    hits: list[str] = []
    seen: set[int] = set()
    by_id: dict[int, dict[str, Any]] = indexes["by_id"]
    # Prefer longer web_names first so "João Pedro" beats "Pedro".
    ranked: list[tuple[str, list[int]]] = []
    for norm, pids in indexes["by_norm"].items():
        if len(norm) < 3:
            continue
        # Only consider keys that are an exact web_name (or multi-token web).
        webs = []
        for pid in pids:
            web = str(by_id[pid].get("web_name") or "").strip()
            if not web:
                continue
            if normalize_name(web) == norm or (
                " " in web and normalize_name(web.split()[-1]) == norm and len(norm) >= 4
            ):
                webs.append((web, pid))
        if not webs:
            continue
        # Use the longest matching web string for scanning.
        web = max((w for w, _ in webs), key=len)
        ranked.append((web, [pid for _, pid in webs]))
    ranked.sort(key=lambda t: len(t[0]), reverse=True)

    for web, pids in ranked:
        if not alias_in_text(blob, web):
            # Trailing token of multi-word web names ("Pedro" for "João Pedro").
            if " " in web:
                tail = web.split()[-1]
                if len(normalize_name(tail)) < 4 or not alias_in_text(blob, tail):
                    continue
            else:
                continue
        for pid in pids:
            if pid in seen:
                continue
            seen.add(pid)
            w = str(by_id[pid].get("web_name") or "")
            if w:
                hits.append(w)
            if len(hits) >= limit:
                return hits
    return hits
