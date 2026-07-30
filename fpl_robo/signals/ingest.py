"""Scrape news + interpret with local LLM into signals.json."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import SIGNAL_INGEST
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.runtime.atomic import atomic_write_json
from fpl_robo.signals.heuristics import heuristic_signals_from_title
from fpl_robo.signals.llm import interpret_article, interpret_lineup_article, ollama_available
from fpl_robo.signals.match import (
    build_player_indexes,
    candidate_web_names_for_text,
    normalize_name,
    resolve_player_name,
)
from fpl_robo.signals.scrape import (
    NewsArticle,
    article_is_actionable,
    articles_payload,
    enrich_articles_with_bodies,
    fetch_all_feeds,
    looks_like_lineup_article,
)
from fpl_robo.signals.validate import (
    build_team_aliases,
    kg_candidate_lines,
    prioritize_kg_candidates,
    score_signal_evidence,
    team_mentioned,
)

logger = logging.getLogger("fpl_robo.signals.ingest")


@dataclass
class IngestResult:
    ok: bool
    dry_run: bool
    applied: bool
    articles: int
    proposed_news: list[dict[str, Any]] = field(default_factory=list)
    proposed_lineups: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    feed_errors: list[dict[str, Any]] = field(default_factory=list)
    signals_path: str | None = None
    raw_cache_path: str | None = None
    reason: str | None = None
    ollama_model: str | None = None
    pre_deadline: bool = False
    gameweek: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "articles": self.articles,
            "proposed_news": self.proposed_news,
            "proposed_lineups": self.proposed_lineups,
            "rejected": self.rejected,
            "unresolved": self.unresolved,
            "feed_errors": self.feed_errors,
            "signals_path": self.signals_path,
            "raw_cache_path": self.raw_cache_path,
            "reason": self.reason,
            "ollama_model": self.ollama_model,
            "pre_deadline": self.pre_deadline,
            "gameweek": self.gameweek,
            "proposed_count": len(self.proposed_news),
            "proposed_lineup_count": len(self.proposed_lineups),
            "rejected_count": len(self.rejected),
            "unresolved_count": len(self.unresolved),
        }


def _load_existing_signals(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "as_of": None,
            "news": [],
            "preseason": [],
            "international": [],
            "predicted_lineups": [],
        }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {
                "as_of": None,
                "news": [],
                "preseason": [],
                "international": [],
                "predicted_lineups": [],
            }
        return {
            "as_of": raw.get("as_of"),
            "news": list(raw.get("news") or []),
            "preseason": list(raw.get("preseason") or []),
            "international": list(raw.get("international") or []),
            "predicted_lineups": list(raw.get("predicted_lineups") or []),
        }
    except Exception:
        return {
            "as_of": None,
            "news": [],
            "preseason": [],
            "international": [],
            "predicted_lineups": [],
        }


def _news_key(item: dict[str, Any]) -> tuple:
    return (
        item.get("player_id"),
        str(item.get("web_name") or "").strip().lower(),
        str(item.get("type") or item.get("kind") or "").strip().lower(),
    )


def _parse_signal_ts(item: dict[str, Any]) -> datetime | None:
    for key in ("ingested_at", "updated_at", "as_of"):
        raw = item.get(key)
        if not raw:
            continue
        text = str(raw).strip()
        try:
            if len(text) == 10 and text[4] == "-":
                return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
    return None


def _prune_stale_auto_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_age = int(SIGNAL_INGEST.get("auto_news_max_age_days") or 0)
    if max_age <= 0:
        return items
    auto_sources = {"ollama", "heuristic", "auto", "fpl"}
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age)
    kept: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("source") not in auto_sources:
            kept.append(item)
            continue
        ts = _parse_signal_ts(item)
        # No timestamp → treat as fresh enough to keep until next successful scrape.
        if ts is None or ts >= cutoff:
            kept.append(item)
    return kept


def merge_news_signals(
    existing_news: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace prior auto news; keep manual entries.

    If the scrape yields no accepted signals, retain prior auto news so a soft
    listicle day does not wipe real injury/transfer/form signals.
    """
    auto_sources = {"ollama", "heuristic", "auto", "fpl"}
    retain_empty = bool(SIGNAL_INGEST.get("retain_auto_news_on_empty", True))

    if not proposed and retain_empty:
        return _prune_stale_auto_news(
            [item for item in existing_news if isinstance(item, dict)]
        )

    keep: list[dict[str, Any]] = []
    for item in existing_news:
        if not isinstance(item, dict):
            continue
        if item.get("source") in auto_sources:
            continue
        keep.append(item)

    seen: set[tuple] = set()
    merged = list(keep)
    now = datetime.now(timezone.utc).isoformat()
    for item in proposed:
        key = _news_key(item)
        if key in seen:
            continue
        seen.add(key)
        row = dict(item)
        row.setdefault("ingested_at", now)
        merged.append(row)
    return _prune_stale_auto_news(merged)


def heuristics_are_sufficient(raw_signals: list[dict[str, Any]], *, title: str) -> bool:
    """True when headline heuristics are concrete enough to skip the LLM."""
    if not raw_signals:
        return False
    if not bool(SIGNAL_INGEST.get("skip_llm_when_heuristic", True)):
        return False
    title_l = (title or "").lower()
    # Roundups with "latest" often bury extra names — always let the LLM help.
    if "latest" in title_l:
        return False
    vague_title = any(
        k in title_l for k in ("look ahead", "round-up", "roundup", "team news only")
    )
    concrete_title = any(
        k in title_l
        for k in (
            "injur",
            "hat-trick",
            "hattrick",
            "sign",
            "deal",
            "miss",
            "absent",
            "groin",
            "months out",
            "ruled out",
            "suspend",
        )
    )
    if vague_title and not concrete_title:
        return False
    strong_note = (
        "injur",
        "hat-trick",
        "hattrick",
        "months out",
        "groin",
        "signs",
        "sign",
        "deal",
        "miss",
        "absent",
        "scored",
        "on target",
        "lively",
        "sharp",
        "suspend",
    )
    for sig in raw_signals:
        note = str(sig.get("note") or "").lower()
        if "latest" in note and not any(k in note for k in strong_note):
            return False
        if not any(k in note for k in strong_note):
            # Soft/unknown heuristic — let LLM try to enrich.
            if str(sig.get("type") or "") in {"form", "positive"} and any(
                k in note for k in ("lively", "sharp", "impressive", "on target", "hat-trick")
            ):
                continue
            if str(sig.get("type") or "") in {"injury", "suspension", "transfer", "rotation"}:
                continue
            return False
    return True


def _player_name_tokens(player: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    web = str(player.get("web_name") or "").strip()
    second = str(player.get("second_name") or "").strip()
    first = str(player.get("first_name") or "").strip()
    for tok in (web, second, f"{first} {second}".strip()):
        if tok and len(tok) >= 3:
            tokens.append(tok)
    if web and " " in web:
        tokens.append(web.split()[-1])
    # Dedupe while preserving order (longer names first for cleaner matches).
    seen: set[str] = set()
    out: list[str] = []
    for tok in sorted(set(tokens), key=len, reverse=True):
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            out.append(tok)
    return out


def _player_mention_spans(text: str, player: dict[str, Any]) -> list[tuple[int, int]]:
    """Character spans of player name mentions (merged overlaps)."""
    import re

    blob = text or ""
    if not blob:
        return []
    spans: list[tuple[int, int]] = []
    for tok in _player_name_tokens(player):
        for m in re.finditer(re.escape(tok), blob, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    if not spans:
        return []
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


def _player_local_window(text: str, player: dict[str, Any], *, radius: int = 160) -> str:
    """Return merged windows around player name mentions in text."""
    blob = text or ""
    if not blob:
        return ""
    intervals: list[tuple[int, int]] = []
    for start, end in _player_mention_spans(blob, player):
        intervals.append((max(0, start - radius), min(len(blob), end + radius)))
    if not intervals:
        return ""
    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return " … ".join(blob[a:b] for a, b in merged)


def _player_near_keywords(
    text: str,
    player: dict[str, Any],
    keywords: tuple[str, ...] | list[str],
    *,
    radius: int = 220,
) -> bool:
    """True if any keyword appears within radius of a player name mention."""
    blob = (text or "").lower()
    if not blob:
        return False
    keys = [k.lower() for k in keywords if k]
    if not keys:
        return False
    for start, end in _player_mention_spans(blob, player):
        lo = max(0, start - radius)
        hi = min(len(blob), end + radius)
        window = blob[lo:hi]
        if any(k in window for k in keys):
            return True
    return False


_INJURY_NEAR_KEYWORDS = (
    "injur",
    "hamstring",
    "groin",
    "knock",
    "surgery",
    "scan",
    "recover",
    "ankle",
    "calf",
    "knee",
    "thigh",
    "fracture",
)


def _soft_form_language(note: str, evidence: str = "") -> bool:
    blob = f"{note} {evidence}".lower()
    strong = (
        "hat-trick",
        "hattrick",
        "hat trick",
        "brace",
        "scored",
        "scores",
        "goals",
        "winner",
        "on target",
    )
    soft = ("lively", "sharp", "impressive", "debut")
    if any(k in blob for k in strong):
        return False
    return any(k in blob for k in soft)


def derive_preseason_from_news(news: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn strong form / confirmed transfer news into preseason role boosts."""
    from fpl_robo.signals.transfer_infer import infer_transfer_impact

    form_boost = float(SIGNAL_INGEST.get("form_signal_boost") or 0.08)
    out: list[dict[str, Any]] = []
    for sig in news:
        if not isinstance(sig, dict):
            continue
        kind = str(sig.get("type") or sig.get("kind") or "").lower()
        # Prefer the per-player note so multi-player headlines don't leak
        # "on target" / "hat-trick" onto teammates mentioned in the same article.
        blob = f"{sig.get('note') or ''}".lower()
        evidence_blob = f"{sig.get('evidence') or ''}".lower()
        is_hattrick = any(k in blob for k in ("hat-trick", "hattrick", "hat trick"))
        is_goals = is_hattrick or any(
            k in blob for k in ("brace", "scored", "scores", "goals", "winner", "on target")
        )
        is_soft_form = any(k in blob for k in ("lively", "sharp", "impressive", "debut"))
        # Fallback: if note is sparse, allow evidence only for single-player articles.
        if not is_hattrick and not is_goals and not is_soft_form and evidence_blob:
            is_hattrick = any(k in evidence_blob for k in ("hat-trick", "hattrick", "hat trick"))
            is_goals = is_hattrick or any(
                k in evidence_blob for k in ("brace", "scored", "scores", "goals", "winner", "on target")
            )
            is_soft_form = any(k in evidence_blob for k in ("lively", "sharp", "impressive", "debut"))
        if kind == "form" or (kind == "positive" and (is_goals or is_soft_form)):
            if is_hattrick:
                boost = 0.12
                minutes, starts, role = 90, 1, "starter"
            elif is_goals:
                boost = form_boost
                minutes, starts, role = 60, 1, "starter"
            else:
                # Soft form (lively/sharp): keep mild and do not claim starts.
                boost = max(0.03, form_boost - 0.04)
                minutes, starts, role = 30, 0, "rotation"
            out.append(
                {
                    "player_id": sig.get("player_id"),
                    "web_name": sig.get("web_name"),
                    "minutes": minutes,
                    "starts": starts,
                    "role": role,
                    "boost": round(boost, 3),
                    "note": f"auto from {kind}: {sig.get('note') or sig.get('evidence') or ''}"[:180],
                    "source": "auto",
                }
            )
        elif kind == "transfer":
            impact = infer_transfer_impact(
                team_short=str(sig.get("team_short") or "") or None,
                position=str(sig.get("position") or "") or None,
                element_type=sig.get("element_type"),
                evidence=str(sig.get("evidence") or ""),
                note=str(sig.get("note") or ""),
                strength_home=sig.get("team_strength_home"),
            )
            out.append(
                {
                    "player_id": sig.get("player_id"),
                    "web_name": sig.get("web_name"),
                    "minutes": impact["minutes"],
                    "starts": impact["starts"],
                    "role": impact["role"],
                    "boost": impact["boost"],
                    "note": (
                        f"auto from transfer ({impact['inference']}): "
                        f"{sig.get('note') or sig.get('evidence') or ''}"
                    )[:180],
                    "source": "auto",
                }
            )
    # Dedupe by player_id / web_name, keep highest boost.
    best: dict[tuple, dict[str, Any]] = {}
    for item in out:
        key = (item.get("player_id"), str(item.get("web_name") or "").lower())
        prev = best.get(key)
        if prev is None or float(item.get("boost") or 0) >= float(prev.get("boost") or 0):
            best[key] = item
    return list(best.values())


def merge_preseason_signals(
    existing: list[dict[str, Any]],
    proposed_auto: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep manual preseason rows; replace prior auto rows for touched players."""
    if not proposed_auto:
        return [item for item in existing if isinstance(item, dict)]

    touched_ids = {p.get("player_id") for p in proposed_auto if p.get("player_id") is not None}
    touched_names = {str(p.get("web_name") or "").strip().lower() for p in proposed_auto}
    keep: list[dict[str, Any]] = []
    for item in existing:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source") or "")
        if src in {"auto", "ollama", "heuristic"}:
            pid = item.get("player_id")
            name = str(item.get("web_name") or "").strip().lower()
            if (pid is not None and pid in touched_ids) or (name and name in touched_names):
                continue
            # Drop stale auto entirely when refreshing from news.
            continue
        keep.append(item)
    seen: set[tuple] = set()
    merged = list(keep)
    for item in proposed_auto:
        key = (item.get("player_id"), str(item.get("web_name") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def merge_lineup_signals(
    existing: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
    *,
    gameweek: int | None,
) -> list[dict[str, Any]]:
    """Keep manual lineups; replace auto rows for this GW (or all auto if GW unknown)."""
    keep: list[dict[str, Any]] = []
    for item in existing:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source") or "")
        if src.startswith("ollama") or src in {"heuristic", "auto"}:
            if gameweek is None or item.get("gameweek") == gameweek or item.get("gameweek") is None:
                continue
        keep.append(item)
    seen: set[tuple] = set()
    merged = list(keep)
    for item in proposed:
        key = (
            item.get("player_id"),
            str(item.get("web_name") or "").lower(),
            str(item.get("role") or "").lower(),
            item.get("gameweek"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _players_and_aliases(
    client: FPLClient | None = None,
) -> tuple[list[dict[str, Any]], dict[int, list[str]]]:
    """Build player rows from the KG plus team alias map from bootstrap."""
    client = client or FPLClient()
    bootstrap = client.bootstrap_static()
    fixtures = []
    try:
        fixtures = client.fixtures()
    except Exception:  # noqa: BLE001
        fixtures = []
    kg = build_knowledge_graph(bootstrap=bootstrap, fixtures=fixtures, client=client)
    players = kg.players()
    # Ensure team_name available when present on bootstrap teams.
    teams = list(bootstrap.get("teams") or [])
    team_name = {int(t["id"]): t.get("name") for t in teams if t.get("id") is not None}
    for p in players:
        tid = p.get("team_id")
        if tid is not None and not p.get("team_name"):
            p["team_name"] = team_name.get(int(tid))
    aliases = build_team_aliases(teams)
    strength_by_short = {
        str(t.get("short_name") or "").strip().upper(): t.get("strength_overall_home")
        for t in teams
        if t.get("short_name")
    }
    for p in players:
        short = str(p.get("team_short") or "").strip().upper()
        if short and short in strength_by_short:
            p["team_strength_home"] = strength_by_short[short]
    return players, aliases


def interpret_articles(
    articles: list[NewsArticle],
    indexes: dict[str, Any],
    team_aliases: dict[int, list[str]],
    *,
    model: str | None = None,
    base_url: str | None = None,
    min_confidence: float | None = None,
    extract_lineups: bool = False,
    gameweek: int | None = None,
    allow_llm: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proposed: list[dict[str, Any]] = []
    proposed_lineups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    min_conf = float(
        SIGNAL_INGEST["min_confidence"] if min_confidence is None else min_confidence
    )

    for art in articles:
        if not article_is_actionable(art):
            unresolved.append(
                {
                    "title": art.title,
                    "source": art.source,
                    "reason": "not_actionable_news",
                }
            )
            continue
        blob = art.text_blob()
        candidate_names = candidate_web_names_for_text(blob, indexes)
        candidate_players: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for name in candidate_names:
            resolved = resolve_player_name(name, indexes)
            if resolved is not None:
                pids = [int(resolved["id"])]
            else:
                key = normalize_name(name)
                pids = list(
                    dict.fromkeys(
                        (indexes["by_web"].get(name.lower()) or [])
                        + (indexes["by_norm"].get(key) or [])
                    )
                )
                if len(pids) > 1:
                    team_pids = [
                        pid
                        for pid in pids
                        if team_mentioned(blob, indexes["by_id"][pid], team_aliases)
                    ]
                    if team_pids:
                        pids = team_pids
            for pid in pids:
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                candidate_players.append(indexes["by_id"][pid])
        focused_players = prioritize_kg_candidates(candidate_players)
        candidate_lines = kg_candidate_lines(focused_players)

        if not candidate_lines:
            unresolved.append(
                {
                    "title": art.title,
                    "source": art.source,
                    "reason": "no_kg_player_mention",
                }
            )
            continue

        raw_signals = heuristic_signals_from_title(
            art.title,
            focused_players,
            summary=art.summary,
        )
        if art.body:
            body_signals = heuristic_signals_from_title(
                art.title,
                focused_players,
                summary=art.body[:900],
            )
            seen_ht = {(s.get("web_name"), s.get("type")) for s in raw_signals}
            for sig in body_signals:
                key = (sig.get("web_name"), sig.get("type"))
                if key in seen_ht:
                    continue
                raw_signals.append(sig)
                seen_ht.add(key)
        # On slow hosts (e.g. Pi CPU Ollama), skip LLM when heuristics already
        # extracted clear signals from the headline — or when LLM is disabled.
        skip_llm = (not allow_llm) or heuristics_are_sufficient(raw_signals, title=art.title)
        llm_summary = art.text_blob()
        if not skip_llm:
            try:
                llm_signals = interpret_article(
                    title=art.title,
                    summary=llm_summary,
                    source=art.source,
                    candidate_names=candidate_names,
                    candidate_lines=candidate_lines,
                    model=model,
                    base_url=base_url,
                )
                seen_ht = {(s.get("web_name"), s.get("type")) for s in raw_signals}
                for sig in llm_signals:
                    key = (sig.get("web_name"), sig.get("type"))
                    if key in seen_ht:
                        continue
                    raw_signals.append(sig)
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM failed on %s: %s", art.title[:80], exc)
                if not raw_signals:
                    unresolved.append(
                        {
                            "title": art.title,
                            "source": art.source,
                            "error": str(exc),
                        }
                    )
                    continue
        else:
            logger.info(
                "Heuristic hit on %s — skipping LLM (%d signal(s))",
                art.title[:80],
                len(raw_signals),
            )

        want_lineups = allow_llm and (extract_lineups or looks_like_lineup_article(art))
        if want_lineups:
            squad_players: list[dict[str, Any]] = []
            squad_ids: set[int] = set()
            for p in indexes["by_id"].values():
                if team_mentioned(blob, p, team_aliases):
                    pid = int(p["id"])
                    if pid in squad_ids:
                        continue
                    squad_ids.add(pid)
                    squad_players.append(p)
            lineup_candidates = prioritize_kg_candidates(
                squad_players or focused_players,
                limit=40,
            )
            try:
                raw_lineups = interpret_lineup_article(
                    title=art.title,
                    summary=art.summary,
                    source=art.source,
                    candidate_lines=kg_candidate_lines(lineup_candidates, limit=40),
                    gameweek=gameweek,
                    model=model,
                    base_url=base_url,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lineup LLM failed on %s: %s", art.title[:80], exc)
                raw_lineups = []

            for sig in raw_lineups:
                player = resolve_player_name(
                    str(sig.get("web_name") or ""),
                    indexes,
                    team_short=str(sig.get("team_short") or "") or None,
                    candidate_ids={int(p["id"]) for p in lineup_candidates} or None,
                )
                if player is None:
                    unresolved.append(
                        {
                            "web_name": sig.get("web_name"),
                            "team_short": sig.get("team_short"),
                            "role": sig.get("role"),
                            "title": art.title,
                            "reason": "unresolved_lineup_player",
                        }
                    )
                    continue
                llm_team = str(sig.get("team_short") or "").strip().upper()
                kg_team = str(player.get("team_short") or "").strip().upper()
                if llm_team and kg_team and llm_team != kg_team:
                    rejected.append(
                        {
                            "web_name": player.get("web_name"),
                            "player_id": int(player["id"]),
                            "role": sig.get("role"),
                            "title": art.title,
                            "reason": "lineup_team_mismatch",
                            "llm_team": llm_team,
                            "kg_team": kg_team,
                        }
                    )
                    continue
                conf = float(sig.get("confidence") or 0.55)
                if conf < min_conf:
                    rejected.append(
                        {
                            "web_name": player.get("web_name"),
                            "player_id": int(player["id"]),
                            "role": sig.get("role"),
                            "title": art.title,
                            "reason": "lineup_confidence_below_threshold",
                            "confidence": conf,
                        }
                    )
                    continue
                proposed_lineups.append(
                    {
                        "player_id": int(player["id"]),
                        "web_name": player.get("web_name"),
                        "first_name": player.get("first_name"),
                        "second_name": player.get("second_name"),
                        "team_short": player.get("team_short"),
                        "role": sig.get("role"),
                        "gameweek": gameweek,
                        "confidence": round(conf, 3),
                        "note": sig.get("note"),
                        "evidence": sig.get("evidence"),
                        "source": sig.get("source") or "ollama_lineup",
                        "article_title": art.title,
                        "article_link": art.link,
                        "article_source": art.source,
                    }
                )

        for sig in raw_signals:
            candidate_ids = {int(p["id"]) for p in candidate_players}
            player = resolve_player_name(
                str(sig.get("web_name") or ""),
                indexes,
                team_short=str(sig.get("team_short") or "") or None,
                candidate_ids=candidate_ids or None,
            )
            if player is None:
                player = resolve_player_name(
                    str(sig.get("web_name") or ""),
                    indexes,
                    team_short=str(sig.get("team_short") or "") or None,
                )
            if player is None:
                unresolved.append(
                    {
                        "web_name": sig.get("web_name"),
                        "team_short": sig.get("team_short"),
                        "type": sig.get("type"),
                        "note": sig.get("note"),
                        "title": art.title,
                        "reason": "unresolved_player",
                    }
                )
                continue

            llm_team = str(sig.get("team_short") or "").strip().upper()
            kg_team = str(player.get("team_short") or "").strip().upper()
            if llm_team and kg_team and llm_team != kg_team:
                rejected.append(
                    {
                        "web_name": player.get("web_name"),
                        "player_id": int(player["id"]),
                        "type": sig.get("type"),
                        "title": art.title,
                        "reason": "team_mismatch",
                        "llm_team": llm_team,
                        "kg_team": kg_team,
                    }
                )
                continue

            evidence = score_signal_evidence(
                article_text=blob,
                player=player,
                kind=str(sig.get("type") or ""),
                team_aliases=team_aliases,
                llm_note=str(sig.get("note") or ""),
            )
            if evidence["confidence"] < min_conf:
                evidence["accepted"] = False
                evidence["reject_reason"] = evidence.get("reject_reason") or "confidence_below_threshold"

            if not evidence["accepted"]:
                rejected.append(
                    {
                        "web_name": player.get("web_name"),
                        "player_id": int(player["id"]),
                        "type": sig.get("type"),
                        "title": art.title,
                        "reason": evidence.get("reject_reason"),
                        "confidence": evidence.get("confidence"),
                        "evidence_flags": {
                            k: evidence.get(k)
                            for k in (
                                "name_hit",
                                "team_hit",
                                "keyword_hit",
                                "proximity",
                                "status_align",
                            )
                        },
                    }
                )
                continue

            severity = float(sig.get("severity") or 0.0)
            kind = str(sig.get("type") or "").lower()
            note_l = str(sig.get("note") or "").lower()

            # Soft "misses/absent" upgraded only when injury language is near this player
            # (not a teammate elsewhere in the same article).
            if kind == "rotation" and _player_near_keywords(
                blob, player, _INJURY_NEAR_KEYWORDS, radius=220
            ):
                kind = "injury"
                severity = max(severity, 0.55)

            # Soft form (lively/sharp): cap confidence and severity.
            # Use the per-player note only — article windows can contain teammates' goals.
            if kind == "form" and _soft_form_language(note_l, ""):
                evidence["confidence"] = min(float(evidence.get("confidence") or 0.7), 0.72)
                severity = min(severity, 0.12)

            # Skip retrospective "after Club transfer" analysis pieces.
            if kind == "transfer" and "after" in note_l and "transfer" in note_l:
                rejected.append(
                    {
                        "web_name": player.get("web_name"),
                        "player_id": int(player["id"]),
                        "type": kind,
                        "title": art.title,
                        "reason": "retrospective_transfer",
                        "confidence": evidence.get("confidence"),
                    }
                )
                continue

            if not evidence.get("status_align") and evidence["confidence"] < 0.75:
                severity = min(severity, 0.55)
            exclude = bool(sig.get("exclude"))
            exclude_min = float(SIGNAL_INGEST.get("exclude_min_confidence") or 0.8)
            strong_evidence = bool(
                evidence.get("status_align")
                or (evidence.get("proximity") and evidence.get("team_hit"))
            )
            if exclude and (
                evidence["confidence"] < exclude_min or not strong_evidence
            ):
                exclude = False

            proposed.append(
                {
                    "player_id": int(player["id"]),
                    "web_name": player.get("web_name"),
                    "first_name": player.get("first_name"),
                    "second_name": player.get("second_name"),
                    "team_short": player.get("team_short"),
                    "element_type": player.get("element_type"),
                    "position": player.get("position"),
                    "team_strength_home": player.get("team_strength_home"),
                    "type": kind,
                    "severity": round(severity, 3),
                    "exclude": exclude,
                    "note": sig.get("note"),
                    "evidence": sig.get("evidence"),
                    "confidence": evidence["confidence"],
                    "evidence_flags": {
                        k: evidence.get(k)
                        for k in (
                            "name_hit",
                            "team_hit",
                            "keyword_hit",
                            "proximity",
                            "status_align",
                            "fpl_status",
                            "fpl_news",
                        )
                    },
                    "source": sig.get("source") or "ollama",
                    "article_title": art.title,
                    "article_link": art.link,
                    "article_source": art.source,
                }
            )
            # Attach transfer inference onto the note for dashboard / preseason.
            if kind == "transfer":
                from fpl_robo.signals.transfer_infer import infer_transfer_impact

                impact = infer_transfer_impact(
                    team_short=str(player.get("team_short") or "") or None,
                    position=str(player.get("position") or "") or None,
                    element_type=player.get("element_type"),
                    evidence=str(sig.get("evidence") or ""),
                    note=str(sig.get("note") or ""),
                    strength_home=player.get("team_strength_home"),
                )
                proposed[-1]["transfer_inference"] = impact
                proposed[-1]["note"] = (
                    f"{sig.get('note') or ''} · infer {impact['inference']} "
                    f"boost={impact['boost']}"
                ).strip(" ·")[:180]

    # Official FPL availability fills gaps headlines never mention.
    from fpl_robo.signals.fpl_status import signals_from_fpl_status

    fpl_seeded = signals_from_fpl_status(list(indexes["by_id"].values()))
    have_ids = {int(p["player_id"]) for p in proposed if p.get("player_id") is not None}
    for item in fpl_seeded:
        pid = int(item["player_id"])
        if pid in have_ids:
            # Prefer FPL exclude/status when we already have softer press signal.
            for prev in proposed:
                if int(prev.get("player_id") or -1) != pid:
                    continue
                if item.get("exclude") and not prev.get("exclude"):
                    prev["exclude"] = True
                    prev["severity"] = max(float(prev.get("severity") or 0), float(item["severity"]))
                    prev["note"] = f"{prev.get('note') or ''} · {item.get('note')}"[:180]
                    prev["evidence_flags"] = {
                        **(prev.get("evidence_flags") or {}),
                        **(item.get("evidence_flags") or {}),
                    }
                break
            continue
        proposed.append(item)
        have_ids.add(pid)

    return proposed, proposed_lineups, rejected, unresolved


def ingest_signals(
    *,
    dry_run: bool = True,
    apply: bool = False,
    signals_path: str | None = None,
    raw_cache_path: str | None = None,
    max_articles: int | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    client: FPLClient | None = None,
    skip_llm: bool = False,
    pre_deadline: bool = False,
    gameweek: int | None = None,
) -> IngestResult:
    """Scrape RSS, interpret with Ollama+KG validation, optionally write signals.json."""
    apply = bool(apply)
    dry_run = bool(dry_run) and not apply
    pre_deadline = bool(pre_deadline)
    signals_path = signals_path or SIGNAL_INGEST["signals_path"]
    raw_cache_path = raw_cache_path or SIGNAL_INGEST["raw_cache_path"]
    model = model or SIGNAL_INGEST["ollama_model"]
    ollama_url = ollama_url or SIGNAL_INGEST["ollama_url"]

    feeds = list(SIGNAL_INGEST["rss_feeds"])
    if pre_deadline:
        feeds = feeds + list(SIGNAL_INGEST.get("lineup_rss_feeds") or [])
        if max_articles is None:
            max_articles = int(SIGNAL_INGEST.get("pre_deadline_max_articles") or 12)

    articles, feed_errors = fetch_all_feeds(
        feeds=feeds,
        max_articles=max_articles,
        prefer_lineups=pre_deadline,
    )
    articles = enrich_articles_with_bodies(articles)
    raw_path = Path(raw_cache_path)
    atomic_write_json(raw_path, articles_payload(articles, feed_errors))

    if not articles:
        return IngestResult(
            ok=False,
            dry_run=dry_run,
            applied=False,
            articles=0,
            feed_errors=feed_errors,
            signals_path=signals_path,
            raw_cache_path=str(raw_path),
            reason="No articles fetched",
            ollama_model=model,
            pre_deadline=pre_deadline,
            gameweek=gameweek,
        )

    use_llm = (not skip_llm) and ollama_available(ollama_url)
    mode_bits = ["heuristic", "fpl"]
    if use_llm:
        mode_bits.append("ollama")
    elif skip_llm:
        logger.info("Skipping LLM (--skip-llm); applying heuristics + FPL status seeds")
    else:
        logger.warning(
            "Ollama not reachable at %s — applying heuristics + FPL status seeds only",
            ollama_url,
        )
    mode_bits.append("bodies")
    ingest_model = "+".join(mode_bits)

    players, team_aliases = _players_and_aliases(client)
    indexes = build_player_indexes(players)
    proposed, proposed_lineups, rejected, unresolved = interpret_articles(
        articles,
        indexes,
        team_aliases,
        model=model,
        base_url=ollama_url,
        extract_lineups=pre_deadline and use_llm,
        gameweek=gameweek,
        allow_llm=use_llm,
    )

    result = IngestResult(
        ok=True,
        dry_run=dry_run,
        applied=False,
        articles=len(articles),
        proposed_news=proposed,
        proposed_lineups=proposed_lineups,
        rejected=rejected,
        unresolved=unresolved,
        feed_errors=feed_errors,
        signals_path=signals_path,
        raw_cache_path=str(raw_path),
        ollama_model=ingest_model if not use_llm else model,
        pre_deadline=pre_deadline,
        gameweek=gameweek,
    )

    if dry_run or not apply:
        result.reason = (
            f"Dry-run only; news={len(proposed)} lineups={len(proposed_lineups)} "
            f"rejected={len(rejected)} model={ingest_model} (signals.json not modified)"
        )
        return result

    existing = _load_existing_signals(Path(signals_path))
    merged_news = merge_news_signals(
        [x for x in existing.get("news") or [] if isinstance(x, dict)],
        proposed,
    )
    merged_lineups = merge_lineup_signals(
        [x for x in existing.get("predicted_lineups") or [] if isinstance(x, dict)],
        proposed_lineups,
        gameweek=gameweek,
    )
    auto_preseason = derive_preseason_from_news(proposed)
    merged_preseason = merge_preseason_signals(
        [x for x in existing.get("preseason") or [] if isinstance(x, dict)],
        auto_preseason,
    )
    payload = {
        "as_of": date.today().isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "news": merged_news,
        "preseason": merged_preseason,
        "international": existing.get("international") or [],
        "predicted_lineups": merged_lineups,
        "ingest": {
            "articles": len(articles),
            "proposed": len(proposed),
            "proposed_lineups": len(proposed_lineups),
            "proposed_preseason": len(auto_preseason),
            "rejected": len(rejected),
            "unresolved": len(unresolved),
            "model": ingest_model,
            "ollama": use_llm,
            "min_confidence": SIGNAL_INGEST.get("min_confidence"),
            "feed_errors": feed_errors,
            "kg_validated": True,
            "pre_deadline": pre_deadline,
            "gameweek": gameweek,
            "bodies": sum(1 for a in articles if getattr(a, "body", "")),
        },
    }
    atomic_write_json(signals_path, payload)
    result.applied = True
    result.dry_run = False
    result.reason = (
        f"Wrote {len(proposed)} news + {len(auto_preseason)} preseason + "
        f"{len(proposed_lineups)} predicted lineup signals to {signals_path} "
        f"({ingest_model})"
    )
    return result
