"""Scrape news + interpret with local LLM into signals.json."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import SIGNAL_INGEST
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.runtime.atomic import atomic_write_json
from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text, heuristic_signals_from_title
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
    looks_like_minutes_wording,
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
    content_changed: bool = True

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
            "content_changed": self.content_changed,
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


def lineup_heuristics_are_sufficient(
    heuristic_rows: list[dict[str, Any]],
    *,
    title: str,
    owned_ids: set[int],
    strong_xi_article: bool,
) -> bool:
    """True when offline lineup heuristics are enough to skip the lineup LLM."""
    if not heuristic_rows:
        return False
    if not bool(SIGNAL_INGEST.get("skip_lineup_llm_when_heuristic", True)):
        return False
    title_l = (title or "").lower()
    # Roundups / predicted-XI pieces can bury extra names — keep LLM unless we
    # already have several starter rows.
    strong_xi = strong_xi_article or any(
        k in title_l
        for k in (
            "starting xi",
            "predicted xi",
            "predicted lineup",
            "predicted line-up",
            "likely xi",
            "expected xi",
            "confirmed xi",
            "team sheet",
        )
    )
    starters = [r for r in heuristic_rows if str(r.get("role") or "") == "start"]
    if strong_xi and len(starters) < 3:
        return False
    if owned_ids:
        owned_covered = [
            r
            for r in heuristic_rows
            if r.get("player_id") is not None and int(r["player_id"]) in owned_ids
        ]
        if owned_covered:
            return True
        # Owned-club article with no owned-player hit yet — let LLM try.
        return False
    return not strong_xi


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
                destination_hint=str(sig.get("destination_hint") or "") or None,
            )
            # Outbound exits must not become positive preseason role boosts.
            if impact.get("outbound") or impact.get("exclude"):
                continue
            out.append(
                {
                    "player_id": sig.get("player_id"),
                    "web_name": sig.get("web_name"),
                    "minutes": impact["minutes"],
                    "starts": impact["starts"],
                    "role": impact["role"],
                    "boost": impact["boost"],
                    "predicted_minutes": impact.get("predicted_minutes"),
                    "destination_team_short": impact.get("destination_team_short"),
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


def derive_lineups_from_transfers(
    news: list[dict[str, Any]],
    *,
    gameweek: int | None = None,
    elements: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Turn inbound transfer news into predicted_minutes lineup rows.

    Depth signings (veteran / backup language / behind a club starter) get low
    minutes so ep_baseline cannot treat them as 90m options.
    """
    from fpl_robo.signals.minutes_estimate import (
        club_depth_minutes,
        destination_to_team_short,
    )
    from fpl_robo.signals.transfer_infer import infer_transfer_impact

    out: list[dict[str, Any]] = []
    for sig in news:
        if not isinstance(sig, dict):
            continue
        kind = str(sig.get("type") or sig.get("kind") or "").lower()
        if kind != "transfer" or sig.get("exclude") or sig.get("outbound"):
            continue
        pid = sig.get("player_id")
        if pid is None:
            continue
        impact = sig.get("transfer_inference")
        if not isinstance(impact, dict):
            impact = infer_transfer_impact(
                team_short=str(sig.get("team_short") or "") or None,
                position=str(sig.get("position") or "") or None,
                element_type=sig.get("element_type"),
                evidence=str(sig.get("evidence") or ""),
                note=str(sig.get("note") or ""),
                strength_home=sig.get("team_strength_home"),
                destination_hint=str(sig.get("destination_hint") or "") or None,
            )
        if impact.get("outbound") or impact.get("exclude"):
            continue

        dest_short = (
            impact.get("destination_team_short")
            or destination_to_team_short(impact.get("destination_hint") or sig.get("destination_hint"))
        )
        pred = impact.get("predicted_minutes")
        if pred is None:
            pred = impact.get("minutes") or 35
        try:
            pred_i = max(0, min(90, int(pred)))
        except (TypeError, ValueError):
            pred_i = 35
        role = str(impact.get("role") or "rotation")
        lineup_role = {
            "starter": "start",
            "start": "start",
            "rotation": "doubt",
            "backup": "bench",
            "unavailable": "bench",
        }.get(role, "doubt")

        # If we know the destination pecking order, prefer depth model minutes.
        if elements and dest_short:
            # Annotate elements with team_short for matching when missing.
            depth = club_depth_minutes(
                {
                    "id": int(pid),
                    "element_type": sig.get("element_type"),
                    "team_short": dest_short,
                    "web_name": sig.get("web_name"),
                },
                elements=_elements_with_team_shorts(elements),
                team_short=dest_short,
                element_type=sig.get("element_type"),
            )
            if depth is not None:
                d_mins, d_role, _d_conf, d_why = depth
                # Take the more conservative (lower) of transfer prior and depth.
                if d_mins < pred_i:
                    pred_i = d_mins
                    lineup_role = d_role
                    role = d_role
                    impact = dict(impact)
                    impact["inference"] = f"{impact.get('inference')}; {d_why}"

        conf = float(SIGNAL_INGEST.get("transfer_lineup_confidence") or 0.78)
        out.append(
            {
                "player_id": int(pid),
                "web_name": sig.get("web_name"),
                "first_name": sig.get("first_name"),
                "second_name": sig.get("second_name"),
                "team_short": dest_short or sig.get("team_short"),
                "role": lineup_role,
                "predicted_minutes": pred_i,
                "confidence": conf,
                "gameweek": gameweek,
                "evidence": (sig.get("evidence") or "")[:220],
                "note": (
                    f"[transfer_lineup] {lineup_role} (~{pred_i}m): "
                    f"{sig.get('web_name')} ({impact.get('inference')})"
                )[:180],
                "source": "transfer_lineup",
                "article_title": sig.get("article_title"),
                "article_link": sig.get("article_link"),
                "article_source": sig.get("article_source"),
            }
        )

    # Dedupe by player, keep lowest minutes (more conservative on depth).
    best: dict[int, dict[str, Any]] = {}
    for row in out:
        pid = int(row["player_id"])
        prev = best.get(pid)
        if prev is None or int(row["predicted_minutes"]) <= int(prev["predicted_minutes"]):
            best[pid] = row
    return list(best.values())


def _annotate_elements_team_short(
    elements: list[dict[str, Any]],
    teams: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Copy bootstrap elements with team_short filled from teams list."""
    id_to_short = {
        int(t["id"]): str(t.get("short_name") or "").strip().upper()
        for t in (teams or [])
        if t.get("id") is not None and t.get("short_name")
    }
    out: list[dict[str, Any]] = []
    for e in elements or []:
        if not isinstance(e, dict) or e.get("id") is None:
            continue
        row = dict(e)
        if not row.get("team_short"):
            tid = row.get("team") if row.get("team") is not None else row.get("team_id")
            try:
                if tid is not None and int(tid) in id_to_short:
                    row["team_short"] = id_to_short[int(tid)]
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


def _elements_with_team_shorts(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure bootstrap elements expose team_short for depth matching."""
    if elements and all(
        (not isinstance(e, dict)) or e.get("team_short") for e in elements
    ):
        return elements
    # Build id→short from any rows that already have both.
    id_to_short: dict[int, str] = {}
    for e in elements:
        if not isinstance(e, dict):
            continue
        tid = e.get("team") if e.get("team") is not None else e.get("team_id")
        short = str(e.get("team_short") or "").strip().upper()
        if tid is not None and short:
            try:
                id_to_short[int(tid)] = short
            except (TypeError, ValueError):
                pass
    if not id_to_short:
        return elements
    out: list[dict[str, Any]] = []
    for e in elements:
        if not isinstance(e, dict):
            continue
        if e.get("team_short"):
            out.append(e)
            continue
        tid = e.get("team") if e.get("team") is not None else e.get("team_id")
        row = dict(e)
        try:
            if tid is not None and int(tid) in id_to_short:
                row["team_short"] = id_to_short[int(tid)]
        except (TypeError, ValueError):
            pass
        out.append(row)
    return out


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
        if src.startswith("ollama") or src in {
            "heuristic",
            "auto",
            "fpl_implied",
            "heuristic_lineup",
            "heuristic_mention",
            "transfer_lineup",
            "depth_model",
        }:
            if gameweek is None or item.get("gameweek") == gameweek or item.get("gameweek") is None:
                continue
        keep.append(item)

    def _quality(row: dict[str, Any]) -> tuple:
        src = str(row.get("source") or "")
        # Prefer transfer/depth priors over soft article heuristics when they conflict.
        src_rank = {
            "transfer_lineup": 3,
            "ollama_lineup": 3,
            "depth_model": 2,
            "heuristic_lineup": 1,
            "heuristic_mention": 0,
            "fpl_implied": 0,
        }.get(src, 1)
        conf = float(row.get("confidence") or 0)
        has_mins = 1 if row.get("predicted_minutes") is not None else 0
        # Among equals, prefer the more conservative minutes estimate.
        try:
            mins = int(row.get("predicted_minutes"))
            cons = -mins
        except (TypeError, ValueError):
            cons = -999
        return (has_mins, src_rank, conf, cons)

    best_by_pid: dict[int, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    for item in proposed:
        if not isinstance(item, dict):
            continue
        pid = item.get("player_id")
        if pid is None:
            extras.append(item)
            continue
        pid_i = int(pid)
        prev = best_by_pid.get(pid_i)
        if prev is None or _quality(item) >= _quality(prev):
            best_by_pid[pid_i] = item

    seen: set[tuple] = set()
    merged = list(keep)
    for item in list(best_by_pid.values()) + extras:
        key = (
            item.get("player_id"),
            str(item.get("web_name") or "").lower(),
            str(item.get("role") or "").lower(),
            item.get("gameweek"),
            str(item.get("source") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _load_owned_squad(
    indexes: dict[str, Any],
    *,
    client: FPLClient | None = None,
) -> tuple[set[int], set[str], list[dict[str, Any]]]:
    """Return (owned_ids, owned_team_shorts, owned_player_rows) from live/public picks."""
    owned_ids: set[int] = set()
    owned_teams: set[str] = set()
    owned_players: list[dict[str, Any]] = []
    try:
        from fpl_robo.config import DEFAULT_MANAGER_ID, DEFAULT_SESSION_PATH
        from fpl_robo.data.live_team import load_live_team_state

        client = client or FPLClient()
        # my-team needs an authenticated session; public entry picks are empty
        # until the first deadline / published GW squad.
        try:
            client.load_session(DEFAULT_SESSION_PATH)
        except Exception:  # noqa: BLE001
            pass

        live = load_live_team_state(manager_id=DEFAULT_MANAGER_ID, client=client)
        pick_ids = [
            int(p["element"])
            for p in (live.picks or [])
            if p.get("element") is not None
        ]
        # Fallback: last optimiser squad if auth/public picks are empty.
        if not pick_ids:
            summary_path = Path("data/last_run_summary.json")
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    squad = (
                        ((summary.get("transfer_plan") or {}).get("solution") or {}).get(
                            "squad"
                        )
                        or []
                    )
                    pick_ids = [
                        int(p["id"])
                        for p in squad
                        if isinstance(p, dict) and p.get("id") is not None
                    ]
                    if pick_ids:
                        logger.info(
                            "Owned squad from last_run_summary (%d players); live picks empty",
                            len(pick_ids),
                        )
                except Exception:  # noqa: BLE001
                    pick_ids = []
        for pid_i in pick_ids:
            owned_ids.add(pid_i)
            row = indexes["by_id"].get(pid_i)
            if row is None:
                continue
            owned_players.append(row)
            short = str(row.get("team_short") or "").strip().upper()
            if short:
                owned_teams.add(short)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load owned squad for lineup minutes check: %s", exc)
    return owned_ids, owned_teams, owned_players


def article_mentions_owned_player(
    text: str,
    owned_players: list[dict[str, Any]],
) -> bool:
    """True if article text mentions an owned player's web/second name."""
    blob = text or ""
    if not blob or not owned_players:
        return False
    lower = blob.lower()
    for p in owned_players:
        for key in ("web_name", "second_name"):
            name = str(p.get(key) or "").strip()
            if len(name) < 3:
                continue
            if name.lower() in lower:
                return True
    return False


def _team_mentioned_in_text(
    text: str,
    team_short: str | None,
    team_aliases: dict[int, list[str]],
    *,
    team_id: int | None = None,
    team_name: str | None = None,
) -> bool:
    blob = (text or "").lower()
    short = str(team_short or "").strip().upper()
    if short and re.search(rf"(?<![a-z0-9]){re.escape(short.lower())}(?![a-z0-9])", blob):
        return True
    name = str(team_name or "").strip().lower()
    if name and len(name) >= 4 and name in blob:
        return True
    if team_id is not None:
        for alias in team_aliases.get(int(team_id), []) or []:
            a = str(alias).strip().lower()
            if not a:
                continue
            if len(a) <= 3:
                if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", blob):
                    return True
            elif a in blob:
                return True
    return False


def article_mentions_owned_club(
    text: str,
    owned_team_shorts: set[str],
    team_aliases: dict[int, list[str]],
    *,
    players_by_id: dict[int, dict[str, Any]] | None = None,
) -> bool:
    """True if article text mentions any owned-squad club."""
    if not owned_team_shorts:
        return False
    blob = (text or "").lower()
    alias_by_short: dict[str, set[str]] = {s: {s.lower()} for s in owned_team_shorts}
    if players_by_id:
        for p in players_by_id.values():
            short = str(p.get("team_short") or "").strip().upper()
            if short not in owned_team_shorts:
                continue
            tid = p.get("team_id")
            if tid is not None:
                for alias in team_aliases.get(int(tid), []) or []:
                    a = str(alias).strip().lower()
                    if a:
                        alias_by_short[short].add(a)
            name = str(p.get("team_name") or "").strip().lower()
            if name:
                alias_by_short[short].add(name)
    for _short, aliases in alias_by_short.items():
        for alias in aliases:
            if not alias:
                continue
            if len(alias) <= 3:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", blob):
                    return True
            elif alias in blob:
                return True
    return False


def prefer_owned_club_articles(
    articles: list[NewsArticle],
    owned_team_shorts: set[str],
    team_aliases: dict[int, list[str]],
    *,
    players_by_id: dict[int, dict[str, Any]] | None = None,
    owned_players: list[dict[str, Any]] | None = None,
) -> list[NewsArticle]:
    """Stable-sort articles so owned-club / minutes wording ranks first."""
    if not articles:
        return articles
    owned_players = owned_players or []

    def key(art: NewsArticle) -> tuple:
        blob = art.text_blob()
        hit = article_mentions_owned_club(
            blob,
            owned_team_shorts,
            team_aliases,
            players_by_id=players_by_id,
        ) or article_mentions_owned_player(blob, owned_players)
        minutesish = looks_like_lineup_article(art) or looks_like_minutes_wording(blob)
        return (0 if hit else 1, 0 if minutesish else 1)

    return sorted(articles, key=key)


def build_squad_minutes_check(
    owned_players: list[dict[str, Any]],
    lineups: list[dict[str, Any]],
    *,
    gameweek: int | None = None,
) -> dict[str, Any]:
    """Summarise predicted minutes/role coverage for the owned 15."""
    by_pid: dict[int, dict[str, Any]] = {}
    for row in lineups:
        if not isinstance(row, dict):
            continue
        pid = row.get("player_id")
        if pid is None:
            continue
        pid_i = int(pid)
        if gameweek is not None and row.get("gameweek") not in (None, gameweek):
            continue
        prev = by_pid.get(pid_i)
        if prev is None:
            by_pid[pid_i] = row
            continue
        # Prefer rows with minutes, then higher confidence, then non-fallback.
        def rank(r: dict[str, Any]) -> tuple:
            return (
                1 if r.get("predicted_minutes") is not None else 0,
                0 if str(r.get("source") or "") == "fpl_implied" else 1,
                float(r.get("confidence") or 0),
            )

        if rank(row) >= rank(prev):
            by_pid[pid_i] = row

    players_out: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for p in owned_players:
        pid = int(p["id"])
        hit = by_pid.get(pid)
        if hit is None or hit.get("predicted_minutes") is None:
            missing.append(
                {
                    "player_id": pid,
                    "web_name": p.get("web_name"),
                    "team_short": p.get("team_short"),
                }
            )
            players_out.append(
                {
                    "player_id": pid,
                    "web_name": p.get("web_name"),
                    "team_short": p.get("team_short"),
                    "covered": False,
                    "role": None,
                    "predicted_minutes": None,
                    "confidence": None,
                    "source": None,
                }
            )
            continue
        players_out.append(
            {
                "player_id": pid,
                "web_name": p.get("web_name"),
                "team_short": p.get("team_short"),
                "covered": True,
                "role": hit.get("role"),
                "predicted_minutes": hit.get("predicted_minutes"),
                "confidence": hit.get("confidence"),
                "source": hit.get("source"),
                "note": hit.get("note") or "",
            }
        )
    covered_n = sum(1 for r in players_out if r.get("covered"))
    news_n = sum(
        1
        for r in players_out
        if r.get("covered")
        and str(r.get("source") or "")
        not in {"fpl_implied", "depth_model"}
    )
    proxy_n = sum(
        1
        for r in players_out
        if r.get("covered") and str(r.get("source") or "") in {"fpl_implied", "depth_model"}
    )
    return {
        "gameweek": gameweek,
        "owned": len(owned_players),
        "covered": covered_n,
        "news_covered": news_n,
        "fpl_implied_covered": proxy_n,
        "missing_count": len(missing),
        "missing": missing,
        "players": players_out,
        "complete": covered_n == len(owned_players) and len(owned_players) > 0,
    }


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
    el_by_id = {
        int(el["id"]): el
        for el in (bootstrap.get("elements") or [])
        if el.get("id") is not None
    }
    for p in players:
        short = str(p.get("team_short") or "").strip().upper()
        if short and short in strength_by_short:
            p["team_strength_home"] = strength_by_short[short]
        el = el_by_id.get(int(p["id"])) if p.get("id") is not None else None
        if not el:
            continue
        for key in (
            "ep_next",
            "ep_this",
            "minutes",
            "starts",
            "status",
            "chance_of_playing_next_round",
            "news",
        ):
            if p.get(key) is None and el.get(key) is not None:
                p[key] = el.get(key)
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
    owned_ids: set[int] | None = None,
    owned_team_shorts: set[str] | None = None,
    owned_lineup_on_daily: bool | None = None,
    owned_players: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proposed: list[dict[str, Any]] = []
    proposed_lineups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    min_conf = float(
        SIGNAL_INGEST["min_confidence"] if min_confidence is None else min_confidence
    )
    owned_ids = owned_ids or set()
    owned_team_shorts = owned_team_shorts or set()
    owned_players = owned_players or []
    if owned_lineup_on_daily is None:
        owned_lineup_on_daily = bool(SIGNAL_INGEST.get("owned_lineup_on_daily", True))
    if extract_lineups:
        lineup_llm_budget = int(
            SIGNAL_INGEST.get("lineup_llm_max_articles_pre_deadline") or 12
        )
    else:
        lineup_llm_budget = int(SIGNAL_INGEST.get("lineup_llm_max_articles") or 8)
    lineup_llm_used = 0

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
                    # Last-chance heuristic pass over whatever text we have.
                    fallback_text = (art.body or art.summary or "")[:1200]
                    raw_signals = heuristic_signals_from_title(
                        art.title,
                        focused_players,
                        summary=fallback_text,
                    )
                    if raw_signals:
                        logger.info(
                            "Heuristic recovery after LLM failure on %s (%d signal(s))",
                            art.title[:80],
                            len(raw_signals),
                        )
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

        club_mentioned = article_mentions_owned_club(
            blob,
            owned_team_shorts,
            team_aliases,
            players_by_id=indexes["by_id"],
        )
        player_mentioned = article_mentions_owned_player(blob, owned_players)
        lineup_article = looks_like_lineup_article(art)
        minutes_wording = looks_like_minutes_wording(blob) or lineup_article
        # Pre-deadline: cast a wider net. Daily: require XI/minutes language —
        # bare club mentions alone were burning sequential 180s LLM calls on gossip.
        if extract_lineups:
            want_lineups = (
                lineup_article
                or minutes_wording
                or club_mentioned
                or player_mentioned
            )
        elif owned_lineup_on_daily:
            want_lineups = lineup_article or (
                minutes_wording and (club_mentioned or player_mentioned)
            )
        else:
            want_lineups = lineup_article
        if want_lineups:
            squad_players: list[dict[str, Any]] = []
            squad_ids: set[int] = set()
            for p in indexes["by_id"].values():
                if team_mentioned(blob, p, team_aliases) or int(p["id"]) in owned_ids:
                    # Only add owned players if their name appears, or any player from mentioned clubs.
                    pid = int(p["id"])
                    if pid in owned_ids and not article_mentions_owned_player(blob, [p]):
                        if not team_mentioned(blob, p, team_aliases):
                            continue
                    if pid in squad_ids:
                        continue
                    if team_mentioned(blob, p, team_aliases) or (
                        pid in owned_ids and article_mentions_owned_player(blob, [p])
                    ):
                        squad_ids.add(pid)
                        squad_players.append(p)
            if owned_ids and squad_players:
                owned_first = [p for p in squad_players if int(p["id"]) in owned_ids]
                rest = [p for p in squad_players if int(p["id"]) not in owned_ids]
                squad_players = owned_first + rest
            # Named owned players must survive the candidate cap (otherwise club-wide
            # alias hits like "United" push Sarr/Mbeumo out of the top 40).
            named_owned = [
                p
                for p in (owned_players or [])
                if int(p["id"]) in owned_ids
                and article_mentions_owned_player(blob, [p])
            ]
            lineup_candidates = prioritize_kg_candidates(
                squad_players or focused_players,
                limit=40,
            )
            if named_owned:
                keep_ids = {int(p["id"]) for p in named_owned}
                lineup_candidates = named_owned + [
                    p for p in lineup_candidates if int(p["id"]) not in keep_ids
                ]
            lineup_summary = art.text_blob()

            # Deterministic role/minutes extraction first (works offline / when LLM times out).
            heuristic_rows = heuristic_lineup_signals_from_text(
                lineup_summary,
                lineup_candidates or owned_players or focused_players,
                prefer_ids=owned_ids,
                source="heuristic_lineup",
                gameweek=gameweek,
                article_title=art.title,
            )
            for row in heuristic_rows:
                row = dict(row)
                row["article_link"] = art.link
                row["article_source"] = art.source
                if float(row.get("confidence") or 0) < min_conf:
                    continue
                proposed_lineups.append(row)

            raw_lineups: list[dict[str, Any]] = []
            skip_lineup_llm = (not allow_llm) or lineup_heuristics_are_sufficient(
                heuristic_rows,
                title=art.title,
                owned_ids=owned_ids,
                strong_xi_article=lineup_article,
            )
            if allow_llm and not skip_lineup_llm and lineup_llm_used >= lineup_llm_budget:
                skip_lineup_llm = True
                logger.info(
                    "Lineup LLM budget exhausted (%d) — heuristics only for %s",
                    lineup_llm_budget,
                    art.title[:80],
                )
            if allow_llm and not skip_lineup_llm:
                try:
                    raw_lineups = interpret_lineup_article(
                        title=art.title,
                        summary=lineup_summary[:1400],
                        source=art.source,
                        candidate_lines=kg_candidate_lines(lineup_candidates, limit=24),
                        gameweek=gameweek,
                        model=model,
                        base_url=base_url,
                    )
                    lineup_llm_used += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Lineup LLM failed on %s: %s", art.title[:80], exc)
                    raw_lineups = []
            elif allow_llm and heuristic_rows:
                logger.info(
                    "Heuristic lineup hit on %s — skipping lineup LLM (%d row(s))",
                    art.title[:80],
                    len(heuristic_rows),
                )

            heuristic_pids = {
                int(r["player_id"])
                for r in heuristic_rows
                if r.get("player_id") is not None
            }
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
                pid = int(player["id"])
                # Skip LLM duplicate when heuristic already covered this owned/player.
                if pid in heuristic_pids:
                    continue
                llm_team = str(sig.get("team_short") or "").strip().upper()
                kg_team = str(player.get("team_short") or "").strip().upper()
                if llm_team and kg_team and llm_team != kg_team:
                    rejected.append(
                        {
                            "web_name": player.get("web_name"),
                            "player_id": pid,
                            "role": sig.get("role"),
                            "title": art.title,
                            "reason": "lineup_team_mismatch",
                            "llm_team": llm_team,
                            "kg_team": kg_team,
                        }
                    )
                    continue
                # Reject hallucinations: player's club must appear in the article
                # (unless they are owned and named in the text).
                team_ok = _team_mentioned_in_text(
                    blob,
                    player.get("team_short"),
                    team_aliases,
                    team_id=player.get("team_id"),
                    team_name=player.get("team_name"),
                )
                named = article_mentions_owned_player(blob, [player])
                if not team_ok and not (pid in owned_ids and named):
                    rejected.append(
                        {
                            "web_name": player.get("web_name"),
                            "player_id": pid,
                            "role": sig.get("role"),
                            "title": art.title,
                            "reason": "lineup_team_not_in_article",
                            "kg_team": kg_team,
                        }
                    )
                    continue
                conf = float(sig.get("confidence") or 0.55)
                if conf < min_conf:
                    rejected.append(
                        {
                            "web_name": player.get("web_name"),
                            "player_id": pid,
                            "role": sig.get("role"),
                            "title": art.title,
                            "reason": "lineup_confidence_below_threshold",
                            "confidence": conf,
                        }
                    )
                    continue
                row = {
                    "player_id": pid,
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
                mins = sig.get("predicted_minutes")
                if mins is not None:
                    try:
                        row["predicted_minutes"] = max(0, min(90, int(mins)))
                    except (TypeError, ValueError):
                        pass
                proposed_lineups.append(row)

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

            # Ollama-only availability calls need proximity or FPL status alignment.
            src = str(sig.get("source") or "ollama")
            if (
                bool(SIGNAL_INGEST.get("require_ollama_injury_proximity_or_status", True))
                and src.startswith("ollama")
                and kind in {"injury", "suspension", "delayed_return"}
                and not evidence.get("status_align")
                and not evidence.get("proximity")
            ):
                rejected.append(
                    {
                        "web_name": player.get("web_name"),
                        "player_id": int(player["id"]),
                        "type": kind,
                        "title": art.title,
                        "reason": "ollama_injury_weak_evidence",
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
                    destination_hint=str(sig.get("destination_hint") or "") or None,
                )
                proposed[-1]["transfer_inference"] = impact
                proposed[-1]["destination_hint"] = sig.get("destination_hint")
                if impact.get("outbound") or impact.get("exclude") or sig.get("outbound"):
                    proposed[-1]["exclude"] = True
                    proposed[-1]["outbound"] = True
                    proposed[-1]["severity"] = max(
                        float(proposed[-1].get("severity") or 0), 0.9
                    )
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
    owned_ids, owned_team_shorts, owned_players = _load_owned_squad(
        indexes, client=client
    )
    articles = prefer_owned_club_articles(
        articles,
        owned_team_shorts,
        team_aliases,
        players_by_id=indexes["by_id"],
        owned_players=owned_players,
    )
    owned_lineup_on_daily = bool(SIGNAL_INGEST.get("owned_lineup_on_daily", True))
    proposed, proposed_lineups, rejected, unresolved = interpret_articles(
        articles,
        indexes,
        team_aliases,
        model=model,
        base_url=ollama_url,
        extract_lineups=pre_deadline and use_llm,
        gameweek=gameweek,
        allow_llm=use_llm,
        owned_ids=owned_ids,
        owned_team_shorts=owned_team_shorts,
        owned_lineup_on_daily=owned_lineup_on_daily,
        owned_players=owned_players,
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
    # Transfer → predicted minutes (depth signings, destination club).
    bootstrap = None
    annotated_elements: list[dict[str, Any]] = []
    try:
        bootstrap = (client or FPLClient()).bootstrap_static()
        annotated_elements = _annotate_elements_team_short(
            list(bootstrap.get("elements") or []),
            list(bootstrap.get("teams") or []),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load bootstrap for transfer/depth minutes: %s", exc)

    transfer_lineups = derive_lineups_from_transfers(
        merged_news,
        gameweek=gameweek,
        elements=annotated_elements or None,
    )
    proposed_lineups = list(proposed_lineups) + transfer_lineups
    merged_lineups = merge_lineup_signals(
        [x for x in existing.get("predicted_lineups") or [] if isinstance(x, dict)],
        proposed_lineups,
        gameweek=gameweek,
    )
    transfer_destinations: dict[int, str] = {}
    for row in transfer_lineups:
        pid = row.get("player_id")
        short = str(row.get("team_short") or "").strip().upper()
        if pid is not None and short:
            transfer_destinations[int(pid)] = short
    if owned_players and bool(SIGNAL_INGEST.get("require_full_squad_minutes", True)):
        from fpl_robo.signals.minutes_estimate import ensure_owned_minutes_coverage

        elements_by_id: dict[int, dict[str, Any]] = {
            int(el["id"]): el for el in annotated_elements if el.get("id") is not None
        }
        merged_lineups = ensure_owned_minutes_coverage(
            owned_players,
            merged_lineups,
            elements_by_id=elements_by_id or None,
            elements=annotated_elements or None,
            gameweek=gameweek,
            transfer_destinations=transfer_destinations or None,
        )
    auto_preseason = derive_preseason_from_news(proposed)
    merged_preseason = merge_preseason_signals(
        [x for x in existing.get("preseason") or [] if isinstance(x, dict)],
        auto_preseason,
    )
    from fpl_robo.runtime.cadence import signals_content_fingerprint

    content_payload = {
        "news": merged_news,
        "preseason": merged_preseason,
        "international": existing.get("international") or [],
        "predicted_lineups": merged_lineups,
    }
    new_fp = signals_content_fingerprint(content_payload)
    old_fp = signals_content_fingerprint(existing)
    squad_minutes_check = build_squad_minutes_check(
        owned_players,
        merged_lineups,
        gameweek=gameweek,
    )
    if (
        old_fp is not None
        and new_fp is not None
        and old_fp == new_fp
        and (existing.get("news") or existing.get("preseason") or existing.get("predicted_lineups"))
    ):
        # Still rewrite when squad minutes coverage metadata was incomplete.
        prev_check = (existing.get("ingest") or {}).get("squad_minutes_check") or {}
        need_check_write = bool(squad_minutes_check.get("complete")) and not bool(
            prev_check.get("complete")
        )
        if not need_check_write and squad_minutes_check.get("covered") == prev_check.get(
            "covered"
        ):
            result.applied = False
            result.content_changed = False
            result.dry_run = False
            result.reason = (
                f"Signals content unchanged (fp={new_fp}); skipped rewrite of {signals_path}"
            )
            logger.info(result.reason)
            return result

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
            "content_fingerprint": new_fp,
            "owned_squad": len(owned_ids),
            "owned_teams": sorted(owned_team_shorts),
            "owned_lineup_on_daily": owned_lineup_on_daily and use_llm,
            "squad_minutes_check": squad_minutes_check,
        },
    }
    atomic_write_json(signals_path, payload)
    result.applied = True
    result.content_changed = True
    result.dry_run = False
    result.reason = (
        f"Wrote {len(proposed)} news + {len(auto_preseason)} preseason + "
        f"{len(proposed_lineups)} predicted lineup signals to {signals_path} "
        f"({ingest_model})"
    )
    return result
