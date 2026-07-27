"""Scrape news + interpret with local LLM into signals.json."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
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
    articles_payload,
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


def merge_news_signals(
    existing_news: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace prior ollama news for same player/type; keep manual entries."""
    keep: list[dict[str, Any]] = []
    for item in existing_news:
        if not isinstance(item, dict):
            continue
        # Drop previous auto signals (fresh scrape + KG validation wins).
        if item.get("source") in {"ollama", "heuristic", "auto"}:
            continue
        keep.append(item)

    seen: set[tuple] = set()
    merged = list(keep)
    for item in proposed:
        key = _news_key(item)
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proposed: list[dict[str, Any]] = []
    proposed_lineups: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    min_conf = float(
        SIGNAL_INGEST["min_confidence"] if min_confidence is None else min_confidence
    )

    for art in articles:
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

        raw_signals = heuristic_signals_from_title(art.title, focused_players)
        try:
            llm_signals = interpret_article(
                title=art.title,
                summary=art.summary,
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

        want_lineups = extract_lineups or looks_like_lineup_article(art)
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
                    "type": sig.get("type"),
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

    if skip_llm:
        return IngestResult(
            ok=True,
            dry_run=True,
            applied=False,
            articles=len(articles),
            feed_errors=feed_errors,
            signals_path=signals_path,
            raw_cache_path=str(raw_path),
            reason="Skipped LLM (--skip-llm); raw articles cached only",
            ollama_model=model,
            pre_deadline=pre_deadline,
            gameweek=gameweek,
        )

    if not ollama_available(ollama_url):
        return IngestResult(
            ok=False,
            dry_run=dry_run,
            applied=False,
            articles=len(articles),
            feed_errors=feed_errors,
            signals_path=signals_path,
            raw_cache_path=str(raw_path),
            reason=f"Ollama not reachable at {ollama_url}",
            ollama_model=model,
            pre_deadline=pre_deadline,
            gameweek=gameweek,
        )

    players, team_aliases = _players_and_aliases(client)
    indexes = build_player_indexes(players)
    proposed, proposed_lineups, rejected, unresolved = interpret_articles(
        articles,
        indexes,
        team_aliases,
        model=model,
        base_url=ollama_url,
        extract_lineups=pre_deadline,
        gameweek=gameweek,
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
        ollama_model=model,
        pre_deadline=pre_deadline,
        gameweek=gameweek,
    )

    if dry_run or not apply:
        result.reason = (
            f"Dry-run only; news={len(proposed)} lineups={len(proposed_lineups)} "
            f"rejected={len(rejected)} (signals.json not modified)"
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
    payload = {
        "as_of": date.today().isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "news": merged_news,
        "preseason": existing.get("preseason") or [],
        "international": existing.get("international") or [],
        "predicted_lineups": merged_lineups,
        "ingest": {
            "articles": len(articles),
            "proposed": len(proposed),
            "proposed_lineups": len(proposed_lineups),
            "rejected": len(rejected),
            "unresolved": len(unresolved),
            "model": model,
            "min_confidence": SIGNAL_INGEST.get("min_confidence"),
            "feed_errors": feed_errors,
            "kg_validated": True,
            "pre_deadline": pre_deadline,
            "gameweek": gameweek,
        },
    }
    atomic_write_json(signals_path, payload)
    result.applied = True
    result.dry_run = False
    result.reason = (
        f"Wrote {len(proposed)} news + {len(proposed_lineups)} predicted lineup "
        f"signals to {signals_path}"
    )
    return result
