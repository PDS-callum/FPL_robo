"""Seed availability signals from official FPL bootstrap status/news."""

from __future__ import annotations

from typing import Any


def signals_from_fpl_status(
    players: list[dict[str, Any]],
    *,
    min_ownership: float = 0.0,
    include_all_unavailable: bool = True,
) -> list[dict[str, Any]]:
    """Turn FPL status / chance / news into news signal dicts.

    These are high-trust (official API) and fill gaps headlines never mention.
    """
    out: list[dict[str, Any]] = []
    for p in players:
        status = str(p.get("status") or "a").lower()
        chance = p.get("chance_of_playing_next_round")
        news = str(p.get("news") or "").strip()
        ownership = float(p.get("selected_by_percent") or 0.0)
        if ownership < min_ownership and not (
            include_all_unavailable and status in {"i", "d", "s"}
        ):
            continue

        kind = None
        severity = 0.0
        exclude = False
        if status == "s" or "suspend" in news.lower():
            kind = "suspension"
            severity = 0.9
            exclude = True
        elif status == "i" or (chance is not None and float(chance) <= 0 and news):
            kind = "injury"
            severity = 0.85
            exclude = True
        elif status == "d" or (chance is not None and float(chance) < 75):
            kind = "injury"
            try:
                severity = max(0.35, min(0.75, 1.0 - float(chance or 50) / 100.0))
            except (TypeError, ValueError):
                severity = 0.5
            exclude = False
        elif news and status == "a":
            # News text without status flag — mild doubt only.
            lower = news.lower()
            if any(k in lower for k in ("injur", "knock", "doubt", "illness", "fitness")):
                kind = "injury"
                severity = 0.4
        if kind is None:
            continue

        first = str(p.get("first_name") or "").strip()
        web = str(p.get("web_name") or "").strip()
        note = news or f"FPL status={status}"
        if chance is not None:
            note = f"{note} (chance {chance}%)"
        out.append(
            {
                "player_id": int(p["id"]),
                "web_name": web,
                "first_name": first or None,
                "second_name": p.get("second_name"),
                "team_short": p.get("team_short"),
                "element_type": p.get("element_type"),
                "position": p.get("position"),
                "type": kind,
                "severity": round(severity, 3),
                "exclude": exclude,
                "note": f"[fpl] {note}"[:180],
                "evidence": news or f"status={status}",
                "confidence": 0.95,
                "evidence_flags": {
                    "name_hit": True,
                    "team_hit": True,
                    "keyword_hit": True,
                    "proximity": True,
                    "status_align": True,
                    "fpl_status": status,
                    "fpl_news": news or None,
                },
                "source": "fpl",
                "article_title": "",
                "article_link": "",
                "article_source": "fpl_bootstrap",
            }
        )
    # Dedupe by player
    seen: set[int] = set()
    uniq = []
    for item in out:
        pid = int(item["player_id"])
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(item)
    return uniq
