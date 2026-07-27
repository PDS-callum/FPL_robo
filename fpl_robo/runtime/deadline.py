"""Deadline helpers for scheduled season runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def next_deadline(bootstrap: dict[str, Any]) -> dict[str, Any] | None:
    """Return next unfinished event with parsed deadline UTC."""
    events = bootstrap.get("events") or []
    upcoming = []
    for ev in events:
        if ev.get("finished"):
            continue
        raw = ev.get("deadline_time")
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        upcoming.append(
            {
                "id": int(ev["id"]),
                "name": ev.get("name"),
                "deadline_time": raw,
                "deadline_utc": dt.astimezone(timezone.utc),
                "is_next": bool(ev.get("is_next")),
                "is_current": bool(ev.get("is_current")),
            }
        )
    if not upcoming:
        return None
    upcoming.sort(key=lambda x: x["deadline_utc"])
    # Prefer flagged next event if present.
    for ev in upcoming:
        if ev["is_next"]:
            return ev
    return upcoming[0]


def hours_to_deadline(deadline_utc: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    return (deadline_utc - now).total_seconds() / 3600.0


def detect_dgw_teams(fixtures: list[dict[str, Any]], gameweek: int) -> set[int]:
    counts: dict[int, int] = {}
    for fix in fixtures:
        if fix.get("event") != gameweek:
            continue
        if fix.get("finished"):
            continue
        for key in ("team_h", "team_a"):
            tid = fix.get(key)
            if tid is None:
                continue
            counts[int(tid)] = counts.get(int(tid), 0) + 1
    return {tid for tid, n in counts.items() if n >= 2}
