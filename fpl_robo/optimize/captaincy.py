"""Captain / vice-captain helpers shared by optimiser and team writes."""

from __future__ import annotations

from typing import Any


def pick_vice_captain_id(
    squad_rows: list[dict[str, Any]],
    captain_id: int | None,
) -> int | None:
    """Choose vice as highest-EV starter who isn't captain (fallback: best other)."""
    if not squad_rows:
        return None
    cap = int(captain_id) if captain_id is not None else None

    def _ev(row: dict[str, Any]) -> float:
        try:
            return float(row.get("ev") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    starters = [
        r for r in squad_rows
        if r.get("start") and (cap is None or int(r["id"]) != cap)
    ]
    starters.sort(key=_ev, reverse=True)
    if starters:
        return int(starters[0]["id"])
    others = [r for r in squad_rows if cap is None or int(r["id"]) != cap]
    others.sort(key=_ev, reverse=True)
    return int(others[0]["id"]) if others else None
