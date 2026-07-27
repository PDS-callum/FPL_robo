"""Chip inventory parsing and season windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CHIP_ALIASES = {
    "3xc": "triple_captain",
    "triplecaptain": "triple_captain",
    "triple_captain": "triple_captain",
    "bboost": "bench_boost",
    "benchboost": "bench_boost",
    "bench_boost": "bench_boost",
    "freehit": "free_hit",
    "free_hit": "free_hit",
    "wildcard": "wildcard",
}


@dataclass
class ChipWindow:
    name: str
    available: bool
    used: bool
    start_event: int | None = None
    stop_event: int | None = None
    played_by_entry: list[int] = field(default_factory=list)
    status_for_entry: str | None = None


@dataclass
class ChipInventory:
    windows: list[ChipWindow] = field(default_factory=list)
    half_split_gw: int = 19

    def available(self, chip: str, gameweek: int | None = None) -> bool:
        key = CHIP_ALIASES.get(chip.lower(), chip.lower())
        for w in self.windows:
            if w.name != key:
                continue
            if not w.available:
                continue
            if gameweek is None:
                return True
            if w.start_event is not None and gameweek < w.start_event:
                continue
            if w.stop_event is not None and gameweek > w.stop_event:
                continue
            return True
        # Fallback when API inventory is empty: assume classic two-half availability.
        if not self.windows:
            return True
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "half_split_gw": self.half_split_gw,
            "chips": [
                {
                    "name": w.name,
                    "available": w.available,
                    "used": w.used,
                    "start_event": w.start_event,
                    "stop_event": w.stop_event,
                    "status_for_entry": w.status_for_entry,
                    "played_by_entry": w.played_by_entry,
                }
                for w in self.windows
            ],
        }


def parse_chip_inventory(my_team_payload: dict[str, Any] | None, *, half_split_gw: int = 19) -> ChipInventory:
    """Parse chips block from authenticated my-team response."""
    inv = ChipInventory(half_split_gw=half_split_gw)
    if not my_team_payload:
        return inv
    raw = my_team_payload.get("chips") or []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name") or "").lower()
        name = CHIP_ALIASES.get(raw_name, raw_name)
        status = str(item.get("status_for_entry") or "").lower()
        played = list(item.get("played_by_entry") or [])
        available = status in {"available", "active"} and not bool(item.get("is_pending") and status == "unavailable")
        if status in {"played", "unavailable"}:
            available = False
        used = bool(played) or status == "played"
        inv.windows.append(
            ChipWindow(
                name=name,
                available=available,
                used=used,
                start_event=_as_optional_int(item.get("start_event")),
                stop_event=_as_optional_int(item.get("stop_event")),
                played_by_entry=[int(x) for x in played],
                status_for_entry=status or None,
            )
        )
    return inv


def half_window(gameweek: int, half_split_gw: int = 19) -> tuple[int, int]:
    if gameweek <= half_split_gw:
        return 1, half_split_gw
    return half_split_gw + 1, 38


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
