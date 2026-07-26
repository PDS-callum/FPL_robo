"""Manager entry helpers (current team, bank, free transfers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpl_robo.config import DEFAULT_MANAGER_ID
from fpl_robo.data.fpl_client import FPLClient


@dataclass
class ManagerState:
    manager_id: int
    name: str
    team_name: str
    bank: float
    team_value: float
    overall_points: int
    overall_rank: int | None
    gameweek: int | None
    picks: list[dict[str, Any]]
    raw_entry: dict[str, Any]
    raw_picks: dict[str, Any] | None

    @property
    def player_ids(self) -> list[int]:
        return [int(p["element"]) for p in self.picks]


def load_manager_state(
    manager_id: int = DEFAULT_MANAGER_ID,
    client: FPLClient | None = None,
    gameweek: int | None = None,
) -> ManagerState:
    client = client or FPLClient()
    entry = client.entry(manager_id)
    picks_payload = client.entry_picks_latest(manager_id, gameweek=gameweek)
    gw = None
    raw_picks = None
    picks: list[dict[str, Any]] = []
    if picks_payload is not None:
        gw, raw_picks = picks_payload
        picks = list(raw_picks.get("picks") or [])

    # bank / value are in tenths of a million
    bank = float(entry.get("last_deadline_bank") or 0) / 10.0
    team_value = float(entry.get("last_deadline_value") or 0) / 10.0
    if raw_picks and isinstance(raw_picks.get("entry_history"), dict):
        hist = raw_picks["entry_history"]
        if hist.get("bank") is not None:
            bank = float(hist["bank"]) / 10.0
        if hist.get("value") is not None:
            team_value = float(hist["value"]) / 10.0

    first = entry.get("player_first_name") or ""
    last = entry.get("player_last_name") or ""
    name = f"{first} {last}".strip() or f"Manager {manager_id}"

    return ManagerState(
        manager_id=manager_id,
        name=name,
        team_name=str(entry.get("name") or ""),
        bank=bank,
        team_value=team_value,
        overall_points=int(entry.get("summary_overall_points") or 0),
        overall_rank=entry.get("summary_overall_rank"),
        gameweek=gw,
        picks=picks,
        raw_entry=entry,
        raw_picks=raw_picks,
    )
