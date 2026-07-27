"""Live manager state from authenticated my-team endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpl_robo.config import DEFAULT_MANAGER_ID
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.manager import ManagerState, load_manager_state


@dataclass
class LiveTeamState:
    manager: ManagerState
    free_transfers: int
    bank: float
    selling_prices: dict[int, float]
    purchase_prices: dict[int, float]
    picks: list[dict[str, Any]]
    transfers_status: str | None
    raw_my_team: dict[str, Any]
    unlimited_transfers: bool = False


def load_live_team_state(
    manager_id: int = DEFAULT_MANAGER_ID,
    client: FPLClient | None = None,
) -> LiveTeamState:
    """Prefer authenticated my-team; fall back to public entry picks."""
    client = client or FPLClient()
    manager = load_manager_state(manager_id=manager_id, client=client)
    try:
        my_team = client.my_team(manager_id)
    except Exception:
        selling = {
            int(p["element"]): float(p.get("selling_price") or 0) / 10.0
            for p in manager.picks
            if p.get("element") is not None
        }
        purchase = {
            int(p["element"]): float(p.get("purchase_price") or 0) / 10.0
            for p in manager.picks
            if p.get("element") is not None
        }
        return LiveTeamState(
            manager=manager,
            free_transfers=1,
            bank=manager.bank,
            selling_prices=selling,
            purchase_prices=purchase,
            picks=manager.picks,
            transfers_status=None,
            raw_my_team={},
            unlimited_transfers=False,
        )

    picks = list(my_team.get("picks") or [])
    transfers = my_team.get("transfers") or {}
    bank_tenths = transfers.get("bank")
    bank = float(bank_tenths) / 10.0 if bank_tenths is not None else manager.bank
    limit = transfers.get("limit")
    made = int(transfers.get("made") or 0)
    unlimited = limit is None
    if unlimited:
        # Unlimited preseason / chip window.
        free_transfers = 15
    else:
        free_transfers = max(0, int(limit) - made)

    selling = {
        int(p["element"]): float(p.get("selling_price") or 0) / 10.0
        for p in picks
        if p.get("element") is not None
    }
    purchase = {
        int(p["element"]): float(p.get("purchase_price") or 0) / 10.0
        for p in picks
        if p.get("element") is not None
    }
    # Keep manager picks in sync with authenticated source of truth.
    manager.picks = picks
    manager.bank = bank
    return LiveTeamState(
        manager=manager,
        free_transfers=free_transfers,
        bank=bank,
        selling_prices=selling,
        purchase_prices=purchase,
        picks=picks,
        transfers_status=str(transfers.get("status") or "") or None,
        raw_my_team=my_team,
        unlimited_transfers=unlimited,
    )
