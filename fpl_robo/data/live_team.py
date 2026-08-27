"""Live manager state from authenticated my-team endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpl_robo.config import DEFAULT_MANAGER_ID
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.manager import ManagerState, load_manager_state
from fpl_robo.data.prices import pick_selling_price_pounds, pounds_from_tenths


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


def _selling_map(picks: list[dict[str, Any]]) -> dict[int, float]:
    """Map element → sell £. Omit unknowns (never store 0 — that breaks capital)."""
    out: dict[int, float] = {}
    for p in picks:
        if p.get("element") is None:
            continue
        sell = pick_selling_price_pounds(p)
        if sell is None:
            continue
        out[int(p["element"])] = sell
    return out


def _purchase_map(picks: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for p in picks:
        if p.get("element") is None:
            continue
        purch = pounds_from_tenths(p.get("purchase_price"))
        if purch is None:
            continue
        out[int(p["element"])] = purch
    return out


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
        picks = list(manager.picks or [])
        return LiveTeamState(
            manager=manager,
            free_transfers=1,
            bank=manager.bank,
            selling_prices=_selling_map(picks),
            purchase_prices=_purchase_map(picks),
            picks=picks,
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

    selling = _selling_map(picks)
    purchase = _purchase_map(picks)
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
