"""Live optimiser adapter: EV predictions → vector team SA (non-MIP)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.config import SEASON_RUNNER, VECTOR_TEAM
from fpl_robo.optimize.single_gw import SquadSolution
from fpl_robo.optimize.transfers import TransferPlan
from fpl_robo.optimize.vector_search import anneal_squad, anneal_transfers
from fpl_robo.predict.vector_features import (
    attach_vector_ev,
    build_feature_matrix,
    hand_model_from_pool,
)


def vector_live_enabled() -> bool:
    return bool(VECTOR_TEAM.get("enabled", False))


def prepare_vector_predictions(
    predictions: pd.DataFrame,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
) -> pd.DataFrame:
    """Anchor on live EV, add vector geometry columns / residual scaling."""
    vp = predictions.copy()
    if gameweek is not None and "gameweek" not in vp.columns:
        vp["gameweek"] = int(gameweek)
    if "rolling_expected_goals" not in vp.columns and "rolling_xg" in vp.columns:
        vp["rolling_expected_goals"] = vp["rolling_xg"]
    if "rolling_expected_assists" not in vp.columns and "rolling_xa" in vp.columns:
        vp["rolling_expected_assists"] = vp["rolling_xa"]
    if "rolling_ict_index" not in vp.columns and "rolling_ict" in vp.columns:
        vp["rolling_ict_index"] = vp["rolling_ict"]
    if "prior_minutes" not in vp.columns:
        vp["prior_minutes"] = vp["minutes"] if "minutes" in vp.columns else 0.0
    if "total_points" not in vp.columns:
        vp["total_points"] = 0.0

    X, _ = build_feature_matrix(vp, team_fixtures=team_fixtures)
    model = hand_model_from_pool(X)  # live: no labelled residual fit yet
    anchor = vp["ev"] if "ev" in vp.columns else None
    out = attach_vector_ev(vp, model, team_fixtures=team_fixtures, xp_anchor=anchor)
    # Prefer live EV as base_points when residual model is unfitted (attach already does this).
    if "ev" in vp.columns and not model.fitted:
        out["base_points"] = vp["ev"].astype(float).clip(lower=0.0)
        out["ev"] = out["base_points"]
    return out


def _effective_ft_hits(
    free_transfers: int,
    max_hits: int,
    *,
    unlimited: bool,
) -> tuple[int, int]:
    """When FT are unlimited, bound SA neighbourhood to preseason transfer cap."""
    if unlimited or int(free_transfers) >= 10:
        cap = int(SEASON_RUNNER.get("preseason_max_transfers_unless_repair") or 2)
        return max(1, cap), 0
    return int(free_transfers), int(max_hits)


def live_optimise_transfers(
    predictions: pd.DataFrame,
    *,
    current_ids: list[int],
    bank: float,
    selling_prices: dict[int, float] | None = None,
    free_transfers: int = 1,
    max_hits: int = 1,
    unlimited_transfers: bool = False,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
    **_ignored: Any,
) -> TransferPlan:
    """Vector SA transfer plan for the live daemon (MIP-compatible TransferPlan)."""
    preds = prepare_vector_predictions(
        predictions, team_fixtures=team_fixtures, gameweek=gameweek
    )
    ft, hits = _effective_ft_hits(
        free_transfers, max_hits, unlimited=unlimited_transfers
    )
    return anneal_transfers(
        preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling_prices,
        free_transfers=ft,
        max_hits=hits,
        n_iters=int(VECTOR_TEAM.get("sa_iters_transfer") or 2200),
        seed=int(VECTOR_TEAM.get("sa_seed") or 42),
    )


def live_optimise_single_gw(
    predictions: pd.DataFrame,
    *,
    budget: float = 100.0,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
    **_ignored: Any,
) -> SquadSolution:
    """Vector SA initial/rebuild squad (MIP-compatible SquadSolution)."""
    preds = prepare_vector_predictions(
        predictions, team_fixtures=team_fixtures, gameweek=gameweek
    )
    return anneal_squad(
        preds,
        budget=budget,
        n_iters=int(VECTOR_TEAM.get("sa_iters_initial") or 3500),
        seed=int(VECTOR_TEAM.get("sa_seed") or 42),
    )
