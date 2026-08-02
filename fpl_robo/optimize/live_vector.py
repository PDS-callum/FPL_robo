"""Live optimiser adapter: EV predictions → vector team SA (non-MIP)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd

from fpl_robo.chips.backtest import load_merged_gw, prepare_gw_table
from fpl_robo.config import (
    BENCH_COVER,
    CHIP_BACKTEST_CACHE,
    SEASON_RUNNER,
    VECTOR_TEAM,
)
from fpl_robo.optimize.single_gw import SquadSolution
from fpl_robo.optimize.transfers import TransferPlan
from fpl_robo.optimize.vector_search import anneal_squad, anneal_transfers
from fpl_robo.predict.ev_modes import attach_lagged_history
from fpl_robo.predict.horizon import team_fixtures_from_merged
from fpl_robo.predict.vector_features import (
    RidgeModel,
    attach_vector_ev,
    build_feature_matrix,
    fit_ridge_from_history,
    hand_model_from_pool,
)

logger = logging.getLogger("fpl_robo.optimize.live_vector")


def vector_live_enabled() -> bool:
    return bool(VECTOR_TEAM.get("enabled", False))


@lru_cache(maxsize=2)
def _cached_prior_season_residual(season: str, cache_path: str) -> RidgeModel:
    """Fit residual ridge on a finished season (cold-start for live GW1)."""
    merged = load_merged_gw(season=season, cache_path=cache_path)
    table = attach_lagged_history(prepare_gw_table(merged, season=season))
    fixtures = team_fixtures_from_merged(table)
    max_gw = int(table["gameweek"].max())
    model = fit_ridge_from_history(
        table,
        before_gw=max_gw + 1,
        team_fixtures=fixtures,
        lookback=int(VECTOR_TEAM.get("fit_lookback_gws") or 16),
    )
    return model


def fit_live_residual_model(
    predictions: pd.DataFrame,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
) -> RidgeModel:
    """Prefer a fitted residual model; fall back to hand weights."""
    X, _ = build_feature_matrix(predictions, team_fixtures=team_fixtures)
    season = str(VECTOR_TEAM.get("live_residual_season") or "2025-26")
    cache = f"data/cache/merged_gw_{season}.csv"
    if season == "2025-26":
        cache = CHIP_BACKTEST_CACHE
    try:
        model = _cached_prior_season_residual(season, cache)
        if model.fitted and int(model.n_train) >= int(VECTOR_TEAM.get("min_ridge_rows") or 120):
            logger.info(
                "Live residual ridge fitted from %s (n=%s)",
                season,
                model.n_train,
            )
            return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live residual fit failed (%s); using hand model", exc)
    return hand_model_from_pool(X)


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

    model = fit_live_residual_model(
        vp, team_fixtures=team_fixtures, gameweek=gameweek
    )
    anchor = vp["ev"] if "ev" in vp.columns else None
    out = attach_vector_ev(vp, model, team_fixtures=team_fixtures, xp_anchor=anchor)
    # Always keep live EV as the primary base; residual is a scaled correction only.
    if "ev" in vp.columns:
        live_ev = vp["ev"].astype(float).clip(lower=0.0)
        if model.fitted and bool(VECTOR_TEAM.get("residual_mode", True)):
            # attach_vector_ev already applied residual onto anchor; keep that.
            out["ev"] = out["base_points"]
        else:
            out["base_points"] = live_ev
            out["ev"] = live_ev
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
    require_cover_eligible: bool = True,
    bench_ev_weight: float | None = None,  # accepted for MIP parity; SA uses VECTOR_TEAM
    enforce_buy_eligible: bool = False,
    min_buy_ev: float | None = None,
    use_tiebreak: Any = None,  # noqa: ARG001 — MIP-only
    **_ignored: Any,
) -> TransferPlan:
    """Vector SA transfer plan for the live daemon (MIP-compatible TransferPlan)."""
    _ = bench_ev_weight if bench_ev_weight is not None else BENCH_COVER.get("bench_ev_weight")
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
        require_cover_eligible=require_cover_eligible,
        enforce_buy_eligible=enforce_buy_eligible,
        min_buy_ev=min_buy_ev,
        exclude_unavailable=bool(VECTOR_TEAM.get("exclude_unavailable", True)),
    )


def live_optimise_single_gw(
    predictions: pd.DataFrame,
    *,
    budget: float = 100.0,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
    require_cover_eligible: bool = True,  # noqa: ARG001 — pool filter in anneal_squad later
    bench_ev_weight: float | None = None,  # noqa: ARG001
    **_ignored: Any,
) -> SquadSolution:
    """Vector SA initial/rebuild squad (MIP-compatible SquadSolution)."""
    preds = prepare_vector_predictions(
        predictions, team_fixtures=team_fixtures, gameweek=gameweek
    )
    if require_cover_eligible and "cover_eligible" in preds.columns:
        # Soft-filter non-cover for rebuilds; keep enough depth by EV rank.
        cover = preds[preds["cover_eligible"].astype(bool)]
        if len(cover) >= 40:
            preds = cover
    return anneal_squad(
        preds,
        budget=budget,
        n_iters=int(VECTOR_TEAM.get("sa_iters_initial") or 3500),
        seed=int(VECTOR_TEAM.get("sa_seed") or 42),
    )
