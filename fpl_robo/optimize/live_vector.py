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
    PosRidgeEnsemble,
    RidgeModel,
    attach_vector_ev,
    blend_residual_models,
    build_feature_matrix,
    fit_ridge_from_history,
    hand_model_from_pool,
)

logger = logging.getLogger("fpl_robo.optimize.live_vector")


def vector_live_enabled() -> bool:
    return bool(VECTOR_TEAM.get("enabled", False))


@lru_cache(maxsize=4)
def _cached_season_residual(
    season: str, cache_path: str, feature_version: str
) -> RidgeModel | PosRidgeEnsemble:
    """Fit residual ridge on a finished (or partial) season table."""
    _ = feature_version  # included so FEATURE_NAMES bumps invalidate the cache
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
    model.source = f"season:{season}"
    return model


@lru_cache(maxsize=8)
def _cached_current_residual(
    season: str, cache_path: str, before_gw: int, feature_version: str
) -> RidgeModel | PosRidgeEnsemble:
    """Fit residual using only GWs before `before_gw` in the live season."""
    _ = feature_version
    merged = load_merged_gw(season=season, cache_path=cache_path)
    table = attach_lagged_history(prepare_gw_table(merged, season=season))
    fixtures = team_fixtures_from_merged(table)
    model = fit_ridge_from_history(
        table,
        before_gw=int(before_gw),
        team_fixtures=fixtures,
        lookback=int(VECTOR_TEAM.get("fit_lookback_gws") or 16),
    )
    model.source = f"current:{season}:gw<{before_gw}"
    return model


def fit_live_residual_model(
    predictions: pd.DataFrame,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
) -> RidgeModel | PosRidgeEnsemble:
    """Prefer current-season residual when enough GW points exist; else prior season.

    From `residual_blend_from_gw` onward, blend prior-season and current-season
    ridges so live adapts without discarding the cold-start prior.
    """
    from fpl_robo.config import KG_ENRICH

    X, _ = build_feature_matrix(predictions, team_fixtures=team_fixtures)
    feat_ver = str(KG_ENRICH.get("feature_version") or "v0")
    min_rows = int(VECTOR_TEAM.get("min_ridge_rows") or 120)
    gw = int(gameweek) if gameweek is not None else None
    if gw is None and "gameweek" in predictions.columns and len(predictions):
        try:
            gw = int(predictions["gameweek"].dropna().iloc[0])
        except (TypeError, ValueError, IndexError):
            gw = None

    prior: RidgeModel | PosRidgeEnsemble | None = None
    season = str(VECTOR_TEAM.get("live_residual_season") or "2025-26")
    cache = f"data/cache/merged_gw_{season}.csv"
    if season == "2025-26":
        cache = CHIP_BACKTEST_CACHE
    try:
        prior = _cached_season_residual(season, cache, feat_ver)
        if not (prior.fitted and int(prior.n_train) >= min_rows):
            prior = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prior-season residual fit failed (%s)", exc)
        prior = None

    current: RidgeModel | PosRidgeEnsemble | None = None
    blend_from = int(VECTOR_TEAM.get("residual_blend_from_gw") or 5)
    if (
        bool(VECTOR_TEAM.get("prefer_current_season_residual", True))
        and gw is not None
        and gw > 1
    ):
        cur = str(VECTOR_TEAM.get("live_current_season") or "2026-27")
        cur_cache = f"data/cache/merged_gw_{cur}.csv"
        try:
            current = _cached_current_residual(cur, cur_cache, gw, feat_ver)
            if not (current.fitted and int(current.n_train) >= min_rows):
                current = None
        except Exception as exc:  # noqa: BLE001
            logger.info("Current-season residual unavailable (%s)", exc)
            current = None

    if current is not None and prior is not None and gw is not None and gw >= blend_from:
        # Ramp current weight: 0 at blend_from-1 → residual_blend_max_current by ~GW15.
        span = max(1, 15 - blend_from)
        w_max = float(VECTOR_TEAM.get("residual_blend_max_current") or 0.75)
        w = min(w_max, w_max * (gw - blend_from + 1) / span)
        blended = blend_residual_models(prior, current, weight_current=w)
        logger.info(
            "Live residual blend prior=%s current=%s w_current=%.2f source=%s",
            getattr(prior, "source", "?"),
            getattr(current, "source", "?"),
            w,
            getattr(blended, "source", "?"),
        )
        return blended
    if current is not None:
        logger.info(
            "Live residual ridge fitted from current season (n=%s source=%s)",
            current.n_train,
            current.source,
        )
        return current
    if prior is not None:
        logger.info(
            "Live residual ridge fitted from %s (n=%s source=%s)",
            season,
            prior.n_train,
            prior.source,
        )
        return prior

    hand = hand_model_from_pool(X)
    hand.source = "hand_fallback"
    return hand


# Back-compat alias for tests / callers.
_cached_prior_season_residual = _cached_season_residual


def prepare_vector_predictions(
    predictions: pd.DataFrame,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    gameweek: int | None = None,
) -> pd.DataFrame:
    """Anchor on live EV, add vector geometry columns / residual scaling."""
    from fpl_robo.config import PRESEASON_HORIZON

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
    gw = int(gameweek) if gameweek is not None else (
        int(vp["gameweek"].dropna().iloc[0])
        if "gameweek" in vp.columns and len(vp) and vp["gameweek"].notna().any()
        else None
    )
    # Opening weeks: look ahead PRESEASON_HORIZON fixtures for run_ease / bundle.
    horizon = int(VECTOR_TEAM.get("horizon") or 3)
    if gw is not None and gw <= 1:
        horizon = max(horizon, int(PRESEASON_HORIZON))
    out = attach_vector_ev(
        vp, model, team_fixtures=team_fixtures, xp_anchor=anchor, horizon=horizon
    )
    out.attrs["residual_fit"] = {
        "source": getattr(model, "source", "unknown"),
        "fitted": bool(model.fitted),
        "n_train": int(model.n_train),
    }
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
    owned_risk_count: int = 0,
) -> tuple[int, int]:
    """When FT are unlimited, bound SA neighbourhood to preseason transfer cap.

    Injury / doubt repairs may widen the neighbourhood up to
    ``preseason_repair_max_transfers`` so SA can propose multi-sell fixes.
    """
    if unlimited or int(free_transfers) >= 10:
        cap = int(SEASON_RUNNER.get("preseason_max_transfers_unless_repair") or 2)
        risk_n = max(0, int(owned_risk_count))
        if risk_n > 0:
            repair_cap = int(SEASON_RUNNER.get("preseason_repair_max_transfers") or 5)
            cap = max(cap, min(repair_cap, max(risk_n, cap)))
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
    owned_risk_count: int = 0,
    **_ignored: Any,
) -> TransferPlan:
    """Vector SA transfer plan for the live daemon (MIP-compatible TransferPlan)."""
    _ = bench_ev_weight if bench_ev_weight is not None else BENCH_COVER.get("bench_ev_weight")
    preds = prepare_vector_predictions(
        predictions, team_fixtures=team_fixtures, gameweek=gameweek
    )
    ft, hits = _effective_ft_hits(
        free_transfers,
        max_hits,
        unlimited=unlimited_transfers,
        owned_risk_count=owned_risk_count,
    )
    plan = anneal_transfers(
        preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling_prices,
        free_transfers=ft,
        max_hits=hits,
        n_iters=int(VECTOR_TEAM.get("sa_iters_transfer") or 2200),
        seed=int(VECTOR_TEAM.get("sa_seed") or 42),
        require_cover_eligible=require_cover_eligible,
        enforce_buy_eligible=(
            True
            if bool(VECTOR_TEAM.get("enforce_buy_eligible", True))
            else bool(enforce_buy_eligible)
        ),
        min_buy_ev=min_buy_ev,
        exclude_unavailable=bool(VECTOR_TEAM.get("exclude_unavailable", True)),
        min_play_chance=(
            float(VECTOR_TEAM["min_play_chance"])
            if VECTOR_TEAM.get("min_play_chance") is not None
            else None
        ),
    )
    residual_fit = dict(getattr(preds, "attrs", {}) or {}).get("residual_fit")
    if residual_fit:
        meta = dict(plan.meta or {})
        meta["residual_fit"] = residual_fit
        plan.meta = meta
    return plan


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


def transfer_plan_from_rebuild(
    sol: SquadSolution,
    *,
    current_ids: list[int],
    bank: float,
    selling_prices: dict[int, float] | None = None,
    predictions: pd.DataFrame,
    chip: str | None = None,
) -> TransferPlan:
    """Convert a full-squad rebuild into a TransferPlan for FH/WC execute."""
    from fpl_robo.optimize.vector_search import _pair_transfers, _pool_from_preds, squad_cost

    preds = prepare_vector_predictions(predictions)
    pool = _pool_from_preds(preds)
    owned = [int(i) for i in current_ids if int(i) in pool.by_id]
    target = [int(p["id"]) for p in sol.squad]
    # Ensure target ids exist in pool (reinject from sol rows if needed).
    for row in sol.squad:
        pid = int(row["id"])
        if pid not in pool.by_id:
            pool.by_id[pid] = pd.Series(row)

    transfers = _pair_transfers(owned, target, pool) if owned else []
    sell = selling_prices or {}
    capital = float(bank) + sum(
        float(sell.get(i, float(pool.by_id[i].get("price") or 4.5))) for i in owned
    ) if owned else float(bank) + 100.0
    budget_left = round(capital - squad_cost(target, pool), 1) if target else float(bank)
    return TransferPlan(
        status=sol.status or "Optimal",
        objective=float(sol.objective),
        transfers=transfers,
        transfers_made=len(transfers),
        hits=0,
        hit_cost=0,
        budget_left=budget_left,
        free_transfers=0,
        solution=sol,
        meta={"rebuild": True, "chip": chip, "optimiser": "vector_team_sa_rebuild"},
    )
