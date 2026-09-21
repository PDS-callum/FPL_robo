"""Live multi-week path-graph transfer planning."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from fpl_robo.config import PATH_PLAN
from fpl_robo.optimize.live_vector import (
    hold_plan_from_current,
    live_optimise_transfers,
    prepare_vector_predictions,
)
from fpl_robo.optimize.transfers import TransferPlan
from fpl_robo.plan.path_graph import PathPlanResult, plan_transfer_path

logger = logging.getLogger(__name__)


def path_plan_live_enabled() -> bool:
    return bool(PATH_PLAN.get("enabled"))


def _predictions_by_gw(
    predictions: pd.DataFrame,
    *,
    start_gw: int,
    horizon: int,
) -> dict[int, pd.DataFrame]:
    """Clone live predictions across the planning horizon.

    Future-GW EV uses the same base projection; `plan_transfer_path` applies
    week-specific FDR scales so entry-week timing is priced correctly.
    """
    out: dict[int, pd.DataFrame] = {}
    for offset in range(int(horizon)):
        gw = int(start_gw) + offset
        frame = predictions.copy()
        if "gameweek" in frame.columns:
            frame["gameweek"] = gw
        else:
            frame["gameweek"] = gw
        out[gw] = frame
    return out


def _plan_from_forced_transfer(
    predictions: pd.DataFrame,
    *,
    current_ids: list[int],
    element_out: int,
    element_in: int,
    bank: float,
    selling_prices: dict[int, float] | None,
    free_transfers: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    gameweek: int | None,
    path: PathPlanResult,
) -> TransferPlan | None:
    from fpl_robo.data.prices import effective_sell_price
    from fpl_robo.optimize.vector_search import _pool_from_preds, _to_solution, squad_cost
    from fpl_robo.predict.vector_team_score import FastTeamScorer

    preds = prepare_vector_predictions(
        predictions, team_fixtures=team_fixtures, gameweek=gameweek
    )
    pool = _pool_from_preds(preds)
    owned = [int(i) for i in current_ids if int(i) in pool.by_id]
    out_id, in_id = int(element_out), int(element_in)
    if out_id not in owned or in_id in owned:
        return None
    if in_id not in pool.by_id:
        return None

    sell = selling_prices or {}
    sell_price = effective_sell_price(
        out_id, sell, fallback=float(pool.by_id[out_id].get("price") or 4.5)
    )
    buy_price = float(pool.by_id[in_id].get("price") or 99.0)
    if sell_price + float(bank) + 1e-9 < buy_price:
        return None

    new_ids = [in_id if i == out_id else i for i in owned]
    if len(new_ids) != 15 or in_id not in new_ids:
        return None

    hits = max(0, 1 - int(free_transfers))
    hit_cost = hits * int(PATH_PLAN.get("hit_cost") or 4)
    if hits > int(PATH_PLAN.get("max_hits_per_week") or 1):
        return None

    capital = float(bank) + sum(
        effective_sell_price(
            i, sell, fallback=float(pool.by_id[i].get("price") or 4.5)
        )
        for i in owned
    )
    # After swap, capital accounting: remove out sell from owned sum conceptually
    # by using new squad cost vs original capital.
    scorer = FastTeamScorer(preds)
    sol = _to_solution(new_ids, preds, scorer)
    budget_left = round(capital - squad_cost(new_ids, pool), 1)
    if budget_left < -1e-6:
        return None

    transfer = {
        "element_out": out_id,
        "element_in": in_id,
        "out_name": str(pool.by_id[out_id].get("web_name") or out_id),
        "in_name": str(pool.by_id[in_id].get("web_name") or in_id),
    }
    return TransferPlan(
        status="Optimal",
        objective=float(scorer.score(new_ids).total) - float(hit_cost),
        transfers=[transfer],
        transfers_made=1,
        hits=hits,
        hit_cost=hit_cost,
        budget_left=max(0.0, budget_left),
        free_transfers=int(free_transfers),
        solution=sol,
        meta={
            "optimiser": "path_graph",
            "path_plan": path.to_dict(),
            "path_explanation": path.explanation,
        },
    )


def live_path_optimise_transfers(
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
    bench_ev_weight: float | None = None,
    owned_risk_count: int = 0,
    **kwargs: Any,
) -> TransferPlan:
    """Plan transfers with the multi-week path graph; fall back to vector SA."""
    if unlimited_transfers or not path_plan_live_enabled():
        return live_optimise_transfers(
            predictions,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling_prices,
            free_transfers=free_transfers,
            max_hits=max_hits,
            unlimited_transfers=unlimited_transfers,
            team_fixtures=team_fixtures,
            gameweek=gameweek,
            require_cover_eligible=require_cover_eligible,
            bench_ev_weight=bench_ev_weight,
            owned_risk_count=owned_risk_count,
            **kwargs,
        )

    if gameweek is None or not team_fixtures:
        logger.warning("Path plan skipped: missing gameweek/fixtures; using vector SA")
        return live_optimise_transfers(
            predictions,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling_prices,
            free_transfers=free_transfers,
            max_hits=max_hits,
            unlimited_transfers=False,
            team_fixtures=team_fixtures,
            gameweek=gameweek,
            require_cover_eligible=require_cover_eligible,
            bench_ev_weight=bench_ev_weight,
            owned_risk_count=owned_risk_count,
            **kwargs,
        )

    horizon = int(PATH_PLAN.get("horizon") or 3)
    preds_by_gw = _predictions_by_gw(
        predictions, start_gw=int(gameweek), horizon=horizon
    )
    path = plan_transfer_path(
        start_gw=int(gameweek),
        current_ids=list(current_ids),
        predictions_by_gw=preds_by_gw,
        team_fixtures=team_fixtures,
        bank=float(bank),
        free_transfers=int(free_transfers),
        selling_prices=selling_prices,
    )
    logger.info(
        "Path plan status=%s week0=%s score=%.2f | %s",
        path.status,
        path.week0_action.kind,
        path.path_score,
        path.explanation,
    )

    if path.status != "Optimal":
        fallback = live_optimise_transfers(
            predictions,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling_prices,
            free_transfers=free_transfers,
            max_hits=max_hits,
            team_fixtures=team_fixtures,
            gameweek=gameweek,
            require_cover_eligible=require_cover_eligible,
            bench_ev_weight=bench_ev_weight,
            owned_risk_count=owned_risk_count,
            **kwargs,
        )
        meta = dict(fallback.meta or {})
        meta["path_plan_fallback"] = path.to_dict()
        fallback.meta = meta
        return fallback

    act = path.week0_action
    if act.kind == "hold" or act.element_out is None or act.element_in is None:
        plan = hold_plan_from_current(
            predictions,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling_prices,
            free_transfers=free_transfers,
            team_fixtures=team_fixtures,
            gameweek=gameweek,
            meta={
                "optimiser": "path_graph",
                "path_plan": path.to_dict(),
                "path_explanation": path.explanation,
            },
        )
        return plan

    forced = _plan_from_forced_transfer(
        predictions,
        current_ids=current_ids,
        element_out=int(act.element_out),
        element_in=int(act.element_in),
        bank=bank,
        selling_prices=selling_prices,
        free_transfers=free_transfers,
        team_fixtures=team_fixtures,
        gameweek=gameweek,
        path=path,
    )
    if forced is not None:
        return forced

    logger.warning(
        "Path transfer %s→%s infeasible live; falling back to vector SA",
        act.element_out,
        act.element_in,
    )
    fallback = live_optimise_transfers(
        predictions,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling_prices,
        free_transfers=free_transfers,
        max_hits=max_hits,
        team_fixtures=team_fixtures,
        gameweek=gameweek,
        require_cover_eligible=require_cover_eligible,
        bench_ev_weight=bench_ev_weight,
        owned_risk_count=owned_risk_count,
        **kwargs,
    )
    meta = dict(fallback.meta or {})
    meta["path_plan_infeasible"] = path.to_dict()
    fallback.meta = meta
    return fallback
