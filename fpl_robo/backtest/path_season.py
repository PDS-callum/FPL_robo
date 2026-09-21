"""Season backtest using multi-week path-graph transfer planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_robo.backtest.season import (
    SeasonBacktestResult,
    _load_average_scores,
    _prediction_frame,
    _score_xi,
    _team_fixtures_for_season,
    write_season_report,
)
from fpl_robo.chips.backtest import load_merged_gw, prepare_gw_table
from fpl_robo.config import (
    BENCH_COVER,
    CHIP_BACKTEST_CACHE,
    CHIP_BACKTEST_SEASON,
    CHIP_DEFAULTS,
    PATH_PLAN,
)
from fpl_robo.optimize.single_gw import optimise_single_gw
from fpl_robo.plan.path_graph import PathAction, plan_transfer_path
from fpl_robo.predict.ev_modes import EvMode, attach_lagged_history


def _apply_path_action(
    *,
    action: PathAction,
    current_ids: list[int],
    preds: pd.DataFrame,
    bank: float,
    selling_prices: dict[int, float],
    free_transfers: int,
    max_hits: int,
) -> tuple[list[int], float, int, int, list[dict[str, int]], str]:
    """Apply week-0 path action; fall back to hold on infeasible transfer."""
    by_id = {int(r["id"]): r for _, r in preds.iterrows()}
    if action.kind != "transfer" or action.element_out is None or action.element_in is None:
        return list(current_ids), bank, free_transfers, 0, [], "hold"

    out_id, in_id = int(action.element_out), int(action.element_in)
    if out_id not in current_ids or in_id in current_ids:
        return list(current_ids), bank, free_transfers, 0, [], "hold_invalid"
    if in_id not in by_id:
        return list(current_ids), bank, free_transfers, 0, [], "hold_missing_in"

    sell = float(
        selling_prices.get(
            out_id,
            (by_id[out_id].get("price") if out_id in by_id else None) or 4.5,
        )
    )
    buy = float(by_id[in_id].get("price") or 99.0)
    if sell + bank + 1e-9 < buy:
        return list(current_ids), bank, free_transfers, 0, [], "hold_unaffordable"

    hits = max(0, 1 - int(free_transfers))
    if hits > int(max_hits):
        return list(current_ids), bank, free_transfers, 0, [], "hold_hit_cap"

    new_ids = [in_id if i == out_id else i for i in current_ids]
    new_bank = round(bank + sell - buy, 1)
    return (
        new_ids,
        new_bank,
        max(0, int(free_transfers) - 1),
        hits,
        [{"element_out": out_id, "element_in": in_id}],
        "transfer",
    )


def run_path_season_backtest(
    season: str = CHIP_BACKTEST_SEASON,
    *,
    max_hits: int = 1,
    use_chips: bool = False,
    half_split_gw: int = CHIP_DEFAULTS["half_split_gw"],
    cache_path: str = CHIP_BACKTEST_CACHE,
    ev_mode: EvMode = "ep_baseline",
    horizon: int | None = None,
    beam_width: int | None = None,
    blank_ahead: bool | None = False,
    fixture_run: bool | None = False,
) -> SeasonBacktestResult:
    """Full-season sim: each week take week-0 action from a 3-GW path plan."""
    cfg = dict(PATH_PLAN)
    if horizon is not None:
        cfg["horizon"] = int(horizon)
    if beam_width is not None:
        cfg["beam_width"] = int(beam_width)
    h = int(cfg["horizon"])

    merged = load_merged_gw(season=season, cache_path=cache_path)
    table = attach_lagged_history(prepare_gw_table(merged, season=season))
    team_fixtures = _team_fixtures_for_season(season, table)
    averages = _load_average_scores() if str(season).startswith("2025") else {}
    gameweeks = sorted(int(g) for g in table["gameweek"].unique())

    weekly: list[dict[str, Any]] = []
    current_ids: list[int] = []
    bank = 0.0
    selling_prices: dict[int, float] = {}
    free_transfers = 0
    total_points = 0.0
    hits_cost = 0
    transfers_made_total = 0
    chips_played: dict[str, list[int]] = {
        "triple_captain": [],
        "bench_boost": [],
        "free_hit": [],
        "wildcard": [],
    }
    inventory = None  # chips unused in this harness
    last_starting_xi: list[dict[str, Any]] = []
    last_bench: list[dict[str, Any]] = []
    last_captain_id: int | None = None
    last_squad: list[dict[str, Any]] = []
    path_holds = 0
    path_transfers = 0

    # Precompute prediction frames (base EV; week-specific FDR applied inside planner).
    pred_cache: dict[int, pd.DataFrame] = {}
    for gw in gameweeks:
        gw_df = table[table["gameweek"] == gw].copy()
        pred_cache[gw] = _prediction_frame(
            gw_df,
            mode=ev_mode,
            team_fixtures=team_fixtures,
            blank_ahead=blank_ahead,
            fixture_run=fixture_run,
            current_ids=None,
        )

    for gw in gameweeks:
        gw_df = table[table["gameweek"] == gw].copy()
        preds = pred_cache[gw]
        points = {int(r["id"]): float(r["total_points"]) for _, r in gw_df.iterrows()}
        minutes = {int(r["id"]): float(r["minutes"]) for _, r in gw_df.iterrows()}
        price_map = {int(r["id"]): float(r["price"]) for _, r in gw_df.iterrows()}

        path_meta: dict[str, Any] = {}
        if gw == gameweeks[0] or not current_ids:
            sol = optimise_single_gw(
                preds,
                budget=100.0,
                min_play_chance=None,
                require_cover_eligible=True,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
            )
            if sol.status != "Optimal":
                weekly.append({"gameweek": gw, "status": sol.status, "points": 0})
                continue
            plan_transfers: list[dict[str, int]] = []
            transfers_made = 0
            hits = 0
            hit_cost = 0
            squad = sol.squad
            captain_id = sol.captain_id
            starting_xi = sol.starting_xi
            bench = sol.bench
            bank = round(100.0 - float(sol.budget_used), 1)
            free_transfers = 1
            path_meta = {"week0": "initial_squad"}
        else:
            selling_prices = {
                pid: float(price_map.get(pid, selling_prices.get(pid, 4.5)))
                for pid in current_ids
            }
            preds_by_gw = {
                g: pred_cache[g]
                for g in range(gw, gw + h)
                if g in pred_cache
            }
            plan = plan_transfer_path(
                start_gw=gw,
                current_ids=current_ids,
                predictions_by_gw=preds_by_gw,
                team_fixtures=team_fixtures,
                bank=bank,
                free_transfers=free_transfers,
                selling_prices=selling_prices,
                config=cfg,
            )
            new_ids, bank, free_transfers, hits, plan_transfers, applied = _apply_path_action(
                action=plan.week0_action,
                current_ids=current_ids,
                preds=preds,
                bank=bank,
                selling_prices=selling_prices,
                free_transfers=free_transfers,
                max_hits=max_hits,
            )
            if applied == "transfer":
                path_transfers += 1
            else:
                path_holds += 1
            hit_cost = hits * 4
            hits_cost += hit_cost
            transfers_made = len(plan_transfers)
            transfers_made_total += transfers_made
            current_ids = new_ids
            path_meta = {
                "status": plan.status,
                "path_score": plan.path_score,
                "explanation": plan.explanation,
                "week0": plan.week0_action.kind,
                "applied": applied,
                "actions": [a.__dict__ for a in plan.actions],
                "graph": plan.graph_summary,
            }

            # Rebuild XI from owned preds.
            owned_preds = preds[preds["id"].isin(current_ids)].copy()
            owned_preds["cover_eligible"] = True
            owned_preds["ev"] = owned_preds["ev"].astype(float).clip(lower=0.01)
            lineup = optimise_single_gw(
                owned_preds,
                budget=1000.0,
                min_play_chance=None,
                require_cover_eligible=False,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
            )
            if lineup.status != "Optimal":
                starting_xi = last_starting_xi
                bench = last_bench
                captain_id = last_captain_id
                squad = last_squad
            else:
                starting_xi = lineup.starting_xi
                bench = lineup.bench
                captain_id = lineup.captain_id
                squad = lineup.squad

        # Chip recommendation left off by default — path A/B isolates transfer timing.
        chip = None
        if use_chips:
            chip = None  # reserved; chip timing not part of this experiment

        scored = _score_xi(
            starting_xi=starting_xi,
            bench=bench,
            captain_id=captain_id,
            points=points,
            minutes=minutes,
            chip=chip,
        )
        gw_pts = float(scored["points"]) - float(hit_cost if gw != gameweeks[0] else 0)
        total_points += gw_pts
        avg = averages.get(int(gw))
        weekly.append(
            {
                "gameweek": gw,
                "status": "Optimal",
                "points": gw_pts,
                "raw_points": scored["points"],
                "hit_cost": hit_cost if gw != gameweeks[0] else 0,
                "hits": hits if gw != gameweeks[0] else 0,
                "transfers_made": transfers_made if gw != gameweeks[0] else 0,
                "transfers": plan_transfers if gw != gameweeks[0] else [],
                "free_transfers": free_transfers,
                "bank": bank,
                "chip": chip,
                "captain_id": captain_id,
                "average_entry_score": avg,
                "delta_vs_average": None if avg is None else round(gw_pts - avg, 1),
                "path": path_meta,
            }
        )
        current_ids = [int(p["id"]) for p in squad]
        last_starting_xi = list(starting_xi)
        last_bench = list(bench)
        last_captain_id = captain_id
        last_squad = list(squad)
        free_transfers = min(5, free_transfers + 1)
        selling_prices = {
            pid: float(price_map.get(pid, selling_prices.get(pid, 4.5)))
            for pid in current_ids
        }

    scored_weeks = [w for w in weekly if w.get("status") == "Optimal"]
    avg_total = None
    if averages:
        avg_total = float(sum(averages.get(int(w["gameweek"]), 0.0) for w in scored_weeks))
    vs_avg = None if avg_total is None else round(float(total_points) - avg_total, 1)

    summary = {
        "ev_mode": ev_mode,
        "planner": "path_graph",
        "path_horizon": h,
        "path_beam": int(cfg["beam_width"]),
        "path_holds": path_holds,
        "path_transfers": path_transfers,
        "blank_ahead": blank_ahead,
        "fixture_run": fixture_run,
        "gameweeks_simulated": len(scored_weeks),
        "mean_gw_points": round(float(total_points) / max(1, len(scored_weeks)), 2),
        "notes": [
            "Week-0 action from beam-searched multi-GW path graph.",
            "Per-GW FDR scaling inside planner (entry-week sensitive).",
            "Chips disabled by default for clean A/B vs MIP baseline.",
        ],
    }
    return SeasonBacktestResult(
        season=season,
        total_points=int(round(total_points)),
        average_manager_points=avg_total,
        points_vs_average=vs_avg,
        hits_cost=int(hits_cost),
        transfers_made=int(transfers_made_total),
        chips_played=chips_played,
        weekly=weekly,
        summary=summary,
    )


def compare_path_vs_baseline(
    season: str = CHIP_BACKTEST_SEASON,
    *,
    max_hits: int = 1,
    report_file: str = "data/path_plan_compare.json",
) -> dict[str, Any]:
    from fpl_robo.backtest.season import run_season_backtest

    base = run_season_backtest(
        season=season,
        max_hits=max_hits,
        use_chips=False,
        ev_mode="ep_baseline",
        blank_ahead=False,
        fixture_run=False,
    )
    path = run_path_season_backtest(
        season=season,
        max_hits=max_hits,
        use_chips=False,
        ev_mode="ep_baseline",
        blank_ahead=False,
        fixture_run=False,
    )
    delta = int(path.total_points) - int(base.total_points)
    report = {
        "season": season,
        "winner": "path_graph" if delta > 0 else ("baseline" if delta < 0 else "tie"),
        "points_delta_path_minus_base": delta,
        "baseline": {
            "total_points": base.total_points,
            "hits_cost": base.hits_cost,
            "transfers_made": base.transfers_made,
        },
        "path_graph": {
            "total_points": path.total_points,
            "hits_cost": path.hits_cost,
            "transfers_made": path.transfers_made,
            "summary": path.summary,
        },
        "recommendation": (
            "Promote PATH_PLAN toward live after more seasons validate"
            if delta > 0
            else "Keep PATH_PLAN.enabled=False — tune beam/candidates/FDR"
        ),
    }
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_season_report(path, report_file.replace(".json", "_path_season.json"))
    return report
