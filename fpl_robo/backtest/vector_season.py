"""Season backtest using vector-team SA (non-MIP) instead of PuLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fpl_robo.backtest.season import (
    SeasonBacktestResult,
    _default_inventory,
    _load_average_scores,
    _mark_chip_used,
    _score_xi,
    write_season_report,
)
from fpl_robo.chips.backtest import load_merged_gw, prepare_gw_table
from fpl_robo.chips.decide import recommend_chips
from fpl_robo.chips.schedule import chip_schedule_adjustments
from fpl_robo.config import CHIP_BACKTEST_CACHE, CHIP_BACKTEST_SEASON, CHIP_DEFAULTS, VECTOR_TEAM
from fpl_robo.optimize.vector_search import anneal_squad, anneal_transfers
from fpl_robo.predict.ev_modes import EvMode, attach_lagged_history, build_predictions
from fpl_robo.predict.horizon import count_blank_owned, team_fixtures_from_merged
from fpl_robo.predict.vector_features import attach_vector_ev, fit_ridge_from_history


def merged_cache_for_season(season: str) -> str:
    """Season-specific merged_gw cache path (avoids always loading 2025-26)."""
    safe = str(season).strip()
    default = Path(CHIP_BACKTEST_CACHE)
    candidate = default.parent / f"merged_gw_{safe}.csv"
    return str(candidate)


def run_vector_season_backtest(
    season: str = CHIP_BACKTEST_SEASON,
    *,
    max_hits: int = 1,
    use_chips: bool = True,
    half_split_gw: int = CHIP_DEFAULTS["half_split_gw"],
    cache_path: str | None = None,
    sa_iters_initial: int | None = None,
    sa_iters_transfer: int | None = None,
    seed: int | None = None,
    blank_ahead: bool | None = True,
    use_horizon: bool = False,
    anchor_mode: EvMode | None = None,
) -> SeasonBacktestResult:
    """Full-season sim with xP-anchored vectors + team geometry + SA search.

    `use_horizon` anchors to `ep_baseline_h` (multi-GW fixture look-ahead).
    Chips use the same recommend_chips heuristics as the MIP season backtest.
    """
    cache = cache_path or merged_cache_for_season(season)
    mode: EvMode = anchor_mode or ("ep_baseline_h" if use_horizon else "ep_baseline")
    merged = load_merged_gw(season=season, cache_path=cache)
    table = attach_lagged_history(prepare_gw_table(merged, season=season))
    team_fixtures = team_fixtures_from_merged(table)
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
    inventory = _default_inventory(half_split_gw)
    hist_tc: list[float] = []
    hist_bb: list[float] = []
    last_starting_xi: list[dict[str, Any]] = []
    last_bench: list[dict[str, Any]] = []
    last_captain_id: int | None = None
    last_squad: list[dict[str, Any]] = []
    ridge_fitted_weeks = 0

    for gw in gameweeks:
        gw_df = table[table["gameweek"] == gw].copy()
        model = fit_ridge_from_history(table, before_gw=gw, team_fixtures=team_fixtures)
        if model.fitted:
            ridge_fitted_weeks += 1
        # Anchor to ep_baseline(+optional horizon/blank-ahead), then residual vector.
        anchor = build_predictions(
            gw_df,
            mode=mode,
            team_fixtures=team_fixtures,
            blank_ahead=blank_ahead,
            current_ids=current_ids or None,
        )
        anchor_by_id = anchor.set_index("id")["ev"].astype(float)
        preds = attach_vector_ev(
            gw_df,
            model,
            team_fixtures=team_fixtures,
            xp_anchor=gw_df["id"].map(anchor_by_id),
        )
        if "blank_ahead_triggered" in anchor.columns:
            ba = anchor.set_index("id")["blank_ahead_triggered"]
            preds["blank_ahead_triggered"] = preds["id"].map(ba).fillna(False)

        points = {int(r["id"]): float(r["total_points"]) for _, r in gw_df.iterrows()}
        minutes = {int(r["id"]): float(r["minutes"]) for _, r in gw_df.iterrows()}
        price_map = {int(r["id"]): float(r["price"]) for _, r in gw_df.iterrows()}

        if gw == gameweeks[0] or not current_ids:
            sol = anneal_squad(
                preds,
                budget=100.0,
                n_iters=sa_iters_initial,
                seed=seed,
            )
            if sol.status != "Optimal":
                weekly.append({"gameweek": gw, "status": sol.status, "points": 0})
                continue
            plan_transfers: list[dict[str, int]] = []
            transfers_made = 0
            hits = 0
            hit_cost = 0
            objective = sol.objective
            squad = sol.squad
            captain_id = sol.captain_id
            bank = round(100.0 - float(sol.budget_used), 1)
            free_transfers = 1
            fallback_used = False
        else:
            selling_prices = {
                pid: float(price_map.get(pid, selling_prices.get(pid, 4.5))) for pid in current_ids
            }
            plan = anneal_transfers(
                preds,
                current_ids=current_ids,
                bank=bank,
                selling_prices=selling_prices,
                free_transfers=free_transfers,
                max_hits=max_hits,
                n_iters=sa_iters_transfer,
                seed=None if seed is None else int(seed) + int(gw),
            )
            if plan.status != "Optimal":
                if not last_starting_xi:
                    weekly.append({"gameweek": gw, "status": plan.status, "points": 0})
                    free_transfers = min(5, free_transfers + 1)
                    continue
                scored = _score_xi(
                    starting_xi=last_starting_xi,
                    bench=last_bench,
                    captain_id=last_captain_id,
                    points=points,
                    minutes=minutes,
                    chip=None,
                )
                gw_points = float(scored["points"])
                total_points += gw_points
                avg = averages.get(gw)
                free_transfers = min(5, free_transfers + 1)
                weekly.append(
                    {
                        "gameweek": gw,
                        "status": "Held",
                        "points": round(gw_points, 1),
                        "raw_xi_points": scored["points"],
                        "hit_cost": 0,
                        "hits": 0,
                        "transfers_made": 0,
                        "transfers": [],
                        "free_transfers_next": free_transfers,
                        "bank": bank,
                        "objective": None,
                        "captain_id": last_captain_id,
                        "chip": None,
                        "fallback_lineup_only": True,
                        "average_entry_score": avg,
                        "delta_vs_average": None if avg is None else round(gw_points - avg, 1),
                        "squad": last_squad,
                    }
                )
                continue
            fallback_used = False
            sol = plan.solution
            plan_transfers = plan.transfers
            transfers_made = plan.transfers_made
            hits = plan.hits
            hit_cost = plan.hit_cost
            objective = plan.objective
            squad = sol.squad
            captain_id = sol.captain_id
            bank = plan.budget_left
            transfers_made_total += transfers_made
            hits_cost += hit_cost
            free_transfers = min(5, max(0, free_transfers - transfers_made) + 1)

        chip = None
        if use_chips:
            n_blank = 0
            dgw_teams: set[int] = set()
            if team_fixtures:
                for tid, by_gw in team_fixtures.items():
                    slot = by_gw.get(int(gw)) or {}
                    if float(slot.get("n_fixtures") or 0) >= 2:
                        dgw_teams.add(int(tid))
            if current_ids:
                n_blank = count_blank_owned(
                    current_ids=current_ids,
                    predictions=preds,
                    team_fixtures=team_fixtures,
                    gameweek=gw,
                )
            chip_cfg, _schedule = chip_schedule_adjustments(
                gameweek=gw,
                starting_xi=sol.starting_xi,
                bench=sol.bench,
                captain_id=captain_id,
                dgw_teams=dgw_teams,
                team_fixtures=team_fixtures,
                blank_owned=n_blank,
            )
            chip_rec = recommend_chips(
                gameweek=gw,
                starting_xi=sol.starting_xi,
                bench=sol.bench,
                captain_id=captain_id,
                inventory=inventory,
                historical_tc_scores=hist_tc,
                historical_bb_scores=hist_bb,
                config=chip_cfg or None,
            )
            hist_tc.append(float(chip_rec.scores["triple_captain"]))
            hist_bb.append(float(chip_rec.scores["bench_boost"]))
            play = chip_rec.play_now
            if play.get("triple_captain"):
                chip = "3xc"
                _mark_chip_used(inventory, "triple_captain", gw)
                chips_played["triple_captain"].append(gw)
            elif play.get("bench_boost"):
                chip = "bboost"
                _mark_chip_used(inventory, "bench_boost", gw)
                chips_played["bench_boost"].append(gw)
            elif play.get("wildcard"):
                _mark_chip_used(inventory, "wildcard", gw)
                chips_played["wildcard"].append(gw)
            elif play.get("free_hit"):
                _mark_chip_used(inventory, "free_hit", gw)
                chips_played["free_hit"].append(gw)

        scored = _score_xi(
            starting_xi=sol.starting_xi,
            bench=sol.bench,
            captain_id=captain_id,
            points=points,
            minutes=minutes,
            chip=chip,
        )
        gw_points = float(scored["points"]) - float(hit_cost)
        total_points += gw_points
        avg = averages.get(gw)

        current_ids = [int(p["id"]) for p in squad]
        selling_prices = {pid: float(price_map.get(pid, 4.5)) for pid in current_ids}
        last_starting_xi = list(sol.starting_xi)
        last_bench = list(sol.bench)
        last_captain_id = captain_id
        last_squad = list(squad)

        weekly.append(
            {
                "gameweek": gw,
                "status": "Optimal",
                "points": round(gw_points, 1),
                "raw_xi_points": scored["points"],
                "hit_cost": hit_cost,
                "hits": hits if gw != gameweeks[0] else 0,
                "transfers_made": transfers_made if gw != gameweeks[0] else 0,
                "transfers": plan_transfers if gw != gameweeks[0] else [],
                "free_transfers_next": free_transfers,
                "bank": bank,
                "objective": objective,
                "captain_id": captain_id,
                "captain_name": sol.captain_name,
                "chip": chip,
                "fallback_lineup_only": bool(fallback_used) if gw != gameweeks[0] else False,
                "average_entry_score": avg,
                "delta_vs_average": None if avg is None else round(gw_points - avg, 1),
                "ridge_fitted": bool(model.fitted),
                "ridge_n_train": int(model.n_train),
                "squad": squad,
                "starting_xi": sol.starting_xi,
                "bench": sol.bench,
            }
        )

    scored_weeks = [w for w in weekly if w.get("status") in ("Optimal", "Held")]
    avg_total = sum(float(averages[g]) for g in averages if g in {w["gameweek"] for w in scored_weeks})
    vs_avg = None
    if averages:
        matched = [w for w in scored_weeks if w.get("average_entry_score") is not None]
        if matched:
            avg_total = sum(float(w["average_entry_score"]) for w in matched)
            vs_avg = round(total_points - avg_total, 1)
            avg_total = round(avg_total, 1)
        else:
            avg_total = None
    else:
        avg_total = None

    summary = {
        "solver": "vector_team_sa",
        "feature_version": "proto-vector-team",
        "anchor_mode": mode,
        "use_horizon": bool(use_horizon) or str(mode).endswith("_h"),
        "blank_ahead": blank_ahead,
        "use_chips": bool(use_chips),
        "max_hits": int(max_hits),
        "mean_gw_points": round(
            sum(float(w.get("points") or 0) for w in scored_weeks) / max(1, len(scored_weeks)), 2
        ),
        "weeks_beat_average": sum(
            1
            for w in scored_weeks
            if w.get("delta_vs_average") is not None and float(w["delta_vs_average"]) > 0
        ),
        "weeks_with_average": sum(1 for w in scored_weeks if w.get("average_entry_score") is not None),
        "ridge_fitted_weeks": ridge_fitted_weeks,
        "sa_iters_initial": sa_iters_initial or VECTOR_TEAM["sa_iters_initial"],
        "sa_iters_transfer": sa_iters_transfer or VECTOR_TEAM["sa_iters_transfer"],
        "chips_played": chips_played,
        "notes": [
            f"base_points = {mode}(+blank-ahead) + residual_scale * ridge(player vector).",
            "Team geometry (fixture portfolio, attack, minutes, club soft) on top of XI sum.",
            "Squad search via SA + hill-climb with constraint repair (non-MIP).",
            "Chips: TC/BB scored; FH/WC tracked when recommended (same limits as MIP harness).",
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


def compare_vector_vs_baseline(
    season: str = CHIP_BACKTEST_SEASON,
    *,
    max_hits: int = 1,
    use_chips: bool = False,
    use_horizon: bool = False,
    blank_ahead: bool | None = True,
    report_path: str = "data/vector_team_comparison.json",
    sa_iters_initial: int | None = None,
    sa_iters_transfer: int | None = None,
    write_side_reports: bool = True,
) -> dict[str, Any]:
    """Run ep_baseline(+optional _h) MIP and vector-team SA; write comparison JSON."""
    import json

    from fpl_robo.backtest.season import run_season_backtest

    ev_mode: EvMode = "ep_baseline_h" if use_horizon else "ep_baseline"
    cache = merged_cache_for_season(season)

    baseline = run_season_backtest(
        season=season,
        max_hits=max_hits,
        use_chips=use_chips,
        ev_mode=ev_mode,
        use_tiebreak=True,
        blank_ahead=blank_ahead,
        cache_path=cache,
    )
    vector = run_vector_season_backtest(
        season=season,
        max_hits=max_hits,
        use_chips=use_chips,
        cache_path=cache,
        sa_iters_initial=sa_iters_initial,
        sa_iters_transfer=sa_iters_transfer,
        blank_ahead=blank_ahead,
        use_horizon=use_horizon,
        anchor_mode=ev_mode,
    )

    b_by_gw = {int(w["gameweek"]): w for w in baseline.weekly if w.get("status") == "Optimal"}
    v_by_gw = {int(w["gameweek"]): w for w in vector.weekly if w.get("status") == "Optimal"}
    gw_delta = []
    for gw in sorted(set(b_by_gw) & set(v_by_gw)):
        gw_delta.append(
            {
                "gameweek": gw,
                "baseline_points": b_by_gw[gw].get("points"),
                "vector_points": v_by_gw[gw].get("points"),
                "delta_vector_minus_baseline": round(
                    float(v_by_gw[gw].get("points") or 0) - float(b_by_gw[gw].get("points") or 0),
                    1,
                ),
                "baseline_transfers": b_by_gw[gw].get("transfers_made"),
                "vector_transfers": v_by_gw[gw].get("transfers_made"),
                "baseline_chip": b_by_gw[gw].get("chip"),
                "vector_chip": v_by_gw[gw].get("chip"),
            }
        )

    payload = {
        "season": season,
        "max_hits": max_hits,
        "use_chips": use_chips,
        "use_horizon": use_horizon,
        "blank_ahead": blank_ahead,
        "ev_mode": ev_mode,
        "baseline": {
            "solver": f"mip_{ev_mode}",
            "total_points": baseline.total_points,
            "hits_cost": baseline.hits_cost,
            "transfers_made": baseline.transfers_made,
            "points_vs_average": baseline.points_vs_average,
            "chips_played": baseline.chips_played,
            "summary": baseline.summary,
        },
        "vector_team": {
            "solver": "sa_vector_team",
            "total_points": vector.total_points,
            "hits_cost": vector.hits_cost,
            "transfers_made": vector.transfers_made,
            "points_vs_average": vector.points_vs_average,
            "chips_played": vector.chips_played,
            "summary": vector.summary,
        },
        "delta_total_points": int(vector.total_points) - int(baseline.total_points),
        "winner": (
            "vector_team"
            if vector.total_points > baseline.total_points
            else "baseline"
            if baseline.total_points > vector.total_points
            else "tie"
        ),
        "weekly_delta": gw_delta,
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if write_side_reports:
        write_season_report(baseline, path.with_name(path.stem + "_baseline.json"))
        write_season_report(vector, path.with_name(path.stem + "_vector.json"))
    return payload


def run_vector_evaluation_suite(
    *,
    seasons: list[str] | None = None,
    report_path: str = "data/vector_team_eval_suite.json",
    sa_iters_initial: int | None = 2000,
    sa_iters_transfer: int | None = 1200,
) -> dict[str, Any]:
    """Expanded A/B: chips on/off, horizon, hit budgets, optional multi-season."""
    import json

    seasons = seasons or ["2025-26"]
    configs: list[dict[str, Any]] = [
        {"label": "no_chips", "use_chips": False, "use_horizon": False, "max_hits": 1},
        {"label": "with_chips", "use_chips": True, "use_horizon": False, "max_hits": 1},
        {"label": "chips_horizon", "use_chips": True, "use_horizon": True, "max_hits": 1},
        {"label": "chips_hits2", "use_chips": True, "use_horizon": False, "max_hits": 2},
        {"label": "horizon_no_chips", "use_chips": False, "use_horizon": True, "max_hits": 1},
    ]

    runs: list[dict[str, Any]] = []
    for season in seasons:
        cache = Path(merged_cache_for_season(season))
        if not cache.exists():
            runs.append(
                {
                    "season": season,
                    "skipped": True,
                    "reason": f"missing cache {cache}",
                }
            )
            continue
        for cfg in configs:
            label = f"{season}/{cfg['label']}"
            print(f"=== Suite run: {label} ===", flush=True)
            try:
                result = compare_vector_vs_baseline(
                    season=season,
                    max_hits=int(cfg["max_hits"]),
                    use_chips=bool(cfg["use_chips"]),
                    use_horizon=bool(cfg["use_horizon"]),
                    blank_ahead=True,
                    report_path=f"data/vector_suite_{season}_{cfg['label']}.json",
                    sa_iters_initial=sa_iters_initial,
                    sa_iters_transfer=sa_iters_transfer,
                    write_side_reports=False,
                )
                runs.append(
                    {
                        "label": label,
                        "season": season,
                        **{k: cfg[k] for k in ("use_chips", "use_horizon", "max_hits")},
                        "winner": result["winner"],
                        "delta_total_points": result["delta_total_points"],
                        "baseline_points": result["baseline"]["total_points"],
                        "vector_points": result["vector_team"]["total_points"],
                        "baseline_hits": result["baseline"]["hits_cost"],
                        "vector_hits": result["vector_team"]["hits_cost"],
                        "baseline_xfers": result["baseline"]["transfers_made"],
                        "vector_xfers": result["vector_team"]["transfers_made"],
                        "baseline_chips": result["baseline"].get("chips_played"),
                        "vector_chips": result["vector_team"].get("chips_played"),
                    }
                )
                print(
                    f"  → {result['winner']} ({result['delta_total_points']:+d}) "
                    f"vec={result['vector_team']['total_points']} "
                    f"base={result['baseline']['total_points']}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                runs.append(
                    {
                        "label": label,
                        "season": season,
                        "error": str(exc),
                        **{k: cfg[k] for k in ("use_chips", "use_horizon", "max_hits")},
                    }
                )
                print(f"  → ERROR: {exc}", flush=True)

    scored = [r for r in runs if "delta_total_points" in r]
    vector_wins = sum(1 for r in scored if r.get("winner") == "vector_team")
    baseline_wins = sum(1 for r in scored if r.get("winner") == "baseline")
    suite = {
        "seasons": seasons,
        "n_runs": len(scored),
        "vector_wins": vector_wins,
        "baseline_wins": baseline_wins,
        "ties": sum(1 for r in scored if r.get("winner") == "tie"),
        "mean_delta": (
            round(sum(r["delta_total_points"] for r in scored) / len(scored), 1) if scored else None
        ),
        "runs": runs,
        "verdict": (
            "vector_team_ahead"
            if vector_wins > baseline_wins
            else "baseline_ahead"
            if baseline_wins > vector_wins
            else "mixed"
        ),
        "notes": [
            "Each run pairs MIP (ep_baseline or ep_baseline_h) vs vector SA with matching knobs.",
            "Horizon = multi-GW fixture look-ahead on the EV anchor.",
            "Chips use shared recommend_chips heuristics; FH/WC are not full one-week rebuilds.",
            "Prior no-chips 2025-26 result was vector +131.",
        ],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite
