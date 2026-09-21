"""Unattended weekly season runner for Raspberry Pi."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.auth import ensure_authenticated, load_auth_config
from fpl_robo.chips.decide import recommend_chips
from fpl_robo.chips.inventory import parse_chip_inventory
from fpl_robo.chips.lineup_gate import apply_lineup_chip_gates
from fpl_robo.chips.schedule import chip_schedule_adjustments
from fpl_robo.config import (
    BENCH_COVER,
    BLANK_AHEAD,
    CHIP_DEFAULTS,
    DEFAULT_MANAGER_ID,
    DEFAULT_SIGNALS_PATH,
    DEFAULT_SUMMARY_PATH,
    EV_MODE,
    FIXTURE_RUN,
    HORIZON_EV,
    KG_ENRICH,
    PATH_PLAN,
    PRESEASON_HORIZON,
    SEASON_RUNNER,
    VECTOR_TEAM,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.live_team import load_live_team_state
from fpl_robo.data.rolling_history import attach_history_to_graph, load_rolling_features
from fpl_robo.data.team_updater import apply_weekly_plan
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.enrich import enrich_knowledge_graph
from fpl_robo.graph.signals import enrich_graph_with_signals
from fpl_robo.optimize.single_gw import optimise_single_gw
from fpl_robo.optimize.transfers import TransferPlan, optimise_transfers
from fpl_robo.optimize.live_vector import (
    hold_plan_from_current,
    live_optimise_single_gw,
    live_optimise_transfers,
    transfer_plan_from_rebuild,
    vector_live_enabled,
)
from fpl_robo.plan.live_path import live_path_optimise_transfers, path_plan_live_enabled
from fpl_robo.predict.ev_modes import make_live_ev_model, transfer_gate_kwargs
from fpl_robo.predict.horizon import (
    apply_blank_ahead_ev,
    apply_horizon_ev,
    count_blank_owned,
    mode_uses_horizon,
    team_fixtures_from_kg,
)
from fpl_robo.runtime.atomic import atomic_write_json
from fpl_robo.runtime.deadline import (
    detect_dgw_teams,
    hours_to_deadline,
    is_preseason,
    next_deadline,
)
from fpl_robo.runtime.interrupt import install_stop_handlers, interruptible_sleep, stop_requested
from fpl_robo.runtime.lockfile import RunLock
from fpl_robo.signals import aggregate_player_signals, load_signals

logger = logging.getLogger("fpl_robo.season")


@dataclass
class SeasonRunResult:
    ok: bool
    skipped: bool
    reason: str | None
    gameweek: int | None
    hours_to_deadline: float | None
    transfer_plan: dict[str, Any] | None
    chips: dict[str, Any] | None
    execution: dict[str, Any] | None
    dgw_teams: list[int]
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def setup_logging(log_file: str | None = None) -> None:
    log_file = log_file or SEASON_RUNNER["log_file"]
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def run_week(
    *,
    manager_id: int = DEFAULT_MANAGER_ID,
    execute: bool = False,
    force: bool = False,
    max_hits: int | None = None,
    signals_path: str | None = DEFAULT_SIGNALS_PATH,
    summary_file: str = DEFAULT_SUMMARY_PATH,
    state_file: str | None = None,
    client: FPLClient | None = None,
    load_session: bool = True,
    save_session: bool = True,
    optimise_triggers: list[str] | None = None,
) -> SeasonRunResult:
    """One weekly decision cycle: predict → transfers → chips → optional execute."""
    setup_logging()
    state_path = Path(state_file or SEASON_RUNNER["state_file"])
    lock = RunLock(Path(SEASON_RUNNER["lock_file"]))
    if not lock.acquire(stale_seconds=int(SEASON_RUNNER["lock_stale_seconds"])):
        return SeasonRunResult(
            ok=False,
            skipped=True,
            reason="Another run is in progress (lockfile)",
            gameweek=None,
            hours_to_deadline=None,
            transfer_plan=None,
            chips=None,
            execution=None,
            dgw_teams=[],
            summary_path=summary_file,
        )

    try:
        lock.touch()
        client = client or FPLClient()
        cfg = load_auth_config()
        auth = ensure_authenticated(
            client,
            cfg,
            load_session=load_session,
            save_session=save_session,
        )
        authed = bool(auth.get("ok"))
        if not authed:
            reason = str(auth.get("reason") or "unknown")
            logger.error("Auth failed: %s", reason)
            try:
                from fpl_robo.notify import notify_auth_failure

                notify_auth_failure(reason=reason, source="season-daemon")
            except Exception:  # noqa: BLE001
                logger.exception("Failed to send auth failure alert")
            if execute:
                return _fail(summary_file, f"Auth failed: {reason}")
            logger.warning("Continuing dry-run with public endpoints only")
        else:
            try:
                from fpl_robo.notify import clear_auth_alert_stamp

                clear_auth_alert_stamp()
            except Exception:  # noqa: BLE001
                pass

        bootstrap = client.bootstrap_static()
        deadline = next_deadline(bootstrap)
        if deadline is None:
            return _fail(summary_file, "No upcoming deadline found")
        gw = int(deadline["id"])
        hours = hours_to_deadline(deadline["deadline_utc"])
        logger.info("Next deadline GW%s in %.2f hours", gw, hours)
        logger.info(
            "EV run context: mode=%s feature=%s",
            EV_MODE,
            KG_ENRICH.get("feature_version"),
        )

        state = _load_state(state_path)
        in_window = 0 < hours <= float(SEASON_RUNNER["execute_within_hours"])

        live = load_live_team_state(manager_id=manager_id, client=client)
        if not live.picks:
            return _fail(summary_file, "No live picks available")

        unlimited = bool(live.unlimited_transfers) and bool(
            SEASON_RUNNER.get("execute_when_unlimited", True)
        )
        can_write = _can_auto_execute(
            hours_to_deadline=hours,
            unlimited_transfers=unlimited,
            force=force,
        )

        # Reconcile interrupted executes before deciding whether to skip.
        if authed:
            _reconcile_execute_in_progress(
                state=state,
                state_path=state_path,
                summary_file=summary_file,
                client=client,
                manager_id=manager_id,
                gw=gw,
                in_window=can_write and hours > 0,
            )

        if (
            not force
            and state.get("last_executed_gw") == gw
            and state.get("last_execute_ok")
            and execute
            and not unlimited
        ):
            return SeasonRunResult(
                ok=True,
                skipped=True,
                reason=f"Already executed for GW{gw}",
                gameweek=gw,
                hours_to_deadline=hours,
                transfer_plan=None,
                chips=None,
                execution=None,
                dgw_teams=[],
                summary_path=summary_file,
            )

        # Normal mode: only auto-execute in the final pre-deadline window.
        # Unlimited FT (preseason / WC / FH): apply as soon as the plan changes.
        if execute and not force and not can_write:
            execute = False
            logger.info(
                "Outside execute window (need 0 < hours <= %s, or unlimited transfers); dry-run only",
                SEASON_RUNNER["execute_within_hours"],
            )
        elif execute and unlimited and not in_window:
            logger.info(
                "Unlimited transfers active — will apply plan changes immediately (chips wait for final %.1fh window)",
                float(SEASON_RUNNER["execute_within_hours"]),
            )

        fixtures = client.fixtures()
        dgw_teams = sorted(detect_dgw_teams(fixtures, gw))

        kg = build_knowledge_graph(bootstrap=bootstrap, fixtures=fixtures, client=client)
        try:
            hist_features = load_rolling_features(
                client=client,
                bootstrap=bootstrap,
                decision_gw=gw,
            )
            n_hist = attach_history_to_graph(kg, hist_features)
            if n_hist:
                logger.info("Attached lagged history features to %s players", n_hist)
        except Exception:  # noqa: BLE001
            logger.exception("Rolling history attach failed; using season-minutes proxy")

        bundle = load_signals(signals_path)
        enrich_graph_with_signals(kg, bundle)
        # Re-run structural enrichment after history so rolling xGI / form sit on players.
        enrich_stats = enrich_knowledge_graph(kg, gameweek=gw)
        logger.info(
            "KG enrich: players=%s teams=%s player_faces=%s set_piece=%s rivals=%s",
            enrich_stats.get("players"),
            enrich_stats.get("teams"),
            enrich_stats.get("player_faces"),
            enrich_stats.get("set_piece_edges"),
            enrich_stats.get("rival_edges"),
        )
        factors = aggregate_player_signals(
            bundle, players_by_id={int(p["id"]): p for p in kg.players()}
        )
        # In-season model selected by EV_MODE (set after compare-ev-models).
        from fpl_robo.predict.phase import phase_kg_enrich

        preseason_now = is_preseason(bootstrap)
        model = make_live_ev_model(EV_MODE)
        # Attach signal multipliers by mutating predictions after predict.
        with phase_kg_enrich(preseason=preseason_now, gameweek=gw):
            predictions = model.predict(kg, gameweek=gw)
            from fpl_robo.optimize.kg_features import enrich_predictions_from_kg

            predictions = enrich_predictions_from_kg(predictions, kg)
            if factors:
                predictions = _apply_signal_factors(predictions, factors)

        weights = list(HORIZON_EV.get("weights") or [])
        ba_look = int(BLANK_AHEAD.get("lookahead") or 0) if BLANK_AHEAD.get("enabled") else 0
        vec_h = int(VECTOR_TEAM.get("horizon") or 3)
        fh_look = int(CHIP_DEFAULTS.get("fh_lookahead_gws") or 6) if CHIP_DEFAULTS.get("fh_plan_ahead", True) else 0
        if preseason_now:
            vec_h = max(vec_h, int(PRESEASON_HORIZON))
        team_fixtures = team_fixtures_from_kg(
            kg,
            start_gw=gw,
            horizon=max(1, len(weights), ba_look + 1, vec_h, fh_look + 1),
        )
        current_ids = [int(p["element"]) for p in live.picks]
        if mode_uses_horizon(EV_MODE):
            predictions = apply_horizon_ev(
                predictions, start_gw=gw, team_fixtures=team_fixtures
            )
        # Fixture-run for vector is applied inside prepare_vector_predictions /
        # live_optimise_transfers (post-residual). MIP path applies below.
        if bool(BLANK_AHEAD.get("enabled")):
            predictions = apply_blank_ahead_ev(
                predictions,
                start_gw=gw,
                team_fixtures=team_fixtures,
                current_ids=current_ids,
            )

        hits_budget = max_hits if max_hits is not None else int(SEASON_RUNNER["max_hits"])
        owned_risk_ids = _owned_risk_element_ids(
            current_ids, predictions=predictions, factors=factors
        )
        if path_plan_live_enabled() and not unlimited:
            plan = live_path_optimise_transfers(
                predictions,
                current_ids=current_ids,
                bank=live.bank,
                selling_prices=live.selling_prices,
                free_transfers=live.free_transfers,
                max_hits=hits_budget,
                unlimited_transfers=False,
                team_fixtures=team_fixtures,
                gameweek=gw,
                require_cover_eligible=True,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                owned_risk_count=len(owned_risk_ids),
                **transfer_gate_kwargs(EV_MODE),
            )
            logger.info(
                "Optimiser=path_graph transfers=%s hits=%s status=%s | %s",
                plan.transfers_made,
                plan.hits,
                plan.status,
                (plan.meta or {}).get("path_explanation") or "",
            )
            shadow_mip_plan = None
            if bool(SEASON_RUNNER.get("shadow_mip", True)):
                try:
                    shadow = optimise_transfers(
                        predictions,
                        current_ids=current_ids,
                        bank=live.bank,
                        selling_prices=live.selling_prices,
                        free_transfers=live.free_transfers,
                        max_hits=hits_budget,
                        require_cover_eligible=True,
                        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                        **transfer_gate_kwargs(EV_MODE),
                        use_tiebreak=None,
                    )
                    shadow_mip_plan = shadow.to_dict()
                    shadow_mip_plan["optimiser"] = "mip_pulp_shadow"
                    logger.info(
                        "Shadow MIP transfers=%s hits=%s status=%s obj=%.3f (path_obj=%.3f)",
                        shadow.transfers_made,
                        shadow.hits,
                        shadow.status,
                        float(shadow.objective),
                        float(plan.objective),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Shadow MIP failed (non-fatal): %s", exc)
                    shadow_mip_plan = {"status": "Failed", "error": str(exc)}
        elif vector_live_enabled():
            plan = live_optimise_transfers(
                predictions,
                current_ids=current_ids,
                bank=live.bank,
                selling_prices=live.selling_prices,
                free_transfers=live.free_transfers,
                max_hits=hits_budget,
                unlimited_transfers=unlimited,
                team_fixtures=team_fixtures,
                gameweek=gw,
                require_cover_eligible=True,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                owned_risk_count=len(owned_risk_ids),
                **transfer_gate_kwargs(EV_MODE),
            )
            logger.info(
                "Optimiser=vector_team_sa transfers=%s hits=%s status=%s",
                plan.transfers_made,
                plan.hits,
                plan.status,
            )
            shadow_mip_plan = None
            if bool(SEASON_RUNNER.get("shadow_mip", True)):
                try:
                    shadow = optimise_transfers(
                        predictions,
                        current_ids=current_ids,
                        bank=live.bank,
                        selling_prices=live.selling_prices,
                        free_transfers=live.free_transfers,
                        max_hits=hits_budget,
                        require_cover_eligible=True,
                        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                        **transfer_gate_kwargs(EV_MODE),
                        use_tiebreak=None,
                    )
                    shadow_mip_plan = shadow.to_dict()
                    shadow_mip_plan["optimiser"] = "mip_pulp_shadow"
                    logger.info(
                        "Shadow MIP transfers=%s hits=%s status=%s obj=%.3f (vector_obj=%.3f)",
                        shadow.transfers_made,
                        shadow.hits,
                        shadow.status,
                        float(shadow.objective),
                        float(plan.objective),
                    )
                    plan = _apply_shadow_mip_hold_gate(
                        plan=plan,
                        shadow=shadow,
                        predictions=predictions,
                        current_ids=current_ids,
                        owned_risk_ids=owned_risk_ids,
                        bank=float(live.bank),
                        selling_prices=live.selling_prices,
                        free_transfers=int(live.free_transfers),
                        team_fixtures=team_fixtures,
                        gameweek=gw,
                    )
                    plan = _apply_shadow_mip_prefer_gate(
                        plan=plan,
                        shadow=shadow,
                        predictions=predictions,
                        owned_risk_ids=owned_risk_ids,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Shadow MIP failed (non-fatal): %s", exc)
                    shadow_mip_plan = {"status": "Failed", "error": str(exc)}
        else:
            shadow_mip_plan = None
            from fpl_robo.predict.horizon import mode_uses_fixture_run
            from fpl_robo.predict.fixture_run import apply_fixture_run_ev

            mip_preds = predictions
            if (
                not mode_uses_horizon(EV_MODE)
                and team_fixtures
                and (bool(FIXTURE_RUN.get("enabled")) or mode_uses_fixture_run(EV_MODE))
            ):
                mip_preds = apply_fixture_run_ev(
                    predictions, start_gw=gw, team_fixtures=team_fixtures
                )
            plan = optimise_transfers(
                mip_preds,
                current_ids=current_ids,
                bank=live.bank,
                selling_prices=live.selling_prices,
                free_transfers=live.free_transfers,
                max_hits=hits_budget,
                require_cover_eligible=True,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                **transfer_gate_kwargs(EV_MODE),
                use_tiebreak=None,  # honour OPTIMISER_TIEBREAK["enabled"]
            )
        if plan.status != "Optimal":
            return _fail(summary_file, f"Transfer optimiser status: {plan.status}", gw, hours)

        # Chip scoring can use horizon-adjusted EV without changing transfer picks.
        chip_predictions = predictions
        if bool(CHIP_DEFAULTS.get("use_horizon_ev", True)) and not mode_uses_horizon(EV_MODE):
            chip_predictions = apply_horizon_ev(
                predictions, start_gw=gw, team_fixtures=team_fixtures
            )
        chip_xi, chip_bench = _remap_squad_ev(
            starting_xi=plan.solution.starting_xi,
            bench=plan.solution.bench,
            predictions=chip_predictions,
        )

        chip_inputs = _compute_chip_opportunity_inputs(
            predictions=chip_predictions,
            current_ids=current_ids,
            bank=live.bank,
            selling_prices=live.selling_prices,
            team_value=float(live.manager.team_value or 0.0),
            state=state,
            gameweek=gw,
        )

        inventory = parse_chip_inventory(live.raw_my_team)
        n_blank = count_blank_owned(
            current_ids=current_ids,
            predictions=chip_predictions,
            team_fixtures=team_fixtures,
            gameweek=gw,
        )
        rebuild_xi_for_chips = None
        unconstrained = chip_inputs.get("unconstrained_solution")
        if unconstrained is not None and getattr(unconstrained, "starting_xi", None):
            rebuild_xi_for_chips, _ = _remap_squad_ev(
                starting_xi=list(unconstrained.starting_xi),
                bench=list(getattr(unconstrained, "bench", None) or []),
                predictions=chip_predictions,
            )

        chip_cfg, schedule_details = chip_schedule_adjustments(
            gameweek=gw,
            starting_xi=chip_xi,
            bench=chip_bench,
            captain_id=plan.solution.captain_id,
            dgw_teams=dgw_teams,
            team_fixtures=team_fixtures,
            blank_owned=n_blank,
        )
        chips = recommend_chips(
            gameweek=gw,
            starting_xi=chip_xi,
            bench=chip_bench,
            captain_id=plan.solution.captain_id,
            inventory=inventory,
            optimal_xi_ev=chip_inputs["optimal_xi_ev"],
            current_xi_ev=chip_inputs["current_xi_ev"],
            rolling_gap=chip_inputs["rolling_gap"],
            historical_tc_scores=chip_inputs["historical_tc_scores"],
            historical_bb_scores=chip_inputs["historical_bb_scores"],
            config=chip_cfg or None,
            team_fixtures=team_fixtures,
            blank_owned=n_blank,
            rebuild_xi=rebuild_xi_for_chips,
        ).to_dict()
        chips["opportunity"] = {
            "current_xi_ev": chip_inputs["current_xi_ev"],
            "optimal_xi_ev": chip_inputs["optimal_xi_ev"],
            "fh_gap": chip_inputs["fh_gap"],
            "rolling_gap": chip_inputs["rolling_gap"],
            "blank_owned": n_blank,
            "horizon_ev": bool(CHIP_DEFAULTS.get("use_horizon_ev", True)),
            "fh_timing": (chips.get("details") or {}).get("fh_timing"),
            "tc_timing": (chips.get("details") or {}).get("tc_timing"),
            "bb_timing": (chips.get("details") or {}).get("bb_timing"),
            "wc_timing": (chips.get("details") or {}).get("wc_timing"),
        }
        chips["schedule"] = schedule_details
        if schedule_details.get("applied"):
            logger.info("Chip schedule tweaks: %s", ", ".join(schedule_details["applied"]))
        if bool(CHIP_DEFAULTS.get("lineup_gate", True)):
            gate = apply_lineup_chip_gates(
                chips,
                gameweek=gw,
                captain_id=plan.solution.captain_id,
                starting_xi=chip_xi,
                bench=chip_bench,
                predicted_lineups=list(bundle.predicted_lineups),
                bb_min_startish=int(CHIP_DEFAULTS.get("bb_lineup_min_startish") or 2),
            )
            if gate.get("suppressed"):
                logger.info("Chip lineup gate: %s", ", ".join(gate["suppressed"]))
        _update_chip_histories(state, gameweek=gw, chips=chips, fh_gap=chip_inputs["fh_gap"])

        chip_to_play = _select_executable_chip(chips.get("play_now") or {})
        # During unlimited early applies, only move the squad — hold chips for the final window.
        if unlimited and not in_window and chip_to_play:
            logger.info(
                "Deferring chip %s until final execute window (unlimited early apply)",
                chip_to_play,
            )
            chip_to_play = None

        # When TC is armed, captain the look-ahead target for this GW (not just
        # whoever the transfer anneal picked on raw EV).
        if chip_to_play == "3xc":
            plan = _apply_tc_captain_from_timing(plan, chips)

        # Snapshot the realistic FT-capped plan before any FH/WC rebuild overwrites it.
        ft_plan = plan
        chip_rebuild_plan = None

        # FH/WC execute needs a full anneal_squad rebuild, not FT-capped transfer SA.
        if (
            chip_to_play in {"freehit", "wildcard"}
            and bool(SEASON_RUNNER.get("chip_rebuild_on_fh_wc", True))
        ):
            rebuild_sol = chip_inputs.get("unconstrained_solution")
            if rebuild_sol is None:
                budget = float(live.manager.team_value or 0.0) + float(live.bank)
                if budget <= 0:
                    budget = 100.0
                rebuild_sol = (
                    live_optimise_single_gw(
                        predictions,
                        budget=budget,
                        team_fixtures=team_fixtures,
                        gameweek=gw,
                        require_cover_eligible=True,
                        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                    )
                    if vector_live_enabled()
                    else optimise_single_gw(
                        predictions,
                        budget=budget,
                        require_cover_eligible=True,
                        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                    )
                )
            if rebuild_sol is not None and rebuild_sol.status == "Optimal":
                if vector_live_enabled():
                    plan = transfer_plan_from_rebuild(
                        rebuild_sol,
                        current_ids=current_ids,
                        bank=live.bank,
                        selling_prices=live.selling_prices,
                        predictions=predictions,
                        chip=chip_to_play,
                    )
                else:
                    from fpl_robo.optimize.vector_search import _pair_transfers, _pool_from_preds

                    pool = _pool_from_preds(predictions)
                    target = [int(p["id"]) for p in rebuild_sol.squad]
                    transfers = _pair_transfers(current_ids, target, pool)
                    plan = TransferPlan(
                        status=rebuild_sol.status,
                        objective=float(rebuild_sol.objective),
                        transfers=transfers,
                        transfers_made=len(transfers),
                        hits=0,
                        hit_cost=0,
                        budget_left=round(100.0 - float(rebuild_sol.budget_used), 1),
                        free_transfers=0,
                        solution=rebuild_sol,
                        meta={"rebuild": True, "chip": chip_to_play},
                    )
                logger.info(
                    "Chip %s rebuild: transfers=%s status=%s",
                    chip_to_play,
                    plan.transfers_made,
                    plan.status,
                )
                chip_rebuild_plan = plan
                # Outside the final execute window, keep the realistic FT plan as the
                # primary recommendation so the dashboard doesn't look like a 12-move hold.
                if not (execute and in_window):
                    plan = ft_plan
                # Remap chip EV onto the rebuild XI for downstream summary consistency.
                chip_xi, chip_bench = _remap_squad_ev(
                    starting_xi=chip_rebuild_plan.solution.starting_xi,
                    bench=chip_rebuild_plan.solution.bench,
                    predictions=chip_predictions,
                )

        if (
            execute
            and unlimited
            and not in_window
            and plan.transfers_made == 0
            and _live_matches_plan(live, plan)
        ):
            logger.info("Unlimited transfers: squad already matches plan; skipping API write")
            execute = False

        if execute and unlimited and not in_window:
            plan_ids = [int(p["id"]) for p in plan.solution.squad]
            players_by_id_early = {int(p["id"]): dict(p) for p in kg.players()}
            plan_q = _count_quality_fwds(
                plan_ids, players_by_id=players_by_id_early, predictions=predictions
            )
            cur_q = _count_quality_fwds(
                current_ids, players_by_id=players_by_id_early, predictions=predictions
            )
            # Hard stop: never sell down the attack floor via unlimited early applies.
            # (Stale long-running daemons / CS-greedy plans used to overwrite quality FWDs.)
            if cur_q > 0 and plan_q < cur_q:
                logger.warning(
                    "Blocking execute: attack quality drop %s -> %s (feature=%s)",
                    cur_q,
                    plan_q,
                    KG_ENRICH.get("feature_version"),
                )
                execute = False
            else:
                risky = _owned_risk_element_ids(
                    current_ids, predictions=predictions, factors=factors
                )
                traps = {
                    int(x)
                    for x in ((plan.meta or {}).get("owned_minutes_traps") or [])
                }
                plan_id_set = {int(x) for x in plan_ids}
                owned_risk_sold = len((risky | traps) - plan_id_set)
                throttle = _preseason_execute_throttle(
                    state=state,
                    preseason=is_preseason(bootstrap),
                    optimise_triggers=optimise_triggers or [],
                    plan_quality_fwds=plan_q,
                    current_quality_fwds=cur_q,
                    force=force,
                    transfers_made=int(plan.transfers_made or 0),
                    owned_risk_sold=owned_risk_sold,
                    plan_objective=float(plan.objective),
                )
                if not throttle["allow"]:
                    logger.info("Preseason execute throttle: %s", throttle["reason"])
                    execute = False
                else:
                    if throttle.get("reason"):
                        logger.info("Preseason execute throttle allow: %s", throttle["reason"])
                    churn = _unlimited_churn_decision(
                        state=state,
                        gameweek=gw,
                        plan_objective=float(plan.objective),
                        locked_objective=chip_inputs.get("locked_objective"),
                        preseason=is_preseason(bootstrap),
                        plan_quality_fwds=plan_q,
                        current_quality_fwds=cur_q,
                    )
                    if not churn["allow"]:
                        logger.info("Unlimited churn guard: %s", churn["reason"])
                        execute = False
                    else:
                        logger.info("Unlimited churn guard allow: %s", churn["reason"])

        execution = None
        players_by_id = {int(p["id"]): dict(p) for p in kg.players()}
        for _, pred in predictions.iterrows():
            pid = int(pred["id"])
            if pid in players_by_id:
                players_by_id[pid]["ev"] = float(pred["ev"])
            else:
                players_by_id[pid] = {
                    "id": pid,
                    "web_name": pred.get("web_name"),
                    "position": pred.get("position"),
                    "team_short": pred.get("team_short"),
                    "price": float(pred.get("price") or 0.0),
                    "ev": float(pred["ev"]),
                }
        transfer_plan = plan.to_dict()
        transfer_plan["transfers"] = _named_transfers(plan.transfers, players_by_id)
        transfer_plan["optimiser"] = (
            ((plan.meta or {}).get("optimiser_override")) or _live_optimiser_name()
        )

        ft_transfer_plan = ft_plan.to_dict()
        ft_transfer_plan["transfers"] = _named_transfers(ft_plan.transfers, players_by_id)
        ft_transfer_plan["optimiser"] = _live_optimiser_name()
        # If vector held but shadow MIP found a clear FT move, surface that as the
        # considered plan (not a multi-GW bank — just the alternate optimiser).
        if (
            int(ft_plan.transfers_made or 0) == 0
            and isinstance(shadow_mip_plan, dict)
            and int(shadow_mip_plan.get("transfers_made") or 0) > 0
            and str(shadow_mip_plan.get("status") or "") == "Optimal"
        ):
            shadow_named = _named_transfers(
                list(shadow_mip_plan.get("transfers") or []),
                players_by_id,
            )
            ft_transfer_plan = {
                **shadow_mip_plan,
                "transfers": shadow_named,
                "optimiser": "mip_pulp_shadow",
                "meta": {
                    **(shadow_mip_plan.get("meta") or {}),
                    "vector_held": True,
                    "source": "shadow_mip_while_vector_hold",
                },
            }

        chip_rebuild_payload = None
        if chip_rebuild_plan is not None:
            chip_rebuild_payload = chip_rebuild_plan.to_dict()
            chip_rebuild_payload["transfers"] = _named_transfers(
                chip_rebuild_plan.transfers, players_by_id
            )
            chip_rebuild_payload["optimiser"] = _live_optimiser_name()
            chip_rebuild_payload["meta"] = {
                **(chip_rebuild_payload.get("meta") or {}),
                "rebuild": True,
                "chip": chip_to_play,
            }

        if execute:
            plan = _freeze_reviewed_execute_plan(
                plan,
                summary_path=summary_file,
                gameweek=int(gw),
                players_by_id=players_by_id,
            )
            if (plan.meta or {}).get("execute_freeze_reviewed_plan"):
                transfer_plan = plan.to_dict()
                transfer_plan["transfers"] = _named_transfers(
                    plan.transfers, players_by_id
                )
                transfer_plan["optimiser"] = (
                    ((plan.meta or {}).get("optimiser_override"))
                    or _live_optimiser_name()
                )
            logger.info(
                "Executing weekly plan: %s transfers, hits=%s, chip=%s freeze=%s",
                plan.transfers_made,
                plan.hits,
                chip_to_play,
                bool((plan.meta or {}).get("execute_freeze_reviewed_plan")),
            )
            # Checkpoint before mutating FPL so a mid-write restart can reconcile.
            state["execute_in_progress"] = {
                "gw": gw,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "chip": chip_to_play,
                "target_ids": [int(r["id"]) for r in plan.solution.squad],
                "planned_xi": [
                    int(r["id"]) for r in plan.solution.squad if r.get("start")
                ],
                "planned_bench": [
                    int(r["id"]) for r in plan.solution.squad if not r.get("start")
                ],
            }
            _write_json(state_path, state)

            execution = apply_weekly_plan(
                client=client,
                manager_id=manager_id,
                squad_rows=plan.solution.squad,
                captain_id=plan.solution.captain_id,
                event_id=gw,
                chip=chip_to_play,
            )
            state["last_executed_gw"] = gw
            state["last_execute_ok"] = bool(execution.get("saved")) and bool(
                execution.get("lineup_verified", True)
            )
            state["last_executed_at"] = datetime.now(timezone.utc).isoformat()
            state["last_execute_lineup"] = {
                "planned_xi": execution.get("planned_xi"),
                "verified_xi": execution.get("verified_xi"),
                "planned_bench": execution.get("planned_bench"),
                "verified_bench": execution.get("verified_bench"),
                "lineup_verified": execution.get("lineup_verified"),
                "frozen": bool((plan.meta or {}).get("execute_freeze_reviewed_plan")),
            }
            state.pop("execute_in_progress", None)
            if execution.get("saved"):
                _append_applied_moves(
                    state,
                    gameweek=gw,
                    applied_at=state["last_executed_at"],
                    transfers=transfer_plan.get("transfers") or [],
                )
            if unlimited and not in_window and execution.get("saved"):
                state["last_unlimited_gw"] = gw
                state["last_unlimited_objective"] = float(plan.objective)
                state["last_unlimited_optimiser"] = _live_optimiser_name()
                state["last_unlimited_squad_ids"] = sorted(
                    int(r["id"]) for r in plan.solution.squad
                )
        else:
            state["last_recommended_gw"] = gw
            state["last_recommended_at"] = datetime.now(timezone.utc).isoformat()

        payload = {
            "gameweek": gw,
            "hours_to_deadline": round(hours, 3),
            "deadline_time": deadline.get("deadline_time"),
            "dgw_teams": dgw_teams,
            "ev_mode": EV_MODE,
            "manager": {
                "manager_id": live.manager.manager_id,
                "name": live.manager.name,
                "team_name": live.manager.team_name,
                "bank": live.bank,
                "free_transfers": live.free_transfers,
                "unlimited_transfers": unlimited,
            },
            "execute_policy": {
                "in_window": in_window,
                "unlimited_transfers": unlimited,
                "can_write": can_write,
                "execute_within_hours": SEASON_RUNNER["execute_within_hours"],
                "unlimited_min_obj_delta": SEASON_RUNNER.get("unlimited_min_obj_delta"),
                "preseason": is_preseason(bootstrap),
                "preseason_skip_churn_guard": bool(
                    SEASON_RUNNER.get("preseason_skip_churn_guard", True)
                ),
            },
            "transfer_plan": transfer_plan,
            "ft_transfer_plan": ft_transfer_plan,
            "chip_rebuild_plan": chip_rebuild_payload,
            "shadow_mip_plan": shadow_mip_plan,
            "chips": chips,
            "chip_to_play": chip_to_play,
            "execution": execution,
            "signals": bundle.summary(),
        }
        _write_json(summary_file, payload)
        _write_json(state_path, state)
        logger.info("Weekly run complete for GW%s (execute=%s)", gw, bool(execution))
        return SeasonRunResult(
            ok=True,
            skipped=False,
            reason=None,
            gameweek=gw,
            hours_to_deadline=hours,
            transfer_plan=transfer_plan,
            chips=chips,
            execution=execution,
            dgw_teams=dgw_teams,
            summary_path=summary_file,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Weekly run failed")
        return _fail(summary_file, f"Unhandled error: {exc}")
    finally:
        lock.release()


def run_daemon(
    *,
    poll_seconds: int | None = None,
    execute: bool = True,
    manager_id: int = DEFAULT_MANAGER_ID,
) -> None:
    """Forever loop: ingest / optimise only when cadence says so."""
    setup_logging()
    install_stop_handlers()
    from fpl_robo.runtime.code_stamp import CodeFreshnessGuard

    code_guard = CodeFreshnessGuard(feature_version=str(KG_ENRICH.get("feature_version") or ""))
    logger.info(
        "Daemon start pid=%s feature=%s ev_mode=%s execute=%s code_mtime=%.0f",
        os.getpid(),
        KG_ENRICH.get("feature_version"),
        EV_MODE,
        execute,
        code_guard.baseline_mtime,
    )
    from fpl_robo.runtime.cadence import (
        evaluate_cadence,
        record_optimise,
        signals_mtime,
    )
    from fpl_robo.runtime.atomic import atomic_write_json
    from fpl_robo.signals.pre_deadline import maybe_run_scheduled_ingest

    default_poll = int(
        poll_seconds
        or SEASON_RUNNER.get("heartbeat_seconds")
        or SEASON_RUNNER.get("poll_seconds")
        or 900
    )
    state_path = Path(SEASON_RUNNER["state_file"])
    logger.info(
        "Starting season daemon (heartbeat=%ss, near_poll=%ss within %.1fh, execute=%s)",
        default_poll,
        SEASON_RUNNER["near_deadline_poll_seconds"],
        SEASON_RUNNER["near_deadline_hours"],
        execute,
    )
    while not stop_requested():
        poll = default_poll
        try:
            code_guard.check_or_exit(exit_code=0)
            client = FPLClient()
            bootstrap = client.bootstrap_static()
            fixtures = client.fixtures()
            state = _load_state(state_path)
            decision = evaluate_cadence(
                bootstrap=bootstrap,
                fixtures=fixtures,
                state=state,
            )
            poll = int(decision.sleep_seconds or default_poll)
            logger.info(
                "Cadence: phase=%s gw=%s hours=%s ingest=%s optimise=%s sleep=%ss",
                decision.phase,
                decision.gameweek,
                decision.hours_to_deadline,
                decision.ingest_kind if decision.should_ingest else "no",
                ",".join(decision.optimise_triggers) if decision.should_optimise else "no",
                poll,
            )

            if decision.should_ingest:
                try:
                    pre = maybe_run_scheduled_ingest(
                        client=client,
                        apply=True,
                        force=False,
                        bootstrap=bootstrap,
                        fixtures=fixtures,
                    )
                    if not pre.get("skipped"):
                        logger.info(
                            "Scheduled ingest: kind=%s ok=%s gw=%s news=%s lineups=%s",
                            pre.get("kind"),
                            pre.get("ok"),
                            pre.get("gameweek"),
                            ((pre.get("ingest") or {}).get("proposed_count")),
                            ((pre.get("ingest") or {}).get("proposed_lineup_count")),
                        )
                        # Re-evaluate so fresh signals trigger optimise this cycle.
                        state = _load_state(state_path)
                        decision = evaluate_cadence(
                            bootstrap=bootstrap,
                            fixtures=fixtures,
                            state=state,
                        )
                        poll = int(decision.sleep_seconds or default_poll)
                except Exception:  # noqa: BLE001
                    logger.exception("Scheduled signal ingest failed; continuing")

            if decision.should_optimise:
                result = run_week(
                    manager_id=manager_id,
                    execute=execute,
                    force=False,
                    optimise_triggers=list(decision.optimise_triggers or []),
                )
                logger.info(
                    "Daemon cycle: ok=%s skipped=%s reason=%s gw=%s hours_to_deadline=%s triggers=%s",
                    result.ok,
                    result.skipped,
                    result.reason,
                    result.gameweek,
                    result.hours_to_deadline,
                    decision.optimise_triggers,
                )
                # Persist cadence marks (run_week rewrites state — merge afterward).
                try:
                    state = _load_state(state_path)
                    record_optimise(
                        state,
                        gameweek=result.gameweek or decision.gameweek,
                        triggers=decision.optimise_triggers,
                        bootstrap_fp=decision.bootstrap_fingerprint,
                        price_status_fp=decision.price_status_fingerprint,
                        signals_m=signals_mtime(),
                        hours_to_deadline_value=result.hours_to_deadline
                        if result.hours_to_deadline is not None
                        else decision.hours_to_deadline,
                    )
                    atomic_write_json(state_path, state)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed to record cadence optimise marks")
            else:
                logger.info(
                    "Optimise skipped: %s (gw=%s)",
                    decision.skip_reason,
                    decision.gameweek,
                )
        except Exception:  # noqa: BLE001
            logger.exception("Daemon cycle crashed; continuing")
            poll = default_poll
        if stop_requested():
            break
        if not interruptible_sleep(poll):
            break
    logger.info("Season daemon drained cleanly; exiting")


def _reconcile_execute_in_progress(
    *,
    state: dict[str, Any],
    state_path: Path,
    summary_file: str,
    client: FPLClient,
    manager_id: int,
    gw: int,
    in_window: bool,
) -> None:
    pending = state.get("execute_in_progress")
    if not isinstance(pending, dict):
        return
    pending_gw = pending.get("gw")
    if pending_gw is None:
        state.pop("execute_in_progress", None)
        _write_json(state_path, state)
        return

    logger.warning("Found execute_in_progress for GW%s; reconciling", pending_gw)
    target_ids = {
        int(x) for x in (pending.get("target_ids") or []) if str(x).isdigit() or isinstance(x, int)
    }
    # Prefer ids from the last summary if checkpoint lacked them.
    if not target_ids:
        summary = _load_state(Path(summary_file))
        squad = ((summary.get("transfer_plan") or {}).get("solution") or {}).get("squad") or []
        target_ids = {int(r["id"]) for r in squad if r.get("id") is not None}

    try:
        my_team = client.my_team(manager_id)
        live_ids = {
            int(p["element"]) for p in (my_team.get("picks") or []) if p.get("element") is not None
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Reconcile failed to load my-team: %s", exc)
        return

    matched = bool(target_ids) and live_ids == target_ids
    if matched or int(pending_gw) != int(gw):
        state["last_executed_gw"] = int(pending_gw)
        state["last_execute_ok"] = matched
        state["last_executed_at"] = datetime.now(timezone.utc).isoformat()
        state["last_reconciled_at"] = datetime.now(timezone.utc).isoformat()
        state.pop("execute_in_progress", None)
        _write_json(state_path, state)
        logger.info(
            "Reconciled GW%s execute_in_progress (matched=%s)",
            pending_gw,
            matched,
        )
        return

    if in_window:
        logger.warning(
            "GW%s execute incomplete and still in window; will retry apply",
            pending_gw,
        )
        return

    state["last_executed_gw"] = int(pending_gw)
    state["last_execute_ok"] = False
    state["last_executed_at"] = datetime.now(timezone.utc).isoformat()
    state.pop("execute_in_progress", None)
    _write_json(state_path, state)
    logger.error(
        "GW%s execute_in_progress cleared outside window without confirmed match",
        pending_gw,
    )


def _xi_ev(rows: list[dict[str, Any]] | None) -> float:
    return float(sum(float(r.get("ev") or 0.0) for r in (rows or [])))


def _remap_squad_ev(
    *,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    predictions,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy XI/bench rows with EV taken from a (possibly horizon-adjusted) frame."""
    ev_by_id: dict[int, float] = {}
    team_by_id: dict[int, int] = {}
    try:
        for _, row in predictions.iterrows():
            if row.get("id") is None:
                continue
            pid = int(row["id"])
            ev_by_id[pid] = float(row.get("ev") or 0.0)
            tid = row.get("team_id", row.get("team"))
            if tid is not None:
                try:
                    team_by_id[pid] = int(tid)
                except (TypeError, ValueError):
                    pass
    except Exception:  # noqa: BLE001
        ev_by_id = {}
        team_by_id = {}

    def _remap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for p in rows:
            row = dict(p)
            try:
                pid = int(p.get("id"))
            except (TypeError, ValueError):
                out.append(row)
                continue
            if pid in ev_by_id:
                row["ev"] = ev_by_id[pid]
            if pid in team_by_id and row.get("team_id") is None and row.get("team") is None:
                row["team_id"] = team_by_id[pid]
            out.append(row)
        return out

    return _remap(starting_xi), _remap(bench)


def _compute_chip_opportunity_inputs(
    *,
    predictions,
    current_ids: list[int],
    bank: float,
    selling_prices: dict[int, float],
    team_value: float,
    state: dict[str, Any],
    gameweek: int,
) -> dict[str, Any]:
    """Build FH/WC gap inputs and TC/BB history for live chip recommendations."""
    locked = (
        live_optimise_transfers(
            predictions,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling_prices,
            free_transfers=0,
            max_hits=0,
            unlimited_transfers=False,
            require_cover_eligible=True,
            bench_ev_weight=BENCH_COVER["bench_ev_weight"],
            **transfer_gate_kwargs(EV_MODE),
        )
        if vector_live_enabled()
        else optimise_transfers(
            predictions,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling_prices,
            free_transfers=0,
            max_hits=0,
            require_cover_eligible=True,
            bench_ev_weight=BENCH_COVER["bench_ev_weight"],
            **transfer_gate_kwargs(EV_MODE),
        )
    )
    current_xi_ev = (
        _xi_ev(locked.solution.starting_xi) if locked.status == "Optimal" else 0.0
    )
    locked_objective = float(locked.objective) if locked.status == "Optimal" else None

    budget = float(team_value) + float(bank)
    if budget <= 0:
        budget = 100.0
    unconstrained = (
        live_optimise_single_gw(
            predictions,
            budget=budget,
            require_cover_eligible=True,
            bench_ev_weight=BENCH_COVER["bench_ev_weight"],
        )
        if vector_live_enabled()
        else optimise_single_gw(
            predictions,
            budget=budget,
            require_cover_eligible=True,
            bench_ev_weight=BENCH_COVER["bench_ev_weight"],
        )
    )
    optimal_xi_ev = (
        _xi_ev(unconstrained.starting_xi) if unconstrained.status == "Optimal" else current_xi_ev
    )
    fh_gap = max(0.0, float(optimal_xi_ev) - float(current_xi_ev))

    fh_hist = [float(x) for x in (state.get("fh_gap_history") or []) if x is not None]
    # Include this GW's gap in the rolling view for decision-time WC without double-writing yet.
    horizon = int(CHIP_DEFAULTS["rolling_gap_horizon"])
    if int(state.get("chip_history_gw") or -1) == int(gameweek) and fh_hist:
        rolling_source = fh_hist[:-1] + [fh_gap]
    else:
        rolling_source = fh_hist + [fh_gap]
    rolling_gap = float(sum(rolling_source[-horizon:]))

    hist_tc = [float(x) for x in (state.get("tc_score_history") or []) if x is not None]
    hist_bb = [float(x) for x in (state.get("bb_score_history") or []) if x is not None]
    # For same-GW re-polls, exclude the already-recorded score for this GW from percentile hist
    # so recommend_chips sees prior weeks only (matches backtest append-after pattern).
    if int(state.get("chip_history_gw") or -1) == int(gameweek):
        if hist_tc:
            hist_tc = hist_tc[:-1]
        if hist_bb:
            hist_bb = hist_bb[:-1]

    return {
        "current_xi_ev": round(current_xi_ev, 3),
        "optimal_xi_ev": round(optimal_xi_ev, 3),
        "fh_gap": round(fh_gap, 3),
        "rolling_gap": round(rolling_gap, 3),
        "historical_tc_scores": hist_tc,
        "historical_bb_scores": hist_bb,
        "locked_objective": locked_objective,
        "unconstrained_solution": unconstrained if unconstrained.status == "Optimal" else None,
    }


def _update_chip_histories(
    state: dict[str, Any],
    *,
    gameweek: int,
    chips: dict[str, Any],
    fh_gap: float,
) -> None:
    """Append chip opportunity scores once per gameweek into season_state."""
    scores = chips.get("scores") or {}
    tc = float(scores.get("triple_captain") or 0.0)
    bb = float(scores.get("bench_boost") or 0.0)
    fh_hist = [float(x) for x in (state.get("fh_gap_history") or [])]
    tc_hist = [float(x) for x in (state.get("tc_score_history") or [])]
    bb_hist = [float(x) for x in (state.get("bb_score_history") or [])]

    if int(state.get("chip_history_gw") or -1) == int(gameweek):
        # Refresh this GW's values in place (daemon re-poll).
        if fh_hist:
            fh_hist[-1] = float(fh_gap)
        else:
            fh_hist.append(float(fh_gap))
        if tc_hist:
            tc_hist[-1] = tc
        else:
            tc_hist.append(tc)
        if bb_hist:
            bb_hist[-1] = bb
        else:
            bb_hist.append(bb)
    else:
        fh_hist.append(float(fh_gap))
        tc_hist.append(tc)
        bb_hist.append(bb)

    # Keep histories bounded (full season is fine; trim extreme growth).
    state["fh_gap_history"] = fh_hist[-40:]
    state["tc_score_history"] = tc_hist[-40:]
    state["bb_score_history"] = bb_hist[-40:]
    state["chip_history_gw"] = int(gameweek)


def _live_optimiser_name() -> str:
    if path_plan_live_enabled():
        return "path_graph"
    return "vector_team_sa" if vector_live_enabled() else "mip_pulp"


def _compatible_last_unlimited_objective(state: dict[str, Any]) -> float | None:
    """Return last unlimited objective only when it matches the live optimiser.

    MIP and vector SA objectives are not comparable; a stale MIP baseline must
    not block the first vector write (or vice versa).
    """
    last_obj = state.get("last_unlimited_objective")
    if last_obj is None:
        return None
    last_opt = state.get("last_unlimited_optimiser")
    if last_opt != _live_optimiser_name():
        return None
    try:
        return float(last_obj)
    except (TypeError, ValueError):
        return None


def _preseason_execute_throttle(
    *,
    state: dict[str, Any],
    preseason: bool,
    optimise_triggers: list[str],
    plan_quality_fwds: int,
    current_quality_fwds: int,
    force: bool = False,
    transfers_made: int = 0,
    owned_risk_sold: int = 0,
    plan_objective: float | None = None,
) -> dict[str, Any]:
    """Limit how often unlimited preseason writes hit the FPL API.

    Always allow attack-quality repairs, owned-risk repairs, and --force.
    Optionally block signal-only cycles, enforce cool-down, and cap transfer
    churn after cool-down unless repairing.
    """
    if force or not preseason:
        return {"allow": True, "reason": "not_throttled"}

    repairing_attack = int(plan_quality_fwds) > int(current_quality_fwds)
    repairing_risk = int(owned_risk_sold) > 0
    repairing = repairing_attack or repairing_risk

    if repairing_attack:
        return {
            "allow": True,
            "reason": f"attack_quality_repair {current_quality_fwds}->{plan_quality_fwds}",
        }
    if repairing_risk:
        return {
            "allow": True,
            "reason": f"owned_risk_repair sold={owned_risk_sold}",
        }

    triggers = {str(t) for t in (optimise_triggers or [])}
    deadlineish = {
        t
        for t in triggers
        if t in {"execute_window", "pre_deadline_plan"}
        or t.startswith("after_ingest:pre_deadline")
    }
    if deadlineish:
        return {
            "allow": True,
            "reason": f"pre_deadline_execute ({','.join(sorted(deadlineish))})",
        }

    signal_only = bool(triggers) and triggers.issubset({"new_signals"})
    if signal_only and not bool(SEASON_RUNNER.get("preseason_execute_on_new_signals", False)):
        return {
            "allow": False,
            "reason": "preseason_skip_execute_on_new_signals",
        }

    min_hours = float(SEASON_RUNNER.get("preseason_execute_min_hours_between") or 0.0)
    age_h = None
    if min_hours > 0:
        last_at = state.get("last_executed_at") or state.get("last_unlimited_at")
        if last_at:
            try:
                last_dt = datetime.fromisoformat(str(last_at).replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                age_h = (
                    datetime.now(timezone.utc) - last_dt.astimezone(timezone.utc)
                ).total_seconds() / 3600.0
            except Exception:  # noqa: BLE001
                age_h = None
            if age_h is not None and age_h < min_hours:
                return {
                    "allow": False,
                    "reason": f"preseason_cooldown {age_h:.2f}h < {min_hours}h",
                }

    max_xfer = int(SEASON_RUNNER.get("preseason_max_transfers_unless_repair") or 0)
    if max_xfer > 0 and int(transfers_made) > max_xfer and not repairing:
        return {
            "allow": False,
            "reason": (
                f"preseason_transfer_cap {transfers_made} > {max_xfer} "
                "(no repair)"
            ),
        }

    # After cool-down, non-repair reshuffles must clear a higher objective bar.
    # Skip when last objective was from a different optimiser (incomparable scales).
    post_delta = float(SEASON_RUNNER.get("preseason_post_cooldown_min_obj_delta") or 0.0)
    last_obj = _compatible_last_unlimited_objective(state)
    if (
        not repairing
        and post_delta > 0
        and plan_objective is not None
        and last_obj is not None
        and int(transfers_made) > 0
    ):
        delta = float(plan_objective) - float(last_obj)
        if delta < post_delta:
            return {
                "allow": False,
                "reason": (
                    f"preseason_post_cooldown_delta {delta:.3f} < {post_delta} "
                    f"(plan={float(plan_objective):.3f} last={float(last_obj):.3f})"
                ),
            }

    if (
        int(transfers_made) > 0
        and state.get("last_unlimited_objective") is not None
        and last_obj is None
    ):
        return {
            "allow": True,
            "reason": (
                f"optimiser_baseline_reset "
                f"(was={state.get('last_unlimited_optimiser')!s} "
                f"now={_live_optimiser_name()})"
            ),
        }

    if age_h is None and min_hours > 0 and not (
        state.get("last_executed_at") or state.get("last_unlimited_at")
    ):
        return {"allow": True, "reason": "first_preseason_execute"}
    if age_h is not None:
        return {"allow": True, "reason": f"cooldown_ok age={age_h:.2f}h"}
    return {"allow": True, "reason": "cooldown_ok"}


def _owned_risk_element_ids(
    current_ids: list[int],
    *,
    predictions=None,
    factors: dict[int, Any] | None = None,
) -> set[int]:
    """Owned players who look injured / doubtful / excluded."""
    chance_floor = float(SEASON_RUNNER.get("owned_risk_chance_below") or 50)
    news_floor = float(SEASON_RUNNER.get("owned_risk_news_mult_below") or 0.75)
    pred_by_id: dict[int, Any] = {}
    if predictions is not None and hasattr(predictions, "iterrows"):
        pred_by_id = {int(r["id"]): r for _, r in predictions.iterrows()}
    risky: set[int] = set()
    for eid in current_ids:
        pid = int(eid)
        row = pred_by_id.get(pid)
        status = str((row.get("status") if row is not None else None) or "a").lower()
        if status in {"i", "d", "s", "u"}:
            risky.add(pid)
            continue
        chance = None
        if row is not None:
            chance = row.get("chance_of_playing_next_round")
        if chance is not None:
            try:
                if float(chance) < chance_floor:
                    risky.add(pid)
                    continue
            except (TypeError, ValueError):
                pass
        fac = (factors or {}).get(pid)
        if fac is not None:
            if bool(getattr(fac, "exclude", False)):
                risky.add(pid)
                continue
            try:
                if float(getattr(fac, "news_mult", 1.0) or 1.0) < news_floor:
                    risky.add(pid)
            except (TypeError, ValueError):
                pass
    return risky


def _unlimited_churn_decision(
    *,
    state: dict[str, Any],
    gameweek: int,
    plan_objective: float,
    locked_objective: float | None = None,
    min_delta: float | None = None,
    preseason: bool = False,
    plan_quality_fwds: int | None = None,
    current_quality_fwds: int | None = None,
) -> dict[str, Any]:
    """Whether an unlimited early apply should write despite noise.

    Preseason: optionally always allow (`preseason_skip_churn_guard`).
    Mid-season unlimited (WC/FH): first write of a GW always allowed; later writes
    need objective improvement of at least `unlimited_min_obj_delta`.

    Attack repair: always allow when the plan has strictly more quality FWDs
    (ep_next >= attack-floor threshold) than the live squad — stops a locked
    high-obj DEF-heavy plan from blocking a genuine attack rebuild.
    """
    if (
        plan_quality_fwds is not None
        and current_quality_fwds is not None
        and int(plan_quality_fwds) > int(current_quality_fwds)
    ):
        return {
            "allow": True,
            "reason": (
                f"attack_quality_repair "
                f"{current_quality_fwds}->{plan_quality_fwds}"
            ),
            "delta": None,
        }

    if preseason and bool(SEASON_RUNNER.get("preseason_skip_churn_guard", True)):
        return {"allow": True, "reason": "preseason_always_apply", "delta": None}

    delta_needed = float(
        SEASON_RUNNER["unlimited_min_obj_delta"] if min_delta is None else min_delta
    )
    last_gw = state.get("last_unlimited_gw")
    last_obj = _compatible_last_unlimited_objective(state)
    if last_gw is None or int(last_gw) != int(gameweek) or last_obj is None:
        baseline = float(locked_objective) if locked_objective is not None else None
        if baseline is None:
            return {"allow": True, "reason": "first_unlimited_write_this_gw", "delta": None}
        delta = float(plan_objective) - baseline
        if delta >= delta_needed:
            return {
                "allow": True,
                "reason": "first_write_improves_locked",
                "delta": round(delta, 3),
            }
        # Still allow the first write of the GW even if delta is small — establishing a plan.
        return {"allow": True, "reason": "first_unlimited_write_this_gw", "delta": round(delta, 3)}

    delta = float(plan_objective) - float(last_obj)
    if delta >= delta_needed:
        return {"allow": True, "reason": "objective_improved", "delta": round(delta, 3)}
    return {
        "allow": False,
        "reason": (
            f"objective delta {delta:.3f} < {delta_needed} "
            f"(plan={plan_objective:.3f} last={float(last_obj):.3f})"
        ),
        "delta": round(delta, 3),
    }


def _count_quality_fwds(
    element_ids: list[int],
    *,
    players_by_id: dict[int, dict[str, Any]] | None = None,
    predictions=None,
) -> int:
    """How many FWDs meet the shared ATTACK_FLOOR quality predicate."""
    from fpl_robo.config import ATTACK_FLOOR
    from fpl_robo.optimize.attack_quality import is_quality_fwd

    if not bool(ATTACK_FLOOR.get("enabled", True)):
        return 0
    n = 0
    pred_by_id = {}
    if predictions is not None and hasattr(predictions, "iterrows"):
        pred_by_id = {int(r["id"]): r for _, r in predictions.iterrows()}
    for eid in element_ids:
        pid = int(eid)
        pos = None
        ep = None
        status = None
        focal = False
        form = None
        rolling = None
        if pid in pred_by_id:
            row = pred_by_id[pid]
            pos = row.get("position")
            ep = row.get("ep_next")
            status = row.get("status")
            focal = bool(row.get("is_focal_attacker"))
            if row.get("form") is not None:
                try:
                    form = float(row.get("form"))
                except (TypeError, ValueError):
                    form = None
            if row.get("rolling_minutes") is not None:
                try:
                    rolling = float(row.get("rolling_minutes"))
                except (TypeError, ValueError):
                    rolling = None
        if (pos is None or ep is None) and players_by_id and pid in players_by_id:
            p = players_by_id[pid]
            pos = pos or p.get("position")
            if ep is None:
                ep = p.get("ep_next")
            if status is None:
                status = p.get("status")
            if not focal:
                focal = bool(p.get("is_focal_attacker"))
            if form is None and p.get("form") is not None:
                try:
                    form = float(p.get("form"))
                except (TypeError, ValueError):
                    form = None
        if str(pos or "").upper() != "FWD":
            continue
        if is_quality_fwd(
            ep_next=ep,
            status=status,
            is_focal=focal,
            form=form,
            rolling_minutes=rolling,
        ):
            n += 1
    return n



def _can_auto_execute(
    *,
    hours_to_deadline: float,
    unlimited_transfers: bool,
    force: bool = False,
    execute_within_hours: float | None = None,
) -> bool:
    """Whether the runner may write transfers/lineup to FPL."""
    if force:
        return True
    if hours_to_deadline <= 0:
        return False
    window = float(
        SEASON_RUNNER["execute_within_hours"]
        if execute_within_hours is None
        else execute_within_hours
    )
    if 0 < hours_to_deadline <= window:
        return True
    return bool(unlimited_transfers) and bool(SEASON_RUNNER.get("execute_when_unlimited", True))


def _live_matches_plan(live: Any, plan: Any) -> bool:
    """True if live 15/XI/captain already match the optimiser plan."""

    def _pid(x: Any) -> int:
        if isinstance(x, dict):
            return int(x.get("id") if x.get("id") is not None else x.get("element"))
        return int(x)

    current_ids = sorted(int(p["element"]) for p in (live.picks or []) if p.get("element") is not None)
    target_ids = sorted(_pid(r) if not isinstance(r, dict) else int(r["id"]) for r in (plan.solution.squad or []))
    if current_ids != target_ids:
        return False
    live_xi = sorted(
        int(p["element"])
        for p in (live.picks or [])
        if p.get("element") is not None and int(p.get("position") or 99) <= 11
    )
    plan_xi = sorted(_pid(x) for x in (plan.solution.starting_xi or []))
    if live_xi != plan_xi:
        return False
    live_cap = next(
        (int(p["element"]) for p in (live.picks or []) if p.get("is_captain")),
        None,
    )
    if live_cap != int(plan.solution.captain_id):
        return False
    plan_vice = getattr(plan.solution, "vice_captain_id", None)
    if plan_vice is None and isinstance(plan.solution, dict):
        plan_vice = plan.solution.get("vice_captain_id")
    if plan_vice is None:
        return True
    live_vice = next(
        (int(p["element"]) for p in (live.picks or []) if p.get("is_vice_captain")),
        None,
    )
    return live_vice == int(plan_vice)


def _transfer_pair_key(transfers: list[dict[str, Any]] | None) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for t in transfers or []:
        pairs.append(
            (
                int(t.get("element_out") or t.get("out") or 0),
                int(t.get("element_in") or t.get("in") or 0),
            )
        )
    return tuple(sorted(pairs))


def _freeze_reviewed_execute_plan(
    plan: TransferPlan,
    *,
    summary_path: Path | str,
    gameweek: int,
    players_by_id: dict[int, dict[str, Any]],
) -> TransferPlan:
    """Reuse the dashboard-reviewed XI when execute-window re-opt keeps the same transfers."""
    if not bool(SEASON_RUNNER.get("execute_freeze_reviewed_plan", True)):
        return plan
    path = Path(summary_path)
    if not path.exists():
        return plan
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Execute freeze: could not read summary: %s", exc)
        return plan
    if int(prev.get("gameweek") or -1) != int(gameweek):
        return plan
    prev_tp = prev.get("transfer_plan") or {}
    if str(prev_tp.get("status") or "") != "Optimal":
        return plan
    if _transfer_pair_key(prev_tp.get("transfers")) != _transfer_pair_key(plan.transfers):
        logger.info(
            "Execute freeze: skipped (transfers changed vs reviewed plan)"
        )
        return plan
    prev_sol = prev_tp.get("solution") or {}
    squad = list(prev_sol.get("squad") or [])
    if len(squad) != 15:
        return plan
    if sum(1 for r in squad if r.get("start")) != 11:
        return plan
    # Ensure names/ids are ints and present.
    for row in squad:
        try:
            row["id"] = int(row["id"])
        except (KeyError, TypeError, ValueError):
            return plan
        if not row.get("web_name") and int(row["id"]) in players_by_id:
            row["web_name"] = players_by_id[int(row["id"])].get("web_name")

    from fpl_robo.optimize.single_gw import SquadSolution

    captain_id = prev_sol.get("captain_id") or plan.solution.captain_id
    solution = SquadSolution(
        status="Optimal",
        objective=float(prev_sol.get("objective") or prev_tp.get("objective") or plan.objective),
        budget_used=float(prev_sol.get("budget_used") or getattr(plan.solution, "budget_used", 0) or 0),
        gameweek=int(gameweek),
        squad=squad,
        starting_xi=[r for r in squad if r.get("start")],
        bench=[r for r in squad if not r.get("start")],
        captain_id=int(captain_id) if captain_id is not None else None,
        captain_name=prev_sol.get("captain_name") or plan.solution.captain_name,
        vice_captain_id=(
            int(prev_sol["vice_captain_id"])
            if prev_sol.get("vice_captain_id") is not None
            else plan.solution.vice_captain_id
        ),
        vice_captain_name=prev_sol.get("vice_captain_name") or plan.solution.vice_captain_name,
    )
    # Prefer raw transfer ids from the fresh plan (same set) for apply/audit.
    raw_transfers = [
        {
            "element_out": int(t.get("element_out") or t.get("out") or 0),
            "element_in": int(t.get("element_in") or t.get("in") or 0),
        }
        for t in (plan.transfers or [])
    ]
    logger.info(
        "Execute freeze: reusing reviewed XI/bench from summary (transfers unchanged)"
    )
    return TransferPlan(
        status="Optimal",
        objective=float(prev_tp.get("objective") or plan.objective),
        budget_left=float(prev_tp.get("budget_left") or plan.budget_left),
        free_transfers=int(prev_tp.get("free_transfers") or plan.free_transfers),
        transfers_made=len(raw_transfers),
        hits=int(prev_tp.get("hits") or plan.hits),
        hit_cost=int(prev_tp.get("hit_cost") or plan.hit_cost),
        transfers=raw_transfers,
        solution=solution,
        meta={
            **(plan.meta or {}),
            "execute_freeze_reviewed_plan": True,
        },
    )


def _transfer_raw_ep_delta(
    transfers: list[dict[str, Any]],
    predictions,
) -> float:
    """Sum (in − out) using official EP / anchor, not residual-boosted vector EV."""
    if predictions is None or not len(predictions):
        return 0.0
    by_id = {int(r["id"]): r for _, r in predictions.iterrows()}

    def _ep(row) -> float:
        if row is None:
            return 0.0
        # Prefer model EV / vector anchor over raw FPL ep_next — in preseason
        # ep_next is often flat across players and would zero-out real upgrades.
        for key in ("ep_anchor", "ev", "xP", "ep_next"):
            if row.get(key) is not None:
                try:
                    return float(row.get(key))
                except (TypeError, ValueError):
                    continue
        return 0.0

    net = 0.0
    for t in transfers or []:
        out_id = int(t.get("element_out") or t.get("out") or 0)
        in_id = int(t.get("element_in") or t.get("in") or 0)
        net += _ep(by_id.get(in_id)) - _ep(by_id.get(out_id))
    return float(net)


def _transfer_play_prob_delta(
    transfers: list[dict[str, Any]],
    predictions,
) -> float:
    """Sum (in − out) play_prob across named transfers."""
    if predictions is None or not len(predictions):
        return 0.0
    by_id = {int(r["id"]): r for _, r in predictions.iterrows()}

    def _pp(row) -> float:
        if row is None:
            return 0.0
        try:
            return float(row.get("play_prob") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    net = 0.0
    for t in transfers or []:
        out_id = int(t.get("element_out") or t.get("out") or 0)
        in_id = int(t.get("element_in") or t.get("in") or 0)
        net += _pp(by_id.get(in_id)) - _pp(by_id.get(out_id))
    return float(net)


def _is_cheap_same_pos_swap(
    transfers: list[dict[str, Any]],
    predictions,
    *,
    max_price: float,
) -> bool:
    """True for a single like-for-like DEF/MID swap at or below max_price."""
    if len(transfers or []) != 1 or predictions is None or not len(predictions):
        return False
    t = transfers[0]
    out_id = int(t.get("element_out") or t.get("out") or 0)
    in_id = int(t.get("element_in") or t.get("in") or 0)
    by_id = {int(r["id"]): r for _, r in predictions.iterrows()}
    out_row = by_id.get(out_id)
    in_row = by_id.get(in_id)
    if out_row is None or in_row is None:
        return False
    opos = str(out_row.get("position") or "").upper()
    ipos = str(in_row.get("position") or "").upper()
    if opos != ipos or opos not in {"DEF", "MID"}:
        return False
    try:
        op = float(out_row.get("price") or 99.0)
        ip = float(in_row.get("price") or 99.0)
    except (TypeError, ValueError):
        return False
    return op <= float(max_price) + 1e-9 and ip <= float(max_price) + 1e-9


def _apply_shadow_mip_prefer_gate(
    *,
    plan: TransferPlan,
    shadow: TransferPlan,
    predictions,
    owned_risk_ids: set[int],
) -> TransferPlan:
    """Adopt MIP when it clearly beats a weak vector thrash (both transferring)."""
    if not bool(SEASON_RUNNER.get("shadow_mip_prefer_gate", True)):
        return plan
    if int(plan.transfers_made or 0) <= 0:
        return plan
    if shadow is None or str(shadow.status) != "Optimal":
        return plan
    if int(shadow.transfers_made or 0) <= 0:
        return plan

    outs = {
        int(t.get("element_out") or t.get("out") or 0)
        for t in (plan.transfers or [])
    }
    if outs & set(owned_risk_ids or ()):
        return plan
    cons = (plan.meta or {}).get("consensus") or {}
    if bool(cons.get("minutes_repair")):
        return plan
    traps = set(int(x) for x in ((plan.meta or {}).get("owned_minutes_traps") or []))
    if outs & traps:
        return plan

    v_net = _transfer_raw_ep_delta(plan.transfers or [], predictions)
    s_net = _transfer_raw_ep_delta(shadow.transfers or [], predictions)
    margin = float(SEASON_RUNNER.get("shadow_prefer_ep_margin") or 0.25)
    min_shadow = float(SEASON_RUNNER.get("shadow_prefer_min_ep") or 0.15)
    max_cheap = float(SEASON_RUNNER.get("shadow_prefer_cheap_swap_max_price") or 5.5)

    prefer = False
    reason = ""
    if s_net - v_net + 1e-9 >= margin:
        prefer = True
        reason = "shadow_net_edge"
    elif (
        _is_cheap_same_pos_swap(plan.transfers or [], predictions, max_price=max_cheap)
        and s_net + 1e-9 >= min_shadow
        and s_net + 1e-9 >= v_net
    ):
        prefer = True
        reason = "cheap_swap_vs_shadow"

    if not prefer:
        return plan

    logger.info(
        "Shadow MIP prefer gate: adopting shadow (%s) vector_net=%.3f shadow_net=%.3f",
        reason,
        v_net,
        s_net,
    )
    return TransferPlan(
        status=shadow.status,
        objective=float(shadow.objective),
        budget_left=float(shadow.budget_left),
        free_transfers=int(shadow.free_transfers),
        transfers_made=int(shadow.transfers_made),
        hits=int(shadow.hits),
        hit_cost=int(shadow.hit_cost),
        transfers=list(shadow.transfers or []),
        solution=shadow.solution,
        meta={
            **(shadow.meta or {}),
            **(plan.meta or {}),
            "shadow_mip_prefer": {
                "applied": True,
                "reason": reason,
                "vector_net_ep": round(v_net, 4),
                "shadow_net_ep": round(s_net, 4),
                "suppressed_vector_transfers": list(plan.transfers or []),
            },
            "optimiser_override": "mip_pulp_shadow_prefer",
        },
    )


def _apply_shadow_mip_hold_gate(
    *,
    plan: TransferPlan,
    shadow: TransferPlan,
    predictions,
    current_ids: list[int],
    owned_risk_ids: set[int],
    bank: float,
    selling_prices: dict[int, float] | None,
    free_transfers: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    gameweek: int | None,
) -> TransferPlan:
    """Force hold when MIP holds and vector's raw EP gain is weak (geometry thrash)."""
    if not bool(SEASON_RUNNER.get("shadow_mip_hold_gate", True)):
        return plan
    if int(plan.transfers_made or 0) <= 0:
        return plan
    if shadow is None or str(shadow.status) != "Optimal":
        return plan
    if int(shadow.transfers_made or 0) != 0:
        return plan

    outs = {
        int(t.get("element_out") or t.get("out") or 0)
        for t in (plan.transfers or [])
    }
    if outs & set(owned_risk_ids or ()):
        return plan  # never block injury / doubt repairs

    cons = (plan.meta or {}).get("consensus") or {}
    # Validated vector paths: minutes repair pass or clear best_obj consensus.
    if bool(cons.get("minutes_repair")) or str(cons.get("mode") or "") == "best_obj":
        logger.info(
            "Shadow MIP hold gate: allowing vector-validated plan (mode=%s minutes_repair=%s)",
            cons.get("mode"),
            bool(cons.get("minutes_repair")),
        )
        return plan

    # Selling owned minutes traps recorded on the vector plan.
    traps = set(int(x) for x in ((plan.meta or {}).get("owned_minutes_traps") or []))
    if outs & traps:
        logger.info(
            "Shadow MIP hold gate: allowing minutes-trap sell (outs=%s)",
            sorted(outs & traps),
        )
        return plan

    # Minutes repairs (sell low-pp residual trap for clearer starter) must not be
    # blocked by the weak-EP hold gate — Le Fée←Buendía was +0.03 EP / +0.39 pp.
    pp_delta = _transfer_play_prob_delta(plan.transfers or [], predictions)
    pp_repair = float(SEASON_RUNNER.get("shadow_hold_minutes_repair_pp") or 0.15)
    if pp_delta + 1e-9 >= pp_repair:
        logger.info(
            "Shadow MIP hold gate: allowing minutes repair (pp_delta=%.3f >= %.3f)",
            pp_delta,
            pp_repair,
        )
        return plan

    # Prefer model EV over flat preseason ep_next (helper already prefers ev).
    min_ev = float(SEASON_RUNNER.get("shadow_hold_min_ev_delta") or 0.02)
    net_ep = _transfer_raw_ep_delta(plan.transfers or [], predictions)
    if net_ep + 1e-9 >= min_ev:
        logger.info(
            "Shadow MIP hold gate: allowing model-EV gain (net=%.3f >= %.3f)",
            net_ep,
            min_ev,
        )
        return plan

    min_delta = float(SEASON_RUNNER.get("shadow_hold_min_ep_delta") or 0.05)
    if net_ep >= min_delta:
        return plan

    logger.info(
        "Shadow MIP hold gate: forcing hold (vector net_ep=%.3f < %.3f, pp_delta=%.3f, mip_xfers=0)",
        net_ep,
        min_delta,
        pp_delta,
    )
    hold = hold_plan_from_current(
        predictions,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling_prices,
        free_transfers=free_transfers,
        team_fixtures=team_fixtures,
        gameweek=gameweek,
        meta={
            **(plan.meta or {}),
            "shadow_mip_hold_gate": {
                "applied": True,
                "net_ep": round(net_ep, 4),
                "min_ep_delta": min_delta,
                "pp_delta": round(pp_delta, 4),
                "suppressed_transfers": int(plan.transfers_made or 0),
            },
        },
    )
    return hold


def _named_transfers(
    transfers: list[dict[str, int]],
    players_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    named: list[dict[str, Any]] = []
    for t in transfers:
        out_id = int(t["element_out"])
        in_id = int(t["element_in"])
        out_p = players_by_id.get(out_id) or {}
        in_p = players_by_id.get(in_id) or {}
        named.append(
            {
                "element_out": out_id,
                "element_in": in_id,
                "out_name": out_p.get("web_name") or str(out_id),
                "in_name": in_p.get("web_name") or str(in_id),
                "out_team": out_p.get("team_short"),
                "in_team": in_p.get("team_short"),
                "out_position": out_p.get("position"),
                "in_position": in_p.get("position"),
                "out_price": out_p.get("price"),
                "in_price": in_p.get("price"),
                "out_ev": out_p.get("ev"),
                "in_ev": in_p.get("ev"),
            }
        )
    return named


def _append_applied_moves(
    state: dict[str, Any],
    *,
    gameweek: int,
    applied_at: str,
    transfers: list[dict[str, Any]],
    max_items: int | None = None,
) -> None:
    """Prepend dated applied transfers onto season state (newest first)."""
    if not transfers:
        return
    if max_items is None:
        max_items = int(SEASON_RUNNER.get("applied_moves_max_items") or 40)
    history = list(state.get("applied_moves") or [])
    keep_keys = (
        "element_out",
        "element_in",
        "out_name",
        "in_name",
        "out_team",
        "in_team",
        "out_position",
        "in_position",
        "out_price",
        "in_price",
        "out_ev",
        "in_ev",
    )
    batch = []
    for t in transfers:
        if not isinstance(t, dict):
            continue
        row = {k: t[k] for k in keep_keys if k in t}
        row["applied_at"] = applied_at
        row["gameweek"] = int(gameweek)
        batch.append(row)
    capped = (batch + history)[: max(1, int(max_items))]
    # Compact legacy fat rows still in history.
    slim: list[dict[str, Any]] = []
    for row in capped:
        if not isinstance(row, dict):
            continue
        slim.append(
            {
                **{k: row[k] for k in keep_keys if k in row},
                "applied_at": row.get("applied_at"),
                "gameweek": row.get("gameweek"),
            }
        )
    state["applied_moves"] = slim


def _apply_tc_captain_from_timing(plan: Any, chips: dict[str, Any]) -> Any:
    """Point captain at this GW's TC look-ahead target when Triple Captain is armed."""
    timing = (chips.get("details") or {}).get("tc_timing") or {}
    target = timing.get("this_player_id")
    if target is None:
        return plan
    try:
        target_id = int(target)
    except (TypeError, ValueError):
        return plan
    sol = getattr(plan, "solution", None)
    if sol is None:
        return plan
    current = getattr(sol, "captain_id", None)
    if current is not None and int(current) == target_id:
        return plan
    squad = list(getattr(sol, "squad", None) or [])
    if not any(int(r.get("id")) == target_id for r in squad if r.get("id") is not None):
        return plan
    from fpl_robo.optimize.captaincy import pick_vice_captain_id

    vice_id = pick_vice_captain_id(squad, target_id)
    name_by_id = {
        int(r["id"]): str(r.get("web_name") or r["id"])
        for r in squad
        if r.get("id") is not None
    }
    for row in squad:
        try:
            pid = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        row["captain"] = pid == target_id
        row["vice_captain"] = vice_id is not None and pid == int(vice_id)
    sol.squad = squad
    sol.starting_xi = [r for r in squad if r.get("start")]
    sol.bench = [r for r in squad if not r.get("start")]
    sol.captain_id = target_id
    sol.captain_name = name_by_id.get(target_id)
    sol.vice_captain_id = vice_id
    sol.vice_captain_name = name_by_id.get(int(vice_id)) if vice_id is not None else None
    logger.info(
        "TC captain set to look-ahead target %s (%s)",
        name_by_id.get(target_id),
        target_id,
    )
    return plan


def _select_executable_chip(play_now: dict[str, bool]) -> str | None:
    # Priority: avoid stacking; lineup chips preferred over squad-reset chips here.
    order = ["triple_captain", "bench_boost", "free_hit", "wildcard"]
    api_name = {
        "triple_captain": "3xc",
        "bench_boost": "bboost",
        "free_hit": "freehit",
        "wildcard": "wildcard",
    }
    for chip in order:
        if play_now.get(chip):
            return api_name[chip]
    return None


def _apply_signal_factors(predictions, factors):
    import pandas as pd

    from fpl_robo.config import BUY_GATES

    df = predictions.copy()
    if "signal_news_mult" not in df.columns:
        df["signal_news_mult"] = 1.0
    if "signal_minutes_mult" not in df.columns:
        df["signal_minutes_mult"] = 1.0
    if "signal_predicted_minutes" not in df.columns:
        df["signal_predicted_minutes"] = pd.NA
    news_floor = float(BUY_GATES.get("min_signal_news_mult") or 0.70)
    for pid, fac in factors.items():
        mask = df["id"] == int(pid)
        if not mask.any():
            continue
        mult = float(fac.combined_mult())
        form_boost = float(getattr(fac, "form_boost", 0.0) or 0.0)
        boost_scale = float(KG_ENRICH.get("form_boost_scale") or 1.0)
        form_boost *= max(0.0, boost_scale)
        news_mult = float(getattr(fac, "news_mult", 1.0) or 1.0)
        minutes_mult = float(getattr(fac, "minutes_mult", 1.0) or 1.0)
        df.loc[mask, "ev"] = (
            df.loc[mask, "ev"].astype(float) * mult + form_boost
        ).clip(lower=0.0)
        df.loc[mask, "signal_news_mult"] = news_mult
        df.loc[mask, "signal_minutes_mult"] = minutes_mult
        if getattr(fac, "minutes_trusted", False):
            pred_mins = getattr(fac, "predicted_minutes", None)
            if pred_mins is not None:
                try:
                    df.loc[mask, "signal_predicted_minutes"] = int(pred_mins)
                except (TypeError, ValueError):
                    pass
        if getattr(fac, "exclude", False) or news_mult + 1e-9 < news_floor:
            if getattr(fac, "exclude", False):
                df.loc[mask, "status"] = "u"
                df.loc[mask, "cover_eligible"] = False
                df.loc[mask, "ev"] = 0.0
            df.loc[mask, "buy_eligible"] = False
    return df.sort_values("ev", ascending=False).reset_index(drop=True)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        # Bound / slim applied_moves so state files don't keep growing fat.
        moves = data.get("applied_moves")
        if isinstance(moves, list) and moves:
            max_items = int(SEASON_RUNNER.get("applied_moves_max_items") or 40)
            keep_keys = (
                "element_out",
                "element_in",
                "out_name",
                "in_name",
                "out_team",
                "in_team",
                "out_position",
                "in_position",
                "out_price",
                "in_price",
                "out_ev",
                "in_ev",
                "applied_at",
                "gameweek",
            )
            slim = []
            for row in moves[: max(1, max_items)]:
                if not isinstance(row, dict):
                    continue
                slim.append({k: row[k] for k in keep_keys if k in row})
            data["applied_moves"] = slim
        return data
    except Exception:
        return {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _fail(
    summary_file: str,
    reason: str,
    gameweek: int | None = None,
    hours: float | None = None,
) -> SeasonRunResult:
    logger.error(reason)
    return SeasonRunResult(
        ok=False,
        skipped=False,
        reason=reason,
        gameweek=gameweek,
        hours_to_deadline=hours,
        transfer_plan=None,
        chips=None,
        execution=None,
        dgw_teams=[],
        summary_path=summary_file,
    )
