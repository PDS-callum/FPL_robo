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
    HORIZON_EV,
    KG_ENRICH,
    SEASON_RUNNER,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.live_team import load_live_team_state
from fpl_robo.data.rolling_history import attach_history_to_graph, load_rolling_features
from fpl_robo.data.team_updater import apply_weekly_plan
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.enrich import enrich_knowledge_graph
from fpl_robo.graph.signals import enrich_graph_with_signals
from fpl_robo.optimize.single_gw import optimise_single_gw
from fpl_robo.optimize.transfers import optimise_transfers
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
        with phase_kg_enrich(preseason=preseason_now):
            predictions = model.predict(kg, gameweek=gw)
        from fpl_robo.optimize.kg_features import enrich_predictions_from_kg

        predictions = enrich_predictions_from_kg(predictions, kg)
        if factors:
            predictions = _apply_signal_factors(predictions, factors)

        weights = list(HORIZON_EV.get("weights") or [])
        ba_look = int(BLANK_AHEAD.get("lookahead") or 0) if BLANK_AHEAD.get("enabled") else 0
        team_fixtures = team_fixtures_from_kg(
            kg, start_gw=gw, horizon=max(1, len(weights), ba_look + 1)
        )
        current_ids = [int(p["element"]) for p in live.picks]
        if mode_uses_horizon(EV_MODE):
            predictions = apply_horizon_ev(
                predictions, start_gw=gw, team_fixtures=team_fixtures
            )
        if bool(BLANK_AHEAD.get("enabled")):
            predictions = apply_blank_ahead_ev(
                predictions,
                start_gw=gw,
                team_fixtures=team_fixtures,
                current_ids=current_ids,
            )

        plan = optimise_transfers(
            predictions,
            current_ids=current_ids,
            bank=live.bank,
            selling_prices=live.selling_prices,
            free_transfers=live.free_transfers,
            max_hits=max_hits if max_hits is not None else int(SEASON_RUNNER["max_hits"]),
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
        ).to_dict()
        chips["opportunity"] = {
            "current_xi_ev": chip_inputs["current_xi_ev"],
            "optimal_xi_ev": chip_inputs["optimal_xi_ev"],
            "fh_gap": chip_inputs["fh_gap"],
            "rolling_gap": chip_inputs["rolling_gap"],
            "blank_owned": n_blank,
            "horizon_ev": bool(CHIP_DEFAULTS.get("use_horizon_ev", True)),
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
                plan_id_set = {int(x) for x in plan_ids}
                owned_risk_sold = len(risky - plan_id_set)
                throttle = _preseason_execute_throttle(
                    state=state,
                    preseason=is_preseason(bootstrap),
                    optimise_triggers=optimise_triggers or [],
                    plan_quality_fwds=plan_q,
                    current_quality_fwds=cur_q,
                    force=force,
                    transfers_made=int(plan.transfers_made or 0),
                    owned_risk_sold=owned_risk_sold,
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
        if execute:
            logger.info(
                "Executing weekly plan: %s transfers, hits=%s, chip=%s",
                plan.transfers_made,
                plan.hits,
                chip_to_play,
            )
            # Checkpoint before mutating FPL so a mid-write restart can reconcile.
            state["execute_in_progress"] = {
                "gw": gw,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "chip": chip_to_play,
                "target_ids": [int(r["id"]) for r in plan.solution.squad],
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
            state["last_execute_ok"] = bool(execution.get("saved"))
            state["last_executed_at"] = datetime.now(timezone.utc).isoformat()
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
    try:
        for _, row in predictions.iterrows():
            if row.get("id") is None:
                continue
            ev_by_id[int(row["id"])] = float(row.get("ev") or 0.0)
    except Exception:  # noqa: BLE001
        ev_by_id = {}

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
    locked = optimise_transfers(
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
    current_xi_ev = (
        _xi_ev(locked.solution.starting_xi) if locked.status == "Optimal" else 0.0
    )
    locked_objective = float(locked.objective) if locked.status == "Optimal" else None

    budget = float(team_value) + float(bank)
    if budget <= 0:
        budget = 100.0
    unconstrained = optimise_single_gw(
        predictions,
        budget=budget,
        require_cover_eligible=True,
        bench_ev_weight=BENCH_COVER["bench_ev_weight"],
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
    last_obj = state.get("last_unlimited_objective")
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
    """How many FWDs meet ATTACK_FLOOR ep_next threshold."""
    from fpl_robo.config import ATTACK_FLOOR

    if not bool(ATTACK_FLOOR.get("enabled", True)):
        return 0
    try:
        min_ep = float(ATTACK_FLOOR.get("min_ep_next") or 2.1)
    except (TypeError, ValueError):
        min_ep = 2.1
    n = 0
    pred_by_id = {}
    if predictions is not None and hasattr(predictions, "iterrows"):
        pred_by_id = {int(r["id"]): r for _, r in predictions.iterrows()}
    for eid in element_ids:
        pid = int(eid)
        pos = None
        ep = None
        if pid in pred_by_id:
            row = pred_by_id[pid]
            pos = row.get("position")
            ep = row.get("ep_next")
        if (pos is None or ep is None) and players_by_id and pid in players_by_id:
            p = players_by_id[pid]
            pos = pos or p.get("position")
            if ep is None:
                ep = p.get("ep_next")
        if str(pos or "").upper() != "FWD":
            continue
        try:
            ep_v = float(ep or 0.0)
        except (TypeError, ValueError):
            ep_v = 0.0
        if ep_v >= min_ep:
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
    max_items: int = 100,
) -> None:
    """Prepend dated applied transfers onto season state (newest first)."""
    if not transfers:
        return
    history = list(state.get("applied_moves") or [])
    batch = []
    for t in transfers:
        if not isinstance(t, dict):
            continue
        row = dict(t)
        row["applied_at"] = applied_at
        row["gameweek"] = int(gameweek)
        batch.append(row)
    state["applied_moves"] = (batch + history)[: max(1, int(max_items))]


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

    df = predictions.copy()
    for pid, fac in factors.items():
        mask = df["id"] == int(pid)
        if not mask.any():
            continue
        mult = float(fac.combined_mult())
        form_boost = float(getattr(fac, "form_boost", 0.0) or 0.0)
        df.loc[mask, "ev"] = (
            df.loc[mask, "ev"].astype(float) * mult + form_boost
        ).clip(lower=0.0)
        if getattr(fac, "exclude", False):
            df.loc[mask, "status"] = "u"
            df.loc[mask, "cover_eligible"] = False
            df.loc[mask, "buy_eligible"] = False
            df.loc[mask, "ev"] = 0.0
    return df.sort_values("ev", ascending=False).reset_index(drop=True)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
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
