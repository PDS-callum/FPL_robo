"""Unattended weekly season runner for Raspberry Pi."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.auth import ensure_authenticated, load_auth_config
from fpl_robo.chips.decide import recommend_chips
from fpl_robo.chips.inventory import parse_chip_inventory
from fpl_robo.config import (
    BENCH_COVER,
    CHIP_DEFAULTS,
    DEFAULT_MANAGER_ID,
    DEFAULT_SIGNALS_PATH,
    DEFAULT_SUMMARY_PATH,
    EV_MODE,
    HORIZON_EV,
    SEASON_RUNNER,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.live_team import load_live_team_state
from fpl_robo.data.rolling_history import attach_history_to_graph, load_rolling_features
from fpl_robo.data.team_updater import apply_weekly_plan
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.signals import enrich_graph_with_signals
from fpl_robo.optimize.single_gw import optimise_single_gw
from fpl_robo.optimize.transfers import optimise_transfers
from fpl_robo.predict.ev_modes import make_live_ev_model, transfer_gate_kwargs
from fpl_robo.predict.horizon import (
    apply_horizon_ev,
    count_blank_owned,
    mode_uses_horizon,
    team_fixtures_from_kg,
)
from fpl_robo.runtime.atomic import atomic_write_json
from fpl_robo.runtime.deadline import detect_dgw_teams, hours_to_deadline, next_deadline
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
            logger.error("Auth failed: %s", auth.get("reason"))
            if execute:
                return _fail(summary_file, f"Auth failed: {auth.get('reason')}")
            logger.warning("Continuing dry-run with public endpoints only")

        bootstrap = client.bootstrap_static()
        deadline = next_deadline(bootstrap)
        if deadline is None:
            return _fail(summary_file, "No upcoming deadline found")
        gw = int(deadline["id"])
        hours = hours_to_deadline(deadline["deadline_utc"])
        logger.info("Next deadline GW%s in %.2f hours", gw, hours)

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
        factors = aggregate_player_signals(
            bundle, players_by_id={int(p["id"]): p for p in kg.players()}
        )
        # In-season model selected by EV_MODE (set after compare-ev-models).
        model = make_live_ev_model(EV_MODE)
        # Attach signal multipliers by mutating predictions after predict.
        predictions = model.predict(kg, gameweek=gw)
        if factors:
            predictions = _apply_signal_factors(predictions, factors)

        weights = list(HORIZON_EV.get("weights") or [])
        team_fixtures = team_fixtures_from_kg(kg, start_gw=gw, horizon=max(1, len(weights)))
        if mode_uses_horizon(EV_MODE):
            predictions = apply_horizon_ev(
                predictions, start_gw=gw, team_fixtures=team_fixtures
            )

        current_ids = [int(p["element"]) for p in live.picks]
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
        )
        if plan.status != "Optimal":
            return _fail(summary_file, f"Transfer optimiser status: {plan.status}", gw, hours)

        chip_inputs = _compute_chip_opportunity_inputs(
            predictions=predictions,
            current_ids=current_ids,
            bank=live.bank,
            selling_prices=live.selling_prices,
            team_value=float(live.manager.team_value or 0.0),
            state=state,
            gameweek=gw,
        )

        inventory = parse_chip_inventory(live.raw_my_team)
        # DGW bump: make TC/BB thresholds easier when doubles exist.
        chip_cfg: dict[str, float] = {}
        if dgw_teams:
            chip_cfg.update(
                {
                    "tc_min_captain_ev": 5.0,
                    "bb_min_bench_ev": 7.0,
                    "tc_min_percentile": 0.75,
                    "bb_min_percentile": 0.75,
                }
            )
        n_blank = count_blank_owned(
            current_ids=current_ids,
            predictions=predictions,
            team_fixtures=team_fixtures,
            gameweek=gw,
        )
        if n_blank >= int(HORIZON_EV.get("fh_blank_squad_threshold") or 3):
            chip_cfg["fh_min_gap"] = max(
                0.0,
                float(CHIP_DEFAULTS["fh_min_gap"])
                - float(HORIZON_EV.get("fh_blank_gap_relief") or 3.0),
            )
        chips = recommend_chips(
            gameweek=gw,
            starting_xi=plan.solution.starting_xi,
            bench=plan.solution.bench,
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
        }
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
            churn = _unlimited_churn_decision(
                state=state,
                gameweek=gw,
                plan_objective=float(plan.objective),
                locked_objective=chip_inputs.get("locked_objective"),
            )
            if not churn["allow"]:
                logger.info("Unlimited churn guard: %s", churn["reason"])
                execute = False

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
    """Forever loop: recommend/execute near each deadline, then sleep."""
    setup_logging()
    install_stop_handlers()
    default_poll = int(poll_seconds or SEASON_RUNNER["poll_seconds"])
    near_hours = float(SEASON_RUNNER["near_deadline_hours"])
    near_poll = int(SEASON_RUNNER["near_deadline_poll_seconds"])
    logger.info(
        "Starting season daemon (poll=%ss, near_deadline_poll=%ss within %.1fh, execute=%s)",
        default_poll,
        near_poll,
        near_hours,
        execute,
    )
    while not stop_requested():
        try:
            # Once inside the ~2h window, refresh news + predicted XIs before planning.
            try:
                from fpl_robo.signals.pre_deadline import maybe_run_pre_deadline_ingest

                pre = maybe_run_pre_deadline_ingest(apply=True, force=False)
                if not pre.get("skipped"):
                    logger.info(
                        "Pre-deadline signal ingest: ok=%s gw=%s news=%s lineups=%s",
                        pre.get("ok"),
                        pre.get("gameweek"),
                        ((pre.get("ingest") or {}).get("proposed_count")),
                        ((pre.get("ingest") or {}).get("proposed_lineup_count")),
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Pre-deadline signal ingest failed; continuing week run")

            result = run_week(manager_id=manager_id, execute=execute, force=False)
            logger.info(
                "Daemon cycle: ok=%s skipped=%s reason=%s gw=%s hours_to_deadline=%s",
                result.ok,
                result.skipped,
                result.reason,
                result.gameweek,
                result.hours_to_deadline,
            )
            hours = result.hours_to_deadline
            if hours is not None and 0 < hours <= near_hours:
                poll = near_poll
            else:
                poll = default_poll
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


def _unlimited_churn_decision(
    *,
    state: dict[str, Any],
    gameweek: int,
    plan_objective: float,
    locked_objective: float | None = None,
    min_delta: float | None = None,
) -> dict[str, Any]:
    """Whether an unlimited early apply should write despite noise.

    First unlimited write of a GW always allowed. Subsequent writes need objective
    improvement of at least `unlimited_min_obj_delta` vs last unlimited write
    (fallback: locked lineup objective).
    """
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
    current_ids = sorted(int(p["element"]) for p in (live.picks or []) if p.get("element") is not None)
    target_ids = sorted(int(r["id"]) for r in (plan.solution.squad or []))
    if current_ids != target_ids:
        return False
    live_xi = sorted(
        int(p["element"])
        for p in (live.picks or [])
        if p.get("element") is not None and int(p.get("position") or 99) <= 11
    )
    plan_xi = sorted(int(x) for x in (plan.solution.starting_xi or []))
    if live_xi != plan_xi:
        return False
    live_cap = next(
        (int(p["element"]) for p in (live.picks or []) if p.get("is_captain")),
        None,
    )
    return live_cap == int(plan.solution.captain_id)


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
        df.loc[mask, "ev"] = (df.loc[mask, "ev"].astype(float) * mult).clip(lower=0.0)
        if getattr(fac, "exclude", False):
            df.loc[mask, "status"] = "u"
            df.loc[mask, "cover_eligible"] = False
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
