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
    DEFAULT_MANAGER_ID,
    DEFAULT_SIGNALS_PATH,
    DEFAULT_SUMMARY_PATH,
    SEASON_RUNNER,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.data.live_team import load_live_team_state
from fpl_robo.data.team_updater import apply_weekly_plan
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.signals import enrich_graph_with_signals
from fpl_robo.optimize.transfers import optimise_transfers
from fpl_robo.predict.baseline import BaselineEVModel
from fpl_robo.runtime.deadline import detect_dgw_teams, hours_to_deadline, next_deadline
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
        if (
            not force
            and state.get("last_executed_gw") == gw
            and state.get("last_execute_ok")
            and execute
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

        # Only auto-execute inside the configured pre-deadline window.
        in_window = 0 < hours <= float(SEASON_RUNNER["execute_within_hours"])
        if execute and not force and not in_window:
            execute = False
            logger.info(
                "Outside execute window (need 0 < hours <= %s); dry-run only",
                SEASON_RUNNER["execute_within_hours"],
            )

        live = load_live_team_state(manager_id=manager_id, client=client)
        if not live.picks:
            return _fail(summary_file, "No live picks available")

        fixtures = client.fixtures()
        dgw_teams = sorted(detect_dgw_teams(fixtures, gw))

        kg = build_knowledge_graph(bootstrap=bootstrap, fixtures=fixtures, client=client)
        bundle = load_signals(signals_path)
        enrich_graph_with_signals(kg, bundle)
        factors = aggregate_player_signals(
            bundle, players_by_id={int(p["id"]): p for p in kg.players()}
        )
        # In-season model; signal factors still apply when present.
        model = BaselineEVModel()
        # Attach signal multipliers by mutating predictions after predict.
        predictions = model.predict(kg, gameweek=gw)
        if factors:
            predictions = _apply_signal_factors(predictions, factors)

        plan = optimise_transfers(
            predictions,
            current_ids=[int(p["element"]) for p in live.picks],
            bank=live.bank,
            selling_prices=live.selling_prices,
            free_transfers=live.free_transfers,
            max_hits=max_hits if max_hits is not None else int(SEASON_RUNNER["max_hits"]),
            require_cover_eligible=True,
            bench_ev_weight=BENCH_COVER["bench_ev_weight"],
        )
        if plan.status != "Optimal":
            return _fail(summary_file, f"Transfer optimiser status: {plan.status}", gw, hours)

        inventory = parse_chip_inventory(live.raw_my_team)
        # DGW bump: make TC/BB thresholds easier when doubles exist.
        chip_cfg = None
        if dgw_teams:
            chip_cfg = {
                "tc_min_captain_ev": 5.0,
                "bb_min_bench_ev": 7.0,
                "tc_min_percentile": 0.75,
                "bb_min_percentile": 0.75,
            }
        chips = recommend_chips(
            gameweek=gw,
            starting_xi=plan.solution.starting_xi,
            bench=plan.solution.bench,
            captain_id=plan.solution.captain_id,
            inventory=inventory,
            config=chip_cfg,
        ).to_dict()

        chip_to_play = _select_executable_chip(chips.get("play_now") or {})
        execution = None
        if execute:
            logger.info(
                "Executing weekly plan: %s transfers, hits=%s, chip=%s",
                plan.transfers_made,
                plan.hits,
                chip_to_play,
            )
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
        else:
            state["last_recommended_gw"] = gw
            state["last_recommended_at"] = datetime.now(timezone.utc).isoformat()

        payload = {
            "gameweek": gw,
            "hours_to_deadline": round(hours, 3),
            "deadline_time": deadline.get("deadline_time"),
            "dgw_teams": dgw_teams,
            "manager": {
                "manager_id": live.manager.manager_id,
                "name": live.manager.name,
                "team_name": live.manager.team_name,
                "bank": live.bank,
                "free_transfers": live.free_transfers,
            },
            "transfer_plan": plan.to_dict(),
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
            transfer_plan=plan.to_dict(),
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
    while True:
        try:
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
        time.sleep(poll)


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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
