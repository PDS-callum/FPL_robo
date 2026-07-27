"""Assemble read-only dashboard payload from bot artifacts + live FPL metadata."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import DASHBOARD, DEFAULT_MANAGER_ID, EV_MODE, SEASON_RUNNER
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.oidc import find_oidc_user, load_session_payload, seconds_until_expiry
from fpl_robo.runtime.deadline import hours_to_deadline, next_deadline
from fpl_robo.runtime.lockfile import RunLock


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tail_lines(path: str | Path, n: int = 40) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _enrich_transfers(
    summary: dict[str, Any],
    players_by_id: dict[int, dict[str, Any]],
    teams: dict[int, str],
) -> list[dict[str, Any]]:
    transfers = ((summary.get("transfer_plan") or {}).get("transfers")) or []
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    out: list[dict[str, Any]] = []

    def _pos(player: dict[str, Any], explicit: Any) -> Any:
        if explicit:
            return explicit
        if player.get("position"):
            return player.get("position")
        if player.get("element_type") is not None:
            return pos_map.get(int(player["element_type"]))
        return None

    def _team(player: dict[str, Any], explicit: Any) -> Any:
        if explicit:
            return explicit
        if player.get("team_short"):
            return player.get("team_short")
        tid = player.get("team")
        if tid is not None:
            return teams.get(int(tid))
        return None

    for t in transfers:
        if not isinstance(t, dict):
            continue
        out_id = int(t.get("element_out") or 0)
        in_id = int(t.get("element_in") or 0)
        out_p = players_by_id.get(out_id) or {}
        in_p = players_by_id.get(in_id) or {}
        out_price = t.get("out_price")
        if out_price is None and out_p.get("now_cost") is not None:
            out_price = float(out_p["now_cost"]) / 10.0
        elif out_price is None and out_p.get("price") is not None:
            out_price = float(out_p["price"])
        in_price = t.get("in_price")
        if in_price is None and in_p.get("now_cost") is not None:
            in_price = float(in_p["now_cost"]) / 10.0
        elif in_price is None and in_p.get("price") is not None:
            in_price = float(in_p["price"])
        out.append(
            {
                "element_out": out_id,
                "element_in": in_id,
                "out_name": t.get("out_name") or out_p.get("web_name") or str(out_id),
                "in_name": t.get("in_name") or in_p.get("web_name") or str(in_id),
                "out_team": _team(out_p, t.get("out_team")),
                "in_team": _team(in_p, t.get("in_team")),
                "out_position": _pos(out_p, t.get("out_position")),
                "in_position": _pos(in_p, t.get("in_position")),
                "out_price": out_price,
                "in_price": in_price,
                "out_ev": t.get("out_ev"),
                "in_ev": t.get("in_ev"),
            }
        )
    return out


def _split_squad(summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    squad = (((summary.get("transfer_plan") or {}).get("solution") or {}).get("squad")) or []
    xi = [p for p in squad if p.get("start")]
    bench = [p for p in squad if not p.get("start")]
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi.sort(key=lambda p: (pos_order.get(str(p.get("position") or ""), 9), -float(p.get("ev") or 0)))
    bench.sort(key=lambda p: (pos_order.get(str(p.get("position") or ""), 9), -float(p.get("ev") or 0)))
    return xi, bench


def build_dashboard_payload(*, manager_id: int = DEFAULT_MANAGER_ID) -> dict[str, Any]:
    summary = _read_json(DASHBOARD["summary_file"])
    state = _read_json(DASHBOARD["state_file"])
    now = datetime.now(timezone.utc)

    client = FPLClient()
    client.load_session(DASHBOARD["session_path"])
    auth_ok = False
    entry_id = None
    me_error = None
    try:
        me = client.me()
        player = me.get("player") if isinstance(me, dict) else None
        entry_id = (player or {}).get("entry") if isinstance(player, dict) else None
        auth_ok = entry_id is not None
    except Exception as exc:  # noqa: BLE001
        me_error = str(exc)

    bootstrap = {}
    try:
        bootstrap = client.bootstrap_static()
    except Exception:  # noqa: BLE001
        bootstrap = {}

    elements = {
        int(e["id"]): e for e in (bootstrap.get("elements") or []) if e.get("id") is not None
    }
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    for e in elements.values():
        if e.get("element_type") is not None and "position" not in e:
            e["position"] = pos_map.get(int(e["element_type"]))
    teams = {
        int(t["id"]): t.get("short_name") or t.get("name")
        for t in (bootstrap.get("teams") or [])
        if t.get("id") is not None
    }

    deadline = next_deadline(bootstrap) if bootstrap else None
    live_hours = hours_to_deadline(deadline["deadline_utc"]) if deadline else None

    auth_seconds = None
    auth_expires_at = None
    try:
        user = find_oidc_user(load_session_payload(DASHBOARD["session_path"]))
        if user is not None:
            auth_seconds = seconds_until_expiry(user)
            auth_expires_at = user.expires_at
    except Exception:  # noqa: BLE001
        pass

    lock = RunLock(Path(SEASON_RUNNER["lock_file"]))
    transfers = _enrich_transfers(summary, elements, teams)
    xi, bench = _split_squad(summary)
    for row in xi + bench:
        tid = row.get("team_id")
        if tid is not None:
            row["team"] = teams.get(int(tid))

    sol = (summary.get("transfer_plan") or {}).get("solution") or {}
    plan = summary.get("transfer_plan") or {}
    chips = summary.get("chips") or {}
    manager = summary.get("manager") or {}

    in_execute_window = (
        live_hours is not None and 0 < live_hours <= float(SEASON_RUNNER["execute_within_hours"])
    )
    near_deadline = (
        live_hours is not None and 0 < live_hours <= float(SEASON_RUNNER["near_deadline_hours"])
    )
    unlimited = bool(manager.get("unlimited_transfers"))
    policy = summary.get("execute_policy") or {}
    can_write_now = bool(policy.get("can_write")) if policy else (
        in_execute_window or (unlimited and live_hours is not None and live_hours > 0)
    )

    return {
        "generated_at": now.isoformat(),
        "brand": "FPL Robo",
        "status": {
            "auth_ok": auth_ok,
            "entry_id": entry_id or manager.get("manager_id") or manager_id,
            "auth_error": me_error,
            "auth_seconds_remaining": auth_seconds,
            "auth_expires_at": auth_expires_at,
            "run_lock_held": lock.held_by_live_process(),
            "near_deadline": near_deadline,
            "in_execute_window": in_execute_window,
            "unlimited_transfers": unlimited,
            "can_write_now": can_write_now,
            "execute_within_hours": SEASON_RUNNER["execute_within_hours"],
        },
        "deadline": {
            "gameweek": int(deadline["id"]) if deadline else summary.get("gameweek"),
            "deadline_time": deadline.get("deadline_time") if deadline else summary.get("deadline_time"),
            "hours_to_deadline": live_hours if live_hours is not None else summary.get("hours_to_deadline"),
            "name": deadline.get("name") if deadline else None,
        },
        "manager": {
            "manager_id": manager.get("manager_id") or manager_id,
            "name": manager.get("name"),
            "team_name": manager.get("team_name"),
            "bank": manager.get("bank"),
            "free_transfers": manager.get("free_transfers")
            if manager.get("free_transfers") is not None
            else plan.get("free_transfers"),
            "unlimited_transfers": unlimited,
        },
        "plan": {
            "status": plan.get("status"),
            "objective": plan.get("objective"),
            "transfers_made": plan.get("transfers_made"),
            "hits": plan.get("hits"),
            "hit_cost": plan.get("hit_cost"),
            "budget_left": plan.get("budget_left"),
            "budget_used": sol.get("budget_used"),
            "captain_name": sol.get("captain_name"),
            "captain_id": sol.get("captain_id"),
            "transfers": transfers,
            "starting_xi": xi,
            "bench": bench,
            "chip_to_play": summary.get("chip_to_play"),
            "chips_play_now": chips.get("play_now") or {},
            "chip_scores": chips.get("scores") or {},
            "chip_reasons": chips.get("reasons") or {},
        },
        "season": state,
        "execution": summary.get("execution"),
        "signals": summary.get("signals"),
        "summary_mtime": _mtime(DASHBOARD["summary_file"]),
        "logs": {
            "season": _tail_lines(DASHBOARD["log_file"], 25),
            "auth": _tail_lines(DASHBOARD["auth_log_file"], 15),
        },
        "meta": {
            "poll_seconds": SEASON_RUNNER["poll_seconds"],
            "near_deadline_poll_seconds": SEASON_RUNNER["near_deadline_poll_seconds"],
            "server_time_unix": time.time(),
            "ev_mode": summary.get("ev_mode") or EV_MODE,
        },
    }


def _mtime(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
