"""Assemble read-only dashboard payload from bot artifacts + live FPL metadata."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import (
    CHIP_DEFAULTS,
    DASHBOARD,
    DEFAULT_MANAGER_ID,
    DEFAULT_SIGNALS_PATH,
    EV_MODE,
    KG_ENRICH,
    SEASON_RUNNER,
    SIGNAL_INGEST,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.oidc import find_oidc_user, load_session_payload, seconds_until_expiry
from fpl_robo.optimize.live_vector import vector_live_enabled
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
    *,
    plan_key: str = "transfer_plan",
) -> list[dict[str, Any]]:
    transfers = ((summary.get(plan_key) or {}).get("transfers")) or []
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
    sol = ((summary.get("transfer_plan") or {}).get("solution") or {})
    squad = list(sol.get("squad") or [])
    xi = [p for p in squad if p.get("start")]
    bench = [p for p in squad if not p.get("start")]
    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi.sort(key=lambda p: (pos_order.get(str(p.get("position") or ""), 9), -float(p.get("ev") or 0)))
    bench.sort(key=lambda p: (pos_order.get(str(p.get("position") or ""), 9), -float(p.get("ev") or 0)))

    # Annotate C / VC for older summaries that only stored captain_id.
    cap_id = sol.get("captain_id")
    vice_id = sol.get("vice_captain_id")
    if vice_id is None and xi:
        from fpl_robo.optimize.captaincy import pick_vice_captain_id

        vice_id = pick_vice_captain_id(squad or (xi + bench), cap_id)
        if vice_id is not None:
            sol["vice_captain_id"] = vice_id
            for p in xi + bench:
                if int(p.get("id") or 0) == int(vice_id):
                    sol["vice_captain_name"] = p.get("web_name")
                    break
    for p in xi + bench:
        pid = p.get("id")
        p["captain"] = cap_id is not None and int(pid or 0) == int(cap_id)
        p["vice_captain"] = (
            vice_id is not None
            and int(pid or 0) == int(vice_id)
            and not p["captain"]
        )
    return xi, bench


def _photo_code(el: dict[str, Any]) -> int | None:
    code = el.get("code")
    if code is not None:
        try:
            return int(code)
        except (TypeError, ValueError):
            pass
    photo = el.get("photo")
    if photo:
        raw = str(photo).rsplit(".", 1)[0]
        try:
            return int(raw.lstrip("p"))
        except ValueError:
            return None
    return None


def _enrich_squad_display(
    players: list[dict[str, Any]],
    elements: dict[int, dict[str, Any]],
    *,
    team_codes: dict[int, int] | None = None,
) -> None:
    """Attach FPL bootstrap fields used by the pitch detail sheet / headshots."""
    team_codes = team_codes or {}
    for row in players:
        if not isinstance(row, dict):
            continue
        try:
            pid = int(row.get("id") or 0)
        except (TypeError, ValueError):
            continue
        el = elements.get(pid) or {}
        tid = row.get("team_id")
        if tid is None and el.get("team") is not None:
            tid = el.get("team")
            try:
                row["team_id"] = int(tid)
            except (TypeError, ValueError):
                tid = None
        if tid is not None:
            try:
                code = team_codes.get(int(tid))
                if code is not None:
                    row["team_code"] = int(code)
            except (TypeError, ValueError):
                pass
        if not el:
            continue
        row["photo_code"] = _photo_code(el)
        row["first_name"] = el.get("first_name") or row.get("first_name")
        row["second_name"] = el.get("second_name") or row.get("second_name")
        for key in (
            "form",
            "points_per_game",
            "selected_by_percent",
            "status",
            "news",
            "chance_of_playing_next_round",
            "chance_of_playing_this_round",
            "minutes",
            "starts",
            "goals_scored",
            "assists",
            "clean_sheets",
            "bonus",
            "bps",
            "ict_index",
            "expected_goals",
            "expected_assists",
            "expected_goal_involvements",
            "ep_next",
            "ep_this",
            "transfers_in_event",
            "transfers_out_event",
        ):
            if el.get(key) is not None:
                row[key] = el.get(key)


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
    team_codes = {
        int(t["id"]): int(t["code"])
        for t in (bootstrap.get("teams") or [])
        if t.get("id") is not None and t.get("code") is not None
    }

    deadline = next_deadline(bootstrap) if bootstrap else None
    live_hours = hours_to_deadline(deadline["deadline_utc"]) if deadline else None

    fixtures: list[dict[str, Any]] = []
    try:
        fixtures = client.fixtures()
    except Exception:  # noqa: BLE001
        fixtures = []

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
    ft_key = "ft_transfer_plan" if isinstance(summary.get("ft_transfer_plan"), dict) else "transfer_plan"
    transfers = _enrich_transfers(summary, elements, teams, plan_key=ft_key)
    chip_rebuild_transfers = (
        _enrich_transfers(summary, elements, teams, plan_key="chip_rebuild_plan")
        if isinstance(summary.get("chip_rebuild_plan"), dict)
        else []
    )
    xi, bench = _split_squad(summary)
    for row in xi + bench:
        tid = row.get("team_id")
        if tid is not None:
            row["team"] = teams.get(int(tid))
    _enrich_squad_display(xi + bench, elements, team_codes=team_codes)

    sol = (summary.get("transfer_plan") or {}).get("solution") or {}
    plan = summary.get("transfer_plan") or {}
    chips = summary.get("chips") or {}
    manager = summary.get("manager") or {}
    applied_moves = _applied_moves_payload(state, summary)
    intel = _load_intel_feed()
    stats = _build_stats(
        xi=xi,
        bench=bench,
        plan=plan,
        manager=manager,
        chips=chips,
        applied_moves=applied_moves,
        hours_to_deadline=live_hours if live_hours is not None else summary.get("hours_to_deadline"),
        signals=intel.get("summary"),
    )
    # Attach chip schedule / lineup gate onto stats for the pulse panel.
    stats["chip_schedule"] = chips.get("schedule") or {}
    stats["chip_lineup_gate"] = chips.get("lineup_gate") or {}
    mid = int(entry_id or manager.get("manager_id") or manager_id)
    scoreboard = _fetch_scoreboard(client, mid, bootstrap=bootstrap)
    history = _record_and_load_history(
        scoreboard=scoreboard,
        stats=stats,
        gameweek=int(deadline["id"]) if deadline else summary.get("gameweek"),
    )

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
    tunnel = _tunnel_status()
    schedule = _bot_schedule_payload(
        hours_to_deadline=live_hours if live_hours is not None else summary.get("hours_to_deadline"),
        unlimited=unlimited,
        can_write_now=can_write_now,
        near_deadline=near_deadline,
        in_execute_window=in_execute_window,
        summary=summary,
        state=state,
        run_lock_held=lock.held_by_live_process(),
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
            "name": manager.get("name") or scoreboard.get("name"),
            "team_name": manager.get("team_name") or scoreboard.get("team_name"),
            "bank": manager.get("bank"),
            "free_transfers": manager.get("free_transfers")
            if manager.get("free_transfers") is not None
            else plan.get("free_transfers"),
            "unlimited_transfers": unlimited,
            "overall_points": scoreboard.get("overall_points"),
            "overall_rank": scoreboard.get("overall_rank"),
            "event_points": scoreboard.get("event_points"),
            "event_rank": scoreboard.get("event_rank"),
            "current_event": scoreboard.get("current_event"),
        },
        "scoreboard": scoreboard,
        "history": history,
        "plan": {
            "status": plan.get("status"),
            "objective": plan.get("objective"),
            "optimiser": plan.get("optimiser"),
            "transfers_made": plan.get("transfers_made"),
            "hits": plan.get("hits"),
            "hit_cost": plan.get("hit_cost"),
            "budget_left": plan.get("budget_left"),
            "budget_used": sol.get("budget_used"),
            "captain_name": sol.get("captain_name"),
            "captain_id": sol.get("captain_id"),
            "vice_captain_name": sol.get("vice_captain_name"),
            "vice_captain_id": sol.get("vice_captain_id"),
            "transfers": transfers,
            "chip_rebuild_transfers": chip_rebuild_transfers,
            "ft_optimiser": (summary.get("ft_transfer_plan") or plan).get("optimiser"),
            "ft_transfers_made": (summary.get("ft_transfer_plan") or plan).get(
                "transfers_made"
            ),
            "starting_xi": xi,
            "bench": bench,
            "chip_to_play": summary.get("chip_to_play"),
            "chips_play_now": chips.get("play_now") or {},
            "chip_scores": chips.get("scores") or {},
            "chip_reasons": chips.get("reasons") or {},
            "chip_schedule": chips.get("schedule") or {},
            "chip_lineup_gate": chips.get("lineup_gate") or {},
        },
        "stats": stats,
        "hub": _build_fpl_hub(
            bootstrap,
            fixtures=fixtures,
            current_gw=int(deadline["id"]) if deadline else summary.get("gameweek"),
            owned_ids=[int(p["id"]) for p in (xi + bench) if p.get("id") is not None],
        ),
        "applied_moves": applied_moves,
        "intel": intel,
        "schedule": schedule,
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
            "feature_version": KG_ENRICH.get("feature_version"),
            "optimiser": plan.get("optimiser")
            or ("vector_team_sa" if vector_live_enabled() else "mip_pulp"),
            "residual_fit": (plan.get("meta") or {}).get("residual_fit"),
            "gate_reject": ((plan.get("meta") or {}).get("consensus") or {}).get(
                "gate_reject"
            ),
            "attack_floor_ok": (plan.get("meta") or {}).get("attack_floor_ok"),
            "consensus_mode": ((plan.get("meta") or {}).get("consensus") or {}).get(
                "mode"
            ),
            "shadow_mip": {
                "status": (summary.get("shadow_mip_plan") or {}).get("status"),
                "transfers_made": (summary.get("shadow_mip_plan") or {}).get(
                    "transfers_made"
                ),
                "objective": (summary.get("shadow_mip_plan") or {}).get("objective"),
                "vector_objective": plan.get("objective"),
            }
            if summary.get("shadow_mip_plan")
            else None,
            "public_tunnel_url": tunnel.get("url"),
            "tunnel_url_mtime": tunnel.get("mtime"),
            "tunnel_age_seconds": tunnel.get("age_seconds"),
        },
    }


def _build_fpl_hub(
    bootstrap: dict[str, Any],
    *,
    fixtures: list[dict[str, Any]],
    current_gw: int | None,
    owned_ids: list[int] | None = None,
    horizon: int = 8,
) -> dict[str, Any]:
    """FPL-app style fixtures / FDR / player stats (separate from bot plan)."""
    from fpl_robo.features.hub_priors import (
        blank_double_by_gw,
        team_run_ease,
    )

    if not bootstrap:
        return {"ok": False, "reason": "bootstrap_unavailable"}

    teams_raw = list(bootstrap.get("teams") or [])
    elements = list(bootstrap.get("elements") or [])
    events = list(bootstrap.get("events") or [])
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    team_by_id = {
        int(t["id"]): {
            "id": int(t["id"]),
            "name": t.get("name"),
            "short": t.get("short_name") or t.get("name"),
            "strength": t.get("strength"),
            "strength_attack_home": t.get("strength_attack_home"),
            "strength_attack_away": t.get("strength_attack_away"),
            "strength_defence_home": t.get("strength_defence_home"),
            "strength_defence_away": t.get("strength_defence_away"),
        }
        for t in teams_raw
        if t.get("id") is not None
    }
    owned = {int(i) for i in (owned_ids or [])}

    gw0 = int(current_gw or 1)
    # Prefer next unfinished event when bootstrap marks one.
    for ev in events:
        if ev.get("is_next"):
            try:
                gw0 = int(ev["id"])
            except (TypeError, ValueError, KeyError):
                pass
            break
    gws = list(range(gw0, gw0 + int(horizon)))

    # team_id → gw → {fdr, opp, home, kickoff}
    by_team_gw: dict[int, dict[int, dict[str, Any]]] = {
        tid: {} for tid in team_by_id
    }
    fixtures_by_gw: dict[int, list[dict[str, Any]]] = {gw: [] for gw in gws}

    for f in fixtures:
        ev = f.get("event")
        if ev is None:
            continue
        try:
            gw = int(ev)
        except (TypeError, ValueError):
            continue
        if gw not in fixtures_by_gw and gw not in gws:
            # Still record into by_team if within extended window later
            pass
        th, ta = f.get("team_h"), f.get("team_a")
        if th is None or ta is None:
            continue
        th, ta = int(th), int(ta)
        kickoff = f.get("kickoff_time")
        finished = bool(f.get("finished"))
        started = bool(f.get("started"))
        score = None
        if finished or started:
            score = {
                "h": f.get("team_h_score"),
                "a": f.get("team_a_score"),
            }
        home_row = {
            "fdr": int(f.get("team_h_difficulty") or 3),
            "opp": team_by_id.get(ta, {}).get("short") or str(ta),
            "opp_id": ta,
            "home": True,
            "kickoff": kickoff,
            "finished": finished,
        }
        away_row = {
            "fdr": int(f.get("team_a_difficulty") or 3),
            "opp": team_by_id.get(th, {}).get("short") or str(th),
            "opp_id": th,
            "home": False,
            "kickoff": kickoff,
            "finished": finished,
        }
        if th in by_team_gw:
            by_team_gw[th][gw] = home_row
        if ta in by_team_gw:
            by_team_gw[ta][gw] = away_row
        if gw in fixtures_by_gw:
            fixtures_by_gw[gw].append(
                {
                    "id": f.get("id"),
                    "kickoff": kickoff,
                    "finished": finished,
                    "started": started,
                    "score": score,
                    "home": {
                        "id": th,
                        "name": team_by_id.get(th, {}).get("name"),
                        "short": team_by_id.get(th, {}).get("short"),
                        "fdr": home_row["fdr"],
                    },
                    "away": {
                        "id": ta,
                        "name": team_by_id.get(ta, {}).get("name"),
                        "short": team_by_id.get(ta, {}).get("short"),
                        "fdr": away_row["fdr"],
                    },
                }
            )

    for gw in fixtures_by_gw:
        fixtures_by_gw[gw].sort(key=lambda r: (r.get("kickoff") or "", r.get("id") or 0))

    fdr_rows = []
    for tid, meta in sorted(team_by_id.items(), key=lambda x: x[1].get("short") or ""):
        cells = []
        for gw in gws:
            slot = (by_team_gw.get(tid) or {}).get(gw)
            if not slot:
                cells.append({"gw": gw, "blank": True, "fdr": None})
                continue
            fdr = int(slot["fdr"])
            cells.append(
                {
                    "gw": gw,
                    "blank": False,
                    "fdr": fdr,
                    "opp": slot.get("opp"),
                    "home": slot.get("home"),
                    "label": f"{'H' if slot.get('home') else 'A'} {slot.get('opp')}",
                }
            )
        fdr_rows.append(
            {
                "id": tid,
                "short": meta.get("short"),
                "name": meta.get("name"),
                "cells": cells,
                "run_ease": team_run_ease(cells),
            }
        )
    fdr_rows.sort(key=lambda r: (-(r.get("run_ease") or 0), r.get("short") or ""))

    players = []
    team_next5: dict[int, float] = {}
    for row in fdr_rows:
        cells = [c for c in (row.get("cells") or [])[:5] if not c.get("blank") and c.get("fdr") is not None]
        if cells:
            team_next5[int(row["id"])] = round(
                sum(max(0.0, 6.0 - float(c["fdr"])) for c in cells) / len(cells), 2
            )

    for el in elements:
        try:
            pid = int(el["id"])
        except (TypeError, ValueError, KeyError):
            continue
        price = float(el.get("now_cost") or 0) / 10.0
        pts = float(el.get("total_points") or 0)
        form = float(el.get("form") or 0)
        ppg = float(el.get("points_per_game") or 0)
        minutes = float(el.get("minutes") or 0)
        if minutes <= 0 and pts <= 0 and form <= 0:
            # Keep premiums / owned even if cold.
            if pid not in owned and price < 5.5:
                continue
        tid = int(el.get("team") or 0)
        tin = int(el.get("transfers_in_event") or 0)
        tout = int(el.get("transfers_out_event") or 0)
        delta_form = round(form - ppg, 2) if ppg or form else 0.0
        price_delta = int(el.get("cost_change_event") or 0)
        price_season = int(el.get("cost_change_start") or 0)
        chance = el.get("chance_of_playing_next_round")
        try:
            chance_n = int(chance) if chance is not None else None
        except (TypeError, ValueError):
            chance_n = None
        news = (el.get("news") or "").strip()
        status = el.get("status") or "a"
        players.append(
            {
                "id": pid,
                "name": el.get("web_name") or el.get("second_name") or str(pid),
                "team": team_by_id.get(tid, {}).get("short"),
                "team_id": tid,
                "pos": pos_map.get(int(el.get("element_type") or 0), "?"),
                "price": price,
                "form": form,
                "pts": pts,
                "ppg": ppg,
                "delta_form": delta_form,
                "own": float(el.get("selected_by_percent") or 0),
                "ep": float(el.get("ep_next") or 0),
                "ep_this": float(el.get("ep_this") or 0),
                "ict": float(el.get("ict_index") or 0),
                "xgi": float(el.get("expected_goal_involvements") or 0),
                "xgi90": float(el.get("expected_goal_involvements_per_90") or 0),
                "goals": int(el.get("goals_scored") or 0),
                "assists": int(el.get("assists") or 0),
                "cs": int(el.get("clean_sheets") or 0),
                "minutes": int(minutes),
                "status": status,
                "news": news or None,
                "chance": chance_n,
                "price_delta": price_delta,
                "price_season": price_season,
                "transfers_in": tin,
                "transfers_out": tout,
                "net_transfers": tin - tout,
                "event_pts": int(el.get("event_points") or 0),
                "value_form": float(el.get("value_form") or 0),
                "fixture_ease": team_next5.get(tid),
                "owned": pid in owned,
                "value": round(pts / price, 2) if price > 0 else 0.0,
            }
        )

    fdr_by_team = {int(r["id"]): r for r in fdr_rows}

    def _top(key: str, n: int = 25, *, reverse: bool = True) -> list[dict[str, Any]]:
        return sorted(players, key=lambda p: float(p.get(key) or 0), reverse=reverse)[:n]

    # Active / relevant pool for trend boards (avoid noise from unused players).
    active = [p for p in players if (p.get("minutes") or 0) >= 90 or p.get("owned") or (p.get("form") or 0) >= 3]
    if len(active) < 40:
        active = players

    def _top_active(key: str, n: int = 8, *, reverse: bool = True) -> list[dict[str, Any]]:
        return sorted(active, key=lambda p: float(p.get(key) or 0), reverse=reverse)[:n]

    owned_players = [p for p in players if p.get("owned")]
    owned_alerts = []
    for p in owned_players:
        flags = []
        if p.get("status") and p["status"] != "a":
            flags.append({"kind": "status", "label": str(p["status"]).upper()})
        if p.get("chance") is not None and int(p["chance"]) < 75:
            flags.append({"kind": "doubt", "label": f"{p['chance']}%"})
        if p.get("news"):
            flags.append({"kind": "news", "label": "news"})
        if (p.get("price_delta") or 0) <= -1:
            flags.append({"kind": "fall", "label": f"−£{abs(p['price_delta']) / 10:.1f}"})
        if (p.get("price_delta") or 0) >= 1:
            flags.append({"kind": "rise", "label": f"+£{p['price_delta'] / 10:.1f}"})
        ease = p.get("fixture_ease")
        if ease is not None and ease >= 3.2:
            flags.append({"kind": "easy", "label": "good run"})
        elif ease is not None and ease <= 2.2:
            flags.append({"kind": "hard", "label": "tough run"})
        if (p.get("delta_form") or 0) <= -2:
            flags.append({"kind": "cold", "label": "form↓"})
        elif (p.get("delta_form") or 0) >= 2:
            flags.append({"kind": "hot", "label": "form↑"})
        if flags:
            owned_alerts.append({**p, "flags": flags})

    # Owned squad fixture strip (next 5 from FDR matrix).
    pos_rank = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad_fixtures = []
    for p in sorted(
        owned_players,
        key=lambda x: (pos_rank.get(str(x.get("pos")), 9), -(x.get("ep") or 0)),
    ):
        row = fdr_by_team.get(int(p.get("team_id") or 0)) or {}
        squad_fixtures.append(
            {
                **{k: p.get(k) for k in ("id", "name", "team", "pos", "price", "form", "ep", "status", "chance")},
                "next5": (row.get("cells") or [])[:5],
                "run_ease": row.get("run_ease"),
            }
        )

    # Captain / vice candidates: high EP, prefer form + soft fixtures.
    def _captain_score(p: dict[str, Any]) -> float:
        return (
            float(p.get("ep") or 0) * 1.4
            + float(p.get("form") or 0) * 0.6
            + float(p.get("fixture_ease") or 0) * 0.8
            - (0.5 if (p.get("chance") is not None and int(p["chance"]) < 75) else 0.0)
            - (2.0 if p.get("status") and p["status"] != "a" else 0.0)
        )

    captain_pool = [
        p
        for p in active
        if p.get("pos") != "GK"
        and (p.get("status") or "a") == "a"
        and (p.get("chance") is None or int(p["chance"]) >= 75)
    ]
    captain_picks = sorted(captain_pool, key=_captain_score, reverse=True)[:8]

    # Differentials: low ownership, meaningful form/EP.
    differentials = sorted(
        [
            p
            for p in active
            if float(p.get("own") or 0) < 8.0
            and float(p.get("form") or 0) >= 4.0
            and (p.get("status") or "a") == "a"
        ],
        key=lambda p: (float(p.get("form") or 0) * 1.2 + float(p.get("ep") or 0) - float(p.get("own") or 0) * 0.15),
        reverse=True,
    )[:8]

    # Availability board (league-wide).
    doubts = []
    for p in players:
        if (p.get("minutes") or 0) < 1 and not p.get("owned"):
            continue
        bad = (p.get("status") or "a") != "a" or (
            p.get("chance") is not None and int(p["chance"]) < 100
        ) or bool(p.get("news"))
        if not bad:
            continue
        if float(p.get("own") or 0) < 1.0 and not p.get("owned") and (p.get("minutes") or 0) < 180:
            continue
        doubts.append(p)
    doubts.sort(
        key=lambda p: (
            0 if p.get("owned") else 1,
            int(p.get("chance") if p.get("chance") is not None else 50),
            -float(p.get("own") or 0),
        )
    )
    doubts = doubts[:12]

    # Best by position (EP next).
    by_pos: dict[str, list[dict[str, Any]]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in _top_active("ep", 40):
        pos = str(p.get("pos") or "")
        if pos in by_pos and len(by_pos[pos]) < 4:
            by_pos[pos].append(p)

    # xGI leaders (attacking underlying).
    xgi_leaders = _top_active("xgi", 8)

    # Blank / double detection across horizon (shared with live bot helpers).
    team_shorts = {tid: meta["short"] for tid, meta in team_by_id.items()}
    blank_double = blank_double_by_gw(
        fixtures_by_gw=fixtures_by_gw,
        team_shorts=team_shorts,
        gws=gws[:6],
    )

    easiest_runs = [
        {
            "id": r["id"],
            "short": r.get("short"),
            "name": r.get("name"),
            "run_ease": r.get("run_ease"),
            "next5": (r.get("cells") or [])[:5],
        }
        for r in fdr_rows[:6]
    ]
    hardest_runs = [
        {
            "id": r["id"],
            "short": r.get("short"),
            "name": r.get("name"),
            "run_ease": r.get("run_ease"),
            "next5": (r.get("cells") or [])[:5],
        }
        for r in list(reversed(fdr_rows))[:6]
    ]

    next_gw_matches = list(fixtures_by_gw.get(gw0) or [])

    return {
        "ok": True,
        "current_gw": gw0,
        "horizon_gws": gws,
        "fdr": {
            "gws": gws,
            "rows": fdr_rows,
            "easiest": easiest_runs,
            "hardest": hardest_runs,
        },
        "fixtures": {
            "by_gw": [
                {"gameweek": gw, "matches": fixtures_by_gw.get(gw) or []}
                for gw in gws[:3]
            ],
            "next_gw": {
                "gameweek": gw0,
                "matches": next_gw_matches,
            },
            "blank_double": blank_double,
        },
        "stats": {
            "top_form": _top("form"),
            "top_points": _top("pts"),
            "top_value": _top("value"),
            "top_ep": _top("ep"),
            "players": players,
        },
        "trends": {
            "price_risers": _top_active("price_delta", 8),
            "price_fallers": _top_active("price_delta", 8, reverse=False),
            "transfer_in": _top_active("net_transfers", 8),
            "transfer_out": _top_active("net_transfers", 8, reverse=False),
            "form_hot": [p for p in _top_active("delta_form", 12) if (p.get("delta_form") or 0) > 0][:8],
            "form_cold": [p for p in _top_active("delta_form", 12, reverse=False) if (p.get("delta_form") or 0) < 0][:8],
            "value_form": _top_active("value_form", 8),
            "owned_alerts": owned_alerts[:20],
            "squad_fixtures": squad_fixtures,
            "captain_picks": captain_picks,
            "differentials": differentials,
            "doubts": doubts,
            "by_pos": by_pos,
            "xgi_leaders": xgi_leaders,
        },
        "owned_ids": sorted(owned),
    }


def _build_stats(
    *,
    xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    plan: dict[str, Any],
    manager: dict[str, Any],
    chips: dict[str, Any],
    applied_moves: list[dict[str, Any]],
    hours_to_deadline: float | None,
    signals: Any,
) -> dict[str, Any]:
    """Derived metrics/charts for the dashboard UI."""
    def _ev(p: dict[str, Any]) -> float:
        try:
            return float(p.get("ev") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _price(p: dict[str, Any]) -> float:
        try:
            return float(p.get("price") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    xi_ev = sum(_ev(p) for p in xi)
    bench_ev = sum(_ev(p) for p in bench)
    captain = next((p for p in xi if p.get("captain") or p.get("id") == plan.get("captain_id")), None)
    captain_ev = _ev(captain) if captain else 0.0
    vice = next(
        (
            p
            for p in xi
            if p.get("vice_captain") or p.get("id") == plan.get("vice_captain_id")
        ),
        None,
    )
    if vice is None and plan.get("vice_captain_id") is not None:
        vice = next((p for p in bench if p.get("id") == plan.get("vice_captain_id")), None)
    # Captain counts twice in FPL scoring; objective often already includes that.
    squad_value = sum(_price(p) for p in xi) + sum(_price(p) for p in bench)
    xi_value = sum(_price(p) for p in xi)

    pos_order = ["GK", "DEF", "MID", "FWD"]
    ev_by_pos = {pos: 0.0 for pos in pos_order}
    count_by_pos = {pos: 0 for pos in pos_order}
    for p in xi:
        pos = str(p.get("position") or "")
        if pos in ev_by_pos:
            ev_by_pos[pos] += _ev(p)
            count_by_pos[pos] += 1
    formation = "-".join(str(count_by_pos[p]) for p in ("DEF", "MID", "FWD"))

    ranked = sorted(xi, key=_ev, reverse=True)
    top_ev = [
        {
            "id": p.get("id"),
            "web_name": p.get("web_name"),
            "position": p.get("position"),
            "team": p.get("team"),
            "ev": round(_ev(p), 2),
            "price": round(_price(p), 1),
            "captain": bool(p.get("captain") or p.get("id") == plan.get("captain_id")),
        }
        for p in ranked[:5]
    ]

    chip_scores = chips.get("scores") or {}
    chip_play = chips.get("play_now") or {}
    chip_thresholds = chips.get("thresholds") if isinstance(chips.get("thresholds"), dict) else {}
    # Bar fill uses the same units as each chip's play check.
    # TC is gated on captain_ev (not 2× tc_score); BB/FH/WC use their score directly.
    chip_display = {
        "triple_captain": {
            "label": "TC",
            "score": float(
                chip_scores.get("captain_ev")
                if chip_scores.get("captain_ev") is not None
                else (float(chip_scores.get("triple_captain") or 0.0) / 2.0)
            ),
            "cap": float(
                chip_thresholds.get("tc_min_captain_ev")
                or CHIP_DEFAULTS.get("tc_min_captain_ev")
                or 6.0
            ),
        },
        "bench_boost": {
            "label": "BB",
            "score": float(chip_scores.get("bench_boost") or 0.0),
            "cap": float(
                chip_thresholds.get("bb_min_bench_ev")
                or CHIP_DEFAULTS.get("bb_min_bench_ev")
                or 8.0
            ),
        },
        "free_hit": {
            "label": "FH",
            "score": float(chip_scores.get("free_hit") or 0.0),
            "cap": float(
                chip_thresholds.get("fh_min_gap")
                or CHIP_DEFAULTS.get("fh_min_gap")
                or 12.0
            ),
        },
        "wildcard": {
            "label": "WC",
            "score": float(chip_scores.get("wildcard") or 0.0),
            "cap": float(
                chip_thresholds.get("wc_min_rolling_gap")
                or CHIP_DEFAULTS.get("wc_min_rolling_gap")
                or 18.0
            ),
        },
    }
    chip_bars = []
    for key, meta in chip_display.items():
        try:
            score = float(meta["score"] or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        cap = max(0.01, float(meta["cap"] or 1.0))
        fill = min(1.0, max(0.0, score / cap))
        chip_bars.append(
            {
                "key": key,
                "label": meta["label"],
                "score": round(score, 2),
                "cap": round(cap, 2),
                "fill": round(fill, 3),
                "play": bool(chip_play.get(key)),
            }
        )

    hours = None
    try:
        hours = float(hours_to_deadline) if hours_to_deadline is not None else None
    except (TypeError, ValueError):
        hours = None
    # Visual fill for countdown ring: full at 14 days out, empty at deadline.
    deadline_fill = None
    if hours is not None and hours > 0:
        deadline_fill = round(min(1.0, max(0.0, hours / (14.0 * 24.0))), 3)

    sig_summary = {}
    if isinstance(signals, dict):
        sig_summary = {
            "count": signals.get("total")
            or signals.get("count")
            or signals.get("n_signals")
            or 0,
            "news": signals.get("news"),
            "excluded": signals.get("excluded") or signals.get("n_excluded"),
            "updated_at": signals.get("updated_at"),
            "model": signals.get("model"),
        }

    return {
        "formation": formation if any(count_by_pos.values()) else None,
        "xi_ev": round(xi_ev, 2),
        "bench_ev": round(bench_ev, 2),
        "captain_ev": round(captain_ev, 2),
        "captain_name": (captain or {}).get("web_name") or plan.get("captain_name"),
        "vice_captain_name": (vice or {}).get("web_name") or plan.get("vice_captain_name"),
        "vice_captain_ev": round(_ev(vice), 2) if vice else None,
        "squad_value": round(squad_value, 1),
        "xi_value": round(xi_value, 1),
        "bank": manager.get("bank"),
        "plan_objective": plan.get("objective"),
        "ev_by_position": [{"position": p, "ev": round(ev_by_pos[p], 2), "n": count_by_pos[p]} for p in pos_order],
        "top_ev": top_ev,
        "xi_ev_bars": [
            {
                "web_name": p.get("web_name"),
                "position": p.get("position"),
                "ev": round(_ev(p), 2),
                "captain": bool(p.get("captain") or p.get("id") == plan.get("captain_id")),
            }
            for p in ranked
        ],
        "chip_bars": chip_bars,
        "deadline_hours": hours,
        "deadline_fill": deadline_fill,
        "applied_moves_count": len(applied_moves),
        "applied_last_at": applied_moves[0].get("applied_at") if applied_moves else None,
        "signals": sig_summary,
    }


def _fetch_scoreboard(
    client: FPLClient,
    manager_id: int,
    *,
    bootstrap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Live FPL score / rank for the managed entry."""
    out: dict[str, Any] = {
        "manager_id": manager_id,
        "overall_points": None,
        "overall_rank": None,
        "event_points": None,
        "event_rank": None,
        "current_event": None,
        "team_name": None,
        "name": None,
        "gw_history": [],
        "past_seasons": [],
        "error": None,
    }
    try:
        entry = client.entry(manager_id)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"entry: {exc}"
        return out

    out["team_name"] = entry.get("name")
    first = entry.get("player_first_name") or ""
    last = entry.get("player_last_name") or ""
    out["name"] = f"{first} {last}".strip() or None
    out["overall_points"] = entry.get("summary_overall_points")
    out["overall_rank"] = entry.get("summary_overall_rank")
    out["event_points"] = entry.get("summary_event_points")
    out["event_rank"] = entry.get("summary_event_rank")
    out["current_event"] = entry.get("current_event")

    try:
        hist = client.entry_history(manager_id)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"history: {exc}"
        return out

    current = hist.get("current") or []
    gw_rows = []
    cumulative = 0
    for row in current:
        if not isinstance(row, dict):
            continue
        pts = int(row.get("points") or 0)
        cumulative += pts
        gw_rows.append(
            {
                "gameweek": int(row.get("event") or 0),
                "points": pts,
                "total_points": int(row.get("total_points") or cumulative),
                "rank": row.get("rank"),
                "overall_rank": row.get("overall_rank"),
                "percentile_rank": row.get("percentile_rank"),
                "bank": (float(row.get("bank") or 0) / 10.0) if row.get("bank") is not None else None,
                "value": (float(row.get("value") or 0) / 10.0) if row.get("value") is not None else None,
                "event_transfers": row.get("event_transfers"),
                "event_transfers_cost": row.get("event_transfers_cost"),
                "points_on_bench": row.get("points_on_bench"),
            }
        )
    out["gw_history"] = gw_rows
    if gw_rows and out["overall_points"] is None:
        out["overall_points"] = gw_rows[-1].get("total_points")
    if gw_rows and out["overall_rank"] is None:
        out["overall_rank"] = gw_rows[-1].get("overall_rank")
    if gw_rows and out["event_points"] is None:
        out["event_points"] = gw_rows[-1].get("points")
    if gw_rows and out["current_event"] is None:
        out["current_event"] = gw_rows[-1].get("gameweek")

    past = []
    for row in hist.get("past") or []:
        if not isinstance(row, dict):
            continue
        past.append(
            {
                "season": row.get("season_name"),
                "total_points": row.get("total_points"),
                "rank": row.get("rank"),
            }
        )
    out["past_seasons"] = past

    # Preseason / before GW1: treat as 0 pts rather than missing forever.
    events = (bootstrap or {}).get("events") or []
    finished = any(bool(e.get("finished")) for e in events if isinstance(e, dict))
    out["season_started"] = bool(finished or gw_rows)
    if not out["season_started"]:
        out["overall_points"] = out["overall_points"] if out["overall_points"] is not None else 0
        out["event_points"] = out["event_points"] if out["event_points"] is not None else 0
    return out


def _metrics_history_path() -> Path:
    return Path(DASHBOARD.get("metrics_history_file") or "data/cache/dashboard_metrics.json")


def _record_and_load_history(
    *,
    scoreboard: dict[str, Any],
    stats: dict[str, Any],
    gameweek: Any,
) -> dict[str, Any]:
    """Append a throttled snapshot and return series for charts."""
    path = _metrics_history_path()
    payload = _read_json(path)
    points = [p for p in (payload.get("points") or []) if isinstance(p, dict)]
    now = datetime.now(timezone.utc)
    min_interval = float(DASHBOARD.get("metrics_min_interval_seconds") or 900)
    max_points = int(DASHBOARD.get("metrics_max_points") or 400)

    snapshot = {
        "ts": now.isoformat(),
        "gameweek": int(gameweek) if gameweek is not None else None,
        "overall_points": scoreboard.get("overall_points"),
        "overall_rank": scoreboard.get("overall_rank"),
        "event_points": scoreboard.get("event_points"),
        "plan_objective": stats.get("plan_objective"),
        "xi_ev": stats.get("xi_ev"),
        "bench_ev": stats.get("bench_ev"),
        "squad_value": stats.get("squad_value"),
        "bank": stats.get("bank"),
    }

    should_append = True
    if points:
        last = points[-1]
        try:
            last_ts = datetime.fromisoformat(str(last.get("ts")).replace("Z", "+00:00"))
            age = (now - last_ts).total_seconds()
        except Exception:  # noqa: BLE001
            age = min_interval + 1
        same_score = (
            last.get("overall_points") == snapshot["overall_points"]
            and last.get("event_points") == snapshot["event_points"]
            and last.get("plan_objective") == snapshot["plan_objective"]
            and last.get("xi_ev") == snapshot["xi_ev"]
        )
        if age < min_interval and same_score:
            should_append = False
        elif age < 60:
            # Hard floor even if values twitch during a run.
            should_append = False

    if should_append:
        points.append(snapshot)
        points = points[-max_points:]

    # Durable GW series from official FPL history (survives restarts / empty seasons).
    gw_history = scoreboard.get("gw_history") or []
    gw_points = [
        {
            "gameweek": g.get("gameweek"),
            "points": g.get("points"),
            "total_points": g.get("total_points"),
            "overall_rank": g.get("overall_rank"),
            "transfers_cost": g.get("event_transfers_cost"),
        }
        for g in gw_history
        if isinstance(g, dict)
    ]
    # Seed bot snapshots from finished GWs when local history is thin.
    if gw_points and len(points) < 2:
        seeded = []
        for g in gw_points:
            seeded.append(
                {
                    "ts": f"gw-{g.get('gameweek')}",
                    "gameweek": g.get("gameweek"),
                    "overall_points": g.get("total_points"),
                    "overall_rank": g.get("overall_rank"),
                    "event_points": g.get("points"),
                    "plan_objective": None,
                    "xi_ev": None,
                    "bench_ev": None,
                    "squad_value": None,
                    "bank": None,
                    "seeded": True,
                }
            )
        # Keep live snapshots after seeds.
        live_only = [p for p in points if not p.get("seeded")]
        points = (seeded + live_only)[-max_points:]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at": now.isoformat(),
                "points": points,
                "gw_points": gw_points,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Prefer live FPL history; fall back to cached gw_points file.
    if not gw_points:
        gw_points = [g for g in (payload.get("gw_points") or []) if isinstance(g, dict)]

    return {
        "snapshots": points,
        "gw_points": gw_points,
        "season_started": bool(scoreboard.get("season_started")),
    }


def _load_intel_feed() -> dict[str, Any]:
    """Load news/LLM signals + recent scraped headlines for the dashboard."""
    signals_path = Path(SIGNAL_INGEST.get("signals_path") or DEFAULT_SIGNALS_PATH)
    raw_path = Path(SIGNAL_INGEST.get("raw_cache_path") or "data/cache/news_raw.json")
    signals = _read_json(signals_path)
    raw = _read_json(raw_path)

    news_rows = []
    for item in signals.get("news") or []:
        if not isinstance(item, dict):
            continue
        news_rows.append(
            {
                "player_id": item.get("player_id"),
                "web_name": item.get("web_name"),
                "first_name": item.get("first_name"),
                "second_name": item.get("second_name"),
                "team_short": item.get("team_short"),
                "type": item.get("type") or item.get("kind") or "news",
                "severity": item.get("severity"),
                "confidence": item.get("confidence"),
                "exclude": bool(item.get("exclude")),
                "note": item.get("note") or "",
                "evidence": item.get("evidence") or "",
                "source": item.get("source") or "",
                "article_title": item.get("article_title") or "",
                "article_link": item.get("article_link") or "",
                "article_source": item.get("article_source") or "",
                "fpl_status": ((item.get("evidence_flags") or {}) if isinstance(item.get("evidence_flags"), dict) else {}).get(
                    "fpl_status"
                ),
                "fpl_news": ((item.get("evidence_flags") or {}) if isinstance(item.get("evidence_flags"), dict) else {}).get(
                    "fpl_news"
                ),
            }
        )
    # Exclude + severity first, then prefer press for ties, then confidence.
    def _rank(r: dict[str, Any]) -> tuple:
        src = str(r.get("source") or "").lower()
        if src in {"heuristic", "ollama", "auto"}:
            press = 0
        elif src == "fpl":
            press = 1
        else:
            press = 2
        return (
            0 if r.get("exclude") else 1,
            -float(r.get("severity") or 0),
            press,
            -float(r.get("confidence") or 0),
        )

    news_rows.sort(key=_rank)

    lineups = []
    for item in signals.get("predicted_lineups") or []:
        if not isinstance(item, dict):
            continue
        lineups.append(
            {
                "web_name": item.get("web_name"),
                "team_short": item.get("team_short"),
                "role": item.get("role"),
                "gameweek": item.get("gameweek"),
                "confidence": item.get("confidence"),
                "predicted_minutes": item.get("predicted_minutes"),
                "note": item.get("note") or "",
                "source": item.get("source") or "",
            }
        )

    preseason_rows = []
    for item in signals.get("preseason") or []:
        if not isinstance(item, dict):
            continue
        preseason_rows.append(
            {
                "player_id": item.get("player_id"),
                "web_name": item.get("web_name"),
                "role": item.get("role"),
                "minutes": item.get("minutes"),
                "starts": item.get("starts"),
                "boost": item.get("boost"),
                "note": item.get("note") or "",
                "source": item.get("source") or "",
            }
        )
    preseason_rows.sort(key=lambda r: float(r.get("boost") or 0), reverse=True)

    # Raw headlines stay off the dashboard; count only for summary metadata.
    articles_cached = len(raw.get("articles") or []) if isinstance(raw.get("articles"), list) else 0

    ingest = signals.get("ingest") if isinstance(signals.get("ingest"), dict) else {}
    squad_minutes_check = ingest.get("squad_minutes_check")
    if not isinstance(squad_minutes_check, dict):
        squad_minutes_check = {}
    excluded = sum(1 for n in news_rows if n.get("exclude"))
    press_n = sum(1 for n in news_rows if str(n.get("source") or "") in {"heuristic", "ollama", "auto"})
    fpl_n = sum(1 for n in news_rows if str(n.get("source") or "") == "fpl")
    summary = {
        "as_of": signals.get("as_of"),
        "updated_at": signals.get("updated_at"),
        "news": len(news_rows),
        "news_press": press_n,
        "news_fpl": fpl_n,
        "preseason": len(signals.get("preseason") or []),
        "international": len(signals.get("international") or []),
        "predicted_lineups": len(lineups),
        "squad_minutes_covered": squad_minutes_check.get("covered"),
        "squad_minutes_owned": squad_minutes_check.get("owned"),
        "squad_minutes_missing": squad_minutes_check.get("missing_count"),
        "squad_minutes_news": squad_minutes_check.get("news_covered"),
        "squad_minutes_fpl": squad_minutes_check.get("fpl_implied_covered"),
        "squad_minutes_complete": squad_minutes_check.get("complete"),
        "total": len(news_rows)
        + len(signals.get("preseason") or [])
        + len(signals.get("international") or [])
        + len(lineups),
        "excluded": excluded,
        "articles_cached": articles_cached,
        "fetched_at": raw.get("fetched_at"),
        "model": ingest.get("model"),
        "kg_validated": ingest.get("kg_validated"),
        "proposed": ingest.get("proposed"),
        "rejected": ingest.get("rejected"),
        "unresolved": ingest.get("unresolved"),
    }
    return {
        "summary": summary,
        "news": news_rows[:40],
        "news_press": [r for r in news_rows if str(r.get("source") or "") in {"heuristic", "ollama", "auto"}][:20],
        "news_fpl": [r for r in news_rows if str(r.get("source") or "") == "fpl"][:20],
        "preseason": preseason_rows[:20],
        "predicted_lineups": lineups[:20],
        "squad_minutes_check": squad_minutes_check,
        "ingest": ingest,
    }


def _applied_moves_payload(
    state: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Newest-first dated applied transfers for the dashboard."""
    history = state.get("applied_moves")
    if isinstance(history, list) and history:
        return [m for m in history if isinstance(m, dict)]

    # One-time fallback: last execute cycle before history was tracked.
    execution = summary.get("execution") or {}
    if not execution.get("saved"):
        return []
    transfers = ((summary.get("transfer_plan") or {}).get("transfers")) or []
    applied_at = state.get("last_executed_at") or summary.get("generated_at")
    gw = summary.get("gameweek")
    out: list[dict[str, Any]] = []
    for t in transfers:
        if not isinstance(t, dict):
            continue
        row = dict(t)
        row["applied_at"] = applied_at
        row["gameweek"] = gw
        out.append(row)
    return out


def _bot_schedule_payload(
    *,
    hours_to_deadline: float | None,
    unlimited: bool,
    can_write_now: bool,
    near_deadline: bool,
    in_execute_window: bool,
    summary: dict[str, Any],
    state: dict[str, Any],
    run_lock_held: bool,
) -> dict[str, Any]:
    """Compact daemon cadence for the dashboard."""
    from fpl_robo.runtime.cadence import load_ingest_stamp, next_daily_ingest_utc

    heartbeat = int(
        SEASON_RUNNER.get("heartbeat_seconds") or SEASON_RUNNER.get("poll_seconds") or 900
    )
    near_h = float(SEASON_RUNNER["near_deadline_hours"])
    near_poll = int(SEASON_RUNNER["near_deadline_poll_seconds"])
    exec_h = float(SEASON_RUNNER["execute_within_hours"])
    daily_h = int(SIGNAL_INGEST.get("daily_ingest_hour") or 8)
    daily_m = int(SIGNAL_INGEST.get("daily_ingest_minute") or 30)
    pre_deadline_h = float(SIGNAL_INGEST.get("pre_deadline_hours") or 0.5)
    tz_name = str(SIGNAL_INGEST.get("timezone") or "Europe/London")

    hours = float(hours_to_deadline) if hours_to_deadline is not None else None
    if hours is None:
        phase_id = "unknown"
    elif hours <= 0:
        phase_id = "locked"
    elif hours <= exec_h:
        phase_id = "execute"
    elif hours <= near_h:
        phase_id = "near_deadline"
    else:
        phase_id = "cruise"

    current_poll = near_poll if phase_id in {"near_deadline", "execute"} else heartbeat
    cadence = state.get("cadence") if isinstance(state.get("cadence"), dict) else {}
    stamp = load_ingest_stamp()

    jobs = [
        {
            "id": "news",
            "title": "News & lineups",
            "when": (
                f"Daily {daily_h:02d}:{daily_m:02d} {tz_name.split('/')[-1]} · "
                f"again {_fmt_hours_short(pre_deadline_h)} before deadline "
                f"(skip daily if deadline is earlier that morning)"
            ),
            "last_at": stamp.get("ran_at"),
            "last_kind": stamp.get("last_kind"),
            "active": phase_id in {"near_deadline", "execute"},
        },
        {
            "id": "optimise",
            "title": "Re-optimise",
            "when": "Only when data changes (news, prices, status, new GW) · again before lock",
            "last_at": cadence.get("last_optimise_at") or summary.get("generated_at"),
            "last_triggers": cadence.get("last_optimise_triggers") or [],
            "active": bool(run_lock_held),
        },
        {
            "id": "write",
            "title": "Write to FPL",
            "when": (
                f"Final {_fmt_hours_short(exec_h)}"
                + (" · or anytime while unlimited" if unlimited else "")
            ),
            "last_at": state.get("last_executed_at"),
            "active": bool(can_write_now) and phase_id != "locked",
        },
    ]

    phase_label = {
        "cruise": "Watching",
        "near_deadline": "Pre-deadline",
        "execute": "Lock window",
        "locked": "Locked",
        "unknown": "—",
    }.get(phase_id, phase_id)

    next_bits: list[dict[str, Any]] = []
    try:
        nxt = next_daily_ingest_utc()
        next_bits.append(
            {
                "title": "Daily news",
                "at": nxt.isoformat(),
                "in_hours": max(0.0, (nxt - datetime.now(timezone.utc)).total_seconds() / 3600.0),
            }
        )
    except Exception:
        pass
    if hours is not None and hours > pre_deadline_h:
        next_bits.append(
            {
                "title": "Final scrape",
                "in_hours": max(0.0, hours - pre_deadline_h),
            }
        )
    if hours is not None and hours > exec_h:
        next_bits.append(
            {
                "title": "Write window",
                "in_hours": max(0.0, hours - exec_h),
            }
        )

    return {
        "phase": phase_id,
        "phase_label": phase_label,
        "hours_to_deadline": round(hours, 3) if hours is not None else None,
        "current_poll_seconds": current_poll,
        "near_deadline": near_deadline,
        "in_execute_window": in_execute_window,
        "unlimited_transfers": unlimited,
        "can_write_now": can_write_now,
        "run_lock_held": run_lock_held,
        "jobs": jobs,
        "next": next_bits,
        "config": {
            "heartbeat_seconds": heartbeat,
            "near_deadline_hours": near_h,
            "near_deadline_poll_seconds": near_poll,
            "execute_within_hours": exec_h,
            "daily_ingest": f"{daily_h:02d}:{daily_m:02d}",
            "timezone": tz_name,
            "pre_deadline_hours": pre_deadline_h,
        },
        "last": {
            "plan_at": cadence.get("last_optimise_at") or summary.get("generated_at"),
            "ingest_at": stamp.get("ran_at"),
            "ingest_kind": stamp.get("last_kind"),
            "execute_at": state.get("last_executed_at"),
            "optimise_triggers": cadence.get("last_optimise_triggers") or [],
        },
    }


def _fmt_seconds(seconds: int | float) -> str:
    s = int(seconds)
    if s >= 3600 and s % 3600 == 0:
        return f"{s // 3600}h"
    if s >= 60:
        m = s // 60
        rem = s % 60
        return f"{m}m" if rem == 0 else f"{m}m {rem}s"
    return f"{s}s"


def _fmt_hours_short(hours: float) -> str:
    h = float(hours)
    if h >= 24:
        return f"{h / 24:g}d"
    if h >= 1:
        return f"{h:g}h"
    return f"{max(1, int(round(h * 60)))}m"


def _read_tunnel_url() -> str | None:
    return _tunnel_status().get("url")


def _tunnel_status() -> dict[str, Any]:
    path = Path(DASHBOARD.get("tunnel_url_file") or "data/cloudflare_tunnel_url.txt")
    out: dict[str, Any] = {"url": None, "mtime": None, "age_seconds": None}
    if not path.exists():
        return out
    try:
        url = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        out["url"] = url or None
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        out["mtime"] = mtime.isoformat()
        out["age_seconds"] = max(0.0, (datetime.now(timezone.utc) - mtime).total_seconds())
    except Exception:
        return out
    return out


def _mtime(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
