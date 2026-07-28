"""Schedule-aware tweaks for chip play thresholds (DGW / easy FDR / blanks)."""

from __future__ import annotations

from typing import Any

from fpl_robo.config import CHIP_DEFAULTS, HORIZON_EV


def _team_id(player: dict[str, Any]) -> int | None:
    for key in ("team_id", "team"):
        val = player.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


def _gw_slot(
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    team_id: int,
    gameweek: int,
) -> dict[str, float]:
    if not team_fixtures:
        return {"n_fixtures": 1.0, "best_fdr": None}
    return (team_fixtures.get(int(team_id)) or {}).get(int(gameweek)) or {
        "n_fixtures": 0.0,
        "best_fdr": None,
    }


def chip_schedule_adjustments(
    *,
    gameweek: int,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    captain_id: int | None,
    dgw_teams: set[int] | list[int] | None = None,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    blank_owned: int = 0,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return (threshold overrides, details) from this-GW schedule context.

    - TC: ease if captain's club has a DGW, or an easy single fixture (low FDR).
    - BB: ease when enough bench players are on DGW teams.
    - FH: ease when many owned players blank (existing blank relief).
    - WC: unchanged here (multi-GW rebuild gap already carries fixture pressure via EV).
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    dgw = {int(t) for t in (dgw_teams or [])}
    overrides: dict[str, float] = {}
    details: dict[str, Any] = {
        "captain_dgw": False,
        "captain_fdr": None,
        "bench_dgw_count": 0,
        "blank_owned": int(blank_owned),
        "dgw_teams": sorted(dgw),
        "applied": [],
    }

    captain = None
    if captain_id is not None:
        for p in starting_xi:
            try:
                if int(p.get("id")) == int(captain_id):
                    captain = p
                    break
            except (TypeError, ValueError):
                continue

    captain_team = _team_id(captain) if captain else None
    captain_slot = (
        _gw_slot(team_fixtures, captain_team, gameweek)
        if captain_team is not None
        else {"n_fixtures": 1.0, "best_fdr": None}
    )
    captain_n = int(float(captain_slot.get("n_fixtures") or 0))
    if captain_team is not None and (captain_team in dgw or captain_n >= 2):
        details["captain_dgw"] = True
    fdr = captain_slot.get("best_fdr")
    if fdr is not None:
        details["captain_fdr"] = float(fdr)

    tc_base = float(cfg["tc_min_captain_ev"])
    tc_min = tc_base
    if details["captain_dgw"]:
        relief = float(cfg.get("dgw_captain_tc_ev_relief") or 1.0)
        tc_min -= relief
        details["applied"].append(f"tc_captain_dgw(-{relief:g})")
        overrides["tc_min_percentile"] = min(
            float(cfg["tc_min_percentile"]),
            float(cfg.get("dgw_tc_min_percentile") or 0.75),
        )
    easy_fdr_max = float(cfg.get("easy_fdr_max") or 2.0)
    if details["captain_fdr"] is not None and float(details["captain_fdr"]) <= easy_fdr_max:
        relief = float(cfg.get("easy_fdr_tc_relief") or 0.5)
        tc_min -= relief
        details["applied"].append(f"tc_easy_fdr(-{relief:g})")

    # Prefer KG attrs on the captain row when present (matchup / set-piece).
    if captain is not None:
        try:
            set_piece = float(captain.get("set_piece_score") or 0.0)
        except (TypeError, ValueError):
            set_piece = 0.0
        details["captain_set_piece"] = set_piece
        if set_piece >= float(cfg.get("set_piece_tc_min_score") or 0.08):
            relief = float(cfg.get("set_piece_tc_relief") or 0.4)
            tc_min -= relief
            details["applied"].append(f"tc_set_piece(-{relief:g})")
        try:
            matchup = float(captain.get("next_matchup_factor") or 1.0)
        except (TypeError, ValueError):
            matchup = 1.0
        details["captain_matchup"] = matchup
        if matchup >= float(cfg.get("strong_matchup_min") or 1.05):
            relief = float(cfg.get("strong_matchup_tc_relief") or 0.35)
            tc_min -= relief
            details["applied"].append(f"tc_strong_matchup(-{relief:g})")
        # Prefer player next_n_fixtures when team_fixtures missed a DGW.
        try:
            n_player = int(captain.get("next_n_fixtures") or 0)
        except (TypeError, ValueError):
            n_player = 0
        if n_player >= 2 and not details["captain_dgw"]:
            details["captain_dgw"] = True
            relief = float(cfg.get("dgw_captain_tc_ev_relief") or 1.0)
            tc_min -= relief
            details["applied"].append(f"tc_captain_dgw_kg(-{relief:g})")

    if tc_min < tc_base:
        overrides["tc_min_captain_ev"] = max(0.0, round(tc_min, 3))

    bench_dgw = 0
    for p in bench:
        tid = _team_id(p)
        if tid is None:
            continue
        n = int(float(_gw_slot(team_fixtures, tid, gameweek).get("n_fixtures") or 0))
        if tid in dgw or n >= 2:
            bench_dgw += 1
    details["bench_dgw_count"] = bench_dgw
    need = int(cfg.get("dgw_bench_min_players") or 2)
    if bench_dgw >= need:
        relief = float(cfg.get("dgw_bench_bb_ev_relief") or 1.0)
        overrides["bb_min_bench_ev"] = max(
            0.0,
            round(float(cfg["bb_min_bench_ev"]) - relief, 3),
        )
        overrides["bb_min_percentile"] = min(
            float(cfg["bb_min_percentile"]),
            float(cfg.get("dgw_bb_min_percentile") or 0.75),
        )
        details["applied"].append(f"bb_bench_dgw(-{relief:g},n={bench_dgw})")

    blank_threshold = int(
        HORIZON_EV.get("fh_blank_squad_threshold")
        or cfg.get("fh_blank_squad_threshold")
        or 3
    )
    if int(blank_owned) >= blank_threshold:
        relief = float(HORIZON_EV.get("fh_blank_gap_relief") or 3.0)
        overrides["fh_min_gap"] = max(
            0.0,
            round(float(cfg["fh_min_gap"]) - relief, 3),
        )
        details["applied"].append(f"fh_blank_owned(-{relief:g},n={blank_owned})")

    return overrides, details
