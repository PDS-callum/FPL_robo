"""Recommend whether/when to play chips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.chips.bb_timing import build_bb_timing_outlook
from fpl_robo.chips.fh_timing import (
    build_fh_timing_outlook,
    owned_team_ids_from_squad,
)
from fpl_robo.chips.inventory import ChipInventory, half_window
from fpl_robo.chips.scoring import ChipOpportunityScores, score_chip_opportunities
from fpl_robo.chips.tc_timing import build_tc_timing_outlook
from fpl_robo.chips.wc_timing import build_wc_timing_outlook
from fpl_robo.config import CHIP_DEFAULTS


@dataclass
class ChipRecommendation:
    gameweek: int
    play_now: dict[str, bool]
    scores: dict[str, float]
    reasons: dict[str, str]
    thresholds: dict[str, float]
    inventory: dict[str, Any]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recommend_chips(
    *,
    gameweek: int,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    captain_id: int | None,
    inventory: ChipInventory | None = None,
    optimal_xi_ev: float | None = None,
    current_xi_ev: float | None = None,
    rolling_gap: float = 0.0,
    historical_tc_scores: list[float] | None = None,
    historical_bb_scores: list[float] | None = None,
    config: dict[str, Any] | None = None,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    owned_team_ids: list[int] | set[int] | None = None,
    blank_owned: int = 0,
    rebuild_xi: list[dict[str, Any]] | None = None,
) -> ChipRecommendation:
    """Decide chip plays for the current GW using EV + look-ahead planners.

    - TC: best owned player × week
    - BB: best owned-bench fixture week
    - FH: one-week rebuild gap + best remaining soft slate / blanks
    - WC: multi-week fixture-run gap (rebuild vs current), timed to best start week
    """
    cfg = {**CHIP_DEFAULTS, **(config or {})}
    inv = inventory or ChipInventory()
    scores = score_chip_opportunities(
        starting_xi=starting_xi,
        bench=bench,
        captain_id=captain_id,
        optimal_xi_ev=optimal_xi_ev,
        current_xi_ev=current_xi_ev,
        rolling_gap=rolling_gap,
    )

    lo, hi = half_window(gameweek, inv.half_split_gw)
    weeks_left = max(1, hi - gameweek + 1)
    owned_ids = (
        list(owned_team_ids)
        if owned_team_ids is not None
        else owned_team_ids_from_squad(starting_xi, bench)
    )
    fh_outlook = build_fh_timing_outlook(
        gameweek=gameweek,
        team_fixtures=team_fixtures,
        owned_team_ids=owned_ids,
        blank_owned=blank_owned,
        window_end=hi,
        config=cfg,
    )
    tc_outlook = build_tc_timing_outlook(
        gameweek=gameweek,
        starting_xi=starting_xi,
        bench=bench,
        team_fixtures=team_fixtures,
        window_end=hi,
        config=cfg,
    )
    bb_outlook = build_bb_timing_outlook(
        gameweek=gameweek,
        bench=bench,
        team_fixtures=team_fixtures,
        window_end=hi,
        config=cfg,
    )
    wc_outlook = build_wc_timing_outlook(
        gameweek=gameweek,
        current_xi=starting_xi,
        rebuild_xi=rebuild_xi,
        team_fixtures=team_fixtures,
        window_end=hi,
        config=cfg,
    )
    # Prefer multi-week WC gap in reported score when plan-ahead is active.
    if wc_outlook is not None:
        scores.wildcard = round(float(wc_outlook.this_gap), 3)

    tc_hist = historical_tc_scores or []
    bb_hist = historical_bb_scores or []
    tc_pct = _percentile(scores.triple_captain, tc_hist)
    bb_pct = _percentile(scores.bench_boost, bb_hist)

    play_now = {
        "triple_captain": False,
        "bench_boost": False,
        "free_hit": False,
        "wildcard": False,
    }
    reasons: dict[str, str] = {}

    # Triple Captain
    if inv.available("triple_captain", gameweek):
        planned_ev = scores.captain_ev
        if tc_outlook is not None and tc_outlook.this_best is not None:
            planned_ev = float(tc_outlook.this_best.projected_ev)
        strong = planned_ev >= float(cfg["tc_min_captain_ev"])
        late_force = weeks_left <= int(cfg["force_within_weeks"]) and strong
        if tc_outlook is not None:
            play = bool((strong and tc_outlook.play_now) or late_force)
            reasons["triple_captain"] = (
                f"captain_ev={scores.captain_ev:.2f}, planned_ev={planned_ev:.2f}, "
                f"tc_score={scores.triple_captain:.2f}, "
                f"weeks_left_in_window={weeks_left} | {tc_outlook.reason}"
                + ("; PLAY" if play else "; HOLD")
            )
        else:
            rare = tc_pct >= float(cfg["tc_min_percentile"]) if tc_hist else False
            elite = scores.captain_ev >= float(cfg["tc_min_captain_ev"]) + 2.0
            play = ((strong and rare) or late_force or (elite and not tc_hist))
            reasons["triple_captain"] = (
                f"captain_ev={scores.captain_ev:.2f}, tc_score={scores.triple_captain:.2f}, "
                f"hist_pct={tc_pct:.2f}, weeks_left_in_window={weeks_left}"
                + ("; PLAY" if play else "; HOLD")
            )
        play_now["triple_captain"] = play
    else:
        reasons["triple_captain"] = "unavailable in current chip window"

    # Bench Boost
    if inv.available("bench_boost", gameweek):
        planned_bench = (
            float(bb_outlook.this_score)
            if bb_outlook is not None
            else float(scores.bench_boost)
        )
        strong = planned_bench >= float(cfg["bb_min_bench_ev"])
        late_force = weeks_left <= int(cfg["force_within_weeks"]) and strong
        if bb_outlook is not None:
            play = bool((strong and bb_outlook.play_now) or late_force)
            reasons["bench_boost"] = (
                f"bench_ev={scores.bench_boost:.2f}, planned_bench={planned_bench:.2f}, "
                f"weeks_left_in_window={weeks_left} | {bb_outlook.reason}"
                + ("; PLAY" if play else "; HOLD")
            )
        else:
            rare = bb_pct >= float(cfg["bb_min_percentile"]) if bb_hist else False
            elite = scores.bench_boost >= float(cfg["bb_min_bench_ev"]) + 4.0
            play = ((strong and rare) or late_force or (elite and not bb_hist))
            reasons["bench_boost"] = (
                f"bench_ev={scores.bench_boost:.2f}, hist_pct={bb_pct:.2f}, "
                f"weeks_left_in_window={weeks_left}"
                + ("; PLAY" if play else "; HOLD")
            )
        play_now["bench_boost"] = play
    else:
        reasons["bench_boost"] = "unavailable in current chip window"

    # Free Hit
    if inv.available("free_hit", gameweek):
        gap_ok = scores.free_hit >= float(cfg["fh_min_gap"])
        late_force = weeks_left <= int(cfg["force_within_weeks"]) and gap_ok
        timing_ok = True
        timing_note = ""
        if fh_outlook is not None:
            timing_ok = bool(fh_outlook.near_best or fh_outlook.blank_force or late_force)
            timing_note = f" | {fh_outlook.reason}"
        play = bool(gap_ok and timing_ok)
        play_now["free_hit"] = play
        reasons["free_hit"] = (
            f"one_week_gap={scores.free_hit:.2f} (need>={cfg['fh_min_gap']})"
            + timing_note
            + ("; PLAY" if play else "; HOLD")
        )
    else:
        reasons["free_hit"] = "unavailable in current chip window"

    # Wildcard — multi-week rebuild run vs current; fall back to rolling one-week gaps.
    if inv.available("wildcard", gameweek):
        min_gap = float(cfg.get("wc_min_horizon_gap") or cfg["wc_min_rolling_gap"])
        late_force = weeks_left <= int(cfg["force_within_weeks"])
        if wc_outlook is not None:
            gap_ok = float(wc_outlook.this_gap) >= min_gap
            play = bool((gap_ok and wc_outlook.play_now) or (late_force and gap_ok))
            reasons["wildcard"] = (
                f"horizon_gap={wc_outlook.this_gap:.2f} (need>={min_gap:g}) "
                f"over {wc_outlook.horizon}gw | {wc_outlook.reason}"
                + ("; PLAY" if play else "; HOLD")
            )
        else:
            play = scores.wildcard >= float(cfg["wc_min_rolling_gap"])
            reasons["wildcard"] = (
                f"rolling_gap={scores.wildcard:.2f} (need>={cfg['wc_min_rolling_gap']})"
                + ("; PLAY" if play else "; HOLD")
            )
        play_now["wildcard"] = play
    else:
        reasons["wildcard"] = "unavailable in current chip window"

    # Avoid stacking disruptive chips in one week.
    if play_now["wildcard"] and play_now["free_hit"]:
        play_now["free_hit"] = False
        reasons["free_hit"] += " | suppressed (wildcard preferred same week)"
    if play_now["triple_captain"] and play_now["bench_boost"]:
        if scores.triple_captain >= scores.bench_boost:
            play_now["bench_boost"] = False
            reasons["bench_boost"] += " | suppressed (TC preferred same week)"
        else:
            play_now["triple_captain"] = False
            reasons["triple_captain"] += " | suppressed (BB preferred same week)"

    return ChipRecommendation(
        gameweek=gameweek,
        play_now=play_now,
        scores={
            "triple_captain": scores.triple_captain,
            "bench_boost": scores.bench_boost,
            "free_hit": scores.free_hit,
            "wildcard": scores.wildcard,
            "captain_ev": scores.captain_ev,
            "bench_ev": scores.bench_ev,
            "xi_ev": scores.xi_ev,
        },
        reasons=reasons,
        thresholds={
            "tc_min_captain_ev": float(cfg["tc_min_captain_ev"]),
            "tc_min_percentile": float(cfg["tc_min_percentile"]),
            "bb_min_bench_ev": float(cfg["bb_min_bench_ev"]),
            "bb_min_percentile": float(cfg["bb_min_percentile"]),
            "fh_min_gap": float(cfg["fh_min_gap"]),
            "wc_min_rolling_gap": float(cfg["wc_min_rolling_gap"]),
            "wc_min_horizon_gap": float(
                cfg.get("wc_min_horizon_gap") or cfg["wc_min_rolling_gap"]
            ),
        },
        inventory=inv.summary(),
        details={
            "window": {"start": lo, "end": hi, "weeks_left": weeks_left},
            "captain_id": scores.captain_id,
            "historical_tc_n": len(tc_hist),
            "historical_bb_n": len(bb_hist),
            "fh_timing": None if fh_outlook is None else fh_outlook.to_dict(),
            "tc_timing": None if tc_outlook is None else tc_outlook.to_dict(),
            "bb_timing": None if bb_outlook is None else bb_outlook.to_dict(),
            "wc_timing": None if wc_outlook is None else wc_outlook.to_dict(),
        },
    )


def _percentile(value: float, history: list[float]) -> float:
    if not history:
        return 1.0
    below = sum(1 for x in history if x <= value)
    return below / len(history)
