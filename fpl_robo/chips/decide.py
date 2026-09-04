"""Recommend whether/when to play chips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.chips.fh_timing import (
    build_fh_timing_outlook,
    owned_team_ids_from_squad,
)
from fpl_robo.chips.inventory import ChipInventory, half_window
from fpl_robo.chips.scoring import ChipOpportunityScores, score_chip_opportunities
from fpl_robo.chips.tc_timing import build_tc_timing_outlook
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
) -> ChipRecommendation:
    """Decide chip plays for the current GW using EV opportunity scores.

    Heuristics (deliberately conservative):
    - TC: play if captain EV is strong absolute AND high vs recent weeks
    - BB: play if bench EV is strong absolute AND high vs recent weeks
    - FH: play if one-week rebuild gap is strong AND this GW ranks near the
      best remaining fixture slate (or owned blanks force the chip)
    - WC: play if rolling multi-GW gap exceeds threshold
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

    # Triple Captain — prefer look-ahead (best owned player × week); fall back
    # to absolute+history only when plan-ahead has no fixture calendar.
    if inv.available("triple_captain", gameweek):
        planned_ev = scores.captain_ev
        if tc_outlook is not None and tc_outlook.this_best is not None:
            planned_ev = float(tc_outlook.this_best.projected_ev)
        strong = planned_ev >= float(cfg["tc_min_captain_ev"])
        late_force = weeks_left <= int(cfg["force_within_weeks"]) and strong
        if tc_outlook is not None:
            timing_ok = bool(tc_outlook.play_now)
            play = bool((strong and timing_ok) or late_force)
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
        strong = scores.bench_boost >= float(cfg["bb_min_bench_ev"])
        rare = bb_pct >= float(cfg["bb_min_percentile"]) if bb_hist else False
        late_force = weeks_left <= int(cfg["force_within_weeks"]) and strong
        elite = scores.bench_boost >= float(cfg["bb_min_bench_ev"]) + 4.0
        play = ((strong and rare) or late_force or (elite and not bb_hist))
        play_now["bench_boost"] = play
        reasons["bench_boost"] = (
            f"bench_ev={scores.bench_boost:.2f}, hist_pct={bb_pct:.2f}, "
            f"weeks_left_in_window={weeks_left}"
            + ("; PLAY" if play else "; HOLD")
        )
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

    # Wildcard
    if inv.available("wildcard", gameweek):
        play = scores.wildcard >= float(cfg["wc_min_rolling_gap"])
        play_now["wildcard"] = play
        reasons["wildcard"] = (
            f"rolling_gap={scores.wildcard:.2f} (need>={cfg['wc_min_rolling_gap']})"
            + ("; PLAY" if play else "; HOLD")
        )
    else:
        reasons["wildcard"] = "unavailable in current chip window"

    # Avoid stacking disruptive chips in one week.
    if play_now["wildcard"] and play_now["free_hit"]:
        play_now["free_hit"] = False
        reasons["free_hit"] += " | suppressed (wildcard preferred same week)"
    if play_now["triple_captain"] and play_now["bench_boost"]:
        # Prefer the larger absolute opportunity.
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
        },
        inventory=inv.summary(),
        details={
            "window": {"start": lo, "end": hi, "weeks_left": weeks_left},
            "captain_id": scores.captain_id,
            "historical_tc_n": len(tc_hist),
            "historical_bb_n": len(bb_hist),
            "fh_timing": None if fh_outlook is None else fh_outlook.to_dict(),
            "tc_timing": None if tc_outlook is None else tc_outlook.to_dict(),
        },
    )


def _percentile(value: float, history: list[float]) -> float:
    if not history:
        return 1.0
    below = sum(1 for x in history if x <= value)
    return below / len(history)
