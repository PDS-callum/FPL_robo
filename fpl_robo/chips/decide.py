"""Recommend whether/when to play chips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fpl_robo.chips.inventory import ChipInventory, half_window
from fpl_robo.chips.scoring import ChipOpportunityScores, score_chip_opportunities
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
) -> ChipRecommendation:
    """Decide chip plays for the current GW using EV opportunity scores.

    Heuristics (deliberately conservative):
    - TC: play if captain EV is strong absolute AND high vs recent weeks
    - BB: play if bench EV is strong absolute AND high vs recent weeks
    - FH: play if one-week rebuild gap exceeds threshold (blank/disaster week)
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
        strong = scores.captain_ev >= float(cfg["tc_min_captain_ev"])
        rare = tc_pct >= float(cfg["tc_min_percentile"]) if tc_hist else False
        late_force = weeks_left <= int(cfg["force_within_weeks"]) and strong
        # Without history, only force near window end or require elite captain EV.
        elite = scores.captain_ev >= float(cfg["tc_min_captain_ev"]) + 2.0
        play = ((strong and rare) or late_force or (elite and not tc_hist))
        play_now["triple_captain"] = play
        reasons["triple_captain"] = (
            f"captain_ev={scores.captain_ev:.2f}, tc_score={scores.triple_captain:.2f}, "
            f"hist_pct={tc_pct:.2f}, weeks_left_in_window={weeks_left}"
            + ("; PLAY" if play else "; HOLD")
        )
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
        play = scores.free_hit >= float(cfg["fh_min_gap"])
        play_now["free_hit"] = play
        reasons["free_hit"] = (
            f"one_week_gap={scores.free_hit:.2f} (need>={cfg['fh_min_gap']})"
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
        },
    )


def _percentile(value: float, history: list[float]) -> float:
    if not history:
        return 1.0
    below = sum(1 for x in history if x <= value)
    return below / len(history)
