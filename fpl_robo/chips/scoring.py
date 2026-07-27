"""Score chip opportunities from a predicted squad."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChipOpportunityScores:
    triple_captain: float
    bench_boost: float
    free_hit: float
    wildcard: float
    captain_id: int | None = None
    captain_ev: float = 0.0
    bench_ev: float = 0.0
    xi_ev: float = 0.0
    notes: dict[str, str] | None = None


def score_chip_opportunities(
    *,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    captain_id: int | None,
    optimal_xi_ev: float | None = None,
    current_xi_ev: float | None = None,
    rolling_gap: float = 0.0,
) -> ChipOpportunityScores:
    """EV-based chip opportunity scores for one gameweek.

    - TC extra value ~= 2x captain EV (normal captain already counts 1x in XI)
    - BB value ~= sum of bench EV
    - FH value ~= optimal one-week XI EV - current XI EV
    - WC value ~= rolling multi-GW gap vs optimal (passed in)
    """
    xi_ev = float(sum(float(p.get("ev") or 0.0) for p in starting_xi))
    bench_ev = float(sum(float(p.get("ev") or 0.0) for p in bench))
    captain_ev = 0.0
    if captain_id is not None:
        for p in starting_xi:
            if int(p["id"]) == int(captain_id):
                captain_ev = float(p.get("ev") or 0.0)
                break
    if captain_ev <= 0 and starting_xi:
        captain_ev = max(float(p.get("ev") or 0.0) for p in starting_xi)

    opt = xi_ev if optimal_xi_ev is None else float(optimal_xi_ev)
    cur = xi_ev if current_xi_ev is None else float(current_xi_ev)
    fh = max(0.0, opt - cur)

    return ChipOpportunityScores(
        triple_captain=round(2.0 * captain_ev, 3),
        bench_boost=round(bench_ev, 3),
        free_hit=round(fh, 3),
        wildcard=round(float(rolling_gap), 3),
        captain_id=captain_id,
        captain_ev=round(captain_ev, 3),
        bench_ev=round(bench_ev, 3),
        xi_ev=round(xi_ev, 3),
        notes={
            "tc": "Extra points vs normal captain ≈ 2× captain EV",
            "bb": "Bench Boost value ≈ sum of bench EV",
            "fh": "Free Hit value ≈ optimal XI EV − locked XI EV",
            "wc": "Wildcard value ≈ rolling multi-GW rebuild gap",
        },
    )
