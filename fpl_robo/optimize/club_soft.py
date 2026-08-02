"""Soft club concentration terms for the MIP secondary objective."""

from __future__ import annotations

from typing import Any

import pulp


def add_club_concentration_penalty(
    prob: pulp.LpProblem,
    *,
    in_squad: dict[Any, pulp.LpVariable],
    ids: list[Any],
    team: dict[Any, Any],
    cfg: dict[str, Any],
    prefix: str = "club",
) -> Any:
    """Return secondary expression penalising 2-of and 3-of club stacks.

    Hard MAX_PER_CLUB still applies separately. Returns 0 if penalties disabled.

    Uses binary indicators (not unbounded excesses) so a legal 3-of-club squad
    stays feasible and only pays the soft secondary penalty.
    """
    try:
        triple_pen = float(cfg.get("club_triple_penalty") or 0.0)
    except (TypeError, ValueError):
        triple_pen = 0.0
    try:
        double_pen = float(cfg.get("club_double_penalty") or 0.0)
    except (TypeError, ValueError):
        double_pen = 0.0
    if triple_pen <= 0 and double_pen <= 0:
        return 0

    term = 0
    clubs = sorted({int(team[i]) for i in ids})
    for club in clubs:
        club_players = [i for i in ids if int(team[i]) == club]
        if len(club_players) < 2:
            continue
        count = pulp.lpSum(in_squad[i] for i in club_players)
        # has2 = 1 iff count >= 2 (valid for count in 0..3 with MAX_PER_CLUB=3)
        if double_pen > 0:
            has2 = pulp.LpVariable(f"{prefix}_ge2_{club}", cat="Binary")
            prob += count <= 1 + 2 * has2
            prob += count >= 2 * has2
            term = term - double_pen * has2
        # has3 = 1 iff count >= 3
        if triple_pen > 0 and len(club_players) >= 3:
            has3 = pulp.LpVariable(f"{prefix}_ge3_{club}", cat="Binary")
            prob += count <= 2 + has3
            prob += count >= 3 * has3
            term = term - triple_pen * has3
    return term
