"""Hard attack-quality floor for the squad MIP."""

from __future__ import annotations

from typing import Any

import pulp

from fpl_robo.config import ATTACK_FLOOR


def apply_attack_floor_constraint(
    prob: pulp.LpProblem,
    *,
    in_squad: dict[Any, pulp.LpVariable],
    ids: list[Any],
    pos: dict[Any, str],
    ep_next: dict[Any, float],
    cfg: dict[str, Any] | None = None,
) -> int:
    """Require min_quality_fwds with ep_next >= threshold.

    Returns the number of eligible quality FWDs in the pool (0 if disabled /
    infeasible pool — constraint skipped when pool cannot satisfy).
    """
    cfg = {**ATTACK_FLOOR, **(cfg or {})}
    if not bool(cfg.get("enabled", True)):
        return 0
    try:
        need = int(cfg.get("min_quality_fwds") or 0)
    except (TypeError, ValueError):
        need = 0
    if need <= 0:
        return 0
    try:
        min_ep = float(cfg.get("min_ep_next") or 2.1)
    except (TypeError, ValueError):
        min_ep = 2.1

    quality = [
        i
        for i in ids
        if str(pos.get(i) or "").upper() == "FWD"
        and float(ep_next.get(i) or 0.0) >= min_ep
    ]
    if len(quality) < need:
        # Pool too thin (injured premium FWDs) — do not make the MIP infeasible.
        return len(quality)
    prob += pulp.lpSum(in_squad[i] for i in quality) >= need
    return len(quality)
