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
    status: dict[Any, str] | None = None,
) -> int:
    """Require min_quality_fwds with ep_next >= threshold.

    When the eligible pool is thinner than `min_quality_fwds` (injuries / bans),
    require as many quality FWDs as remain available instead of dropping the
    constraint entirely.

    Returns the effective required count (0 if disabled).
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

    quality = []
    for i in ids:
        if str(pos.get(i) or "").upper() != "FWD":
            continue
        if float(ep_next.get(i) or 0.0) < min_ep:
            continue
        st = str((status or {}).get(i) or "a").lower()
        if st in {"u", "s"}:  # unavailable / suspended — not selectable quality
            continue
        quality.append(i)

    if not quality:
        return 0
    if len(quality) >= need:
        req = need
    elif bool(cfg.get("soft_when_pool_thin", True)):
        req = len(quality)
    else:
        return len(quality)

    prob += pulp.lpSum(in_squad[i] for i in quality) >= req
    return req
