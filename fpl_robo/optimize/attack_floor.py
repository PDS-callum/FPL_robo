"""Hard attack-quality floor for the squad MIP."""

from __future__ import annotations

from typing import Any, Mapping

import pulp

from fpl_robo.config import ATTACK_FLOOR
from fpl_robo.optimize.attack_quality import quality_fwd_ids


def apply_attack_floor_constraint(
    prob: pulp.LpProblem,
    *,
    in_squad: dict[Any, pulp.LpVariable],
    ids: list[Any],
    pos: dict[Any, str],
    ep_next: dict[Any, float],
    cfg: dict[str, Any] | None = None,
    status: dict[Any, str] | None = None,
    is_focal: Mapping[Any, bool] | None = None,
    form: Mapping[Any, float] | None = None,
    rolling_minutes: Mapping[Any, float] | None = None,
) -> int:
    """Require min_quality_fwds meeting the shared quality predicate.

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

    quality = quality_fwd_ids(
        ids,
        pos=pos,
        ep_next=ep_next,
        status=status,
        is_focal=is_focal,
        form=form,
        rolling_minutes=rolling_minutes,
        cfg=cfg,
    )

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
