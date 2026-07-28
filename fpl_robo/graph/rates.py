"""Gate per-90 attacking / defensive rates until sample size is trustworthy.

FPL publishes expected_goal_involvements_per_90 from season totals. A cameo of a
few minutes after injury can produce absurd rates (e.g. xGI90=3.6 from 2 minutes)
that then dominate EV blends and optimiser tie-breaks.
"""

from __future__ import annotations

from typing import Any

from fpl_robo.config import KG_ENRICH


def season_minutes(player: dict[str, Any] | Any) -> float:
    """Best available season / prior minutes on a player dict or Series-like."""
    if player is None:
        return 0.0
    get = player.get if hasattr(player, "get") else lambda k, d=None: d
    for key in ("minutes", "prior_minutes", "season_minutes"):
        raw = get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return 0.0


def rate_minutes_floor(cfg: dict[str, Any] | None = None) -> float:
    merged = {**KG_ENRICH, **(cfg or {})}
    return float(merged.get("rate_feature_min_minutes") or 180.0)


def rate_sample_ok(player: dict[str, Any] | Any, *, cfg: dict[str, Any] | None = None) -> bool:
    """True when season minutes are large enough to trust per-90 rates."""
    return season_minutes(player) >= rate_minutes_floor(cfg)


def _raw_xgi90(player: dict[str, Any] | Any) -> float | None:
    get = player.get if hasattr(player, "get") else lambda k, d=None: d
    for key in ("xgi90_raw", "xgi90", "expected_goal_involvements_per_90"):
        raw = get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    xg = get("expected_goals_per_90")
    xa = get("expected_assists_per_90")
    try:
        xg_f = float(xg) if xg is not None else 0.0
        xa_f = float(xa) if xa is not None else 0.0
    except (TypeError, ValueError):
        return None
    if xg_f or xa_f:
        return xg_f + xa_f
    return None


def reliable_xgi90(player: dict[str, Any] | Any, *, cfg: dict[str, Any] | None = None) -> float | None:
    """xGI/90 for EV use, or None when the sample is too thin / absurd."""
    merged = {**KG_ENRICH, **(cfg or {})}
    if not rate_sample_ok(player, cfg=merged):
        return None
    raw = _raw_xgi90(player)
    if raw is None or raw <= 0:
        return None
    hard_cap = float(merged.get("xgi90_hard_cap") or 1.25)
    return min(float(raw), hard_cap)


def reliable_cs_per_90(player: dict[str, Any] | Any, *, cfg: dict[str, Any] | None = None) -> float | None:
    """Clean-sheet rate for EV use, gated on minutes."""
    if not rate_sample_ok(player, cfg=cfg):
        return None
    get = player.get if hasattr(player, "get") else lambda k, d=None: d
    for key in ("cs_per_90", "clean_sheets_per_90"):
        raw = get(key)
        if raw is None or raw == "":
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val < 0:
            continue
        return min(val, 1.0)
    return None
