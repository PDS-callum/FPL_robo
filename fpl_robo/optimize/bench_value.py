"""Penalise parked premium value on the outfield bench."""

from __future__ import annotations

from typing import Any, Mapping

from fpl_robo.config import BENCH_COVER


def expensive_bench_settings(config: Mapping[str, Any] | None = None) -> tuple[float, float]:
    """Return (penalty_per_pound, free_threshold)."""
    cfg = {**BENCH_COVER, **(config or {})}
    pen = float(cfg.get("expensive_bench_penalty") or 0.0)
    free = float(cfg.get("expensive_bench_free") or 4.5)
    return max(0.0, pen), max(0.0, free)


def outfield_bench_price_excess(price: float, *, free: float) -> float:
    """£ above the free cheap-bench threshold (0 for fodder)."""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, p - float(free))


def is_outfield_position(position: str | None) -> bool:
    return str(position or "").upper() not in {"GK", "GKP"}


def expensive_bench_penalty_total(
    *,
    bench_rows: list[tuple[str | None, float]],
    config: Mapping[str, Any] | None = None,
) -> float:
    """EV-unit penalty for outfield bench prices above the free threshold.

    `bench_rows` is a list of (position, price) for bench players.
    """
    pen, free = expensive_bench_settings(config)
    if pen <= 0:
        return 0.0
    excess = 0.0
    for pos, price in bench_rows:
        if not is_outfield_position(pos):
            continue
        excess += outfield_bench_price_excess(price, free=free)
    return pen * excess
