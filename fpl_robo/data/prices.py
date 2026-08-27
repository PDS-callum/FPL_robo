"""Selling-price helpers (FPL my-team vs public fallbacks)."""

from __future__ import annotations

from typing import Any, Mapping


def pounds_from_tenths(raw: Any) -> float | None:
    """Parse FPL API tenths (75 → £7.5); treat missing/non-positive as unknown."""
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val <= 0:
        return None
    return val / 10.0


def pick_selling_price_pounds(pick: Mapping[str, Any]) -> float | None:
    """Best available sell price from a my-team / entry pick row."""
    for key in ("selling_price", "purchase_price"):
        got = pounds_from_tenths(pick.get(key))
        if got is not None:
            return got
    return None


def effective_sell_price(
    pid: int,
    selling_prices: Mapping[int, float] | None,
    *,
    fallback: float,
) -> float:
    """Capital sell price: ignore 0/missing entries and use pool/now_cost fallback."""
    if selling_prices:
        raw = selling_prices.get(int(pid))
        if raw is not None:
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = 0.0
            if val > 0:
                return val
    try:
        fb = float(fallback)
    except (TypeError, ValueError):
        fb = 4.5
    return fb if fb > 0 else 4.5
