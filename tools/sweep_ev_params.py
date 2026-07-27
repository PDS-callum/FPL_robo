"""Sweep BUY_GATES / open-EV formula knobs on 2025-26 season backtest.

Does not change live EV_MODE. Writes data/ev_param_sweep.json.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fpl_robo.config as config
import fpl_robo.predict.ev_modes as ev_modes
from fpl_robo.backtest.season import run_season_backtest
from fpl_robo.predict.ev_modes import select_winner


@contextmanager
def patched_gates(gates: dict[str, Any]):
    old = copy.deepcopy(config.BUY_GATES)
    config.BUY_GATES.clear()
    config.BUY_GATES.update(gates)
    try:
        yield
    finally:
        config.BUY_GATES.clear()
        config.BUY_GATES.update(old)


@contextmanager
def patched_open_formula(**kwargs: float):
    """Temporarily replace _open_ev_from_row with a parameterized version."""
    involv_floor = kwargs.get("involv_floor", 0.35)
    involv_span = kwargs.get("involv_span", 0.80)
    price_coef = kwargs.get("price_coef", 0.15)
    xg_coef = kwargs.get("xg_coef", 2.5)
    xa_coef = kwargs.get("xa_coef", 1.5)
    ict_coef = kwargs.get("ict_coef", 0.02)
    cold_price_coef = kwargs.get("cold_price_coef", 0.05)
    cold_cap = kwargs.get("cold_cap", 1.0)
    pts_weight = kwargs.get("pts_weight", 1.0)

    old = ev_modes._open_ev_from_row

    def _formula(rows):
        import pandas as pd

        pts = rows["rolling_points"].astype(float) if "rolling_points" in rows.columns else 0.0
        mins = rows["rolling_minutes"].astype(float) if "rolling_minutes" in rows.columns else 0.0
        price = rows["price"].astype(float)
        involv = (involv_floor + (mins / 90.0).clip(lower=0.0, upper=1.0) * involv_span).astype(float)
        price_prior = (price_coef * (price - 4.5).clip(lower=0.0, upper=8.0)).astype(float)
        xg = rows["rolling_expected_goals"].astype(float) if "rolling_expected_goals" in rows.columns else 0.0
        xa = (
            rows["rolling_expected_assists"].astype(float)
            if "rolling_expected_assists" in rows.columns
            else 0.0
        )
        ict = rows["rolling_ict_index"].astype(float) if "rolling_ict_index" in rows.columns else 0.0
        open_bonus = xg_coef * xg + xa_coef * xa + ict_coef * ict
        ev = (pts_weight * pts * involv + price_prior + open_bonus).clip(lower=0.0)
        cold = ((pts == 0) & (mins == 0)) if hasattr(pts, "__eq__") else False
        if hasattr(cold, "astype"):
            ev = ev.where(~cold, (cold_price_coef * price).clip(lower=0.05, upper=cold_cap))
        return ev.astype(float)

    ev_modes._open_ev_from_row = _formula  # type: ignore[assignment]
    try:
        yield
    finally:
        ev_modes._open_ev_from_row = old  # type: ignore[assignment]


def run_one(
    *,
    label: str,
    mode: str,
    gates: dict[str, Any] | None = None,
    open_kwargs: dict[str, float] | None = None,
    window: int = 4,
    max_hits: int = 1,
) -> dict[str, Any]:
    import fpl_robo.backtest.season as season_mod

    gates = gates or copy.deepcopy(config.BUY_GATES)
    print(f"--- {label} ({mode}) ---", flush=True)

    old_attach = ev_modes.attach_lagged_history
    old_season_attach = season_mod.attach_lagged_history

    def attach_with_window(table, *, window=window):  # noqa: B023
        return old_attach(table, window=window)

    if window != 4:
        ev_modes.attach_lagged_history = attach_with_window  # type: ignore[assignment]
        season_mod.attach_lagged_history = attach_with_window  # type: ignore[assignment]

    try:
        with patched_gates(gates):
            if open_kwargs:
                with patched_open_formula(**open_kwargs):
                    result = run_season_backtest(ev_mode=mode, max_hits=max_hits)  # type: ignore[arg-type]
            else:
                result = run_season_backtest(ev_mode=mode, max_hits=max_hits)  # type: ignore[arg-type]
    finally:
        ev_modes.attach_lagged_history = old_attach  # type: ignore[assignment]
        season_mod.attach_lagged_history = old_season_attach  # type: ignore[assignment]

    row = {
        "label": label,
        "ev_mode": mode,
        "gates": gates,
        "open_kwargs": open_kwargs or {},
        "window": window,
        "max_hits": max_hits,
        "total_points": result.total_points,
        "points_vs_average": result.points_vs_average,
        "hits_cost": result.hits_cost,
        "transfers_made": result.transfers_made,
    }
    print(
        f"  → {result.total_points} pts | vs avg {result.points_vs_average} | "
        f"hits -{result.hits_cost} | xfers {result.transfers_made}",
        flush=True,
    )
    return row


def main() -> None:
    default_gates = copy.deepcopy(config.BUY_GATES)
    rows: list[dict[str, Any]] = []

    # Always include baseline control.
    rows.append(run_one(label="baseline_control", mode="ep_baseline", gates=default_gates))

    hardened_variants = [
        ("hardened_default", default_gates),
        (
            "hardened_soft",
            {"min_buy_ev": 1.5, "min_prior_minutes": 90, "min_rolling_minutes": 60, "require_minutes_or_rolling": True},
        ),
        (
            "hardened_medium_soft",
            {"min_buy_ev": 1.8, "min_prior_minutes": 120, "min_rolling_minutes": 90, "require_minutes_or_rolling": True},
        ),
        (
            "hardened_ev_only_2p0",
            {"min_buy_ev": 2.0, "min_prior_minutes": 0, "min_rolling_minutes": 0, "require_minutes_or_rolling": True},
        ),
        (
            "hardened_mins_only",
            {"min_buy_ev": 0.0, "min_prior_minutes": 180, "min_rolling_minutes": 120, "require_minutes_or_rolling": True},
        ),
        (
            "hardened_strict",
            {"min_buy_ev": 2.5, "min_prior_minutes": 270, "min_rolling_minutes": 180, "require_minutes_or_rolling": True},
        ),
        (
            "hardened_and_gate",
            {"min_buy_ev": 2.0, "min_prior_minutes": 90, "min_rolling_minutes": 60, "require_minutes_or_rolling": False},
        ),
        (
            "hardened_high_ev_soft_mins",
            {"min_buy_ev": 2.5, "min_prior_minutes": 90, "min_rolling_minutes": 60, "require_minutes_or_rolling": True},
        ),
    ]
    for label, gates in hardened_variants:
        rows.append(run_one(label=label, mode="ep_hardened", gates=gates))

    open_variants: list[tuple[str, dict[str, Any], dict[str, float] | None, int, int]] = [
        ("open_default", default_gates, None, 4, 1),
        (
            "open_soft_gates",
            {"min_buy_ev": 1.5, "min_prior_minutes": 90, "min_rolling_minutes": 60, "require_minutes_or_rolling": True},
            None,
            4,
            1,
        ),
        (
            "open_no_buy_ev_floor",
            {"min_buy_ev": 0.0, "min_prior_minutes": 90, "min_rolling_minutes": 60, "require_minutes_or_rolling": True},
            None,
            4,
            1,
        ),
        (
            "open_points_dominant",
            default_gates,
            {"involv_floor": 0.7, "involv_span": 0.4, "price_coef": 0.05, "pts_weight": 1.15, "cold_cap": 1.5},
            4,
            1,
        ),
        (
            "open_strong_price",
            default_gates,
            {"price_coef": 0.35, "involv_floor": 0.4, "pts_weight": 1.0, "cold_cap": 2.0, "cold_price_coef": 0.12},
            4,
            1,
        ),
        (
            "open_window6",
            default_gates,
            None,
            6,
            1,
        ),
        (
            "open_window3",
            default_gates,
            None,
            3,
            1,
        ),
        (
            "open_soft_no_hits",
            {"min_buy_ev": 1.5, "min_prior_minutes": 90, "min_rolling_minutes": 60, "require_minutes_or_rolling": True},
            {"involv_floor": 0.55, "price_coef": 0.1, "pts_weight": 1.1, "cold_cap": 1.5},
            4,
            0,
        ),
        (
            "open_xg_heavy",
            {
                "min_buy_ev": 1.5,
                "min_prior_minutes": 90,
                "min_rolling_minutes": 60,
                "require_minutes_or_rolling": True,
            },
            {"xg_coef": 4.0, "xa_coef": 2.5, "ict_coef": 0.05, "price_coef": 0.1, "cold_cap": 1.5},
            4,
            1,
        ),
    ]
    for label, gates, open_kw, window, max_hits in open_variants:
        rows.append(
            run_one(
                label=label,
                mode="open_ev",
                gates=gates,
                open_kwargs=open_kw,
                window=window,
                max_hits=max_hits,
            )
        )

    # Rank by same fixed rule as the main comparison.
    ranked = sorted(
        rows,
        key=lambda r: (
            float(r["total_points"]),
            float(r["points_vs_average"] or -1e18),
            -float(r["hits_cost"]),
        ),
        reverse=True,
    )
    winner = select_winner(rows)
    baseline_pts = next(r["total_points"] for r in rows if r["label"] == "baseline_control")

    report = {
        "season": "2025-26",
        "baseline_points": baseline_pts,
        "winner_label": winner["label"],
        "winner": winner,
        "ranked": [
            {
                "label": r["label"],
                "ev_mode": r["ev_mode"],
                "total_points": r["total_points"],
                "delta_vs_baseline": r["total_points"] - baseline_pts,
                "points_vs_average": r["points_vs_average"],
                "hits_cost": r["hits_cost"],
                "transfers_made": r["transfers_made"],
            }
            for r in ranked
        ],
        "results": rows,
    }
    out = Path("data/ev_param_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RANKING (pts / vs_avg / hits) ===")
    print(f"{'label':<28} {'mode':<12} {'pts':>6} {'Δbase':>7} {'vs_avg':>8} {'hits':>5}")
    for r in ranked:
        print(
            f"{r['label']:<28} {r['ev_mode']:<12} {r['total_points']:>6} "
            f"{r['total_points'] - baseline_pts:>+7} {str(r['points_vs_average']):>8} {r['hits_cost']:>5}"
        )
    print(f"\nBest overall: {winner['label']} ({winner['total_points']} pts)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
