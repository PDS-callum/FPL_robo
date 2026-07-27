"""Sweep HORIZON_EV knobs on 2025-26 season backtest (ep_minutes_scaled_h).

Writes data/horizon_param_sweep.json. Does not change live EV_MODE unless a
variant clearly beats the no-horizon baseline.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fpl_robo.config as config
from fpl_robo.backtest.season import run_season_backtest


@contextmanager
def patched_horizon(overrides: dict[str, Any]):
    old = copy.deepcopy(config.HORIZON_EV)
    config.HORIZON_EV.update(overrides)
    try:
        yield
    finally:
        config.HORIZON_EV.clear()
        config.HORIZON_EV.update(old)


def run_one(*, label: str, mode: str, horizon: dict[str, Any] | None = None) -> dict[str, Any]:
    if horizon is None:
        result = run_season_backtest(season="2025-26", max_hits=1, use_chips=True, ev_mode=mode)
    else:
        with patched_horizon(horizon):
            result = run_season_backtest(
                season="2025-26", max_hits=1, use_chips=True, ev_mode=mode
            )
    chips = result.chips_played or {}
    return {
        "label": label,
        "ev_mode": mode,
        "horizon": horizon,
        "total_points": result.total_points,
        "points_vs_average": result.points_vs_average,
        "hits_cost": result.hits_cost,
        "transfers_made": result.transfers_made,
        "chips_played": {k: list(v) for k, v in chips.items()},
        "chip_count": sum(len(v) for v in chips.values()),
        "summary": result.summary,
    }


def main() -> None:
    variants: list[tuple[str, str, dict[str, Any] | None]] = [
        ("baseline_no_horizon", "ep_minutes_scaled", None),
        (
            "h_asym_default",
            "ep_minutes_scaled_h",
            {
                "weights": [0.70, 0.20, 0.10],
                "blank_mult": 0.20,
                "dgw_mult": 1.45,
                "fdr_weight": 0.06,
                "unknown_mult": 1.0,
                "asymmetric_lookahead": True,
            },
        ),
        (
            "h_asym_strong_dgw",
            "ep_minutes_scaled_h",
            {
                "weights": [0.60, 0.25, 0.15],
                "blank_mult": 0.15,
                "dgw_mult": 1.70,
                "fdr_weight": 0.05,
                "unknown_mult": 1.0,
                "asymmetric_lookahead": True,
            },
        ),
        (
            "h_asym_this_week_blank",
            "ep_minutes_scaled_h",
            {
                "weights": [0.85, 0.10, 0.05],
                "blank_mult": 0.10,
                "dgw_mult": 1.55,
                "fdr_weight": 0.04,
                "unknown_mult": 1.0,
                "asymmetric_lookahead": True,
            },
        ),
        (
            "h_asym_pull_hard",
            "ep_minutes_scaled_h",
            {
                "weights": [0.55, 0.30, 0.15],
                "blank_mult": 0.25,
                "dgw_mult": 1.85,
                "fdr_weight": 0.05,
                "unknown_mult": 1.0,
                "asymmetric_lookahead": True,
            },
        ),
        (
            "h_sym_control",
            "ep_minutes_scaled_h",
            {
                "weights": [0.70, 0.20, 0.10],
                "blank_mult": 0.20,
                "dgw_mult": 1.45,
                "fdr_weight": 0.06,
                "unknown_mult": 1.0,
                "asymmetric_lookahead": False,
            },
        ),
        (
            "h_gw0_only_dgw_blank",
            "ep_minutes_scaled_h",
            {
                "weights": [1.0],
                "blank_mult": 0.15,
                "dgw_mult": 1.65,
                "fdr_weight": 0.05,
                "unknown_mult": 1.0,
                "asymmetric_lookahead": True,
            },
        ),
    ]

    rows: list[dict[str, Any]] = []
    for label, mode, horizon in variants:
        print(f"=== {label} ({mode}) ===")
        row = run_one(label=label, mode=mode, horizon=horizon)
        rows.append(row)
        print(
            f"{label}: {row['total_points']} pts | vs avg {row['points_vs_average']} | "
            f"hits -{row['hits_cost']} | chips {row['chip_count']} | xfers {row['transfers_made']}"
        )
        print(f"  chips: {row['chips_played']}")

    ranked = sorted(
        rows,
        key=lambda r: (
            float(r["total_points"]),
            float(r.get("points_vs_average") or -1e18),
            -float(r.get("hits_cost") or 0),
        ),
        reverse=True,
    )
    winner = ranked[0]
    baseline = next(r for r in rows if r["label"] == "baseline_no_horizon")
    beat = (
        winner["label"] != "baseline_no_horizon"
        and float(winner["total_points"]) > float(baseline["total_points"])
    )
    report = {
        "season": "2025-26",
        "baseline_points": baseline["total_points"],
        "winner": winner["label"],
        "winner_points": winner["total_points"],
        "winner_horizon": winner.get("horizon"),
        "delta_vs_baseline": float(winner["total_points"]) - float(baseline["total_points"]),
        "results": rows,
        "recommendation": (
            f"Adopt {winner['label']}: set EV_MODE=ep_minutes_scaled_h and HORIZON_EV overrides"
            if beat
            else "Keep EV_MODE=ep_minutes_scaled (no horizon config beat baseline)"
        ),
        "adopt": beat,
    }
    out = Path("data/horizon_param_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Winner: {winner['label']} ({winner['total_points']} pts)")
    print(f"Delta vs baseline: {report['delta_vs_baseline']:+.0f}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
