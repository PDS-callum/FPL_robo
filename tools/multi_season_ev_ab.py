"""Multi-season A/B of EV / optimiser stacks.

Runs a focused set of variants across seasons (default: 2023-24, 2024-25, 2025-26)
and writes data/multi_season_ev_ab.json.

Does not change live EV_MODE.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fpl_robo.backtest.season as season_mod
from fpl_robo.backtest.season import run_season_backtest
from fpl_robo.chips.backtest import load_merged_gw
from fpl_robo.config import BLANK_AHEAD
from fpl_robo.data.ep_history import load_ep_history
from fpl_robo.predict.horizon import apply_blank_ahead_ev
from tools.sweep_anti_ep_models import (
    _attach_extra,
    model_ep_capped_by_involvement,
    model_ep_reliability,
    model_ep_shrink_to_form,
)

OUT = ROOT / "data" / "multi_season_ev_ab.json"
SEASONS = ("2023-24", "2024-25", "2025-26")


def _cache(season: str) -> str:
    return str(ROOT / "data" / "cache" / f"merged_gw_{season}.csv")


def _ensure_data(season: str) -> None:
    load_merged_gw(season=season, cache_path=_cache(season))
    try:
        load_ep_history(season=season)
    except Exception as exc:  # noqa: BLE001 — optional repair for older seasons
        print(f"  (ep_history optional fail for {season}: {exc})", flush=True)


def _run_native(
    *,
    season: str,
    label: str,
    ev_mode: str,
    max_hits: int = 1,
    use_tiebreak: bool = True,
    blank_ahead: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"=== {season} :: {label} ===", flush=True)
    result = run_season_backtest(
        season=season,
        cache_path=_cache(season),
        ev_mode=ev_mode,  # type: ignore[arg-type]
        max_hits=max_hits,
        use_chips=True,
        use_tiebreak=use_tiebreak,
        blank_ahead=blank_ahead,
    )
    row = {
        "season": season,
        "label": label,
        "ev_mode": ev_mode,
        "max_hits": max_hits,
        "use_tiebreak": use_tiebreak,
        "blank_ahead": blank_ahead,
        "total_points": result.total_points,
        "points_vs_average": result.points_vs_average,
        "hits_cost": result.hits_cost,
        "transfers_made": result.transfers_made,
        "chips_played": result.chips_played,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(
        f"  → {result.total_points} pts | vs_avg {result.points_vs_average} | "
        f"hits -{result.hits_cost} | xfers {result.transfers_made} | {row['elapsed_s']}s",
        flush=True,
    )
    return row


def _run_custom(
    *,
    season: str,
    label: str,
    builder: Callable[[pd.DataFrame], pd.DataFrame],
    max_hits: int = 1,
    use_tiebreak: bool = True,
    blank_ahead: bool = True,
) -> dict[str, Any]:
    """Monkeypatch prediction builders for one season run (same pattern as anti-EP sweep)."""
    t0 = time.time()
    print(f"=== {season} :: {label} ===", flush=True)

    old_attach = season_mod.attach_lagged_history
    old_build = season_mod.build_predictions
    old_pf = season_mod._prediction_frame

    season_mod.attach_lagged_history = _attach_extra  # type: ignore[assignment]

    def build(gw_df, *, mode="ep_baseline", team_fixtures=None, blank_ahead=None, current_ids=None):  # noqa: ARG001
        rows = builder(gw_df)
        use_blank = bool(BLANK_AHEAD.get("enabled")) if blank_ahead is None else bool(blank_ahead)
        gw = None
        if "gameweek" in rows.columns and len(rows) and not rows["gameweek"].isna().all():
            gw = int(rows["gameweek"].dropna().iloc[0])
        if use_blank and team_fixtures is not None and gw is not None:
            rows = apply_blank_ahead_ev(
                rows,
                start_gw=gw,
                team_fixtures=team_fixtures,
                current_ids=current_ids,
            )
        return rows

    def prediction_frame(
        gw_df,
        *,
        mode="ep_baseline",
        team_fixtures=None,
        blank_ahead=None,
        current_ids=None,
    ):
        return build(
            gw_df,
            mode=mode,
            team_fixtures=team_fixtures,
            blank_ahead=blank_ahead,
            current_ids=current_ids,
        )

    season_mod.build_predictions = build  # type: ignore[assignment]
    season_mod._prediction_frame = prediction_frame  # type: ignore[assignment]

    try:
        result = run_season_backtest(
            season=season,
            cache_path=_cache(season),
            ev_mode="ep_baseline",
            max_hits=max_hits,
            use_chips=True,
            use_tiebreak=use_tiebreak,
            blank_ahead=blank_ahead,
        )
    finally:
        season_mod.attach_lagged_history = old_attach  # type: ignore[assignment]
        season_mod.build_predictions = old_build  # type: ignore[assignment]
        season_mod._prediction_frame = old_pf  # type: ignore[assignment]

    row = {
        "season": season,
        "label": label,
        "ev_mode": label,
        "max_hits": max_hits,
        "use_tiebreak": use_tiebreak,
        "blank_ahead": blank_ahead,
        "total_points": result.total_points,
        "points_vs_average": result.points_vs_average,
        "hits_cost": result.hits_cost,
        "transfers_made": result.transfers_made,
        "chips_played": result.chips_played,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(
        f"  → {result.total_points} pts | vs_avg {result.points_vs_average} | "
        f"hits -{result.hits_cost} | xfers {result.transfers_made} | {row['elapsed_s']}s",
        flush=True,
    )
    return row


def main() -> None:
    rows: list[dict[str, Any]] = []
    for season in SEASONS:
        print(f"\n##### Preparing {season} #####", flush=True)
        _ensure_data(season)

        # A) Current live-ish stack: baseline + blank-ahead + tiebreak
        rows.append(
            _run_native(
                season=season,
                label="baseline_blank_tiebreak",
                ev_mode="ep_baseline",
                max_hits=1,
                use_tiebreak=True,
                blank_ahead=True,
            )
        )
        # B) Same but allow 2-hit weeks
        rows.append(
            _run_native(
                season=season,
                label="baseline_blank_tiebreak_hits2",
                ev_mode="ep_baseline",
                max_hits=2,
                use_tiebreak=True,
                blank_ahead=True,
            )
        )
        # C) Minutes-scaled (tuned divisor) + blank + tiebreak
        rows.append(
            _run_native(
                season=season,
                label="minutes_scaled_blank_tiebreak",
                ev_mode="ep_minutes_scaled",
                max_hits=1,
                use_tiebreak=True,
                blank_ahead=True,
            )
        )
        # D) Shrink EP toward lagged form when they disagree
        rows.append(
            _run_custom(
                season=season,
                label="ep_shrink_to_form",
                builder=model_ep_shrink_to_form,
                max_hits=1,
                use_tiebreak=True,
                blank_ahead=True,
            )
        )
        # E) Reliability gate: trust xP only with minutes evidence
        rows.append(
            _run_custom(
                season=season,
                label="ep_reliability",
                builder=model_ep_reliability,
                max_hits=1,
                use_tiebreak=True,
                blank_ahead=True,
            )
        )
        # F) Cap xP by involvement ceiling
        rows.append(
            _run_custom(
                season=season,
                label="ep_capped_by_involvement",
                builder=model_ep_capped_by_involvement,
                max_hits=1,
                use_tiebreak=True,
                blank_ahead=True,
            )
        )

    # Aggregate: mean pts and win counts vs baseline stack
    by_label: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    baseline_by_season = {
        r["season"]: r["total_points"] for r in rows if r["label"] == "baseline_blank_tiebreak"
    }
    summary = []
    for label, group in by_label.items():
        pts = [r["total_points"] for r in group]
        deltas = [r["total_points"] - baseline_by_season[r["season"]] for r in group]
        summary.append(
            {
                "label": label,
                "seasons": len(group),
                "mean_points": round(sum(pts) / len(pts), 1),
                "mean_delta_vs_baseline_stack": round(sum(deltas) / len(deltas), 1),
                "wins_vs_baseline_stack": sum(1 for d in deltas if d > 0),
                "losses_vs_baseline_stack": sum(1 for d in deltas if d < 0),
                "ties_vs_baseline_stack": sum(1 for d in deltas if d == 0),
                "per_season": {
                    r["season"]: {
                        "pts": r["total_points"],
                        "delta": r["total_points"] - baseline_by_season[r["season"]],
                        "hits_cost": r["hits_cost"],
                        "xfers": r["transfers_made"],
                    }
                    for r in group
                },
            }
        )
    summary.sort(key=lambda s: (-s["mean_delta_vs_baseline_stack"], -s["mean_points"]))

    report = {
        "baseline_label": "baseline_blank_tiebreak",
        "seasons": list(SEASONS),
        "notes": [
            "vs_average uses 2025-26 GW averages cache for all seasons — treat total_points as primary.",
            "FH/WC are tracked when recommended but not fully resimulated as rebuilds.",
            "Live KG enrichment / news signals are not in the season harness.",
        ],
        "summary": summary,
        "results": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== SUMMARY (delta vs baseline_blank_tiebreak) =====")
    print(f"{'label':<36} {'mean':>7} {'Δmean':>7} {'W-L-T':>7}")
    for s in summary:
        wlt = f"{s['wins_vs_baseline_stack']}-{s['losses_vs_baseline_stack']}-{s['ties_vs_baseline_stack']}"
        print(
            f"{s['label']:<36} {s['mean_points']:>7.1f} "
            f"{s['mean_delta_vs_baseline_stack']:>+7.1f} {wlt:>7}"
        )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
