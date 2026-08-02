#!/usr/bin/env python3
"""Sweep VECTOR_TEAM residual_scale on a historical season (vector SA).

Usage:
  .venv/bin/python tools/residual_scale_sweep.py --season 2025-26
  .venv/bin/python tools/residual_scale_sweep.py --season 2025-26 --apply-best
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpl_robo.backtest.vector_season import run_vector_season_backtest
from fpl_robo.config import VECTOR_TEAM


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument(
        "--scales",
        default="0.25,0.35,0.50,0.70,1.0",
        help="Comma-separated residual_scale values",
    )
    ap.add_argument("--max-hits", type=int, default=1)
    ap.add_argument("--sa-iters-initial", type=int, default=1200)
    ap.add_argument("--sa-iters-transfer", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-file", default="data/residual_scale_sweep.json")
    ap.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best residual_scale into fpl_robo/config.py VECTOR_TEAM",
    )
    args = ap.parse_args()
    scales = [float(x.strip()) for x in str(args.scales).split(",") if x.strip()]

    rows = []
    baseline_scale = float(VECTOR_TEAM.get("residual_scale") or 0.35)
    for scale in scales:
        VECTOR_TEAM["residual_scale"] = float(scale)
        t0 = time.time()
        result = run_vector_season_backtest(
            args.season,
            max_hits=int(args.max_hits),
            use_chips=False,
            sa_iters_initial=int(args.sa_iters_initial),
            sa_iters_transfer=int(args.sa_iters_transfer),
            seed=int(args.seed),
        )
        elapsed = round(time.time() - t0, 1)
        row = {
            "residual_scale": scale,
            "total_points": float(result.total_points),
            "hits_cost": int(result.hits_cost),
            "transfers_made": int(result.transfers_made),
            "elapsed_sec": elapsed,
        }
        rows.append(row)
        print(
            f"scale={scale:.2f} pts={row['total_points']:.0f} "
            f"hits={row['hits_cost']} xfers={row['transfers_made']} ({elapsed}s)",
            flush=True,
        )

    VECTOR_TEAM["residual_scale"] = baseline_scale
    rows.sort(key=lambda r: (-r["total_points"], r["residual_scale"]))
    best = rows[0]
    report = {
        "season": args.season,
        "max_hits": args.max_hits,
        "sa_iters_initial": args.sa_iters_initial,
        "sa_iters_transfer": args.sa_iters_transfer,
        "seed": args.seed,
        "baseline_scale": baseline_scale,
        "best_scale": best["residual_scale"],
        "best_points": best["total_points"],
        "rows": rows,
        "notes": [
            "Relative ranking with reduced SA iters; confirm winner with full suite.",
            "Does not change live EV_MODE (still ep_baseline anchor).",
        ],
    }
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best scale={best['residual_scale']} pts={best['total_points']} → {out}")

    if args.apply_best:
        cfg = Path("fpl_robo/config.py")
        text = cfg.read_text(encoding="utf-8")
        old = f'"residual_scale": {baseline_scale}'
        # tolerate 0.35 vs 0.350
        import re

        text2, n = re.subn(
            r'("residual_scale":\s*)([0-9.]+)',
            rf'\g<1>{best["residual_scale"]}',
            text,
            count=1,
        )
        if n != 1:
            print("Failed to patch config residual_scale")
            return 1
        cfg.write_text(text2, encoding="utf-8")
        print(f"Patched VECTOR_TEAM residual_scale → {best['residual_scale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
