#!/usr/bin/env python3
"""Preseason A/B: baseline SA vs MD-style energy bursts (no FPL writes).

Compares hold-escape behaviour under unlimited FT using the live squad +
current vector predictions. Live config keeps sa_energy_bursts=False; this
tool only enables bursts inside a temporary patch.

Usage:
  .venv/bin/python tools/sa_burst_preseason_test.py
  .venv/bin/python tools/sa_burst_preseason_test.py --seeds 5 --iters 1800
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _squad_names(plan) -> list[str]:
    return [str(p.get("web_name") or p.get("id")) for p in (plan.solution.squad or [])]


def _xi_names(plan) -> list[str]:
    return [str(p.get("web_name") or p.get("id")) for p in (plan.solution.starting_xi or [])]


def _run_variant(
    *,
    label: str,
    preds,
    current_ids: list[int],
    bank: float,
    selling: dict[int, float],
    free_transfers: int,
    max_hits: int,
    n_iters: int,
    seed: int,
    n_seeds: int,
    bursts: bool,
    burst_cfg: dict[str, Any],
) -> dict[str, Any]:
    from fpl_robo.config import VECTOR_TEAM
    from fpl_robo.optimize.vector_search import anneal_transfers

    overlay = {
        "sa_energy_bursts": bursts,
        "sa_consensus_seeds": int(n_seeds),
        "sa_consensus_min_agree": max(1, (int(n_seeds) + 1) // 2),
        **burst_cfg,
    }
    t0 = time.perf_counter()
    with patch.dict(VECTOR_TEAM, overlay, clear=False):
        plan = anneal_transfers(
            preds,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling,
            free_transfers=free_transfers,
            max_hits=max_hits,
            n_iters=n_iters,
            seed=seed,
        )
    elapsed = time.perf_counter() - t0
    cons = (plan.meta or {}).get("consensus") or {}
    seed_stats = cons.get("seed_stats") or []
    accepted = [s for s in seed_stats if s.get("accepted")]
    rejected = [s for s in seed_stats if not s.get("accepted")]
    raw_best = max((float(s.get("raw_obj") or -1e18) for s in seed_stats), default=None)
    return {
        "label": label,
        "bursts": bursts,
        "objective": float(plan.objective),
        "transfers_made": int(plan.transfers_made),
        "hits": int(plan.hits),
        "captain": plan.solution.captain_name,
        "squad": _squad_names(plan),
        "xi": _xi_names(plan),
        "transfers": plan.transfers,
        "consensus": {
            "mode": cons.get("mode"),
            "votes": cons.get("votes"),
            "candidates": cons.get("candidates"),
            "unique_squads": cons.get("unique_squads"),
        },
        "seed_stats": seed_stats,
        "accepted_seeds": len(accepted),
        "rejected_seeds": len(rejected),
        "reject_reasons": {
            r: sum(1 for s in rejected if s.get("reject") == r)
            for r in sorted({str(s.get("reject")) for s in rejected})
        },
        "raw_best_obj": raw_best,
        "burst_best_improves": sum(
            int((s.get("sa") or {}).get("burst_best_improves") or 0) for s in seed_stats
        ),
        "burst_kicks": sum(int((s.get("sa") or {}).get("burst_kicks") or 0) for s in seed_stats),
        "seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=1800)
    parser.add_argument("--seeds", type=int, default=3, help="consensus SA seeds")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-hits", type=int, default=2)
    parser.add_argument(
        "--ft-mode",
        choices=("live", "unlimited", "capped"),
        default="unlimited",
        help="preseason default=unlimited (FT=15)",
    )
    parser.add_argument("--report-file", default="data/sa_burst_preseason_test.json")
    parser.add_argument("--burst-every", type=int, default=250)
    parser.add_argument("--burst-length", type=int, default=50)
    parser.add_argument("--burst-t0", type=float, default=10.0)
    parser.add_argument("--kick-swaps", type=int, default=3)
    parser.add_argument("--kick-prob", type=float, default=0.45)
    args = parser.parse_args()

    # Reuse readiness live context builder.
    sys.path.insert(0, str(ROOT / "tools"))
    from vector_readiness_tests import _build_live_preds  # type: ignore

    ctx = _build_live_preds()
    preds = ctx["vec_preds"]
    current_ids = list(ctx["current_ids"])
    bank = float(ctx["bank"])
    selling = dict(ctx["selling"] or {})
    live_ft = int(ctx["ft"])
    if args.ft_mode == "unlimited":
        ft = max(live_ft, 15)
    elif args.ft_mode == "capped":
        ft = 1
    else:
        ft = live_ft

    burst_cfg = {
        "sa_burst_every": int(args.burst_every),
        "sa_burst_length": int(args.burst_length),
        "sa_burst_t0": float(args.burst_t0),
        "sa_burst_kick_swaps": int(args.kick_swaps),
        "sa_burst_kick_prob": float(args.kick_prob),
    }

    common = dict(
        preds=preds,
        current_ids=current_ids,
        bank=bank,
        selling=selling,
        free_transfers=ft,
        max_hits=int(args.max_hits),
        n_iters=int(args.iters),
        seed=int(args.base_seed),
        n_seeds=int(args.seeds),
        burst_cfg=burst_cfg,
    )

    print(
        f"Preseason SA burst A/B: gw={ctx['gw']} preseason={ctx['preseason']} "
        f"ft={ft} iters={args.iters} seeds={args.seeds}"
    )
    baseline = _run_variant(label="baseline", bursts=False, **common)
    print(
        f"  baseline: obj={baseline['objective']:.3f} xfers={baseline['transfers_made']} "
        f"accepted={baseline['accepted_seeds']}/{args.seeds} "
        f"mode={baseline['consensus'].get('mode')} ({baseline['seconds']}s)"
    )
    burst = _run_variant(label="energy_bursts", bursts=True, **common)
    print(
        f"  bursts:   obj={burst['objective']:.3f} xfers={burst['transfers_made']} "
        f"accepted={burst['accepted_seeds']}/{args.seeds} "
        f"mode={burst['consensus'].get('mode')} "
        f"kicks={burst['burst_kicks']} improves={burst['burst_best_improves']} "
        f"({burst['seconds']}s)"
    )

    base_set = set(baseline["squad"])
    burst_set = set(burst["squad"])
    delta_obj = float(burst["objective"]) - float(baseline["objective"])
    raw_delta = None
    if baseline.get("raw_best_obj") is not None and burst.get("raw_best_obj") is not None:
        raw_delta = float(burst["raw_best_obj"]) - float(baseline["raw_best_obj"])

    summary = {
        "beneficial_final_obj": delta_obj > 0.05,
        "beneficial_raw_explore": (raw_delta is not None and raw_delta > 0.05),
        "same_squad": base_set == burst_set,
        "delta_objective": round(delta_obj, 4),
        "delta_raw_best_obj": None if raw_delta is None else round(raw_delta, 4),
        "squad_only_in_burst": sorted(burst_set - base_set),
        "squad_only_in_baseline": sorted(base_set - burst_set),
        "recommendation": (
            "enable_for_preseason_trial"
            if delta_obj > 0.05
            or (
                burst["transfers_made"] > baseline["transfers_made"]
                and burst["accepted_seeds"] > baseline["accepted_seeds"]
            )
            else "keep_disabled"
        ),
    }
    print(
        f"  delta_obj={summary['delta_objective']:+.3f} "
        f"raw_delta={summary['delta_raw_best_obj']} "
        f"same_squad={summary['same_squad']} → {summary['recommendation']}"
    )

    report = {
        "gameweek": ctx["gw"],
        "preseason": ctx["preseason"],
        "free_transfers": ft,
        "ft_mode": args.ft_mode,
        "iters": args.iters,
        "consensus_seeds": args.seeds,
        "base_seed": args.base_seed,
        "burst_cfg": burst_cfg,
        "baseline": baseline,
        "bursts": burst,
        "summary": summary,
    }
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
