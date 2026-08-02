#!/usr/bin/env python3
"""Pre-merge readiness checks for vector-team vs live MIP architecture.

Runs (no FPL writes):
  1. Live dry-run parity (MIP run-week vs vector SA)
  2. Stability / thrash (repeated SA with same + varied seeds)
  3. Unit regression guards (+ vector unit tests)
  4. Chip fidelity notes from last eval suite / live chip recs
  5. Preseason unlimited FT vs capped FT=1 plans
  6. Latency timings

Usage:
  .venv/bin/python tools/vector_readiness_tests.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "vector_readiness_report.json"


def _ok(name: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "pass", **(detail or {})}


def _fail(name: str, reason: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "fail", "reason": reason, **(detail or {})}


def _warn(name: str, reason: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "warn", "reason": reason, **(detail or {})}


def test_units() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    names = [
        "tests.test_vector_team",
        "tests.test_ops_hardening",
        "tests.test_stability_guards",
        "tests.test_preseason_churn",
        "tests.test_kg_enrich",
        "tests.test_lockfile",
        "tests.test_season_hardening",
    ]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for n in names:
        suite.addTests(loader.loadTestsFromName(n))
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    detail = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }
    if not result.wasSuccessful():
        return _fail("unit_regression", "unit suite failed", detail)
    return _ok("unit_regression", detail)


_LIVE_CTX: dict[str, Any] | None = None


def _build_live_preds():
    """Shared live KG + predictions for MIP and vector (cached per process)."""
    global _LIVE_CTX
    if _LIVE_CTX is not None:
        return _LIVE_CTX
    from fpl_robo.auth import ensure_authenticated, load_auth_config
    from fpl_robo.config import (
        BENCH_COVER,
        BLANK_AHEAD,
        EV_MODE,
        HORIZON_EV,
        SEASON_RUNNER,
        VECTOR_TEAM,
    )
    from fpl_robo.data.fpl_client import FPLClient
    from fpl_robo.data.live_team import load_live_team_state
    from fpl_robo.graph.builder import build_knowledge_graph
    from fpl_robo.graph.enrich import enrich_knowledge_graph
    from fpl_robo.optimize.kg_features import enrich_predictions_from_kg
    from fpl_robo.predict.ev_modes import make_live_ev_model, transfer_gate_kwargs
    from fpl_robo.predict.horizon import (
        apply_blank_ahead_ev,
        apply_horizon_ev,
        mode_uses_horizon,
        team_fixtures_from_kg,
    )
    from fpl_robo.predict.phase import phase_kg_enrich
    from fpl_robo.predict.vector_features import (
        attach_vector_ev,
        build_feature_matrix,
        hand_model_from_pool,
    )
    from fpl_robo.runtime.deadline import is_preseason

    client = FPLClient()
    cfg = load_auth_config()
    ensure_authenticated(client, cfg, load_session=True, save_session=False)
    live = load_live_team_state(client=client)
    gw = int(live.manager.gameweek or 1)
    bootstrap = client.bootstrap_static()
    kg = build_knowledge_graph(client=client)
    enrich_knowledge_graph(kg, gameweek=gw)

    preseason = is_preseason(bootstrap)
    model = make_live_ev_model(EV_MODE)
    with phase_kg_enrich(preseason=preseason):
        mip_preds = model.predict(kg, gameweek=gw)
    mip_preds = enrich_predictions_from_kg(mip_preds, kg)

    weights = list(HORIZON_EV.get("weights") or [])
    ba_look = int(BLANK_AHEAD.get("lookahead") or 0) if BLANK_AHEAD.get("enabled") else 0
    fixtures = team_fixtures_from_kg(
        kg, start_gw=gw, horizon=max(1, len(weights), ba_look + 1, int(VECTOR_TEAM["horizon"]))
    )
    current_ids = [int(p["element"]) for p in live.picks if p.get("element") is not None]
    if mode_uses_horizon(EV_MODE):
        mip_preds = apply_horizon_ev(mip_preds, start_gw=gw, team_fixtures=fixtures)
    if BLANK_AHEAD.get("enabled"):
        mip_preds = apply_blank_ahead_ev(
            mip_preds, start_gw=gw, team_fixtures=fixtures, current_ids=current_ids
        )

    vp = mip_preds.copy()
    if "gameweek" not in vp.columns:
        vp["gameweek"] = gw
    if "rolling_expected_goals" not in vp.columns and "rolling_xg" in vp.columns:
        vp["rolling_expected_goals"] = vp["rolling_xg"]
    if "rolling_expected_assists" not in vp.columns and "rolling_xa" in vp.columns:
        vp["rolling_expected_assists"] = vp["rolling_xa"]
    if "rolling_ict_index" not in vp.columns and "rolling_ict" in vp.columns:
        vp["rolling_ict_index"] = vp["rolling_ict"]
    if "prior_minutes" not in vp.columns:
        vp["prior_minutes"] = vp["minutes"] if "minutes" in vp.columns else 0.0
    if "total_points" not in vp.columns:
        vp["total_points"] = 0.0
    X, _ = build_feature_matrix(vp, team_fixtures=fixtures)
    vmodel = hand_model_from_pool(X)
    vec_preds = attach_vector_ev(
        vp, vmodel, team_fixtures=fixtures, xp_anchor=vp["ev"] if "ev" in vp.columns else None
    )

    max_hits = int(SEASON_RUNNER.get("max_hits") or 1)
    _LIVE_CTX = {
        "live": live,
        "mip_preds": mip_preds,
        "vec_preds": vec_preds,
        "current_ids": current_ids,
        "bank": float(live.bank),
        "ft": int(live.free_transfers),
        "selling": live.selling_prices,
        "max_hits": max_hits,
        "gate_kwargs": transfer_gate_kwargs(EV_MODE),
        "bench_w": float(BENCH_COVER["bench_ev_weight"]),
        "preseason": preseason,
        "gw": gw,
    }
    return _LIVE_CTX


def test_live_parity() -> dict[str, Any]:
    from fpl_robo.optimize.transfers import optimise_transfers
    from fpl_robo.optimize.vector_search import anneal_transfers

    t0 = time.perf_counter()
    ctx = _build_live_preds()
    live = ctx["live"]
    mip_preds = ctx["mip_preds"]
    vec_preds = ctx["vec_preds"]
    current_ids = ctx["current_ids"]
    bank = ctx["bank"]
    ft = ctx["ft"]
    selling = ctx["selling"]
    max_hits = ctx["max_hits"]
    gate_kwargs = ctx["gate_kwargs"]
    bench_w = ctx["bench_w"]
    preseason = ctx["preseason"]
    gw = ctx["gw"]

    t_mip0 = time.perf_counter()
    mip = optimise_transfers(
        mip_preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling,
        free_transfers=ft,
        max_hits=max_hits,
        require_cover_eligible=True,
        bench_ev_weight=bench_w,
        **gate_kwargs,
        use_tiebreak=None,
    )
    mip_s = time.perf_counter() - t_mip0

    t_v0 = time.perf_counter()
    vec_live = anneal_transfers(
        vec_preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling,
        free_transfers=ft,
        max_hits=max_hits,
        n_iters=1500,
        seed=42,
    )
    vec_s = time.perf_counter() - t_v0

    mip_transfers = mip.transfers or []
    if mip_transfers and "element_out" in (mip_transfers[0] or {}):
        mip_outs = {(int(t["element_out"]), int(t["element_in"])) for t in mip_transfers}
    else:
        mip_outs = {(int(t.get("out")), int(t.get("in"))) for t in mip_transfers}
    vec_pairs = {
        (int(t.get("element_out", t.get("out"))), int(t.get("element_in", t.get("in"))))
        for t in (vec_live.transfers or [])
    }
    overlap = len(mip_outs & vec_pairs)
    detail = {
        "gameweek": gw,
        "preseason": preseason,
        "free_transfers": ft,
        "bank": bank,
        "mip": {
            "status": mip.status,
            "transfers_made": mip.transfers_made,
            "hits": mip.hits,
            "hit_cost": mip.hit_cost,
            "captain": mip.solution.captain_name,
            "xi": [p.get("web_name") for p in mip.solution.starting_xi],
            "transfers": mip.transfers,
            "seconds": round(mip_s, 3),
        },
        "vector": {
            "status": vec_live.status,
            "transfers_made": vec_live.transfers_made,
            "hits": vec_live.hits,
            "hit_cost": vec_live.hit_cost,
            "captain": vec_live.solution.captain_name,
            "xi": [p.get("web_name") for p in vec_live.solution.starting_xi],
            "transfers": vec_live.transfers,
            "seconds": round(vec_s, 3),
        },
        "transfer_overlap": overlap,
        "total_elapsed_s": round(time.perf_counter() - t0, 2),
    }
    if mip.status != "Optimal" or vec_live.status != "Optimal":
        return _fail("live_parity", "non-optimal plan", detail)
    if ft >= 10 and vec_live.transfers_made >= 8:
        return _warn(
            "live_parity",
            "vector proposes large rebuild under unlimited FT (expected in preseason; gate before go-live)",
            detail,
        )
    return _ok("live_parity", detail)


def test_stability() -> dict[str, Any]:
    from fpl_robo.optimize.vector_search import anneal_transfers

    ctx = _build_live_preds()
    vec_preds = ctx["vec_preds"]
    current_ids = ctx["current_ids"]
    bank = ctx["bank"]
    ft = ctx["ft"]
    selling = ctx["selling"]
    max_hits = ctx["max_hits"]

    ft_cap = min(int(ft), 2)
    plans = []
    for seed in (42, 42, 43, 99, 7):
        t0 = time.perf_counter()
        plan = anneal_transfers(
            vec_preds,
            current_ids=current_ids,
            bank=bank,
            selling_prices=selling,
            free_transfers=ft_cap,
            max_hits=max_hits,
            n_iters=1200,
            seed=seed,
        )
        plans.append(
            {
                "seed": seed,
                "transfers_made": plan.transfers_made,
                "hits": plan.hits,
                "objective": round(float(plan.objective), 4),
                "captain": plan.solution.captain_name,
                "transfer_set": sorted(
                    (t.get("element_out", t.get("out")), t.get("element_in", t.get("in")))
                    for t in plan.transfers
                ),
                "seconds": round(time.perf_counter() - t0, 3),
            }
        )

    same_seed = [p for p in plans if p["seed"] == 42]
    same_seed_stable = (
        same_seed[0]["transfer_set"] == same_seed[1]["transfer_set"]
        and same_seed[0]["captain"] == same_seed[1]["captain"]
    )
    unique_sets = {tuple(p["transfer_set"]) for p in plans}
    xfer_counts = [p["transfers_made"] for p in plans]
    detail = {
        "ft_cap": ft_cap,
        "plans": plans,
        "same_seed_stable": same_seed_stable,
        "unique_transfer_sets": len(unique_sets),
        "xfer_count_range": [min(xfer_counts), max(xfer_counts)],
        "mean_seconds": round(sum(p["seconds"] for p in plans) / len(plans), 3),
    }
    if not same_seed_stable:
        return _fail("stability", "same seed produced different plans", detail)
    if max(xfer_counts) - min(xfer_counts) > 2:
        return _warn("stability", "high transfer-count variance across seeds", detail)
    if len(unique_sets) > 4:
        return _warn("stability", "many distinct plans across seeds (SA noise)", detail)
    return _ok("stability", detail)


def test_ft_cap_vs_unlimited() -> dict[str, Any]:
    from fpl_robo.optimize.vector_search import anneal_transfers

    ctx = _build_live_preds()
    vec_preds = ctx["vec_preds"]
    current_ids = ctx["current_ids"]
    bank = ctx["bank"]
    ft = ctx["ft"]
    selling = ctx["selling"]
    max_hits = ctx["max_hits"]

    unlimited = anneal_transfers(
        vec_preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling,
        free_transfers=max(ft, 15),
        max_hits=0,
        n_iters=1200,
        seed=1,
    )
    capped = anneal_transfers(
        vec_preds,
        current_ids=current_ids,
        bank=bank,
        selling_prices=selling,
        free_transfers=1,
        max_hits=1,
        n_iters=1200,
        seed=1,
    )
    detail = {
        "live_ft": ft,
        "unlimited": {
            "transfers_made": unlimited.transfers_made,
            "hits": unlimited.hits,
            "captain": unlimited.solution.captain_name,
        },
        "ft1_hit1": {
            "transfers_made": capped.transfers_made,
            "hits": capped.hits,
            "captain": capped.solution.captain_name,
            "transfers": capped.transfers,
        },
    }
    if capped.transfers_made > 2:
        return _fail("preseason_ft_cap", "FT=1/max_hits=1 made >2 transfers", detail)
    if unlimited.transfers_made < capped.transfers_made:
        return _warn("preseason_ft_cap", "unlimited made fewer transfers than capped (odd)", detail)
    return _ok("preseason_ft_cap", detail)


def test_chip_fidelity() -> dict[str, Any]:
    """Check chip recommendations exist and suite evidence for TC/BB."""
    suite_path = ROOT / "data" / "vector_team_eval_suite.json"
    detail: dict[str, Any] = {"suite_path": str(suite_path)}
    if suite_path.exists():
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        chip_runs = [
            r
            for r in suite.get("runs") or []
            if r.get("use_chips") and "vector_chips" in r
        ]
        detail["suite_chip_runs"] = [
            {
                "label": r.get("label"),
                "vector_chips": r.get("vector_chips"),
                "baseline_chips": r.get("baseline_chips"),
                "delta": r.get("delta_total_points"),
            }
            for r in chip_runs
        ]
        used_tc = any((r.get("vector_chips") or {}).get("triple_captain") for r in chip_runs)
        used_bb = any((r.get("vector_chips") or {}).get("bench_boost") for r in chip_runs)
        detail["vector_used_tc"] = used_tc
        detail["vector_used_bb"] = used_bb
        detail["fh_wc_note"] = (
            "FH/WC are tracked when recommended but not full one-week rebuilds "
            "(same limitation as MIP season harness)."
        )
        if not used_tc and not used_bb:
            return _warn("chip_fidelity", "suite shows no TC/BB plays", detail)
        return _ok("chip_fidelity", detail)
    return _warn("chip_fidelity", "eval suite missing; skip historical chip check", detail)


def test_latency(parity_detail: dict[str, Any] | None, stability_detail: dict[str, Any] | None) -> dict[str, Any]:
    mip_s = ((parity_detail or {}).get("mip") or {}).get("seconds")
    vec_s = ((parity_detail or {}).get("vector") or {}).get("seconds")
    mean_sa = (stability_detail or {}).get("mean_seconds")
    detail = {
        "mip_transfer_solve_s": mip_s,
        "vector_transfer_solve_s": vec_s,
        "vector_mean_stability_solve_s": mean_sa,
        "budget_s": 30.0,
    }
    worst = max(x for x in (vec_s, mean_sa) if x is not None) if (vec_s or mean_sa) else None
    if worst is None:
        return _warn("latency", "no timing samples", detail)
    if worst > 30:
        return _fail("latency", f"vector solve {worst}s exceeds 30s budget", detail)
    if worst > 15:
        return _warn("latency", f"vector solve {worst}s is slow but under 30s", detail)
    return _ok("latency", detail)


def test_shadow_gate() -> dict[str, Any]:
    """Confirm live season path uses vector SA when VECTOR_TEAM.enabled."""
    from fpl_robo.config import EV_MODE, VECTOR_TEAM
    from fpl_robo.optimize import transfers as tr
    from fpl_robo.optimize import vector_search as vs
    from fpl_robo.optimize.live_vector import vector_live_enabled
    from fpl_robo.runtime import season_runner as sr

    src = Path(sr.__file__).read_text(encoding="utf-8")
    wired = "live_optimise_transfers" in src and "vector_live_enabled" in src
    enabled = bool(VECTOR_TEAM.get("enabled")) and vector_live_enabled()
    detail = {
        "live_EV_MODE": EV_MODE,
        "VECTOR_TEAM.enabled": enabled,
        "live_transfer_solver": (
            "live_optimise_transfers (vector_team_sa)" if enabled else "optimise_transfers (PuLP MIP)"
        ),
        "season_runner_wired": wired,
        "vector_module_present": hasattr(vs, "anneal_transfers"),
        "mip_module_present": hasattr(tr, "optimise_transfers"),
    }
    if enabled and wired:
        return _ok("live_wiring", detail)
    if wired and not enabled:
        return _warn(
            "live_wiring",
            "season_runner wired but VECTOR_TEAM.enabled=False (MIP still live)",
            detail,
        )
    return _fail("live_wiring", "vector live path not wired into season_runner", detail)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    print("=== vector readiness tests ===", flush=True)
    results: list[dict[str, Any]] = []

    print("\n[1/7] unit regression", flush=True)
    results.append(test_units())

    parity = None
    stability = None
    try:
        print("\n[2/7] live parity (MIP vs vector)", flush=True)
        parity = test_live_parity()
        results.append(parity)
    except Exception as exc:  # noqa: BLE001
        results.append(_fail("live_parity", str(exc), {"traceback": traceback.format_exc()[-800:]}))

    try:
        print("\n[3/7] stability / thrash", flush=True)
        stability = test_stability()
        results.append(stability)
    except Exception as exc:  # noqa: BLE001
        results.append(_fail("stability", str(exc), {"traceback": traceback.format_exc()[-800:]}))

    try:
        print("\n[4/7] preseason FT cap vs unlimited", flush=True)
        results.append(test_ft_cap_vs_unlimited())
    except Exception as exc:  # noqa: BLE001
        results.append(_fail("preseason_ft_cap", str(exc)))

    print("\n[5/7] chip fidelity", flush=True)
    results.append(test_chip_fidelity())

    print("\n[6/7] latency", flush=True)
    results.append(
        test_latency(
            parity if parity and parity.get("status") != "fail" else None,
            stability if stability and stability.get("status") != "fail" else None,
        )
    )

    print("\n[7/7] live wiring", flush=True)
    results.append(test_shadow_gate())

    passes = sum(1 for r in results if r["status"] == "pass")
    warns = sum(1 for r in results if r["status"] == "warn")
    fails = sum(1 for r in results if r["status"] == "fail")
    wiring_ok = any(
        r["name"] == "live_wiring" and r["status"] == "pass" for r in results
    )
    go_live = fails == 0 and not any(
        r["name"] == "live_parity" and r["status"] == "fail" for r in results
    )
    replace_live = go_live and wiring_ok

    report = {
        "ready_for_pr_prototype": go_live,
        "ready_to_replace_live": replace_live,
        "summary": {"pass": passes, "warn": warns, "fail": fails},
        "results": results,
        "blockers_for_live_replace": [
            r["name"] + ": " + str(r.get("reason") or r["status"])
            for r in results
            if r["status"] in ("fail", "warn")
            and r["name"] in ("live_parity", "stability", "live_wiring", "latency")
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== summary ===", flush=True)
    for r in results:
        flag = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[r["status"]]
        extra = f" — {r.get('reason')}" if r.get("reason") else ""
        print(f"  [{flag}] {r['name']}{extra}", flush=True)
    print(
        f"\npass={passes} warn={warns} fail={fails} "
        f"ready_for_pr_prototype={go_live} ready_to_replace_live={replace_live}",
        flush=True,
    )
    print(f"Report: {REPORT}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
