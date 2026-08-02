#!/usr/bin/env python3
"""Fast regression smoke: unit guards + one-season backtest checks.

Usage:
  .venv/bin/python tools/regression_smoke.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "regression_ep_baseline_2025-26.json"


def _run_units() -> None:
    suite_names = [
        "tests.test_ops_hardening",
        "tests.test_stability_guards",
        "tests.test_preseason_churn",
        "tests.test_kg_enrich",
        "tests.test_lockfile",
        "tests.test_season_hardening",
        "tests.test_independent_xi",
    ]
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in suite_names:
        suite.addTests(loader.loadTestsFromName(name))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def _run_backtest() -> dict:
    cmd = [
        sys.executable,
        "-m",
        "fpl_robo.cli",
        "backtest-season",
        "--season",
        "2025-26",
        "--ev-mode",
        "ep_baseline",
        "--tiebreak",
        "--blank-ahead",
        "--max-hits",
        "2",
        "--no-chips",
        "--report-file",
        str(REPORT),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))
    return json.loads(REPORT.read_text(encoding="utf-8"))


def main() -> None:
    sys.path.insert(0, str(ROOT))
    print("=== unit guards ===")
    _run_units()
    print("=== season backtest (ep_baseline, ~20s) ===")
    payload = _run_backtest()
    summary = payload.get("summary") or {}
    total = int(payload.get("total_points") or 0)
    mean_q = summary.get("mean_quality_fwds")
    below = int(summary.get("weeks_below_attack_floor") or 0)
    print(
        f"total_points={total} mean_quality_fwds={mean_q} "
        f"weeks_below_attack_floor={below}"
    )
    # Floor: no-chips 2025-26 ep_baseline should stay competitive (prior ~2.4k+).
    if total < 2000:
        raise SystemExit(f"FAIL: total_points {total} < 2000")
    if mean_q is not None and float(mean_q) < 2.0:
        raise SystemExit(f"FAIL: mean_quality_fwds {mean_q} < 2.0")
    # Allow some soft-floor weeks (injuries) but not a season of junk attack.
    if below > 12:
        raise SystemExit(f"FAIL: weeks_below_attack_floor {below} > 12")
    print("PASS")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
