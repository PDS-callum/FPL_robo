"""Tests for symmetric fixture-run EV planning."""

from __future__ import annotations

import unittest

import pandas as pd

from fpl_robo.predict.fixture_run import apply_fixture_run_ev, team_fixtures_with_fdr
from fpl_robo.predict.horizon import mode_uses_fixture_run, strip_horizon_suffix


class FixtureRunTests(unittest.TestCase):
    def test_mode_suffix(self):
        self.assertTrue(mode_uses_fixture_run("ep_baseline_run"))
        self.assertFalse(mode_uses_fixture_run("ep_baseline"))
        self.assertEqual(strip_horizon_suffix("ep_baseline_run"), "ep_baseline")

    def test_hard_run_haircuts_hot_form(self):
        """Isak-style: same base EV, hard upcoming FDR must lose to easy run."""
        preds = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "HotHard",
                    "position": "FWD",
                    "team_id": 10,
                    "price": 10.0,
                    "ev": 8.0,
                },
                {
                    "id": 2,
                    "web_name": "CoolEasy",
                    "position": "FWD",
                    "team_id": 20,
                    "price": 8.0,
                    "ev": 8.0,
                },
            ]
        )
        fixtures = {
            10: {
                1: {"n_fixtures": 1.0, "best_fdr": 5.0},
                2: {"n_fixtures": 1.0, "best_fdr": 5.0},
                3: {"n_fixtures": 1.0, "best_fdr": 4.0},
                4: {"n_fixtures": 1.0, "best_fdr": 5.0},
            },
            20: {
                1: {"n_fixtures": 1.0, "best_fdr": 2.0},
                2: {"n_fixtures": 1.0, "best_fdr": 2.0},
                3: {"n_fixtures": 1.0, "best_fdr": 1.0},
                4: {"n_fixtures": 1.0, "best_fdr": 2.0},
            },
        }
        out = apply_fixture_run_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            config={
                "weights": [0.60, 0.22, 0.12, 0.06],
                "fdr_weight": 0.08,
                "symmetric_fdr": True,
                "fixture_power": 1.0,
                "blend_gw0": 0.50,
                "factor_min": 0.94,
                "factor_max": 1.10,
            },
        )
        by_id = {int(r["id"]): float(r["ev"]) for _, r in out.iterrows()}
        self.assertGreater(by_id[2], by_id[1])
        # Clamped+blended: haircut/boost is mild but directionally correct.
        self.assertLess(by_id[1], 8.0)
        self.assertGreater(by_id[2], 8.0)
        factors = {int(r["id"]): float(r["fixture_run_factor"]) for _, r in out.iterrows()}
        self.assertGreaterEqual(factors[1], 0.94)
        self.assertLessEqual(factors[2], 1.10)
        self.assertGreater(factors[2], factors[1])

    def test_asymmetric_legacy_would_not_haircut(self):
        preds = pd.DataFrame(
            [{"id": 1, "team_id": 10, "ev": 10.0, "web_name": "A", "position": "MID", "price": 7.0}]
        )
        fixtures = {
            10: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 1.0, "best_fdr": 5.0},
                3: {"n_fixtures": 1.0, "best_fdr": 5.0},
            }
        }
        sym = apply_fixture_run_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            config={
                "weights": [0.5, 0.3, 0.2],
                "fdr_weight": 0.24,
                "symmetric_fdr": True,
            },
        )
        asym = apply_fixture_run_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            config={
                "weights": [0.5, 0.3, 0.2],
                "fdr_weight": 0.24,
                "symmetric_fdr": False,
            },
        )
        self.assertLess(float(sym.iloc[0]["ev"]), float(asym.iloc[0]["ev"]))

    def test_season_fdr_map_loaded(self):
        fx = team_fixtures_with_fdr("2025-26")
        self.assertEqual(len(fx), 20)
        sample = next(iter(fx.values()))
        fdrs = [s.get("best_fdr") for s in sample.values() if s.get("best_fdr") is not None]
        self.assertTrue(fdrs)
        self.assertTrue(all(1.0 <= float(d) <= 5.0 for d in fdrs))


if __name__ == "__main__":
    unittest.main()
