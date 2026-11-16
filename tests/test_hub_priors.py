"""Tests for Hub soft priors shared with the live bot."""

from __future__ import annotations

import unittest

import pandas as pd

from fpl_robo.features.hub_priors import (
    apply_hub_priors,
    blank_double_by_gw,
    differential_score,
    form_heat,
    hub_prior_multiplier,
    team_run_ease,
)


class HubPriorsTests(unittest.TestCase):
    def test_form_heat_signed(self):
        self.assertGreater(form_heat(6.0, 4.0), 0.0)
        self.assertLess(form_heat(3.0, 5.0), 0.0)
        self.assertEqual(form_heat(4.0, 4.0), 0.0)
        self.assertEqual(form_heat(10.0, 0.0), 1.0)

    def test_differential_requires_low_own_and_form(self):
        self.assertEqual(differential_score(20.0, 6.0), 0.0)
        self.assertEqual(differential_score(3.0, 2.0), 0.0)
        self.assertGreater(differential_score(3.0, 6.0), 0.0)

    def test_multiplier_clamped_and_blocked_on_doubt(self):
        hot, _ = hub_prior_multiplier(form=7.0, ppg=4.0, ownership=2.0)
        self.assertGreater(hot, 1.0)
        self.assertLessEqual(hot, 1.04)
        blocked, _ = hub_prior_multiplier(
            form=7.0, ppg=4.0, ownership=2.0, chance=50
        )
        self.assertEqual(blocked, 1.0)

    def test_apply_hub_priors_scales_ev(self):
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "ev": 10.0,
                    "form": 7.0,
                    "points_per_game": 4.0,
                    "selected_by_percent": 2.0,
                    "status": "a",
                },
                {
                    "id": 2,
                    "ev": 10.0,
                    "form": 2.0,
                    "points_per_game": 5.0,
                    "selected_by_percent": 40.0,
                    "status": "a",
                },
            ]
        )
        out = apply_hub_priors(df)
        self.assertGreater(float(out.loc[0, "ev"]), 10.0)
        self.assertLess(float(out.loc[1, "ev"]), 10.0)
        self.assertIn("hub_prior_mult", out.columns)

    def test_team_run_ease_and_blank_double(self):
        cells = [
            {"fdr": 2, "blank": False},
            {"fdr": 2, "blank": False},
            {"blank": True, "fdr": None},
        ]
        self.assertAlmostEqual(team_run_ease(cells), 4.0)
        bd = blank_double_by_gw(
            fixtures_by_gw={
                1: [{"home": {"id": 1}, "away": {"id": 2}}],
            },
            team_shorts={1: "AAA", 2: "BBB", 3: "CCC"},
            gws=[1],
        )
        self.assertEqual(bd[0]["blanks"], ["CCC"])
        self.assertEqual(bd[0]["doubles"], [])


if __name__ == "__main__":
    unittest.main()
