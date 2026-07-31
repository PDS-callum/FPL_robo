import unittest

import pandas as pd

from fpl_robo.predict.horizon import (
    apply_horizon_ev,
    fixture_scale,
    mode_uses_horizon,
    strip_horizon_suffix,
)


class HorizonEvTests(unittest.TestCase):
    def test_fixture_scale_blank_and_dgw(self):
        self.assertEqual(fixture_scale(n_fixtures=0, blank_mult=0.05), 0.05)
        single = fixture_scale(n_fixtures=1, fdr=3.0)
        self.assertAlmostEqual(single, 1.0)
        dgw = fixture_scale(n_fixtures=2, fdr=3.0, dgw_mult=1.75)
        self.assertAlmostEqual(dgw, 1.75)
        easy = fixture_scale(n_fixtures=1, fdr=2.0)
        self.assertGreater(easy, single)

    def test_apply_horizon_blank_near_zero_gw0(self):
        preds = pd.DataFrame(
            [
                {"id": 1, "team_id": 10, "ev": 10.0, "web_name": "A", "position": "MID", "price": 7.0},
                {"id": 2, "team_id": 20, "ev": 10.0, "web_name": "B", "position": "MID", "price": 7.0},
                {"id": 3, "team_id": 30, "ev": 10.0, "web_name": "C", "position": "MID", "price": 7.0},
            ]
        )
        fixtures = {
            10: {1: {"n_fixtures": 0.0, "best_fdr": None}},  # explicit blank
            20: {1: {"n_fixtures": 1.0, "best_fdr": 3.0}},
            # team 30 missing → unknown → neutral (not blank)
        }
        out = apply_horizon_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            weights=[1.0],
            blank_mult=0.05,
            unknown_mult=1.0,
        )
        by_id = {int(r["id"]): float(r["ev"]) for _, r in out.iterrows()}
        self.assertAlmostEqual(by_id[1], 0.5)  # 10 * 0.05
        self.assertAlmostEqual(by_id[2], 10.0)
        self.assertAlmostEqual(by_id[3], 10.0)  # unknown stays neutral
        self.assertIn("ev_gw0", out.columns)

    def test_asymmetric_lookahead_ignores_future_blank(self):
        preds = pd.DataFrame(
            [
                {
                    "id": 1,
                    "team_id": 10,
                    "ev": 10.0,
                    "web_name": "A",
                    "position": "MID",
                    "price": 7.0,
                },
            ]
        )
        fixtures = {
            10: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 0.0, "best_fdr": None},  # blank next week
            }
        }
        out = apply_horizon_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            weights=[0.7, 0.3],
            blank_mult=0.05,
            asymmetric_lookahead=True,
        )
        # Future blank must not haircut: 0.7*10*1 + 0.3*10*1 = 10
        self.assertAlmostEqual(float(out.iloc[0]["ev"]), 10.0)

    def test_asymmetric_lookahead_keeps_future_dgw_boost(self):
        preds = pd.DataFrame(
            [
                {
                    "id": 1,
                    "team_id": 10,
                    "ev": 10.0,
                    "web_name": "A",
                    "position": "MID",
                    "price": 7.0,
                },
            ]
        )
        fixtures = {
            10: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 2.0, "best_fdr": 3.0},
            }
        }
        out = apply_horizon_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            weights=[0.7, 0.3],
            dgw_mult=1.5,
            asymmetric_lookahead=True,
        )
        # 0.7*10*1 + 0.3*10*1.5 = 11.5
        self.assertAlmostEqual(float(out.iloc[0]["ev"]), 11.5)

    def test_blank_ahead_haircuts_upcoming_blank_only(self):
        from fpl_robo.predict.horizon import apply_blank_ahead_ev

        preds = pd.DataFrame(
            [
                {"id": 1, "team_id": 10, "ev": 10.0},
                {"id": 2, "team_id": 20, "ev": 10.0},
                {"id": 3, "team_id": 30, "ev": 10.0},
            ]
        )
        fixtures = {
            10: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 0.0, "best_fdr": None},  # blanks next GW
            },
            20: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 1.0, "best_fdr": 3.0},
                3: {"n_fixtures": 0.0, "best_fdr": None},  # blanks in +2
            },
            30: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 1.0, "best_fdr": 3.0},
            },
        }
        out = apply_blank_ahead_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            lookahead=2,
            offset_mults=[0.80, 0.90],
            only_if_owned_blank=False,
        )
        by_id = {int(r["id"]): r for _, r in out.iterrows()}
        self.assertAlmostEqual(float(by_id[1]["ev"]), 8.0)
        self.assertAlmostEqual(float(by_id[2]["ev"]), 9.0)
        self.assertAlmostEqual(float(by_id[3]["ev"]), 10.0)

    def test_blank_ahead_gated_on_owned_exposure(self):
        from fpl_robo.predict.horizon import apply_blank_ahead_ev

        preds = pd.DataFrame(
            [
                {"id": 1, "team_id": 10, "ev": 10.0},
                {"id": 2, "team_id": 20, "ev": 10.0},
            ]
        )
        fixtures = {
            10: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 0.0, "best_fdr": None},
            },
            20: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 1.0, "best_fdr": 3.0},
            },
        }
        # Own only non-blanking team → no haircut.
        skipped = apply_blank_ahead_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            current_ids=[2],
            only_if_owned_blank=True,
            owned_blank_threshold=1,
            offset_mults=[0.80, 0.90],
        )
        self.assertAlmostEqual(float(skipped.loc[skipped["id"] == 1, "ev"].iloc[0]), 10.0)
        self.assertFalse(bool(skipped["blank_ahead_triggered"].iloc[0]))

        # Own blanking team → haircut fires for all blank-club players.
        fired = apply_blank_ahead_ev(
            preds,
            start_gw=1,
            team_fixtures=fixtures,
            current_ids=[1],
            only_if_owned_blank=True,
            owned_blank_threshold=1,
            offset_mults=[0.80, 0.90],
        )
        self.assertAlmostEqual(float(fired.loc[fired["id"] == 1, "ev"].iloc[0]), 8.0)
        self.assertTrue(bool(fired["blank_ahead_triggered"].iloc[0]))

    def test_mode_suffix_helpers(self):
        self.assertTrue(mode_uses_horizon("ep_minutes_scaled_h"))
        self.assertFalse(mode_uses_horizon("ep_minutes_scaled"))
        self.assertEqual(strip_horizon_suffix("ep_minutes_scaled_h"), "ep_minutes_scaled")


if __name__ == "__main__":
    unittest.main()
