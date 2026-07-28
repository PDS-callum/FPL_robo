"""Tests for Core Insights ep_this fill of blank vaastav xP weeks."""

from __future__ import annotations

import unittest

import pandas as pd

from fpl_robo.data.ep_history import fill_blank_gw_xp, season_to_core_insights_dir


class EpHistoryTests(unittest.TestCase):
    def test_season_dir_mapping(self):
        self.assertEqual(season_to_core_insights_dir("2025-26"), "2025-2026")
        self.assertEqual(season_to_core_insights_dir("2024-25"), "2024-2025")

    def test_fills_only_blank_gameweeks(self):
        table = pd.DataFrame(
            [
                {"id": 1, "gameweek": 5, "xP": 4.0, "web_name": "A"},
                {"id": 2, "gameweek": 5, "xP": 3.0, "web_name": "B"},
                {"id": 1, "gameweek": 12, "xP": 0.0, "web_name": "A"},
                {"id": 2, "gameweek": 12, "xP": 0.0, "web_name": "B"},
            ]
        )
        ep = pd.DataFrame(
            [
                {"element": 1, "gameweek": 5, "ep_this": 9.9, "ep_next": 1.0},
                {"element": 2, "gameweek": 5, "ep_this": 8.8, "ep_next": 1.0},
                {"element": 1, "gameweek": 12, "ep_this": 5.5, "ep_next": 1.0},
                {"element": 2, "gameweek": 12, "ep_this": 6.6, "ep_next": 1.0},
            ]
        )
        out = fill_blank_gw_xp(table, ep_source=ep, blank_frac=0.90)
        # GW5 kept vaastav xP
        self.assertAlmostEqual(float(out[(out.id == 1) & (out.gameweek == 5)].iloc[0]["xP"]), 4.0)
        # GW12 filled from ep_this
        self.assertAlmostEqual(float(out[(out.id == 1) & (out.gameweek == 12)].iloc[0]["xP"]), 5.5)
        self.assertAlmostEqual(float(out[(out.id == 2) & (out.gameweek == 12)].iloc[0]["xP"]), 6.6)


if __name__ == "__main__":
    unittest.main()
