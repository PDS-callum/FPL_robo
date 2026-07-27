"""Tests for EV modes: lag leakage and hardened buy gates."""

from __future__ import annotations

import unittest

import pandas as pd

from fpl_robo.optimize.transfers import optimise_transfers
from fpl_robo.predict.ev_modes import attach_lagged_history, build_predictions, select_winner


class MinutesScaledTests(unittest.TestCase):
    def test_thin_minutes_haircut_xP(self):
        from fpl_robo.predict.ev_modes import play_prob_from_minutes

        self.assertAlmostEqual(play_prob_from_minutes(0.0, prior_minutes=0.0), 0.80)
        self.assertAlmostEqual(play_prob_from_minutes(40.0, prior_minutes=100.0), 0.50)
        self.assertAlmostEqual(play_prob_from_minutes(90.0, prior_minutes=500.0), 1.0)

        gw = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "Hype",
                    "position": "MID",
                    "team": "A",
                    "price": 7.0,
                    "xP": 5.0,
                    "total_points": 0,
                    "minutes": 0,
                    "gameweek": 8,
                    "prior_minutes": 40,
                    "rolling_minutes": 20,
                }
            ]
        )
        base = build_predictions(gw, mode="ep_baseline")
        scaled = build_predictions(gw, mode="ep_minutes_scaled")
        self.assertAlmostEqual(float(base.iloc[0]["ev"]), 5.0)
        # 20/80 = 0.25 → floored to 0.30 → EV 1.5
        self.assertAlmostEqual(float(scaled.iloc[0]["ev"]), 1.5)


class OpenEVLagTests(unittest.TestCase):
    def test_rolling_features_exclude_same_gw(self):
        table = pd.DataFrame(
            [
                {
                    "id": 1,
                    "gameweek": 1,
                    "total_points": 10,
                    "minutes": 90,
                    "price": 6.0,
                    "position": "MID",
                    "team": "A",
                    "web_name": "P1",
                    "xP": 4.0,
                },
                {
                    "id": 1,
                    "gameweek": 2,
                    "total_points": 2,
                    "minutes": 10,
                    "price": 6.0,
                    "position": "MID",
                    "team": "A",
                    "web_name": "P1",
                    "xP": 3.0,
                },
            ]
        )
        hist = attach_lagged_history(table)
        gw2 = hist[hist["gameweek"] == 2].iloc[0]
        # GW2 features must reflect GW1 only, not GW2's own 2 pts / 10 mins.
        self.assertEqual(float(gw2["rolling_points"]), 10.0)
        self.assertEqual(float(gw2["rolling_minutes"]), 90.0)
        self.assertEqual(float(gw2["prior_minutes"]), 90.0)

        preds = build_predictions(hist[hist["gameweek"] == 2], mode="open_ev")
        # Open EV must not equal xP.
        self.assertNotAlmostEqual(float(preds.iloc[0]["ev"]), 3.0)


class HardenedBuyGateTests(unittest.TestCase):
    def test_low_minute_player_cannot_be_bought(self):
        # Owned midfielder keeps squad legal; non-owned low-involvement star blocked.
        players = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "Owned",
                    "position": "MID",
                    "team_id": 1,
                    "price": 5.0,
                    "ev": 3.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 2,
                    "web_name": "OwnedGK",
                    "position": "GK",
                    "team_id": 2,
                    "price": 4.5,
                    "ev": 2.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 3,
                    "web_name": "OwnedDEF1",
                    "position": "DEF",
                    "team_id": 3,
                    "price": 4.5,
                    "ev": 2.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 4,
                    "web_name": "OwnedDEF2",
                    "position": "DEF",
                    "team_id": 4,
                    "price": 4.5,
                    "ev": 2.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 5,
                    "web_name": "OwnedDEF3",
                    "position": "DEF",
                    "team_id": 5,
                    "price": 4.5,
                    "ev": 2.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 6,
                    "web_name": "OwnedDEF4",
                    "position": "DEF",
                    "team_id": 6,
                    "price": 4.5,
                    "ev": 2.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 7,
                    "web_name": "OwnedDEF5",
                    "position": "DEF",
                    "team_id": 7,
                    "price": 4.5,
                    "ev": 2.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 8,
                    "web_name": "OwnedMID2",
                    "position": "MID",
                    "team_id": 8,
                    "price": 5.0,
                    "ev": 3.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 9,
                    "web_name": "OwnedMID3",
                    "position": "MID",
                    "team_id": 9,
                    "price": 5.0,
                    "ev": 3.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 10,
                    "web_name": "OwnedMID4",
                    "position": "MID",
                    "team_id": 10,
                    "price": 5.0,
                    "ev": 3.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 11,
                    "web_name": "OwnedMID5",
                    "position": "MID",
                    "team_id": 11,
                    "price": 5.0,
                    "ev": 3.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 12,
                    "web_name": "OwnedFWD1",
                    "position": "FWD",
                    "team_id": 12,
                    "price": 6.0,
                    "ev": 3.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 13,
                    "web_name": "OwnedFWD2",
                    "position": "FWD",
                    "team_id": 13,
                    "price": 6.0,
                    "ev": 3.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 14,
                    "web_name": "OwnedFWD3",
                    "position": "FWD",
                    "team_id": 14,
                    "price": 6.0,
                    "ev": 3.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                {
                    "id": 15,
                    "web_name": "OwnedGK2",
                    "position": "GK",
                    "team_id": 15,
                    "price": 4.0,
                    "ev": 2.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": True,
                    "gameweek": 5,
                },
                # Tempting non-owned buy with high EV but buy_eligible=False
                {
                    "id": 99,
                    "web_name": "Hype",
                    "position": "MID",
                    "team_id": 16,
                    "price": 5.5,
                    "ev": 9.0,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "buy_eligible": False,
                    "gameweek": 5,
                },
            ]
        )
        owned = list(range(1, 16))
        plan = optimise_transfers(
            players,
            current_ids=owned,
            bank=5.0,
            selling_prices={i: float(players.loc[players["id"] == i, "price"].iloc[0]) for i in owned},
            free_transfers=1,
            max_hits=0,
            min_play_chance=None,
            enforce_buy_eligible=True,
            min_buy_ev=2.0,
        )
        bought = {t["in_id"] for t in plan.transfers}
        self.assertNotIn(99, bought)

    def test_hardened_prediction_marks_low_minutes_ineligible(self):
        gw = pd.DataFrame(
            [
                {
                    "id": 1,
                    "web_name": "Bench",
                    "position": "MID",
                    "team": "A",
                    "price": 5.0,
                    "xP": 3.5,
                    "total_points": 0,
                    "minutes": 0,
                    "gameweek": 8,
                    "prior_minutes": 40,
                    "rolling_minutes": 10,
                }
            ]
        )
        preds = build_predictions(gw, mode="ep_hardened")
        self.assertFalse(bool(preds.iloc[0]["buy_eligible"]))
        self.assertFalse(bool(preds.iloc[0]["cover_eligible"]))


class SelectWinnerTests(unittest.TestCase):
    def test_tie_breakers(self):
        rows = [
            {"ev_mode": "a", "total_points": 100, "points_vs_average": 1, "hits_cost": 8},
            {"ev_mode": "b", "total_points": 100, "points_vs_average": 5, "hits_cost": 12},
            {"ev_mode": "c", "total_points": 99, "points_vs_average": 20, "hits_cost": 0},
        ]
        self.assertEqual(select_winner(rows)["ev_mode"], "b")
        rows2 = [
            {"ev_mode": "a", "total_points": 100, "points_vs_average": 5, "hits_cost": 8},
            {"ev_mode": "b", "total_points": 100, "points_vs_average": 5, "hits_cost": 4},
        ]
        self.assertEqual(select_winner(rows2)["ev_mode"], "b")


if __name__ == "__main__":
    unittest.main()
