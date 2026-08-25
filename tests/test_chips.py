import unittest

from fpl_robo.chips.decide import recommend_chips
from fpl_robo.chips.inventory import ChipInventory, ChipWindow, parse_chip_inventory
from fpl_robo.chips.scoring import score_chip_opportunities


class ChipLogicTests(unittest.TestCase):
    def test_score_chip_opportunities(self):
        xi = [{"id": 1, "ev": 8.0}, {"id": 2, "ev": 5.0}]
        bench = [{"id": 3, "ev": 3.0}, {"id": 4, "ev": 2.5}]
        scores = score_chip_opportunities(
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            optimal_xi_ev=20.0,
            current_xi_ev=13.0,
            rolling_gap=15.0,
        )
        self.assertEqual(scores.triple_captain, 16.0)
        self.assertEqual(scores.bench_boost, 5.5)
        self.assertEqual(scores.free_hit, 7.0)
        self.assertEqual(scores.wildcard, 15.0)

    def test_parse_inventory_and_recommend_hold(self):
        inv = parse_chip_inventory(
            {
                "chips": [
                    {
                        "name": "3xc",
                        "status_for_entry": "available",
                        "start_event": 1,
                        "stop_event": 19,
                        "played_by_entry": [],
                    },
                    {
                        "name": "bboost",
                        "status_for_entry": "available",
                        "start_event": 1,
                        "stop_event": 19,
                        "played_by_entry": [],
                    },
                ]
            }
        )
        self.assertTrue(inv.available("triple_captain", 1))
        xi = [{"id": 1, "ev": 4.0}] + [{"id": i, "ev": 3.0} for i in range(2, 12)]
        bench = [{"id": i, "ev": 1.0} for i in range(12, 16)]
        rec = recommend_chips(
            gameweek=1,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            historical_tc_scores=[20, 18, 16, 14],
            historical_bb_scores=[12, 10, 9, 8],
        )
        self.assertFalse(rec.play_now["triple_captain"])
        self.assertFalse(rec.play_now["bench_boost"])

    def test_recommend_plays_strong_tc(self):
        inv = ChipInventory(
            windows=[ChipWindow("triple_captain", True, False, 1, 19)]
        )
        xi = [{"id": 1, "ev": 9.0}] + [{"id": i, "ev": 3.0} for i in range(2, 12)]
        bench = [{"id": i, "ev": 2.0} for i in range(12, 16)]
        rec = recommend_chips(
            gameweek=10,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            historical_tc_scores=[8, 9, 10, 11, 12],
        )
        self.assertTrue(rec.play_now["triple_captain"])

    def test_recommend_plays_free_hit_on_rebuild_gap(self):
        inv = ChipInventory(windows=[ChipWindow("free_hit", True, False, 1, 19)])
        xi = [{"id": i, "ev": 2.0} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0} for i in range(12, 16)]
        # No fixtures → plan-ahead inactive; gap-only behaviour.
        rec = recommend_chips(
            gameweek=8,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            optimal_xi_ev=40.0,
            current_xi_ev=22.0,  # gap 18 >= fh_min_gap 12
        )
        self.assertTrue(rec.play_now["free_hit"])
        self.assertEqual(rec.scores["free_hit"], 18.0)

    def test_recommend_holds_free_hit_flat_fixture_calendar(self):
        from fpl_robo.chips.fh_timing import build_fh_timing_outlook

        inv = ChipInventory(windows=[ChipWindow("free_hit", True, False, 1, 19)])
        xi = [{"id": i, "ev": 2.0, "team_id": 1 + (i % 5)} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0, "team_id": 1} for i in range(12, 16)]
        # Flat SGW FDR=3 across look-ahead → no schedule edge → HOLD.
        fixtures = {
            tid: {
                gw: {"n_fixtures": 1.0, "best_fdr": 3.0}
                for gw in range(2, 9)
            }
            for tid in range(1, 6)
        }
        outlook = build_fh_timing_outlook(
            gameweek=2,
            team_fixtures=fixtures,
            owned_team_ids=[1, 2, 3],
            blank_owned=0,
        )
        self.assertIsNotNone(outlook)
        self.assertFalse(outlook.near_best)
        rec = recommend_chips(
            gameweek=2,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            optimal_xi_ev=43.0,
            current_xi_ev=28.0,
            team_fixtures=fixtures,
            blank_owned=0,
        )
        self.assertFalse(rec.play_now["free_hit"])
        self.assertIn("no schedule edge", rec.reasons["free_hit"])

    def test_recommend_plays_free_hit_on_best_soft_slate(self):
        inv = ChipInventory(windows=[ChipWindow("free_hit", True, False, 1, 19)])
        xi = [{"id": i, "ev": 2.0, "team_id": 1 + (i % 4)} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0, "team_id": 1} for i in range(12, 16)]
        fixtures = {
            tid: {
                gw: {
                    "n_fixtures": 1.0,
                    # GW 5 is clearly softest.
                    "best_fdr": 2.0 if gw == 5 else 4.0,
                }
                for gw in range(5, 11)
            }
            for tid in range(1, 5)
        }
        rec = recommend_chips(
            gameweek=5,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            optimal_xi_ev=40.0,
            current_xi_ev=22.0,
            team_fixtures=fixtures,
            blank_owned=0,
        )
        self.assertTrue(rec.play_now["free_hit"])
        self.assertIn("best remaining slate", rec.reasons["free_hit"])

    def test_recommend_holds_free_hit_for_better_future_slate(self):
        inv = ChipInventory(windows=[ChipWindow("free_hit", True, False, 1, 19)])
        xi = [{"id": i, "ev": 2.0, "team_id": 1 + (i % 4)} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0, "team_id": 1} for i in range(12, 16)]
        fixtures = {
            tid: {
                gw: {
                    "n_fixtures": 1.0,
                    "best_fdr": 4.0 if gw == 5 else (2.0 if gw == 8 else 3.5),
                }
                for gw in range(5, 11)
            }
            for tid in range(1, 5)
        }
        rec = recommend_chips(
            gameweek=5,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            optimal_xi_ev=40.0,
            current_xi_ev=22.0,
            team_fixtures=fixtures,
            blank_owned=0,
        )
        self.assertFalse(rec.play_now["free_hit"])
        self.assertEqual(rec.details["fh_timing"]["best_gw"], 8)
        self.assertRegex(rec.reasons["free_hit"], r"save for better slate|no schedule edge")
        self.assertIn("best=8", rec.reasons["free_hit"])

    def test_recommend_plays_free_hit_on_blank_force(self):
        inv = ChipInventory(windows=[ChipWindow("free_hit", True, False, 1, 19)])
        xi = [{"id": i, "ev": 2.0, "team_id": i} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0, "team_id": i} for i in range(12, 16)]
        fixtures = {
            tid: {
                gw: {
                    "n_fixtures": 0.0 if (gw == 10 and tid <= 4) else 1.0,
                    "best_fdr": 3.0,
                }
                for gw in range(10, 16)
            }
            for tid in range(1, 16)
        }
        # Soft future week 12 exists, but 4 owned blanks force FH now.
        for tid in range(1, 16):
            fixtures[tid][12] = {"n_fixtures": 1.0, "best_fdr": 2.0}
        rec = recommend_chips(
            gameweek=10,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            optimal_xi_ev=40.0,
            current_xi_ev=22.0,
            team_fixtures=fixtures,
            blank_owned=4,
        )
        self.assertTrue(rec.play_now["free_hit"])
        self.assertIn("blank_force", rec.reasons["free_hit"])

    def test_recommend_holds_free_hit_below_gap(self):
        inv = ChipInventory(windows=[ChipWindow("free_hit", True, False, 1, 19)])
        xi = [{"id": i, "ev": 3.0} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0} for i in range(12, 16)]
        rec = recommend_chips(
            gameweek=8,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            optimal_xi_ev=30.0,
            current_xi_ev=25.0,  # gap 5 < 12
        )
        self.assertFalse(rec.play_now["free_hit"])

    def test_recommend_plays_wildcard_on_rolling_gap(self):
        inv = ChipInventory(windows=[ChipWindow("wildcard", True, False, 1, 19)])
        xi = [{"id": i, "ev": 2.0} for i in range(1, 12)]
        bench = [{"id": i, "ev": 1.0} for i in range(12, 16)]
        rec = recommend_chips(
            gameweek=12,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            inventory=inv,
            rolling_gap=22.0,  # >= wc_min_rolling_gap 18
        )
        self.assertTrue(rec.play_now["wildcard"])
        self.assertEqual(rec.scores["wildcard"], 22.0)

    def test_schedule_captain_dgw_eases_tc(self):
        from fpl_robo.chips.schedule import chip_schedule_adjustments

        xi = [{"id": 1, "ev": 5.0, "team_id": 10}] + [
            {"id": i, "ev": 3.0, "team_id": 1} for i in range(2, 12)
        ]
        bench = [{"id": i, "ev": 2.0, "team_id": 2} for i in range(12, 16)]
        overrides, details = chip_schedule_adjustments(
            gameweek=20,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            dgw_teams={10},
        )
        self.assertTrue(details["captain_dgw"])
        self.assertEqual(overrides["tc_min_captain_ev"], 5.0)
        self.assertIn("tc_captain_dgw", "".join(details["applied"]))

    def test_schedule_bench_dgw_eases_bb(self):
        from fpl_robo.chips.schedule import chip_schedule_adjustments

        xi = [{"id": i, "ev": 3.0, "team_id": 1} for i in range(1, 12)]
        bench = [
            {"id": 12, "ev": 2.0, "team_id": 7},
            {"id": 13, "ev": 2.0, "team_id": 8},
            {"id": 14, "ev": 1.0, "team_id": 1},
            {"id": 15, "ev": 1.0, "team_id": 1},
        ]
        overrides, details = chip_schedule_adjustments(
            gameweek=20,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            dgw_teams={7, 8},
        )
        self.assertEqual(details["bench_dgw_count"], 2)
        self.assertEqual(overrides["bb_min_bench_ev"], 7.0)

    def test_schedule_easy_fdr_and_blanks(self):
        from fpl_robo.chips.schedule import chip_schedule_adjustments

        xi = [{"id": 1, "ev": 5.0, "team_id": 3}] + [
            {"id": i, "ev": 3.0, "team_id": 1} for i in range(2, 12)
        ]
        bench = [{"id": i, "ev": 1.0, "team_id": 1} for i in range(12, 16)]
        fixtures = {3: {11: {"n_fixtures": 1.0, "best_fdr": 2.0}}}
        overrides, details = chip_schedule_adjustments(
            gameweek=11,
            starting_xi=xi,
            bench=bench,
            captain_id=1,
            team_fixtures=fixtures,
            blank_owned=4,
        )
        self.assertEqual(details["captain_fdr"], 2.0)
        self.assertEqual(overrides["tc_min_captain_ev"], 5.5)
        self.assertEqual(overrides["fh_min_gap"], 9.0)

    def test_schedule_set_piece_eases_tc(self):
        from fpl_robo.chips.schedule import chip_schedule_adjustments

        xi = [
            {
                "id": 1,
                "ev": 7.0,
                "team_id": 3,
                "set_piece_score": 0.08,
                "next_matchup_factor": 1.0,
            }
        ] + [{"id": i, "ev": 3.0, "team_id": 1} for i in range(2, 12)]
        overrides, details = chip_schedule_adjustments(
            gameweek=11,
            starting_xi=xi,
            bench=[{"id": i, "ev": 1.0, "team_id": 1} for i in range(12, 16)],
            captain_id=1,
            team_fixtures={3: {11: {"n_fixtures": 1.0, "best_fdr": 3.0}}},
        )
        self.assertIn("tc_set_piece", " ".join(details["applied"]))
        self.assertLess(overrides["tc_min_captain_ev"], 6.0)

    def test_lineup_gate_suppresses_tc_on_doubt(self):
        from fpl_robo.chips.lineup_gate import apply_lineup_chip_gates
        from fpl_robo.signals.schema import PredictedLineupSignal

        chips = {
            "play_now": {"triple_captain": True, "bench_boost": False},
            "reasons": {"triple_captain": "PLAY"},
        }
        xi = [{"id": 1, "web_name": "Salah", "ev": 8.0}]
        gate = apply_lineup_chip_gates(
            chips,
            gameweek=5,
            captain_id=1,
            starting_xi=xi,
            bench=[],
            predicted_lineups=[
                PredictedLineupSignal(
                    player_id=1,
                    web_name="Salah",
                    team_short="LIV",
                    role="doubt",
                    gameweek=5,
                    confidence=0.9,
                )
            ],
        )
        self.assertFalse(chips["play_now"]["triple_captain"])
        self.assertIn("tc_lineup_doubt", gate["suppressed"])

    def test_lineup_gate_suppresses_bb_without_startish_bench(self):
        from fpl_robo.chips.lineup_gate import apply_lineup_chip_gates
        from fpl_robo.signals.schema import PredictedLineupSignal

        chips = {
            "play_now": {"triple_captain": False, "bench_boost": True},
            "reasons": {"bench_boost": "PLAY"},
        }
        bench = [
            {"id": 12, "web_name": "A"},
            {"id": 13, "web_name": "B"},
            {"id": 14, "web_name": "C"},
            {"id": 15, "web_name": "D"},
        ]
        lineups = [
            PredictedLineupSignal(12, "A", "ARS", "bench", 5, 0.8),
            PredictedLineupSignal(13, "B", "CHE", "doubt", 5, 0.8),
            PredictedLineupSignal(14, "C", "MCI", "bench", 5, 0.8),
        ]
        gate = apply_lineup_chip_gates(
            chips,
            gameweek=5,
            captain_id=1,
            starting_xi=[{"id": 1, "web_name": "X"}],
            bench=bench,
            predicted_lineups=lineups,
            bb_min_startish=2,
        )
        self.assertFalse(chips["play_now"]["bench_boost"])
        self.assertTrue(any(s.startswith("bb_lineup") for s in gate["suppressed"]))


if __name__ == "__main__":
    unittest.main()
