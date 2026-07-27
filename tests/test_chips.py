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


if __name__ == "__main__":
    unittest.main()
