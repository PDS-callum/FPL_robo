import unittest

import pandas as pd

from fpl_robo.optimize.transfers import optimise_transfers


class TransferOptimiserTests(unittest.TestCase):
    def test_no_transfer_when_squad_already_strong(self):
        rows = []
        pid = 1
        owned = []
        # Assign unique-ish clubs so the owned 15 is already legal (<=3 per club).
        club = 1
        for position, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
            for _ in range(count):
                rows.append(
                    {
                        "id": pid,
                        "web_name": f"P{pid}",
                        "team_id": club,
                        "position": position,
                        "price": 5.0,
                        "ev": 5.0,
                        "status": "a",
                        "chance_of_playing_next_round": 100,
                        "cover_eligible": True,
                        "gameweek": 2,
                    }
                )
                owned.append(pid)
                pid += 1
                club = club + 1 if club < 15 else 1
        for position in ("GK", "DEF", "MID", "FWD"):
            for i in range(8):
                rows.append(
                    {
                        "id": pid,
                        "web_name": f"W{pid}",
                        "team_id": (i % 10) + 1,
                        "position": position,
                        "price": 4.5,
                        "ev": 1.0,
                        "status": "a",
                        "chance_of_playing_next_round": 100,
                        "cover_eligible": True,
                        "gameweek": 2,
                    }
                )
                pid += 1

        plan = optimise_transfers(
            pd.DataFrame(rows),
            current_ids=owned,
            bank=0.0,
            selling_prices={i: 5.0 for i in owned},
            free_transfers=1,
            max_hits=0,
        )
        self.assertEqual(plan.status, "Optimal")
        self.assertEqual(plan.transfers_made, 0)
        self.assertEqual(plan.hits, 0)

    def test_transfer_upgrades_when_beneficial(self):
        owned = list(range(1, 16))
        rows = []
        for idx, pid in enumerate(owned):
            position = ["GK", "GK", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"][idx]
            rows.append(
                {
                    "id": pid,
                    "web_name": f"P{pid}",
                    "team_id": (idx % 10) + 1,
                    "position": position,
                    "price": 5.0,
                    "ev": 2.0 if pid != 15 else 0.5,
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "cover_eligible": True,
                    "gameweek": 2,
                }
            )
        # Strong replacement forward.
        rows.append(
            {
                "id": 99,
                "web_name": "Upgrade",
                "team_id": 11,
                "position": "FWD",
                "price": 5.0,
                "ev": 6.0,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "cover_eligible": True,
                "gameweek": 2,
            }
        )
        # Extra pool filler for legality.
        pid = 100
        for position in ("GK", "DEF", "MID", "FWD"):
            for i in range(10):
                rows.append(
                    {
                        "id": pid,
                        "web_name": f"F{pid}",
                        "team_id": (i % 12) + 1,
                        "position": position,
                        "price": 4.5,
                        "ev": 1.2,
                        "status": "a",
                        "chance_of_playing_next_round": 100,
                        "cover_eligible": True,
                        "gameweek": 2,
                    }
                )
                pid += 1

        plan = optimise_transfers(
            pd.DataFrame(rows),
            current_ids=owned,
            bank=0.0,
            selling_prices={i: 5.0 for i in owned},
            free_transfers=1,
            max_hits=0,
        )
        self.assertEqual(plan.status, "Optimal")
        self.assertGreaterEqual(plan.transfers_made, 1)
        self.assertIn(99, [p["id"] for p in plan.solution.squad])


if __name__ == "__main__":
    unittest.main()
