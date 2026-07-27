import unittest

import pandas as pd

from fpl_robo.optimize.single_gw import optimise_single_gw


class OptimizerSanityTests(unittest.TestCase):
    def _pool(self) -> pd.DataFrame:
        rows = []
        player_id = 1
        structure = [("GK", 20), ("DEF", 45), ("MID", 45), ("FWD", 35)]
        for position, count in structure:
            for idx in range(count):
                rows.append(
                    {
                        "id": player_id,
                        "web_name": f"P{player_id}",
                        "team_id": (idx % 20) + 1,
                        "position": position,
                        "price": 4.0 + (idx % 8) * 0.5,
                        "ev": 2.0 + (idx % 10) * 0.2,
                        "status": "a",
                        "chance_of_playing_next_round": 100,
                        "cover_eligible": True,
                        "gameweek": 1,
                    }
                )
                player_id += 1
        return pd.DataFrame(rows)

    def test_optimizer_returns_legal_squad(self):
        result = optimise_single_gw(self._pool(), budget=100.0, min_play_chance=50)
        self.assertEqual(result.status, "Optimal")
        self.assertEqual(len(result.squad), 15)
        self.assertEqual(len(result.starting_xi), 11)
        self.assertEqual(len(result.bench), 4)
        self.assertIsNotNone(result.captain_id)

    def test_optimizer_excludes_non_cover_players(self):
        df = self._pool()
        # Mark a chunk of cheap low-EV players as dead fodder in every position.
        df["cover_eligible"] = True
        for position in ("GK", "DEF", "MID", "FWD"):
            pos_idx = df.index[df["position"] == position]
            # Keep most of the pool cover-eligible; blacklist the last 5.
            df.loc[pos_idx[-5:], "cover_eligible"] = False
        result = optimise_single_gw(df, budget=100.0, require_cover_eligible=True)
        self.assertEqual(result.status, "Optimal")
        selected = {r["id"] for r in result.squad}
        non_cover_ids = set(df.loc[~df["cover_eligible"], "id"].tolist())
        self.assertTrue(selected.isdisjoint(non_cover_ids))


if __name__ == "__main__":
    unittest.main()
