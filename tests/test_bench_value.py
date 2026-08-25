"""Tests for expensive outfield-bench price tax."""

from __future__ import annotations

import unittest

import pandas as pd

from fpl_robo.optimize.bench_value import expensive_bench_penalty_total
from fpl_robo.predict.vector_team_score import FastTeamScorer


def _snap_row(
    pid: int,
    *,
    position: str,
    price: float,
    ev: float,
    team_id: int = 1,
) -> dict:
    return {
        "id": pid,
        "position": position,
        "team_id": team_id,
        "price": price,
        "ev": ev,
        "base_points": ev,
        "play_prob": 0.9,
        "run_ease": 0.5,
        "v_rolling_xgi": 0.1,
        "horizon_bundle": 1.0,
        "ep_next": ev,
        "form": 3.0,
    }


class ExpensiveBenchPenaltyTests(unittest.TestCase):
    def test_penalty_scales_with_outfield_excess(self):
        cheap = expensive_bench_penalty_total(
            bench_rows=[("DEF", 4.5), ("MID", 4.5), ("FWD", 4.5), ("GK", 4.0)],
            config={"expensive_bench_penalty": 0.40, "expensive_bench_free": 4.5},
        )
        premium = expensive_bench_penalty_total(
            bench_rows=[("DEF", 4.5), ("MID", 4.5), ("FWD", 7.5), ("GK", 4.0)],
            config={"expensive_bench_penalty": 0.40, "expensive_bench_free": 4.5},
        )
        self.assertEqual(cheap, 0.0)
        self.assertAlmostEqual(premium, 0.40 * 3.0)  # £7.5 − £4.5

    def test_gk_not_taxed(self):
        pen = expensive_bench_penalty_total(
            bench_rows=[("GK", 5.5)],
            config={"expensive_bench_penalty": 0.40, "expensive_bench_free": 4.5},
        )
        self.assertEqual(pen, 0.0)

    def test_scorer_prefers_cheap_outfield_bench(self):
        # Valid 15: 2 GK, 5 DEF, 5 MID, 3 FWD. Same EV; only bench FWD price differs.
        rows = []
        pid = 1
        for i in range(2):
            rows.append(_snap_row(pid, position="GK", price=4.5, ev=2.0, team_id=1 + i))
            pid += 1
        for i in range(5):
            rows.append(
                _snap_row(pid, position="DEF", price=4.5, ev=3.0 - 0.01 * i, team_id=1 + i)
            )
            pid += 1
        for i in range(5):
            rows.append(
                _snap_row(pid, position="MID", price=6.0, ev=5.0 - 0.01 * i, team_id=1 + i)
            )
            pid += 1
        # Three FWDs: two strong starters + one weak (benched).
        rows.append(_snap_row(pid, position="FWD", price=8.0, ev=6.0, team_id=1))
        pid += 1
        rows.append(_snap_row(pid, position="FWD", price=7.5, ev=5.5, team_id=2))
        pid += 1
        cheap_bench_fwd = pid
        rows.append(_snap_row(cheap_bench_fwd, position="FWD", price=4.5, ev=1.0, team_id=3))
        cheap_df = pd.DataFrame(rows)
        premium_rows = [dict(r) for r in rows]
        for r in premium_rows:
            if r["id"] == cheap_bench_fwd:
                r["price"] = 7.5
        premium_df = pd.DataFrame(premium_rows)

        ids = [int(r["id"]) for r in rows]
        cheap = FastTeamScorer(cheap_df).score(ids, geometry=False)
        premium = FastTeamScorer(premium_df).score(ids, geometry=False)
        self.assertIn(cheap_bench_fwd, cheap.bench_ids)
        self.assertIn(cheap_bench_fwd, premium.bench_ids)
        self.assertLess(premium.total, cheap.total)
        self.assertAlmostEqual(premium.bench_price, -0.40 * 3.0, places=5)
        self.assertEqual(cheap.bench_price, 0.0)


if __name__ == "__main__":
    unittest.main()
