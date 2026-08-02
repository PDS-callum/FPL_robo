"""Tests for vector-team prototype (features, repair, score, smoke)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fpl_robo.graph.schema import SQUAD_COUNTS, SQUAD_SIZE
from fpl_robo.optimize.vector_search import (
    _pool_from_preds,
    _random_valid_squad,
    anneal_squad,
    squad_is_valid,
)
from fpl_robo.predict.vector_features import (
    FEATURE_NAMES,
    attach_vector_ev,
    build_feature_matrix,
    fit_ridge,
    hand_model_from_pool,
)
from fpl_robo.predict.vector_team_score import score_team, select_xi_greedy


def _toy_gw(n_per_pos: dict[str, int] | None = None) -> pd.DataFrame:
    counts = n_per_pos or {"GK": 4, "DEF": 12, "MID": 12, "FWD": 8}
    rows = []
    pid = 1
    team = 1
    for pos, n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "id": pid,
                    "web_name": f"{pos}{i}",
                    "position": pos,
                    "team_id": ((team - 1) % 20) + 1,
                    "price": 4.5 + (i % 6) * 0.5,
                    "gameweek": 5,
                    "rolling_points": float(i % 8),
                    "rolling_minutes": 60.0 + i,
                    "prior_minutes": 200.0,
                    "rolling_expected_goals": 0.1 * (i % 5),
                    "rolling_expected_assists": 0.05 * (i % 4),
                    "rolling_ict_index": float(i),
                    "total_points": float(i % 10),
                    "minutes": 90.0,
                    "n_fixtures": 1,
                }
            )
            pid += 1
            team += 1
    return pd.DataFrame(rows)


class VectorFeatureTests(unittest.TestCase):
    def test_feature_matrix_shape_and_finite(self):
        df = _toy_gw()
        X, meta = build_feature_matrix(df, team_fixtures=None)
        self.assertEqual(X.shape, (len(df), len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(X).all())
        self.assertIn("play_prob", meta.columns)
        self.assertFalse(meta["play_prob"].isna().any())

    def test_xp_anchor_dominates_unfitted(self):
        df = _toy_gw()
        df["xP"] = 5.0
        X, _ = build_feature_matrix(df)
        model = hand_model_from_pool(X)
        self.assertFalse(model.fitted)
        preds = attach_vector_ev(df, model, xp_anchor=df["xP"])
        self.assertTrue((preds["base_points"] - 5.0).abs().max() < 1e-6)


class TeamScoreTests(unittest.TestCase):
    def test_xi_selection_valid_size(self):
        df = _toy_gw()
        model = hand_model_from_pool(build_feature_matrix(df)[0])
        preds = attach_vector_ev(df, model)
        pool = _pool_from_preds(preds)
        import random

        squad = _random_valid_squad(pool, random.Random(0), 100.0)
        self.assertIsNotNone(squad)
        assert squad is not None
        by_id = {int(r["id"]): r for _, r in preds.iterrows()}
        start, bench = select_xi_greedy(squad, by_id)
        self.assertEqual(len(start), 11)
        self.assertEqual(len(start) + len(bench), 15)

    def test_score_rises_when_starter_base_rises(self):
        df = _toy_gw()
        model = hand_model_from_pool(build_feature_matrix(df)[0])
        preds = attach_vector_ev(df, model)
        pool = _pool_from_preds(preds)
        import random

        squad = _random_valid_squad(pool, random.Random(1), 100.0)
        assert squad is not None
        base = score_team(squad, preds, geometry=False)
        # Boost one starter's base_points.
        starter = base.starting_ids[0]
        preds2 = preds.copy()
        mask = preds2["id"] == starter
        preds2.loc[mask, "base_points"] = preds2.loc[mask, "base_points"] + 5.0
        preds2.loc[mask, "ev"] = preds2.loc[mask, "base_points"]
        boosted = score_team(squad, preds2, geometry=False)
        self.assertGreater(boosted.total, base.total)


class SearchRepairTests(unittest.TestCase):
    def test_random_and_anneal_valid(self):
        df = _toy_gw()
        model = hand_model_from_pool(build_feature_matrix(df)[0])
        preds = attach_vector_ev(df, model)
        pool = _pool_from_preds(preds)
        import random

        rng = random.Random(2)
        for _ in range(10):
            squad = _random_valid_squad(pool, rng, 100.0)
            self.assertIsNotNone(squad)
            assert squad is not None
            self.assertTrue(squad_is_valid(squad, pool, budget=100.0))
            self.assertEqual(len(squad), SQUAD_SIZE)
            counts = {pos: 0 for pos in SQUAD_COUNTS}
            for pid in squad:
                counts[str(pool.by_id[pid]["position"])] += 1
            self.assertEqual(counts, SQUAD_COUNTS)

        sol = anneal_squad(preds, budget=100.0, n_iters=200, seed=3)
        self.assertEqual(sol.status, "Optimal")
        self.assertEqual(len(sol.squad), 15)
        self.assertEqual(len(sol.starting_xi), 11)
        self.assertEqual(sum(1 for r in sol.squad if r.get("start")), 11)
        self.assertEqual(sum(1 for r in sol.squad if not r.get("start")), 4)
        self.assertTrue(any(r.get("captain") for r in sol.squad))


if __name__ == "__main__":
    unittest.main()
