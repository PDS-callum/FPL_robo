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

    def test_filter_drops_unavailable_non_owned(self):
        from fpl_robo.optimize.vector_search import filter_vector_transfer_pool

        df = _toy_gw()
        df["status"] = "a"
        df["cover_eligible"] = True
        df["buy_eligible"] = True
        owned = list(range(1, 16))
        df.loc[df["id"] == 20, "status"] = "u"
        out = filter_vector_transfer_pool(
            df, current_ids=owned, require_cover_eligible=True, exclude_unavailable=True
        )
        ids = set(int(x) for x in out["id"])
        self.assertNotIn(20, ids)
        self.assertTrue(set(owned).issubset(ids))

    def test_transfer_sanity_rejects_bench_buy(self):
        from fpl_robo.optimize.vector_search import (
            _transfer_sanity_ok,
            anneal_transfers,
        )
        from fpl_robo.predict.vector_team_score import FastTeamScorer

        df = _toy_gw()
        # Give clear EV separation so XI selection is deterministic.
        df["ev"] = 1.0
        df["xP"] = 1.0
        for pos, base in (("GK", 3.0), ("DEF", 2.5), ("MID", 4.0), ("FWD", 3.5)):
            mask = df["position"] == pos
            ranks = df.loc[mask].index
            for n, idx in enumerate(ranks):
                df.at[idx, "ev"] = base + 0.2 * (len(ranks) - n)
                df.at[idx, "xP"] = df.at[idx, "ev"]
        model = hand_model_from_pool(build_feature_matrix(df)[0])
        preds = attach_vector_ev(df, model, xp_anchor=df["ev"])
        sol = anneal_squad(preds, budget=100.0, n_iters=150, seed=1)
        owned = [int(p["id"]) for p in sol.squad]
        owned_def = next(int(p["id"]) for p in sol.starting_xi if p["position"] == "DEF")
        defs = preds[(preds["position"] == "DEF") & (~preds["id"].isin(owned))]
        weak = int(defs.sort_values("base_points").iloc[0]["id"])
        preds.loc[preds["id"] == weak, "base_points"] = 0.01
        preds.loc[preds["id"] == weak, "ev"] = 0.01
        scorer = FastTeamScorer(preds)
        cand = [weak if i == owned_def else i for i in owned]
        self.assertFalse(
            _transfer_sanity_ok(
                owned,
                cand,
                scorer,
                require_buy_starts=True,
                min_transfer_net_ev=0.0,
                min_xi_ev_delta=0.0,
            )
        )
        plan = anneal_transfers(
            preds,
            current_ids=owned,
            bank=0.0,
            free_transfers=2,
            max_hits=0,
            n_iters=200,
            seed=5,
        )
        self.assertEqual(plan.status, "Optimal")
        if plan.transfers_made:
            start_ids = {int(p["id"]) for p in plan.solution.starting_xi}
            for t in plan.transfers:
                self.assertIn(int(t["element_in"]), start_ids)
        self.assertIsInstance(plan.meta, dict)
        self.assertIn("consensus", plan.meta)


class ConsensusAndFloorTests(unittest.TestCase):
    def test_consensus_holds_without_majority(self):
        from fpl_robo.optimize.vector_search import _consensus_squad

        owned = list(range(1, 16))
        a = list(owned)
        b = [99 if i == 1 else i for i in owned]
        c = [88 if i == 1 else i for i in owned]
        squad, obj, meta = _consensus_squad(
            [(a, 1.0), (b, 2.0), (c, 3.0)],
            owned=owned,
            min_agree=2,
            owned_obj=1.0,
        )
        self.assertEqual(squad, owned)
        self.assertEqual(meta["mode"], "hold_no_consensus")
        self.assertEqual(obj, 1.0)

    def test_consensus_picks_majority(self):
        from fpl_robo.optimize.vector_search import _consensus_squad

        owned = list(range(1, 16))
        winner = [99 if i == 1 else i for i in owned]
        squad, obj, meta = _consensus_squad(
            [(winner, 5.0), (winner, 4.5), (owned, 1.0)],
            owned=owned,
            min_agree=2,
            owned_obj=1.0,
        )
        self.assertEqual(set(squad), set(winner))
        self.assertEqual(meta["mode"], "consensus")
        self.assertEqual(obj, 5.0)

    def test_attack_floor_rejects_weak_fwd_swap(self):
        from fpl_robo.optimize.vector_search import _attack_floor_ok, _pool_from_preds

        df = _toy_gw({"GK": 2, "DEF": 5, "MID": 5, "FWD": 6})
        df["status"] = "a"
        df["form"] = 0.0
        df["is_focal_attacker"] = False
        df["ep_next"] = 1.0
        # Make three quality FWDs and three weak ones.
        fwds = df[df["position"] == "FWD"].index.tolist()
        for n, idx in enumerate(fwds):
            if n < 3:
                df.at[idx, "ep_next"] = 6.0
                df.at[idx, "form"] = 6.0
                df.at[idx, "rolling_minutes"] = 80.0
                df.at[idx, "is_focal_attacker"] = True
            else:
                df.at[idx, "ep_next"] = 0.5
                df.at[idx, "form"] = 0.1
                df.at[idx, "rolling_minutes"] = 10.0
        pool = _pool_from_preds(df)
        quality = [int(df.at[i, "id"]) for i in fwds[:3]]
        weak = [int(df.at[i, "id"]) for i in fwds[3:]]
        # Build a 15-id squad with 3 quality FWDs (pad with other positions).
        others = [int(x) for x in df[df["position"] != "FWD"]["id"].tolist()][:12]
        strong = others + quality
        self.assertTrue(_attack_floor_ok(strong, pool, current=strong))
        weak_squad = others + quality[:2] + [weak[0]]
        self.assertFalse(_attack_floor_ok(weak_squad, pool, current=strong))

    def test_price_aware_prefers_keeping_riser(self):
        from fpl_robo.optimize.vector_search import _pool_from_preds, _price_aware_term

        df = _toy_gw()
        df["cost_change_event"] = 0
        owned = list(range(1, 16))
        df.loc[df["id"] == 1, "cost_change_event"] = 1
        pool = _pool_from_preds(df)
        keep = _price_aware_term(owned, owned, pool, capital=100.0)
        alt = int(df[~df["id"].isin(owned)].iloc[0]["id"])
        sold = [alt if i == 1 else i for i in owned]
        sell = _price_aware_term(owned, sold, pool, capital=100.0)
        self.assertGreater(keep, sell)

    def test_filter_blocks_news_risk_and_low_chance(self):
        from fpl_robo.optimize.vector_search import filter_vector_transfer_pool

        df = _toy_gw()
        df["status"] = "a"
        df["cover_eligible"] = True
        df["buy_eligible"] = True
        df["chance_of_playing_next_round"] = 100.0
        df["news_risk"] = 0.0
        df["signal_news_mult"] = 1.0
        owned = list(range(1, 16))
        df.loc[df["id"] == 20, "chance_of_playing_next_round"] = 25.0
        df.loc[df["id"] == 21, "news_risk"] = 0.20
        df.loc[df["id"] == 22, "signal_news_mult"] = 0.40
        df.loc[df["id"] == 23, "buy_eligible"] = False
        out = filter_vector_transfer_pool(
            df,
            current_ids=owned,
            enforce_buy_eligible=True,
            min_play_chance=50,
            max_news_risk=0.08,
            min_signal_news_mult=0.70,
        )
        ids = set(int(x) for x in out["id"])
        self.assertNotIn(20, ids)
        self.assertNotIn(21, ids)
        self.assertNotIn(22, ids)
        self.assertNotIn(23, ids)
        self.assertTrue(set(owned).issubset(ids))


if __name__ == "__main__":
    unittest.main()
