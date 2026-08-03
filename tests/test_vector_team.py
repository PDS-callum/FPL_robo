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
        self.assertIn("rolling_minutes", FEATURE_NAMES)
        self.assertIn("set_piece_score", FEATURE_NAMES)
        self.assertIn("club_pos_threat", FEATURE_NAMES)

    def test_xp_anchor_dominates_unfitted(self):
        df = _toy_gw()
        df["xP"] = 5.0
        X, _ = build_feature_matrix(df)
        model = hand_model_from_pool(X)
        self.assertFalse(model.fitted)
        preds = attach_vector_ev(df, model, xp_anchor=df["xP"])
        self.assertTrue((preds["base_points"] - 5.0).abs().max() < 1e-6)
        self.assertTrue((preds["ep_anchor"] - 5.0).abs().max() < 1e-6)

    def test_positive_residual_dampened_by_play_prob(self):
        from fpl_robo.config import VECTOR_TEAM
        from fpl_robo.predict.vector_features import RidgeModel

        n = len(FEATURE_NAMES)
        model = RidgeModel(
            coef=np.array([1.0] + [0.0] * n),
            feature_mean=np.zeros(n),
            feature_std=np.ones(n),
            fitted=True,
            n_train=50,
            source="test",
        )
        # Low rolling minutes → low play_prob; high minutes → ~1.0.
        rows = []
        for pid, mins, name in ((1, 30.0, "low"), (2, 90.0, "hi")):
            rows.append(
                {
                    "id": pid,
                    "web_name": name,
                    "position": "MID",
                    "team_id": pid,
                    "price": 5.0,
                    "gameweek": 1,
                    "rolling_points": 2.0,
                    "rolling_minutes": mins,
                    "prior_minutes": mins * 4,
                    "rolling_expected_goals": 0.1,
                    "rolling_expected_assists": 0.05,
                    "rolling_ict_index": 10.0,
                    "total_points": 10.0,
                    "minutes": mins,
                    "n_fixtures": 1,
                    "xP": 3.0,
                }
            )
        df = pd.DataFrame(rows)
        preds = attach_vector_ev(df, model, xp_anchor=df["xP"])
        scale = float(VECTOR_TEAM.get("residual_scale") or 0.55)
        ref = float(VECTOR_TEAM.get("residual_play_prob_ref") or 0.80)
        low = preds.loc[preds["web_name"] == "low"].iloc[0]
        hi = preds.loc[preds["web_name"] == "hi"].iloc[0]
        self.assertAlmostEqual(float(hi["residual"]), 1.0, places=5)
        self.assertAlmostEqual(float(hi["residual_scaled"]), scale, places=5)
        expected_low = scale * min(1.0, float(low["play_prob"]) / ref)
        self.assertAlmostEqual(float(low["residual_scaled"]), expected_low, places=5)
        self.assertLess(float(low["residual_scaled"]), float(hi["residual_scaled"]))

        # Negative residuals are not haircut by play_prob.
        model_neg = RidgeModel(
            coef=np.array([-1.0] + [0.0] * n),
            feature_mean=np.zeros(n),
            feature_std=np.ones(n),
            fitted=True,
            n_train=50,
            source="test",
        )
        preds_neg = attach_vector_ev(df, model_neg, xp_anchor=df["xP"])
        for _, row in preds_neg.iterrows():
            self.assertAlmostEqual(float(row["residual_scaled"]), -scale, places=5)

    def test_position_residual_ensemble_predicts(self):
        from fpl_robo.chips.backtest import load_merged_gw, prepare_gw_table
        from fpl_robo.config import CHIP_BACKTEST_CACHE
        from fpl_robo.predict.ev_modes import attach_lagged_history
        from fpl_robo.predict.horizon import team_fixtures_from_merged
        from fpl_robo.predict.vector_features import PosRidgeEnsemble, fit_ridge_from_history

        merged = load_merged_gw(season="2025-26", cache_path=CHIP_BACKTEST_CACHE)
        table = attach_lagged_history(prepare_gw_table(merged, season="2025-26"))
        fixtures = team_fixtures_from_merged(table)
        model = fit_ridge_from_history(table, before_gw=20, team_fixtures=fixtures)
        self.assertTrue(isinstance(model, PosRidgeEnsemble) or model.fitted)
        gw = table[table["gameweek"] == 20].copy()
        preds = attach_vector_ev(gw, model, team_fixtures=fixtures)
        self.assertIn("ep_anchor", preds.columns)
        self.assertIn("residual_scaled", preds.columns)
        self.assertTrue(np.isfinite(preds["ev"]).all())


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

    def test_transfer_sanity_rejects_minutes_regression_buy(self):
        """Buendía-style: residual lifts EV but play_prob / EP drop → reject."""
        from fpl_robo.optimize.vector_search import _transfer_sanity_ok
        from fpl_robo.predict.vector_team_score import FastTeamScorer

        df = _toy_gw()
        df["ev"] = 3.0
        df["base_points"] = 3.0
        df["ep_next"] = 3.0
        df["ep_anchor"] = 3.0
        df["play_prob"] = 0.95
        # Owned starter MID with strong minutes.
        owned = []
        for pos, need in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            ids = df.loc[df["position"] == pos, "id"].astype(int).tolist()[:need]
            owned.extend(ids)
        leave = int(df.loc[df["position"] == "MID", "id"].iloc[0])
        # Outside buy target: higher residual EV, worse minutes + EP.
        enter = int(df.loc[(df["position"] == "MID") & (~df["id"].isin(owned)), "id"].iloc[0])
        df.loc[df["id"] == leave, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]] = [
            2.54, 2.54, 2.58, 2.58, 0.96
        ]
        df.loc[df["id"] == enter, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]] = [
            2.63, 2.63, 2.55, 2.55, 0.58
        ]
        scorer = FastTeamScorer(df)
        cand = [enter if i == leave else i for i in owned]
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

    def test_transfer_sanity_allows_residual_sell_for_minutes(self):
        """Reverse Buendía trap: sell residual-inflated low-pp for higher EP starter."""
        from fpl_robo.optimize.vector_search import _transfer_sanity_ok
        from fpl_robo.predict.vector_team_score import FastTeamScorer

        df = _toy_gw()
        df["ev"] = 1.0
        df["base_points"] = 1.0
        df["ep_next"] = 1.0
        df["ep_anchor"] = 1.0
        df["play_prob"] = 0.95
        owned = []
        for pos, need in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            ids = df.loc[df["position"] == pos, "id"].astype(int).tolist()[:need]
            owned.extend(ids)
        for i, did in enumerate(df.loc[df["position"] == "DEF", "id"].astype(int).tolist()[:5]):
            v = 1.2 - 0.05 * i
            df.loc[df["id"] == did, ["base_points", "ev", "ep_next", "ep_anchor"]] = [v, v, v, v]
        for fid in df.loc[df["position"] == "FWD", "id"].astype(int).tolist()[:3]:
            df.loc[df["id"] == fid, ["base_points", "ev", "ep_next", "ep_anchor"]] = [
                3.5, 3.5, 3.5, 3.5
            ]
        for i, gid in enumerate(df.loc[df["position"] == "GK", "id"].astype(int).tolist()[:2]):
            v = 4.0 if i == 0 else 0.5
            df.loc[df["id"] == gid, ["base_points", "ev", "ep_next", "ep_anchor"]] = [v, v, v, v]
        leave = int(df.loc[df["position"] == "MID", "id"].iloc[0])
        enter = int(df.loc[(df["position"] == "MID") & (~df["id"].isin(owned)), "id"].iloc[0])
        owned_mids = [
            int(x) for x in owned if str(df.loc[df["id"] == x, "position"].iloc[0]) == "MID"
        ]
        premiums = [m for m in owned_mids if m != leave][:3]
        bench_mid = [m for m in owned_mids if m != leave and m not in premiums][0]
        for mid_id in premiums:
            df.loc[
                df["id"] == mid_id, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]
            ] = [5.0, 5.0, 5.0, 5.0, 0.95]
        df.loc[
            df["id"] == bench_mid, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]
        ] = [0.4, 0.4, 0.4, 0.4, 0.9]
        df.loc[df["id"] == leave, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]] = [
            2.63, 2.63, 2.55, 2.55, 0.58
        ]
        df.loc[df["id"] == enter, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]] = [
            2.54, 2.54, 2.58, 2.58, 0.96
        ]
        scorer = FastTeamScorer(df)
        cand = [enter if i == leave else i for i in owned]
        self.assertIn(enter, scorer.score(cand, geometry=False).starting_ids)
        self.assertTrue(
            _transfer_sanity_ok(
                owned,
                cand,
                scorer,
                require_buy_starts=True,
                min_transfer_net_ev=0.0,
                min_xi_ev_delta=0.0,
            )
        )

    def test_transfer_sanity_multi_uses_xi_level_gates(self):
        """Multi-xfer plans pass when XI EP/pp improve (paired nets not required)."""
        from fpl_robo.optimize.vector_search import _transfer_sanity_ok
        from fpl_robo.predict.vector_team_score import FastTeamScorer

        df = _toy_gw()
        df["ev"] = 2.0
        df["base_points"] = 2.0
        df["ep_next"] = 2.0
        df["ep_anchor"] = 2.0
        df["play_prob"] = 0.95
        owned = []
        for pos, need in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            ids = df.loc[df["position"] == pos, "id"].astype(int).tolist()[:need]
            owned.extend(ids)
        # Weak DEFs → prefer 3-4-3 so four MIDs start.
        for i, did in enumerate(df.loc[df["position"] == "DEF", "id"].astype(int).tolist()[:5]):
            v = 1.0 - 0.05 * i
            df.loc[df["id"] == did, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
                v, v, v, 0.95
            ]
        for fid in df.loc[df["position"] == "FWD", "id"].astype(int).tolist()[:3]:
            df.loc[df["id"] == fid, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
                3.5, 3.5, 3.5, 0.95
            ]
        for i, gid in enumerate(df.loc[df["position"] == "GK", "id"].astype(int).tolist()[:2]):
            v = 4.0 if i == 0 else 0.5
            df.loc[df["id"] == gid, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
                v, v, v, 0.95
            ]
        mids = [i for i in owned if str(df.loc[df["id"] == i, "position"].iloc[0]) == "MID"]
        leave_a, leave_b = mids[0], mids[1]
        for mid_id in mids[2:]:
            df.loc[df["id"] == mid_id, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
                5.0, 5.0, 5.0, 0.95
            ]
        df.loc[df["id"] == leave_a, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
            2.2, 2.2, 2.2, 0.55
        ]
        df.loc[df["id"] == leave_b, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
            2.1, 2.1, 2.1, 0.50
        ]
        free = df.loc[(df["position"] == "MID") & (~df["id"].isin(owned)), "id"].astype(int).tolist()
        enter_a, enter_b = free[0], free[1]
        df.loc[df["id"] == enter_a, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
            4.2, 4.2, 4.2, 0.95
        ]
        df.loc[df["id"] == enter_b, ["base_points", "ev", "ep_anchor", "play_prob"]] = [
            4.3, 4.3, 4.3, 0.95
        ]
        scorer = FastTeamScorer(df)
        cand = [
            enter_a if i == leave_a else enter_b if i == leave_b else i for i in owned
        ]
        starts = set(scorer.score(cand, geometry=False).starting_ids)
        self.assertTrue({enter_a, enter_b}.issubset(starts))
        self.assertTrue(
            _transfer_sanity_ok(
                owned,
                cand,
                scorer,
                require_buy_starts=True,
                min_transfer_net_ev=0.0,
                min_xi_ev_delta=0.0,
            )
        )

    def test_effective_ft_hits_uses_sa_explore_cap(self):
        from fpl_robo.optimize.live_vector import _effective_ft_hits

        ft, hits = _effective_ft_hits(15, 2, unlimited=True, owned_risk_count=0)
        self.assertEqual(hits, 0)
        self.assertGreaterEqual(ft, 4)
        ft2, _ = _effective_ft_hits(15, 2, unlimited=True, owned_risk_count=3)
        self.assertGreaterEqual(ft2, 3)
        self.assertLessEqual(ft2, 5)

    def test_minutes_repair_pass_sells_low_pp_starter(self):
        """Deterministic post-SA pass undoes residual-trap ownership."""
        from fpl_robo.optimize.vector_search import (
            _minutes_repair_pass,
            _pool_from_preds,
            _transfer_sanity_ok,
            squad_is_valid,
        )
        from fpl_robo.predict.vector_team_score import FastTeamScorer

        df = _toy_gw()
        # placeholder — existing body continues below
        df["ev"] = 1.0
        df["base_points"] = 1.0
        df["ep_next"] = 1.0
        df["ep_anchor"] = 1.0
        df["play_prob"] = 0.95
        owned = []
        for pos, need in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            ids = df.loc[df["position"] == pos, "id"].astype(int).tolist()[:need]
            owned.extend(ids)
        for i, did in enumerate(df.loc[df["position"] == "DEF", "id"].astype(int).tolist()[:5]):
            v = 1.2 - 0.05 * i
            df.loc[df["id"] == did, ["base_points", "ev", "ep_next", "ep_anchor"]] = [v, v, v, v]
        for fid in df.loc[df["position"] == "FWD", "id"].astype(int).tolist()[:3]:
            df.loc[df["id"] == fid, ["base_points", "ev", "ep_next", "ep_anchor"]] = [
                3.5,
                3.5,
                3.5,
                3.5,
            ]
        for i, gid in enumerate(df.loc[df["position"] == "GK", "id"].astype(int).tolist()[:2]):
            v = 4.0 if i == 0 else 0.5
            df.loc[df["id"] == gid, ["base_points", "ev", "ep_next", "ep_anchor"]] = [v, v, v, v]
        leave = int(df.loc[df["position"] == "MID", "id"].iloc[0])
        enter = int(df.loc[(df["position"] == "MID") & (~df["id"].isin(owned)), "id"].iloc[0])
        owned_mids = [
            int(x)
            for x in owned
            if str(df.loc[df["id"] == x, "position"].iloc[0]) == "MID"
        ]
        premiums = [m for m in owned_mids if m != leave][:3]
        bench_mid = [m for m in owned_mids if m != leave and m not in premiums][0]
        for mid_id in premiums:
            df.loc[
                df["id"] == mid_id, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]
            ] = [5.0, 5.0, 5.0, 5.0, 0.95]
        df.loc[
            df["id"] == bench_mid, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]
        ] = [0.4, 0.4, 0.4, 0.4, 0.9]
        df.loc[df["id"] == leave, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]] = [
            2.63,
            2.63,
            2.55,
            2.55,
            0.58,
        ]
        df.loc[df["id"] == enter, ["base_points", "ev", "ep_next", "ep_anchor", "play_prob"]] = [
            2.54,
            2.54,
            2.58,
            2.58,
            0.96,
        ]
        pool = _pool_from_preds(df)
        scorer = FastTeamScorer(df)

        def feasible(squad):
            return squad_is_valid(squad, pool, budget=1e9)

        def hits_for(_squad):
            return 0

        def obj_fn(squad):
            return scorer.score(squad).total

        repaired, changed = _minutes_repair_pass(
            owned,
            list(owned),
            pool=pool,
            scorer=scorer,
            free_left=2,
            max_hits=0,
            hits_for=hits_for,
            feasible=feasible,
            obj_fn=obj_fn,
            require_starts=True,
            min_net=0.0,
            min_xi=0.0,
        )
        self.assertTrue(changed)
        self.assertIn(enter, repaired)
        self.assertNotIn(leave, repaired)
        self.assertTrue(
            _transfer_sanity_ok(
                owned,
                repaired,
                scorer,
                require_buy_starts=True,
                min_transfer_net_ev=0.0,
                min_xi_ev_delta=0.0,
            )
        )

    def test_consensus_holds_without_majority(self):
        from fpl_robo.optimize.vector_search import _consensus_squad

        owned = list(range(1, 16))
        a = list(owned)
        b = [99 if i == 1 else i for i in owned]
        c = [88 if i == 1 else i for i in owned]
        # Tiny gains under margin → still require majority; none → hold.
        squad, obj, meta = _consensus_squad(
            [(a, 1.0), (b, 1.005), (c, 1.008)],
            owned=owned,
            min_agree=2,
            owned_obj=1.0,
            obj_margin=0.02,
        )
        self.assertEqual(squad, owned)
        self.assertEqual(meta["mode"], "hold_no_consensus")
        self.assertEqual(obj, 1.0)

    def test_consensus_best_obj_beats_majority_hold(self):
        """One seed clear upgrade wins even if two seeds hold."""
        from fpl_robo.optimize.vector_search import _consensus_squad

        owned = list(range(1, 16))
        upgrade = [99 if i == 1 else i for i in owned]
        squad, obj, meta = _consensus_squad(
            [(owned, 1.0), (owned, 1.0), (upgrade, 1.05)],
            owned=owned,
            min_agree=2,
            owned_obj=1.0,
            obj_margin=0.02,
        )
        self.assertEqual(set(squad), set(upgrade))
        self.assertEqual(meta["mode"], "best_obj")
        self.assertAlmostEqual(obj, 1.05)

    def test_consensus_picks_majority(self):
        from fpl_robo.optimize.vector_search import _consensus_squad

        owned = list(range(1, 16))
        winner = [99 if i == 1 else i for i in owned]
        # Under margin vs hold: majority among near-ties still picks the repeated squad.
        squad, obj, meta = _consensus_squad(
            [(winner, 1.01), (winner, 1.009), (owned, 1.0)],
            owned=owned,
            min_agree=2,
            owned_obj=1.0,
            obj_margin=0.02,
        )
        self.assertEqual(set(squad), set(winner))
        self.assertEqual(meta["mode"], "consensus")
        self.assertEqual(obj, 1.01)

    def test_sa_burst_active_schedule(self):
        from fpl_robo.optimize.vector_search import _sa_burst_active

        self.assertEqual(_sa_burst_active(0, every=300, length=40), (True, 1.0))
        self.assertTrue(_sa_burst_active(20, every=300, length=40)[0])
        self.assertEqual(_sa_burst_active(40, every=300, length=40), (False, 0.0))
        self.assertTrue(_sa_burst_active(300, every=300, length=40)[0])

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


class RebuildPlanTests(unittest.TestCase):
    def test_transfer_plan_from_rebuild_pairs_positions(self):
        from fpl_robo.optimize.live_vector import transfer_plan_from_rebuild
        from fpl_robo.optimize.vector_search import anneal_squad
        from fpl_robo.predict.vector_features import (
            attach_vector_ev,
            build_feature_matrix,
            hand_model_from_pool,
        )

        df = _toy_gw()
        df["ev"] = 3.0
        df["xP"] = 3.0
        df["status"] = "a"
        df["cover_eligible"] = True
        df["buy_eligible"] = True
        model = hand_model_from_pool(build_feature_matrix(df)[0])
        preds = attach_vector_ev(df, model, xp_anchor=df["ev"])
        sol = anneal_squad(preds, budget=100.0, n_iters=120, seed=1)
        owned = [int(p["id"]) for p in sol.squad]
        # Force a different rebuild by boosting a non-owned mid.
        outs = preds[~preds["id"].isin(owned) & (preds["position"] == "MID")]
        if outs.empty:
            return
        boost_id = int(outs.iloc[0]["id"])
        preds2 = preds.copy()
        preds2.loc[preds2["id"] == boost_id, "base_points"] = 20.0
        preds2.loc[preds2["id"] == boost_id, "ev"] = 20.0
        sol2 = anneal_squad(preds2, budget=100.0, n_iters=200, seed=2)
        plan = transfer_plan_from_rebuild(
            sol2,
            current_ids=owned,
            bank=0.0,
            predictions=preds2,
            chip="freehit",
        )
        self.assertEqual(plan.status, "Optimal")
        self.assertEqual(plan.hits, 0)
        self.assertTrue(plan.meta and plan.meta.get("rebuild"))
        self.assertEqual(len(plan.solution.squad), 15)


class HorizonBundleTests(unittest.TestCase):
    def test_dgw_ahead_raises_xi_score(self):
        from fpl_robo.predict.horizon import horizon_bundle_factor
        from fpl_robo.predict.vector_team_score import score_team

        df = _toy_gw()
        df["ev"] = 4.0
        df["base_points"] = 4.0
        df["play_prob"] = 0.9
        df["run_ease"] = 0.5
        df["horizon_bundle"] = 1.0
        owned = list(range(1, 16))
        base = score_team(owned, df, geometry=True)
        df2 = df.copy()
        df2["horizon_bundle"] = 1.45  # upcoming DGW pull
        boosted = score_team(owned, df2, geometry=True)
        self.assertGreater(boosted.total, base.total)
        self.assertGreater(boosted.horizon_bundle, 0.0)

    def test_horizon_bundle_asym_ignores_future_blank(self):
        from fpl_robo.predict.horizon import horizon_bundle_factor

        fixtures = {
            1: {
                1: {"n_fixtures": 1.0, "best_fdr": 3.0},
                2: {"n_fixtures": 0.0, "best_fdr": None},  # blank
                3: {"n_fixtures": 1.0, "best_fdr": 3.0},
            }
        }
        # Asym → blanks don't pull below 1.0.
        self.assertAlmostEqual(
            horizon_bundle_factor(fixtures, 1, 1, horizon=3), 1.0, places=5
        )
        fixtures[1][2] = {"n_fixtures": 2.0, "best_fdr": 3.0}  # DGW
        self.assertGreater(horizon_bundle_factor(fixtures, 1, 1, horizon=3), 1.0)


if __name__ == "__main__":
    unittest.main()
