"""Tests for v13 independent XI optimisations."""

from __future__ import annotations

import unittest

from fpl_robo.config import CAPTAINCY, KG_ENRICH, OPTIMISER_TIEBREAK, SIGNAL_INGEST
from fpl_robo.optimize.bench_order import order_bench_for_autosub, order_squad_for_fpl
from fpl_robo.optimize.captaincy import captain_score, pick_vice_captain_id, play_factor_for_captain
from fpl_robo.predict.phase import phase_kg_enrich
from fpl_robo.signals.minutes_policy import club_rank_play_prob, mid_absolute_ep_play_prob


class BenchOrderTests(unittest.TestCase):
    def test_outfield_before_gk_and_by_ev(self):
        bench = [
            {"id": 1, "web_name": "Dubravka", "position": "GK", "ev": 0.8},
            {"id": 2, "web_name": "Dalot", "position": "DEF", "ev": 1.8, "cover_eligible": True},
            {"id": 3, "web_name": "Iwobi", "position": "MID", "ev": 2.25, "cover_eligible": True},
            {"id": 4, "web_name": "N.Williams", "position": "DEF", "ev": 1.81, "cover_eligible": True},
        ]
        ordered = order_bench_for_autosub(bench)
        names = [r["web_name"] for r in ordered]
        self.assertEqual(names[0], "Iwobi")
        self.assertEqual(names[-1], "Dubravka")
        self.assertLess(names.index("Iwobi"), names.index("Dalot"))

    def test_squad_starters_formation_then_autosub_bench(self):
        rows = [
            {"id": 1, "web_name": "Raya", "position": "GK", "ev": 4.0, "start": True},
            {"id": 2, "web_name": "DEF1", "position": "DEF", "ev": 2.0, "start": True},
            {"id": 3, "web_name": "MID1", "position": "MID", "ev": 3.0, "start": True},
            {"id": 4, "web_name": "Iwobi", "position": "MID", "ev": 2.2, "start": False},
            {"id": 5, "web_name": "Dalot", "position": "DEF", "ev": 1.8, "start": False},
            {"id": 6, "web_name": "BK", "position": "GK", "ev": 0.5, "start": False},
        ]
        # Pad to make function happy for partial lists — order_squad only sorts given rows.
        ordered = order_squad_for_fpl(rows)
        starts = [r for r in ordered if r["start"]]
        bench = [r for r in ordered if not r["start"]]
        self.assertEqual(starts[0]["position"], "GK")
        self.assertEqual(bench[0]["web_name"], "Iwobi")
        self.assertEqual(bench[-1]["web_name"], "BK")


class CaptaincyTests(unittest.TestCase):
    def test_doubt_and_lineup_haircut(self):
        base = {"ev": 5.0, "status": "a", "chance_of_playing_next_round": 100}
        self.assertAlmostEqual(captain_score(base), 5.0, places=3)
        doubt = {**base, "status": "d"}
        self.assertLess(captain_score(doubt), 5.0)
        bench_role = {
            **base,
            "lineup_role": "bench",
            "lineup_confidence": 0.9,
        }
        self.assertLess(captain_score(bench_role), 5.0)

    def test_vice_prefers_other_club_within_tolerance(self):
        rows = [
            {"id": 1, "web_name": "Cap", "position": "MID", "ev": 5.0, "start": True, "team_id": 10},
            {"id": 2, "web_name": "SameClub", "position": "MID", "ev": 4.5, "start": True, "team_id": 10},
            {"id": 3, "web_name": "Other", "position": "MID", "ev": 4.4, "start": True, "team_id": 20},
        ]
        # SameClub best raw; Other within 0.15 → prefer Other.
        self.assertEqual(CAPTAINCY.get("vice_other_club_tolerance"), 0.15)
        vid = pick_vice_captain_id(rows, captain_id=1)
        self.assertEqual(vid, 3)

    def test_play_factor_floor(self):
        row = {"ev": 4.0, "status": "i", "chance_of_playing_next_round": 0}
        self.assertGreaterEqual(play_factor_for_captain(row), float(CAPTAINCY["play_factor_floor"]))


class MidDepthTests(unittest.TestCase):
    def test_mid_rank2_not_blanket_near(self):
        p = club_rank_play_prob(rank=2, position="MID", ep_gap_to_leader=0.5, ep_next=2.3)
        self.assertLess(p, 0.96)
        self.assertAlmostEqual(p, float(SIGNAL_INGEST["depth_mid_rank2"]), places=3)

    def test_mid_rank2_near_peer_bump(self):
        p = club_rank_play_prob(rank=2, position="MID", ep_gap_to_leader=0.05, ep_next=2.5)
        self.assertGreaterEqual(p, float(SIGNAL_INGEST["depth_mid_near"]))

    def test_mid_absolute_ep_ladder(self):
        self.assertEqual(mid_absolute_ep_play_prob(2.6), 1.0)
        self.assertLess(mid_absolute_ep_play_prob(1.5), mid_absolute_ep_play_prob(2.1))


class ClubSoftFeasibilityTests(unittest.TestCase):
    def test_three_of_club_remains_feasible(self):
        import pulp
        from fpl_robo.config import OPTIMISER_TIEBREAK
        from fpl_robo.optimize.club_soft import add_club_concentration_penalty

        ids = [1, 2, 3, 4]
        team = {1: 10, 2: 10, 3: 10, 4: 20}
        prob = pulp.LpProblem("club_soft_feas", pulp.LpMaximize)
        in_squad = pulp.LpVariable.dicts("s", ids, cat="Binary")
        prob += pulp.lpSum(in_squad[i] for i in ids) == 3
        prob += in_squad[1] + in_squad[2] + in_squad[3] == 3
        term = add_club_concentration_penalty(
            prob,
            in_squad=in_squad,
            ids=ids,
            team=team,
            cfg=OPTIMISER_TIEBREAK,
            prefix="feas",
        )
        prob += pulp.lpSum(i * in_squad[i] for i in ids) + term
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pulp.LpStatus[status], "Optimal")


class PhaseAndConfigTests(unittest.TestCase):
    def test_feature_version(self):
        self.assertEqual(KG_ENRICH.get("feature_version"), "v13.1-preseason-nocd")

    def test_preseason_form_blend_higher(self):
        with phase_kg_enrich(preseason=True) as pre:
            self.assertGreaterEqual(float(pre["form_blend"]), 0.28)
            self.assertGreaterEqual(float(pre["xgi90_blend"]), 0.16)
        with phase_kg_enrich(preseason=False) as inn:
            self.assertLessEqual(float(inn["form_blend"]), 0.22)
            self.assertGreater(float(inn["cs_boost_scale"]), 0.0)
        # Module defaults restored after context (preseason-safe DEF; base form blend).
        self.assertAlmostEqual(float(KG_ENRICH["form_blend"]), 0.22, places=3)
        self.assertAlmostEqual(float(KG_ENRICH["xgi90_blend"]), 0.12, places=3)

    def test_tiebreak_value_and_club_pens(self):
        self.assertGreater(float(OPTIMISER_TIEBREAK.get("value_weight") or 0), 0)
        self.assertGreater(float(OPTIMISER_TIEBREAK.get("club_triple_penalty") or 0), 0)


if __name__ == "__main__":
    unittest.main()
