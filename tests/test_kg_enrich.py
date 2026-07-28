"""Tests for knowledge-graph enrichment and EV knock-on effects."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.graph.enrich import (
    base_ev_with_form_blend,
    context_multiplier,
    enrich_knowledge_graph,
    xgi_ev_boost,
)
from fpl_robo.graph.schema import EdgeType, NodeType
from fpl_robo.predict.baseline import BaselineEVModel


def _mini_bootstrap() -> dict:
    return {
        "teams": [
            {
                "id": 1,
                "name": "Arsenal",
                "short_name": "ARS",
                "strength": 4,
                "strength_attack_home": 1300,
                "strength_attack_away": 1250,
                "strength_defence_home": 1280,
                "strength_defence_away": 1220,
            },
            {
                "id": 2,
                "name": "Burnley",
                "short_name": "BUR",
                "strength": 2,
                "strength_attack_home": 1050,
                "strength_attack_away": 1000,
                "strength_defence_home": 1020,
                "strength_defence_away": 980,
            },
        ],
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2026-08-15T17:30:00Z",
                "finished": False,
                "is_current": False,
                "is_next": True,
                "average_entry_score": None,
            }
        ],
        "elements": [
            {
                "id": 10,
                "web_name": "Saka",
                "first_name": "Bukayo",
                "second_name": "Saka",
                "team": 1,
                "element_type": 3,
                "now_cost": 100,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "form": "6.0",
                "points_per_game": "5.5",
                "total_points": 20,
                "minutes": 270,
                "ep_next": "5.5",
                "ep_this": "0",
                "ict_index": "40.0",
                "expected_goals": "2.0",
                "expected_assists": "1.5",
                "selected_by_percent": "40.0",
                "cost_change_event": 0,
                "cost_change_start": 0,
                "news": "",
            },
            {
                "id": 20,
                "web_name": "Foster",
                "first_name": "Lyle",
                "second_name": "Foster",
                "team": 2,
                "element_type": 4,
                "now_cost": 55,
                "status": "a",
                "chance_of_playing_next_round": None,
                "form": "2.0",
                "points_per_game": "2.0",
                "total_points": 5,
                "minutes": 90,
                "ep_next": "2.5",
                "ep_this": "0",
                "ict_index": "8.0",
                "expected_goals": "0.4",
                "expected_assists": "0.1",
                "selected_by_percent": "1.0",
                "cost_change_event": 0,
                "cost_change_start": 0,
                "news": "Knock - 75% chance of playing",
            },
        ],
    }


def _mini_fixtures() -> list[dict]:
    return [
        {
            "id": 100,
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "team_h_difficulty": 2,
            "team_a_difficulty": 4,
            "finished": False,
            "kickoff_time": "2026-08-16T14:00:00Z",
        }
    ]


class KnowledgeGraphEnrichTests(unittest.TestCase):
    def test_enrich_attaches_player_context(self):
        kg = build_knowledge_graph(bootstrap=_mini_bootstrap(), fixtures=_mini_fixtures())
        # builder already enriches; force again for explicit gw
        stats = enrich_knowledge_graph(kg, gameweek=1)
        self.assertGreater(stats["players"], 0)
        saka = next(p for p in kg.players() if p["id"] == 10)
        self.assertTrue(saka.get("kg_enriched"))
        self.assertEqual(saka.get("next_n_fixtures"), 1)
        self.assertIsNotNone(saka.get("next_matchup_factor"))
        self.assertIsNotNone(saka.get("next_context_factor"))
        self.assertGreater(float(saka.get("xgi") or 0), 0)

    def test_faces_edges_get_strength_fields(self):
        kg = build_knowledge_graph(bootstrap=_mini_bootstrap(), fixtures=_mini_fixtures())
        found = False
        for u, v, _, data in kg.g.edges(keys=True, data=True):
            if data.get("edge_type") != EdgeType.FACES.value:
                continue
            if u[0] != NodeType.TEAM.value or v[0] != NodeType.TEAM.value:
                continue
            if data.get("gameweek") != 1:
                continue
            self.assertIn("opp_defence", data)
            self.assertIn("matchup_factor", data)
            found = True
        self.assertTrue(found)

    def test_context_multiplier_uses_enrichment(self):
        player = {
            "kg_enriched": True,
            "next_context_factor": 1.1,
            "next_fixture_scale": 1.0,
            "fixture_congestion": 0,
            "news_risk": 0.0,
            "role_stability": 1.0,
            "opp_pressure": 1.0,
            "market_momentum": 0.0,
            "price_momentum": 0.0,
            "ict_share": 0.1,
            "position": "MID",
        }
        mult, meta = context_multiplier(player)
        self.assertAlmostEqual(mult, 1.1, places=3)
        self.assertIn("context", meta)

    def test_form_blend_and_xgi_boost(self):
        player = {
            "rolling_points": 7.0,
            "history_attached": True,
            "xgi": 5.0,
            "xgi_per_gw": 0.5,
            "minutes": 900,
        }
        blended = base_ev_with_form_blend(player, ep_next=5.0, form=7.0)
        self.assertNotEqual(blended, 5.0)
        # Preseason form=0 must not haircut EP.
        self.assertEqual(base_ev_with_form_blend({"form": 0.0}, ep_next=5.0, form=0.0), 5.0)
        self.assertGreater(
            xgi_ev_boost({"xgi_per_gw": 0.8, "ict_share": 0.2, "minutes": 900}),
            0.0,
        )

    def test_tiny_minutes_xgi90_cannot_inflate_ev(self):
        """Injury cameo (2 mins, absurd xGI90) must not beat official EP."""
        cameo = {
            "position": "MID",
            "minutes": 2,
            "xgi90": 3.6,
            "expected_goal_involvements_per_90": 3.6,
            "xgi90_raw": 3.6,
        }
        # Blend must ignore the rate — stay at ep_next.
        self.assertEqual(base_ev_with_form_blend(cameo, ep_next=1.5, form=0.0), 1.5)
        self.assertEqual(xgi_ev_boost(cameo), 0.0)

        solid = {
            "position": "MID",
            "minutes": 2700,
            "xgi90": 0.55,
            "expected_goal_involvements_per_90": 0.55,
        }
        solid_ev = base_ev_with_form_blend(solid, ep_next=2.3, form=0.0)
        cameo_ev = base_ev_with_form_blend(cameo, ep_next=1.5, form=0.0)
        self.assertGreater(solid_ev, cameo_ev)
        self.assertLess(cameo_ev, 2.0)

    def test_set_piece_and_xgi90_features(self):
        bootstrap = _mini_bootstrap()
        bootstrap["elements"][0]["penalties_order"] = 1
        bootstrap["elements"][0]["corners_and_indirect_freekicks_order"] = 1
        bootstrap["elements"][0]["expected_goal_involvements_per_90"] = "0.8"
        bootstrap["elements"][0]["starts_per_90"] = "1.0"
        bootstrap["elements"][0]["clean_sheets_per_90"] = "0.4"
        bootstrap["elements"][1]["clean_sheets_per_90"] = "0.1"
        kg = build_knowledge_graph(bootstrap=bootstrap, fixtures=_mini_fixtures())
        saka = next(p for p in kg.players() if p["id"] == 10)
        self.assertGreater(float(saka.get("set_piece_factor") or 1.0), 1.0)
        self.assertIsNotNone(saka.get("xgi90"))
        # SET_PIECE edges exist
        sp = sum(
            1
            for u, v, _, d in kg.g.edges(keys=True, data=True)
            if d.get("edge_type") == EdgeType.SET_PIECE.value
        )
        self.assertGreaterEqual(sp, 1)

    def test_derive_nulls_xgi90_on_cameo_minutes(self):
        bootstrap = _mini_bootstrap()
        bootstrap["elements"][0]["minutes"] = 2
        bootstrap["elements"][0]["expected_goal_involvements_per_90"] = "3.6"
        kg = build_knowledge_graph(bootstrap=bootstrap, fixtures=_mini_fixtures())
        saka = next(p for p in kg.players() if p["id"] == 10)
        self.assertIsNone(saka.get("xgi90"))
        self.assertEqual(float(saka.get("xgi90_raw") or 0), 3.6)

    def test_cs_boost_prefers_defenders(self):
        from fpl_robo.graph.derive import cs_ev_boost

        def_boost = cs_ev_boost(
            {
                "position": "DEF",
                "cs_per_90": 0.5,
                "next_matchup_factor": 1.1,
                "minutes": 900,
            }
        )
        mid_boost = cs_ev_boost({"position": "MID", "cs_per_90": 0.5, "minutes": 900})
        thin = cs_ev_boost(
            {
                "position": "DEF",
                "cs_per_90": 0.9,
                "next_matchup_factor": 1.1,
                "minutes": 10,
            }
        )
        self.assertGreater(def_boost, 0.0)
        self.assertEqual(mid_boost, 0.0)
        self.assertEqual(thin, 0.0)

    def test_faces_keep_both_meetings(self):
        """Same club pair across two GWs must not overwrite (key collision bug)."""
        bootstrap = _mini_bootstrap()
        fixtures = _mini_fixtures() + [
            {
                "id": 101,
                "event": 20,
                "team_h": 2,
                "team_a": 1,
                "team_h_difficulty": 4,
                "team_a_difficulty": 2,
                "finished": False,
                "kickoff_time": "2027-01-01T15:00:00Z",
            }
        ]
        kg = build_knowledge_graph(bootstrap=bootstrap, fixtures=fixtures)
        gws = set()
        for u, v, _, data in kg.g.edges(keys=True, data=True):
            if data.get("edge_type") != EdgeType.FACES.value:
                continue
            if u[0] != NodeType.TEAM.value or v[0] != NodeType.TEAM.value:
                continue
            if {int(u[1]), int(v[1])} == {1, 2}:
                gws.add(data.get("gameweek"))
        self.assertEqual(gws, {1, 20})


    def test_player_faces_edges_written(self):
        kg = build_knowledge_graph(bootstrap=_mini_bootstrap(), fixtures=_mini_fixtures())
        enrich_knowledge_graph(kg, gameweek=1)
        found = 0
        for u, v, _, data in kg.g.edges(keys=True, data=True):
            if data.get("edge_type") != EdgeType.FACES.value:
                continue
            if u[0] != NodeType.PLAYER.value:
                continue
            if data.get("enriched_faces"):
                found += 1
        self.assertGreaterEqual(found, 2)

    def test_baseline_predict_consumes_kg_context(self):
        kg = build_knowledge_graph(bootstrap=_mini_bootstrap(), fixtures=_mini_fixtures())
        df = BaselineEVModel().predict(kg, gameweek=1)
        self.assertFalse(df.empty)
        self.assertIn("context_factor", df.columns)
        saka = df[df["id"] == 10].iloc[0]
        foster = df[df["id"] == 20].iloc[0]
        # Arsenal home vs weak Burnley should outrank Burnley away.
        self.assertGreater(float(saka["ev"]), float(foster["ev"]))
        self.assertTrue(bool(saka.get("context_factor")))
        # News text without chance should haircut Foster's context vs Saka's.
        foster_p = next(p for p in kg.players() if p["id"] == 20)
        self.assertGreater(float(foster_p.get("news_risk") or 0), 0)

if __name__ == "__main__":
    unittest.main()
