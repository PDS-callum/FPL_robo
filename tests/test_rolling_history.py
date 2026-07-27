import unittest

from fpl_robo.data.rolling_history import (
    PlayerHistoryFeatures,
    aggregate_player_history,
    attach_history_to_graph,
    finished_gameweeks_before,
)
from fpl_robo.graph.schema import GraphNode, NodeId, NodeType
from fpl_robo.graph.store import KnowledgeGraph


class RollingHistoryTests(unittest.TestCase):
    def test_finished_gameweeks_before(self):
        bootstrap = {
            "events": [
                {"id": 1, "finished": True},
                {"id": 2, "finished": True},
                {"id": 3, "finished": True},
                {"id": 4, "finished": True},
                {"id": 5, "finished": True},
                {"id": 6, "finished": False, "is_next": True},
            ]
        }
        self.assertEqual(
            finished_gameweeks_before(bootstrap, before_gw=6, limit=4),
            [2, 3, 4, 5],
        )
        self.assertEqual(
            finished_gameweeks_before(bootstrap, before_gw=3, limit=4),
            [1, 2],
        )

    def test_aggregate_player_history_mean_and_prior(self):
        gw_stats = {
            1: {10: {"minutes": 90.0, "total_points": 6.0}},
            2: {10: {"minutes": 60.0, "total_points": 2.0}},
            3: {10: {"minutes": 0.0, "total_points": 0.0}},
            4: {10: {"minutes": 90.0, "total_points": 8.0}, 11: {"minutes": 45.0, "total_points": 1.0}},
        }
        feats = aggregate_player_history(gw_stats, window=4)
        self.assertIn(10, feats)
        self.assertEqual(feats[10].prior_minutes, 240.0)
        self.assertEqual(feats[10].rolling_minutes, 60.0)  # (90+60+0+90)/4
        self.assertEqual(feats[10].prior_points, 16.0)
        self.assertEqual(feats[10].rolling_points, 4.0)
        self.assertEqual(feats[11].prior_minutes, 45.0)
        self.assertEqual(feats[11].rolling_minutes, 11.25)  # 0+0+0+45 over 4 GWs

    def test_aggregate_uses_last_window_only(self):
        gw_stats = {
            1: {1: {"minutes": 90.0, "total_points": 10.0}},
            2: {1: {"minutes": 90.0, "total_points": 10.0}},
            3: {1: {"minutes": 0.0, "total_points": 0.0}},
            4: {1: {"minutes": 0.0, "total_points": 0.0}},
            5: {1: {"minutes": 90.0, "total_points": 5.0}},
        }
        feats = aggregate_player_history(gw_stats, window=2)
        self.assertEqual(feats[1].prior_minutes, 90.0)  # GW4=0 + GW5=90
        self.assertEqual(feats[1].rolling_minutes, 45.0)

    def test_attach_history_to_graph_mutates_nodes(self):
        kg = KnowledgeGraph()
        kg.add_node(
            GraphNode(
                NodeId(NodeType.PLAYER, 7),
                {"web_name": "Test", "minutes": 999},
            )
        )
        n = attach_history_to_graph(
            kg,
            {7: PlayerHistoryFeatures(prior_minutes=180, rolling_minutes=70, rolling_points=4)},
        )
        self.assertEqual(n, 1)
        players = {p["id"]: p for p in kg.players()}
        self.assertTrue(players[7]["history_attached"])
        self.assertEqual(players[7]["prior_minutes"], 180)
        self.assertEqual(players[7]["rolling_minutes"], 70)


if __name__ == "__main__":
    unittest.main()
