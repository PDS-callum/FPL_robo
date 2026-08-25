import tempfile
import unittest
from pathlib import Path

from fpl_robo.predict.baseline import PreseasonEVModel
from fpl_robo.signals import aggregate_player_signals, load_signals
from fpl_robo.signals.aggregate import PlayerSignalFactors


class SignalsTests(unittest.TestCase):
    def test_load_and_aggregate_by_web_name(self):
        payload = """
        {
          "as_of": "2026-07-27",
          "news": [
            {"web_name": "Injured", "type": "injury", "severity": 0.9, "exclude": true}
          ],
          "preseason": [
            {"web_name": "Starter", "minutes": 180, "starts": 2, "role": "starter", "boost": 0.1}
          ],
          "international": [
            {"web_name": "Fatigued", "tournament": "World Cup", "minutes": 600, "fatigue": 0.8}
          ]
        }
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.json"
            path.write_text(payload, encoding="utf-8")
            bundle = load_signals(path)

        self.assertEqual(bundle.count, 3)
        players = {
            1: {"id": 1, "web_name": "Injured"},
            2: {"id": 2, "web_name": "Starter"},
            3: {"id": 3, "web_name": "Fatigued"},
            4: {"id": 4, "web_name": "Clean"},
        }
        factors = aggregate_player_signals(bundle, players_by_id=players)
        self.assertTrue(factors[1].exclude)
        # Starter preseason tag is intentionally minutes-neutral (noisy/asymmetric).
        self.assertEqual(factors[2].minutes_mult, 1.0)
        self.assertGreater(factors[2].form_boost, 0.0)
        self.assertLess(factors[3].fatigue_mult, 1.0)
        self.assertNotIn(4, factors)

    def test_preseason_model_applies_signal_multipliers(self):
        # Minimal fake KG via monkeypatch-style stub attributes on a simple object.
        class FakeKG:
            def current_gameweek(self):
                return 1

            def players(self):
                return [
                    {
                        "id": 10,
                        "web_name": "A",
                        "team_id": 1,
                        "team_short": "AAA",
                        "position": "MID",
                        "price": 8.0,
                        "status": "a",
                        "chance_of_playing_next_round": 100,
                        "ep_next": 5.0,
                        "points_per_game": 5.0,
                        "ict_index": 50.0,
                        "selected_by_percent": 10.0,
                        "minutes": 0,
                    },
                    {
                        "id": 11,
                        "web_name": "B",
                        "team_id": 1,
                        "team_short": "AAA",
                        "position": "MID",
                        "price": 8.0,
                        "status": "a",
                        "chance_of_playing_next_round": 100,
                        "ep_next": 5.0,
                        "points_per_game": 5.0,
                        "ict_index": 50.0,
                        "selected_by_percent": 10.0,
                        "minutes": 0,
                    },
                ]

            @property
            def g(self):
                class Edges:
                    def edges(self, keys=True, data=True):
                        return []

                return Edges()

        model = PreseasonEVModel(
            gameweek_weights=[1.0],
            signal_factors={
                10: PlayerSignalFactors(news_mult=0.5, fatigue_mult=0.8),
            },
        )
        df = model.predict(FakeKG())
        by_id = {int(r["id"]): r for _, r in df.iterrows()}
        self.assertTrue(by_id[10]["signal_applied"])
        self.assertFalse(by_id[11]["signal_applied"])
        self.assertLess(by_id[10]["ev"], by_id[11]["ev"])


if __name__ == "__main__":
    unittest.main()
