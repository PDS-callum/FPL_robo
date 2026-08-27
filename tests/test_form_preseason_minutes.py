"""Regression: form news must not invent backup minutes haircuts."""

from __future__ import annotations

import unittest

from fpl_robo.signals.aggregate import aggregate_player_signals
from fpl_robo.signals.ingest import _form_text_cues, derive_preseason_from_news
from fpl_robo.signals.loader import SignalBundle
from fpl_robo.signals.schema import NewsSignal, PreseasonSignal


class FormPreseasonMinutesTests(unittest.TestCase):
    def test_score_singular_counts_as_goal_cue(self):
        ht, goals, soft = _form_text_cues(
            "took just 31 seconds to score his opening goal of the season"
        )
        self.assertFalse(ht)
        self.assertTrue(goals)
        self.assertFalse(soft)

    def test_goalkeeper_and_scoreline_are_not_goal_cues(self):
        _, goals_gk, _ = _form_text_cues("new goalkeeper signed for depth")
        _, goals_sl, _ = _form_text_cues("final scoreline was 2-1")
        self.assertFalse(goals_gk)
        self.assertFalse(goals_sl)

    def test_sets_up_is_soft_boost_only_not_rotation(self):
        rows = derive_preseason_from_news(
            [
                {
                    "player_id": 165,
                    "web_name": "João Pedro",
                    "type": "form",
                    "note": "[bbc] sets up Joao Pedro · linked form subject→João Pedro",
                    "evidence": "sets up Joao Pedro",
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["role"], "unknown")
        self.assertEqual(rows[0]["minutes"], 0)
        self.assertGreater(rows[0]["boost"], 0)

    def test_score_form_becomes_starter_not_rotation(self):
        rows = derive_preseason_from_news(
            [
                {
                    "player_id": 165,
                    "web_name": "João Pedro",
                    "type": "form",
                    "note": (
                        "[bbc_premier_league] took just 31 seconds to score "
                        "his opening goal of the season"
                    ),
                    "evidence": "score his opening goal",
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["role"], "starter")
        self.assertGreaterEqual(rows[0]["minutes"], 60)

    def test_stale_soft_form_rotation_row_does_not_haircut_minutes(self):
        # Legacy bad row shape that previously applied minutes_mult=0.55.
        bundle = SignalBundle(
            news=[
                NewsSignal(
                    player_id=165,
                    web_name="João Pedro",
                    kind="form",
                    severity=0.3,
                    note="sets up Joao Pedro",
                    source="ollama",
                )
            ],
            preseason=[
                PreseasonSignal(
                    player_id=165,
                    web_name="João Pedro",
                    minutes=30,
                    starts=0,
                    role="rotation",
                    boost=0.04,
                    note=(
                        "auto from form: [bbc_premier_league] took just 31 seconds "
                        "to score his opening goal of the season"
                    ),
                )
            ],
        )
        factors = aggregate_player_signals(
            bundle,
            players_by_id={165: {"id": 165, "web_name": "João Pedro"}},
        )
        self.assertEqual(factors[165].minutes_mult, 1.0)
        self.assertGreater(factors[165].form_boost, 0.0)

    def test_explicit_backup_preseason_still_haircuts(self):
        bundle = SignalBundle(
            preseason=[
                PreseasonSignal(
                    player_id=1,
                    web_name="Depth",
                    minutes=20,
                    starts=0,
                    role="backup",
                    boost=0.02,
                    note="auto from transfer (tier=elite): depth signing",
                )
            ]
        )
        factors = aggregate_player_signals(
            bundle,
            players_by_id={1: {"id": 1, "web_name": "Depth"}},
        )
        self.assertLess(factors[1].minutes_mult, 0.7)


if __name__ == "__main__":
    unittest.main()
