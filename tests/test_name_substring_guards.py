"""Regression tests for substring name false-positives (Raya ⊂ Rayan, etc.)."""

from __future__ import annotations

import unittest

from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text
from fpl_robo.signals.match import (
    alias_in_text,
    build_player_indexes,
    candidate_web_names_for_text,
)
from fpl_robo.signals.minutes_policy import lineup_row_applies_to_ev
from fpl_robo.signals.validate import player_mentioned


class NameSubstringGuardTests(unittest.TestCase):
    def test_alias_in_text_rejects_raya_inside_rayan(self):
        blob = (
            "Erling Haaland blanking despite several openings, "
            "Rayan Cherki transforming the game from the bench"
        )
        self.assertFalse(alias_in_text(blob, "Raya"))
        self.assertTrue(alias_in_text(blob, "Rayan"))
        self.assertTrue(alias_in_text("David Raya kept a clean sheet", "Raya"))

    def test_heuristic_lineup_ignores_rayan_cherki_for_raya(self):
        text = (
            "FPL notes: Guehi OOP, Haaland blank + Anderson injury update. "
            "midfield, erling haaland (£15.5m) blanking despite several openings, "
            "rayan cherki (£7.5m) transforming the game from the bench and an injury scare"
        )
        raya = {
            "id": 1,
            "web_name": "Raya",
            "first_name": "David",
            "second_name": "Raya Martín",
            "team_short": "ARS",
            "team_id": 1,
            "team_name": "Arsenal",
        }
        rows = heuristic_lineup_signals_from_text(
            text,
            [raya],
            prefer_ids={1},
            source="heuristic_lineup",
            gameweek=2,
            article_title="FPL notes: Guehi OOP, Haaland blank + Anderson injury update",
        )
        self.assertEqual(rows, [])

    def test_heuristic_lineup_still_matches_real_raya_bench(self):
        text = "Arsenal team news: David Raya is expected to start on the bench behind Kelleher."
        raya = {
            "id": 1,
            "web_name": "Raya",
            "first_name": "David",
            "second_name": "Raya Martín",
            "team_short": "ARS",
            "team_id": 1,
            "team_name": "Arsenal",
        }
        rows = heuristic_lineup_signals_from_text(
            text,
            [raya],
            prefer_ids={1},
            source="heuristic_lineup",
            gameweek=2,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["web_name"], "Raya")
        self.assertEqual(rows[0]["role"], "bench")

    def test_player_mentioned_and_candidates_reject_substring(self):
        blob = "Rayan Cherki came on from the bench for Manchester City"
        raya = {
            "id": 1,
            "web_name": "Raya",
            "first_name": "David",
            "second_name": "Raya Martín",
            "team_short": "ARS",
        }
        self.assertFalse(player_mentioned(blob, raya))
        indexes = build_player_indexes([raya])
        self.assertNotIn("Raya", candidate_web_names_for_text(blob, indexes))

    def test_minutes_policy_ignores_role_label_only_bench_note(self):
        # Note contains "bench" from our own label; evidence is Cherki, not Raya.
        row = {
            "source": "heuristic_lineup",
            "role": "bench",
            "predicted_minutes": 20,
            "confidence": 0.76,
            "note": "[heuristic_lineup] bench (~20m): Raya",
            "evidence": "rayan cherki transforming the game from the bench",
        }
        # Even with "from the bench" in evidence, this row should not apply once
        # matching is fixed; policy still requires explicit bench cue in evidence
        # (which is present) — so this alone would still apply. The matcher is
        # the primary guard. Policy must not treat the note's "bench" alone as enough.
        row_note_only = {
            **row,
            "evidence": "rayan cherki transforming the game late on",
        }
        self.assertFalse(lineup_row_applies_to_ev(row_note_only))

    def test_owned_player_mention_rejects_substring(self):
        from fpl_robo.signals.ingest import article_mentions_owned_player

        owned = [
            {
                "id": 1,
                "web_name": "Raya",
                "second_name": "Raya Martín",
                "team_short": "ARS",
            }
        ]
        self.assertFalse(
            article_mentions_owned_player(
                "Rayan Cherki transforming the game from the bench",
                owned,
            )
        )
        self.assertTrue(
            article_mentions_owned_player(
                "David Raya kept a clean sheet for Arsenal",
                owned,
            )
        )


if __name__ == "__main__":
    unittest.main()
