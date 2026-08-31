"""Surname collisions must use club hints (Man Utd → Bruno, not Mateus)."""

from __future__ import annotations

import unittest

from fpl_robo.signals.heuristics import heuristic_signals_from_title
from fpl_robo.signals.match import (
    build_player_indexes,
    candidate_web_names_for_text,
    club_hint_in_text,
    resolve_player_name,
)
from fpl_robo.signals.validate import player_mentioned, score_signal_evidence, team_mentioned


def _bruno() -> dict:
    return {
        "id": 328,
        "web_name": "B.Fernandes",
        "first_name": "Bruno",
        "second_name": "Borges Fernandes",
        "team_short": "MUN",
        "team_name": "Manchester United",
        "team_id": 14,
        "selected_by_percent": 20.0,
        "status": "a",
    }


def _mateus() -> dict:
    return {
        "id": 525,
        "web_name": "Fernandes",
        "first_name": "Mateus",
        "second_name": "Fernandes",
        "team_short": "TOT",
        "team_name": "Tottenham Hotspur",
        "team_id": 17,
        "selected_by_percent": 1.0,
        "status": "a",
    }


TITLE = "Fernandes hat-trick inspires five-goal Man Utd comeback against Ipswich"
SUMMARY = (
    "Bruno Fernandes scores a hat-trick as Manchester United overcome an early "
    "scare to beat Ipswich Town 5-2 at Old Trafford."
)
BLOB = f"{TITLE}. {SUMMARY}"


class ClubHintTests(unittest.TestCase):
    def test_man_utd_stub_matches_bruno_not_mateus(self):
        self.assertTrue(club_hint_in_text(TITLE, _bruno()))
        self.assertFalse(club_hint_in_text(TITLE, _mateus()))
        self.assertTrue(team_mentioned(TITLE, _bruno(), {}))
        self.assertFalse(team_mentioned(TITLE, _mateus(), {}))


class FernandesDisambiguationTests(unittest.TestCase):
    def test_heuristic_credits_bruno_not_mateus(self):
        rows = heuristic_signals_from_title(TITLE, [_bruno(), _mateus()], summary=SUMMARY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["web_name"], "B.Fernandes")
        self.assertEqual(rows[0]["team_short"], "MUN")
        self.assertIn("hat-trick", rows[0]["note"])

    def test_resolve_uses_article_club_hint(self):
        indexes = build_player_indexes([_bruno(), _mateus()])
        # Exact web_name "Fernandes" is Mateus — must not win when article is Man Utd.
        hit = resolve_player_name("Fernandes", indexes, article_text=BLOB)
        self.assertIsNotNone(hit)
        self.assertEqual(int(hit["id"]), 328)
        hit_mun = resolve_player_name("Fernandes", indexes, team_short="MUN")
        self.assertEqual(int(hit_mun["id"]), 328)
        self.assertEqual(
            int(resolve_player_name("Bruno Fernandes", indexes)["id"]),
            328,
        )
        names = candidate_web_names_for_text(BLOB, indexes)
        self.assertIn("B.Fernandes", names)

    def test_mateus_not_mentioned_when_only_bruno_named(self):
        # Bare leading "Fernandes" + later "Bruno Fernandes" should not count as Mateus.
        self.assertTrue(player_mentioned(BLOB, _bruno()))
        # Leading bare surname still exists; club-less Mateus form must fail evidence.
        scored = score_signal_evidence(
            article_text=BLOB,
            player=_mateus(),
            kind="form",
            team_aliases={},
            llm_note="hat-trick",
        )
        self.assertFalse(scored["accepted"])
        self.assertIn(scored["reject_reason"], {
            "form_missing_club_or_firstname",
            "player_not_in_article",
        })
        scored_b = score_signal_evidence(
            article_text=BLOB,
            player=_bruno(),
            kind="form",
            team_aliases={},
            llm_note="hat-trick",
        )
        self.assertTrue(scored_b["team_hit"])
        self.assertTrue(scored_b["accepted"])


if __name__ == "__main__":
    unittest.main()
