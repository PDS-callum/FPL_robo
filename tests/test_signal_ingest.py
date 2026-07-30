"""Tests for RSS parse, name match, and signal merge (no live network/LLM)."""

from __future__ import annotations

import unittest

from fpl_robo.signals.ingest import merge_news_signals
from fpl_robo.signals.llm import _extract_json
from fpl_robo.signals.match import build_player_indexes, normalize_name, resolve_player_name
from fpl_robo.signals.scrape import parse_rss


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Football</title>
    <item>
      <title>Salah injury concern for Liverpool</title>
      <description>Mohamed Salah picked up a knock in training and is a doubt.</description>
      <link>https://example.test/salah</link>
      <pubDate>Mon, 27 Jul 2026 10:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Transfer rumour mill</title>
      <description>Generic chatter with no clear Premier League availability angle.</description>
      <link>https://example.test/rumour</link>
      <pubDate>Mon, 27 Jul 2026 09:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


class ScrapeParseTests(unittest.TestCase):
    def test_parse_rss_items(self):
        articles = parse_rss(SAMPLE_RSS, source="test_feed")
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0].source, "test_feed")
        self.assertIn("Salah", articles[0].title)
        self.assertTrue(articles[0].summary)


class MatchTests(unittest.TestCase):
    def test_resolve_web_name_and_surname(self):
        players = [
            {
                "id": 1,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
            },
            {
                "id": 2,
                "web_name": "Palmer",
                "first_name": "Cole",
                "second_name": "Palmer",
            },
            {
                "id": 3,
                "web_name": "J.Murphy",
                "first_name": "Jacob",
                "second_name": "Murphy",
            },
            {
                "id": 4,
                "web_name": "Josh Murphy",
                "first_name": "Josh",
                "second_name": "Murphy",
            },
        ]
        indexes = build_player_indexes(players)
        self.assertEqual(normalize_name("Mohamed Salah"), "mohamedsalah")
        hit = resolve_player_name("Salah", indexes)
        self.assertIsNotNone(hit)
        self.assertEqual(int(hit["id"]), 1)
        hit2 = resolve_player_name("Mohamed Salah", indexes)
        self.assertEqual(int(hit2["id"]), 1)
        # Ambiguous surname should not resolve.
        self.assertIsNone(resolve_player_name("Murphy", indexes))
        self.assertEqual(int(resolve_player_name("J.Murphy", indexes)["id"]), 3)

    def test_team_disambiguates_surname(self):
        players = [
            {
                "id": 1,
                "web_name": "J.Gomez",
                "first_name": "Joe",
                "second_name": "Gomez",
                "team_short": "BHA",
            },
            {
                "id": 2,
                "web_name": "P.Gomez",
                "first_name": "Papu",
                "second_name": "Gomez",
                "team_short": "SEV",
            },
        ]
        indexes = build_player_indexes(players)
        self.assertIsNone(resolve_player_name("Gomez", indexes))
        hit = resolve_player_name("Gomez", indexes, team_short="BHA")
        self.assertIsNotNone(hit)
        self.assertEqual(int(hit["id"]), 1)
        hit2 = resolve_player_name("Gomez", indexes, candidate_ids={1})
        self.assertEqual(int(hit2["id"]), 1)



class LlmJsonTests(unittest.TestCase):
    def test_extract_json_from_fence(self):
        raw = 'Here you go:\n```json\n{"signals":[{"web_name":"Salah","kind":"injury","severity":0.7}]}\n```'
        parsed = _extract_json(raw)
        self.assertEqual(parsed["signals"][0]["web_name"], "Salah")


class MergeTests(unittest.TestCase):
    def test_merge_replaces_ollama_keeps_manual(self):
        existing = [
            {"web_name": "Salah", "type": "injury", "severity": 0.9, "source": "ollama"},
            {"web_name": "Palmer", "type": "positive", "severity": 0.1, "note": "manual"},
        ]
        proposed = [
            {
                "player_id": 1,
                "web_name": "Salah",
                "type": "injury",
                "severity": 0.6,
                "exclude": False,
                "source": "ollama",
                "note": "updated",
            }
        ]
        merged = merge_news_signals(existing, proposed)
        self.assertEqual(len(merged), 2)
        names = {m["web_name"]: m for m in merged}
        self.assertEqual(names["Salah"]["severity"], 0.6)
        self.assertEqual(names["Palmer"]["note"], "manual")
        self.assertNotIn("source", names["Palmer"])
        self.assertIn("ingested_at", names["Salah"])

    def test_merge_retains_auto_when_proposed_empty(self):
        existing = [
            {"web_name": "Salah", "type": "injury", "severity": 0.9, "source": "ollama"},
            {"web_name": "Palmer", "type": "form", "severity": 0.1, "source": "heuristic"},
            {"web_name": "Haaland", "type": "positive", "note": "manual"},
        ]
        merged = merge_news_signals(existing, [])
        self.assertEqual(len(merged), 3)
        names = {m["web_name"] for m in merged}
        self.assertEqual(names, {"Salah", "Palmer", "Haaland"})

    def test_merge_prunes_stale_auto_news(self):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        fresh = datetime.now(timezone.utc).isoformat()
        existing = [
            {"web_name": "Salah", "type": "injury", "source": "ollama", "ingested_at": old},
            {"web_name": "Palmer", "type": "form", "source": "heuristic", "ingested_at": fresh},
            {"web_name": "Haaland", "type": "positive", "note": "manual"},
        ]
        with patch.dict("fpl_robo.signals.ingest.SIGNAL_INGEST", {"auto_news_max_age_days": 10, "retain_auto_news_on_empty": True}):
            merged = merge_news_signals(existing, [])
        names = {m["web_name"] for m in merged}
        self.assertEqual(names, {"Palmer", "Haaland"})


class HeuristicPolicyTests(unittest.TestCase):
    def test_soft_form_and_local_injury_helpers(self):
        from fpl_robo.signals.ingest import (
            _player_local_window,
            _player_near_keywords,
            _soft_form_language,
            _INJURY_NEAR_KEYWORDS,
        )

        self.assertTrue(_soft_form_language("[heuristic] sharp in headline", ""))
        self.assertFalse(_soft_form_language("[heuristic] hat-trick in headline", ""))
        player = {"web_name": "Ndiaye", "first_name": "Iliman", "second_name": "Ndiaye"}
        text = (
            "Pedro scored a hat-trick early on for Chelsea in New Jersey. "
            "Separately, Ndiaye was absent with a knock after warm-up."
        )
        window = _player_local_window(text, player, radius=40)
        self.assertIn("absent", window.lower())
        self.assertIn("knock", window.lower())
        self.assertNotIn("hat-trick", window.lower())
        self.assertTrue(_player_near_keywords(text, player, _INJURY_NEAR_KEYWORDS, radius=80))
        pedro = {"web_name": "Pedro", "first_name": "João", "second_name": "Pedro"}
        self.assertFalse(_player_near_keywords(text, pedro, _INJURY_NEAR_KEYWORDS, radius=80))

        grealish = {"web_name": "Grealish", "first_name": "Jack", "second_name": "Grealish"}
        g_text = (
            "Grealish misses Man City tour despite training return. "
            "Jack Grealish will not make his return in a Manchester City shirt after "
            "being left out of the clubs pre-season tour of Asia as he continues to "
            "recover from a foot injury."
        )
        self.assertTrue(_player_near_keywords(g_text, grealish, _INJURY_NEAR_KEYWORDS, radius=220))
        # Teammate injury language far from another player should not match.
        haaland = {"web_name": "Haaland", "first_name": "Erling", "second_name": "Haaland"}
        far = "Haaland scored twice. Later that week Grealish recovered from a foot injury."
        self.assertFalse(_player_near_keywords(far, haaland, _INJURY_NEAR_KEYWORDS, radius=40))

    def test_vague_latest_needs_llm(self):
        from fpl_robo.signals.ingest import heuristics_are_sufficient

        self.assertFalse(
            heuristics_are_sufficient(
                [{"web_name": "Matusiwa", "type": "form", "note": "[heuristic] latest"}],
                title="FPL pre-season: Tel lively, Matusiwa latest",
            )
        )
        self.assertTrue(
            heuristics_are_sufficient(
                [{"web_name": "Kroupi.Jr", "type": "injury", "note": "[heuristic] months out in headline"}],
                title="Kroupi set for 3-4 months out with foot injury",
            )
        )

    def test_fpl_official_skips_hard_news_mult(self):
        from fpl_robo.signals.aggregate import aggregate_player_signals
        from fpl_robo.signals.loader import SignalBundle
        from fpl_robo.signals.schema import NewsSignal

        players = {1: {"id": 1, "web_name": "Saliba"}}
        bundle = SignalBundle(
            news=[
                NewsSignal(
                    player_id=1,
                    web_name="Saliba",
                    kind="injury",
                    severity=0.85,
                    exclude=True,
                    note="[fpl] Back injury",
                    source="fpl",
                    status_align=True,
                )
            ]
        )
        factors = aggregate_player_signals(bundle, players_by_id=players)
        self.assertTrue(factors[1].exclude)
        self.assertEqual(factors[1].news_mult, 1.0)

    def test_article_body_helpers(self):
        from fpl_robo.signals.scrape import NewsArticle, needs_article_body, _host_allowlisted

        thin = NewsArticle(
            source="ffs_feed",
            title="Chelsea complete deal for Maxime Lacroix",
            summary="Short blurb.",
            link="https://www.fantasyfootballscout.co.uk/article/lacroix",
        )
        self.assertTrue(thin.summary_is_thin)
        self.assertTrue(needs_article_body(thin))
        self.assertTrue(_host_allowlisted(thin.link))
        self.assertFalse(_host_allowlisted("https://evil.example/x"))
        fat = NewsArticle(
            source="ffs_feed",
            title="Random listicle",
            summary="x" * 200,
            link="https://www.fantasyfootballscout.co.uk/article/x",
        )
        self.assertFalse(needs_article_body(fat))


class HeuristicTests(unittest.TestCase):
    def test_title_injury_seeds_kg_player(self):
        from fpl_robo.signals.heuristics import heuristic_signals_from_title

        players = [
            {
                "id": 1,
                "web_name": "Gomez",
                "first_name": "Joe",
                "second_name": "Gomez",
                "team_short": "LIV",
                "status": "i",
                "chance_of_playing_next_round": 0,
            },
            {
                "id": 2,
                "web_name": "Szoboszlai",
                "first_name": "Dominik",
                "second_name": "Szoboszlai",
                "team_short": "LIV",
                "status": "a",
            },
        ]
        hits = heuristic_signals_from_title(
            "Gomez injury overshadows winning start for Iraola",
            players,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["web_name"], "Gomez")
        self.assertEqual(hits[0]["first_name"], "Joe")
        self.assertEqual(hits[0]["type"], "injury")
        self.assertEqual(hits[0]["source"], "heuristic")
        self.assertTrue(hits[0]["exclude"])

    def test_title_form_and_transfer(self):
        from fpl_robo.signals.heuristics import heuristic_signals_from_title

        players = [
            {
                "id": 10,
                "web_name": "João Pedro",
                "first_name": "João",
                "second_name": "Pedro",
                "team_short": "CHE",
                "status": "a",
            },
            {
                "id": 11,
                "web_name": "Ekitike",
                "first_name": "Hugo",
                "second_name": "Ekitike",
                "team_short": "LIV",
                "status": "a",
            },
        ]
        form_hits = heuristic_signals_from_title(
            "João Pedro scores hat-trick in Chelsea pre-season friendly",
            players,
        )
        self.assertEqual(len(form_hits), 1)
        self.assertEqual(form_hits[0]["type"], "form")
        self.assertEqual(form_hits[0]["web_name"], "João Pedro")

        transfer_hits = heuristic_signals_from_title(
            "Ekitike completes move to Liverpool",
            players,
        )
        self.assertEqual(len(transfer_hits), 1)
        self.assertEqual(transfer_hits[0]["type"], "transfer")

    def test_club_signs_and_multi_missed_out(self):
        from fpl_robo.signals.heuristics import heuristic_signals_from_title

        players = [
            {
                "id": 564,
                "web_name": "Scherpen",
                "first_name": "Kjell",
                "second_name": "Scherpen",
                "team_short": "IPS",
                "element_type": 1,
                "position": "GKP",
                "status": "a",
                "selected_by_percent": 1.0,
            },
            {
                "id": 40,
                "web_name": "Rogers",
                "first_name": "Morgan",
                "second_name": "Rogers",
                "team_short": "CHE",
                "status": "a",
                "selected_by_percent": 20.0,
            },
            {
                "id": 526,
                "web_name": "Solanke",
                "first_name": "Dominic",
                "second_name": "Solanke",
                "team_short": "TOT",
                "status": "a",
                "selected_by_percent": 10.0,
            },
            {
                "id": 100,
                "web_name": "Fernandes",
                "first_name": "Mateus",
                "second_name": "Fernandes",
                "team_short": "TOT",
                "status": "a",
                "selected_by_percent": 5.0,
            },
            {
                "id": 514,
                "web_name": "Tel",
                "first_name": "Mathys",
                "second_name": "Tel",
                "team_short": "TOT",
                "status": "a",
                "selected_by_percent": 8.0,
            },
        ]
        scherpen = heuristic_signals_from_title(
            "Ipswich sign goalkeeper Scherpen from Union Saint-Gilloise",
            players,
            summary="The 26-year-old is expected to come straight in as Ipswich’s No 1",
        )
        self.assertTrue(any(h["web_name"] == "Scherpen" and h["type"] == "transfer" for h in scherpen))

        rogers = heuristic_signals_from_title(
            "Can Rogers be a good FPL asset after Chelsea transfer?",
            players,
        )
        self.assertTrue(any(h["web_name"] == "Rogers" and h["type"] == "transfer" for h in rogers))

        multi = heuristic_signals_from_title(
            "FPL pre-season: Tel lively, why Solanke + Fernandes missed out, Matusiwa latest",
            players,
        )
        kinds = {(h["web_name"], h["type"]) for h in multi}
        self.assertIn(("Tel", "form"), kinds)
        self.assertIn(("Solanke", "rotation"), kinds)
        self.assertIn(("Fernandes", "rotation"), kinds)

    def test_transfer_inference_elite_gk(self):
        from fpl_robo.signals.transfer_infer import infer_transfer_impact

        impact = infer_transfer_impact(
            team_short="CHE",
            position="MID",
            evidence="after Chelsea transfer",
            note="Rogers",
            strength_home=4,
        )
        self.assertEqual(impact["tier"], "elite")
        self.assertGreaterEqual(impact["boost"], 0.08)

        gk = infer_transfer_impact(
            team_short="IPS",
            element_type=1,
            evidence="Ipswich sign goalkeeper Scherpen. expected to come straight in as No 1",
            note="",
            strength_home=2,
        )
        self.assertEqual(gk["role"], "starter")
        self.assertGreaterEqual(gk["minutes"], 45)

    def test_disambiguates_two_gomez_via_fpl_status(self):
        from fpl_robo.signals.heuristics import heuristic_signals_from_title

        players = [
            {
                "id": 127,
                "web_name": "Gomez",
                "first_name": "Diego",
                "second_name": "Gómez Amarilla",
                "team_short": "BHA",
                "status": "a",
            },
            {
                "id": 359,
                "web_name": "Gomez",
                "first_name": "Joe",
                "second_name": "Gomez",
                "team_short": "LIV",
                "status": "i",
                "chance_of_playing_next_round": 0,
                "news": "Muscular injury - Unknown return date",
            },
        ]
        hits = heuristic_signals_from_title("Gomez injury overshadows winning start", players)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["first_name"], "Joe")
        self.assertEqual(hits[0]["team_short"], "LIV")

    def test_prioritize_injured_candidates(self):
        from fpl_robo.signals.validate import kg_candidate_lines, prioritize_kg_candidates

        players = [
            {"web_name": "Salah", "first_name": "Mohamed", "second_name": "Salah", "team_short": "LIV", "position": "MID", "status": "a"},
            {
                "web_name": "Gomez",
                "first_name": "Joe",
                "second_name": "Gomez",
                "team_short": "LIV",
                "position": "DEF",
                "status": "i",
                "chance_of_playing_next_round": 0,
                "age": 29,
                "news": "Muscular injury - Unknown return date",
            },
            {"web_name": "Chiesa", "first_name": "Federico", "second_name": "Chiesa", "team_short": "LIV", "position": "MID", "status": "a"},
        ]
        ranked = prioritize_kg_candidates(players, limit=2)
        self.assertEqual(ranked[0]["web_name"], "Gomez")
        lines = kg_candidate_lines(players, limit=2)
        self.assertIn("Joe Gomez", lines[0])
        self.assertIn("age=29", lines[0])
        self.assertIn("Muscular injury", lines[0])


class ValidateEvidenceTests(unittest.TestCase):
    def test_rejects_player_not_in_article(self):
        from fpl_robo.signals.validate import score_signal_evidence

        player = {
            "id": 1,
            "web_name": "Adli",
            "first_name": "Amine",
            "second_name": "Adli",
            "team_id": 1,
            "team_short": "BOU",
            "status": "a",
        }
        aliases = {1: ["Bournemouth", "BOU"]}
        score = score_signal_evidence(
            article_text="Indian league seeks to ride World Cup wave",
            player=player,
            kind="rotation",
            team_aliases=aliases,
            llm_note="short reason",
        )
        self.assertFalse(score["accepted"])
        self.assertEqual(score["reject_reason"], "player_not_in_article")

    def test_accepts_injury_with_name_team_proximity(self):
        from fpl_robo.signals.validate import score_signal_evidence

        player = {
            "id": 359,
            "web_name": "Gomez",
            "first_name": "Joe",
            "second_name": "Gomez",
            "team_id": 3,
            "team_short": "BHA",
            "team_name": "Brighton",
            "status": "a",
        }
        aliases = {3: ["Brighton", "BHA", "Brighton & Hove Albion"]}
        text = "Gomez injury overshadows winning start for Iraola at Brighton"
        score = score_signal_evidence(
            article_text=text,
            player=player,
            kind="injury",
            team_aliases=aliases,
            llm_note="[bbc_football] Gomez injury",
        )
        self.assertTrue(score["name_hit"])
        self.assertTrue(score["team_hit"])
        self.assertTrue(score["keyword_hit"])
        self.assertTrue(score["accepted"])
        self.assertGreaterEqual(score["confidence"], 0.65)


class LineupSignalTests(unittest.TestCase):
    def test_aggregate_predicted_start_boosts_minutes(self):
        from fpl_robo.signals.aggregate import aggregate_player_signals
        from fpl_robo.signals.loader import SignalBundle
        from fpl_robo.signals.schema import PredictedLineupSignal

        bundle = SignalBundle(
            predicted_lineups=[
                PredictedLineupSignal(
                    player_id=1,
                    web_name="Salah",
                    team_short="LIV",
                    role="start",
                    gameweek=1,
                    confidence=1.0,
                    note="predicted XI",
                )
            ]
        )
        factors = aggregate_player_signals(
            bundle,
            players_by_id={1: {"id": 1, "web_name": "Salah"}},
        )
        self.assertGreater(factors[1].minutes_mult, 1.0)

    def test_aggregate_skips_low_confidence_lineup(self):
        from fpl_robo.signals.aggregate import aggregate_player_signals
        from fpl_robo.signals.loader import SignalBundle
        from fpl_robo.signals.schema import PredictedLineupSignal

        bundle = SignalBundle(
            predicted_lineups=[
                PredictedLineupSignal(
                    player_id=1,
                    web_name="Salah",
                    team_short="LIV",
                    role="start",
                    gameweek=1,
                    confidence=0.5,
                    note="weak tip",
                )
            ]
        )
        factors = aggregate_player_signals(
            bundle,
            players_by_id={1: {"id": 1, "web_name": "Salah"}},
        )
        # Low-conf lineups should not create a minutes bump entry (or leave default).
        self.assertTrue(1 not in factors or factors[1].minutes_mult == 1.0)

    def test_merge_lineups_replaces_auto_for_gw(self):
        from fpl_robo.signals.ingest import merge_lineup_signals

        existing = [
            {
                "player_id": 1,
                "web_name": "Salah",
                "role": "start",
                "gameweek": 1,
                "source": "ollama_lineup",
            },
            {
                "player_id": 2,
                "web_name": "Palmer",
                "role": "start",
                "gameweek": 1,
                "note": "manual",
            },
        ]
        proposed = [
            {
                "player_id": 1,
                "web_name": "Salah",
                "role": "bench",
                "gameweek": 1,
                "source": "ollama_lineup",
            }
        ]
        merged = merge_lineup_signals(existing, proposed, gameweek=1)
        roles = {m["web_name"]: m["role"] for m in merged}
        self.assertEqual(roles["Salah"], "bench")
        self.assertEqual(roles["Palmer"], "start")

    def test_looks_like_lineup_article(self):
        from fpl_robo.signals.scrape import NewsArticle, looks_like_lineup_article

        art = NewsArticle(
            source="test",
            title="Predicted XI: Liverpool vs Sunderland",
            summary="Likely starting lineup for gameweek 1",
            link="https://example.test/xi",
        )
        self.assertTrue(looks_like_lineup_article(art))

        ffs = NewsArticle(
            source="ffs_feed",
            title="FPL pre-season: Pedro hat-trick, Alonso's tactics + Ndiaye absent",
            summary=(
                "The latest summer friendlies of Chelsea. The post FPL pre-season "
                "appeared first on Best FPL Tips, Advice, Team News, Picks, and "
                "Statistics from Fantasy Football Scout ."
            ),
            link="https://example.test/ffs",
        )
        self.assertFalse(looks_like_lineup_article(ffs))


class PreDeadlineGateTests(unittest.TestCase):
    def test_outside_window_skips(self):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import MagicMock

        from fpl_robo.signals.pre_deadline import should_run_pre_deadline_ingest

        future = (datetime.now(timezone.utc) + timedelta(hours=48)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        client = MagicMock()
        client.bootstrap_static.return_value = {
            "events": [
                {
                    "id": 1,
                    "name": "Gameweek 1",
                    "deadline_time": future,
                    "finished": False,
                    "is_next": True,
                    "is_current": False,
                }
            ]
        }
        decision = should_run_pre_deadline_ingest(client=client)
        self.assertFalse(decision["run"])
        self.assertIn("outside_window", decision["reason"])

    def test_force_runs(self):
        from datetime import datetime, timedelta, timezone
        from unittest.mock import MagicMock

        from fpl_robo.signals.pre_deadline import should_run_pre_deadline_ingest

        future = (datetime.now(timezone.utc) + timedelta(hours=48)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        client = MagicMock()
        client.bootstrap_static.return_value = {
            "events": [
                {
                    "id": 3,
                    "name": "Gameweek 3",
                    "deadline_time": future,
                    "finished": False,
                    "is_next": True,
                    "is_current": False,
                }
            ]
        }
        decision = should_run_pre_deadline_ingest(client=client, force=True)
        self.assertTrue(decision["run"])
        self.assertEqual(decision["gameweek"], 3)


if __name__ == "__main__":
    unittest.main()
