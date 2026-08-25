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

        join_hits = heuristic_signals_from_title(
            "Kerkez joins Liverpool from Bournemouth",
            [
                {
                    "id": 99,
                    "web_name": "Kerkez",
                    "first_name": "Milos",
                    "second_name": "Kerkez",
                    "team_short": "LIV",
                    "element_type": 2,
                    "position": "DEF",
                    "status": "a",
                    "selected_by_percent": 15.0,
                }
            ],
        )
        self.assertEqual(len(join_hits), 1)
        self.assertEqual(join_hits[0]["type"], "transfer")
        self.assertEqual(join_hits[0].get("destination_hint"), "Liverpool")

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
        self.assertFalse(impact.get("outbound"))

        gk = infer_transfer_impact(
            team_short="IPS",
            element_type=1,
            evidence="Ipswich sign goalkeeper Scherpen. expected to come straight in as No 1",
            note="",
            strength_home=2,
        )
        self.assertEqual(gk["role"], "starter")
        self.assertGreaterEqual(gk["minutes"], 45)

    def test_transfer_inference_outbound_non_pl(self):
        from fpl_robo.signals.transfer_infer import infer_transfer_impact
        from fpl_robo.signals.heuristics import heuristic_signals_from_title

        impact = infer_transfer_impact(
            team_short="CHE",
            evidence="has joined Al-Hilal",
            note="permanent move",
            destination_hint="Al-Hilal",
        )
        self.assertTrue(impact["outbound"])
        self.assertTrue(impact["exclude"])
        self.assertLess(impact["boost"], 0)
        self.assertEqual(impact["minutes"], 0)

        players = [
            {
                "id": 1,
                "web_name": "Jackson",
                "first_name": "Nicolas",
                "second_name": "Jackson",
                "team_short": "CHE",
                "status": "a",
            }
        ]
        hits = heuristic_signals_from_title(
            "Jackson sold to Al-Hilal in permanent move", players
        )
        self.assertTrue(hits)
        self.assertTrue(any(h.get("exclude") or h.get("outbound") for h in hits))

    def test_fpl_status_u_seeded_as_exclude(self):
        from fpl_robo.signals.fpl_status import signals_from_fpl_status

        seeded = signals_from_fpl_status(
            [
                {
                    "id": 9,
                    "web_name": "Gone",
                    "team_short": "MUN",
                    "status": "u",
                    "selected_by_percent": 0.1,
                    "news": "Transferred to another club",
                },
                {
                    "id": 10,
                    "web_name": "Loaned",
                    "team_short": "CHE",
                    "status": "i",
                    "selected_by_percent": 0.2,
                    "chance_of_playing_next_round": 0,
                    "news": "Has joined KVC Westerlo on loan for the rest of the season",
                },
            ]
        )
        by_name = {s["web_name"]: s for s in seeded}
        self.assertTrue(by_name["Gone"]["exclude"])
        self.assertEqual(by_name["Gone"]["type"], "transfer")
        self.assertTrue(by_name["Loaned"]["outbound"])
        self.assertEqual(by_name["Loaned"]["type"], "transfer")

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
                    source="ollama_lineup",
                    predicted_minutes=85,
                )
            ]
        )
        factors = aggregate_player_signals(
            bundle,
            players_by_id={1: {"id": 1, "web_name": "Salah"}},
        )
        self.assertGreaterEqual(factors[1].minutes_mult, 0.9)
        self.assertLessEqual(factors[1].minutes_mult, 1.05)

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

    def test_aggregate_prefers_predicted_minutes_over_role(self):
        from fpl_robo.signals.aggregate import aggregate_player_signals
        from fpl_robo.signals.loader import SignalBundle
        from fpl_robo.signals.schema import PredictedLineupSignal

        # Bench role alone would be ~0.88; 20 predicted minutes → ~0.35 floor path.
        bundle = SignalBundle(
            predicted_lineups=[
                PredictedLineupSignal(
                    player_id=1,
                    web_name="Welbeck",
                    team_short="BHA",
                    role="start",
                    gameweek=1,
                    confidence=1.0,
                    note="rotation risk",
                    predicted_minutes=45,
                )
            ]
        )
        factors = aggregate_player_signals(
            bundle,
            players_by_id={1: {"id": 1, "web_name": "Welbeck"}},
        )
        self.assertIn(1, factors)
        # Untrusted / soft numeric minutes (no source, role=start @45) do not move EV.
        self.assertAlmostEqual(factors[1].minutes_mult, 1.0, places=3)
        self.assertTrue(any("skipped_untrusted" in n for n in factors[1].notes))

    def test_welbeck_chelsea_transfer_gets_bench_minutes(self):
        from fpl_robo.signals.heuristics import heuristic_signals_from_title
        from fpl_robo.signals.ingest import (
            derive_lineups_from_transfers,
            _annotate_elements_team_short,
        )
        from fpl_robo.signals.transfer_infer import infer_transfer_impact
        from fpl_robo.signals.aggregate import aggregate_player_signals
        from fpl_robo.signals.loader import SignalBundle
        from fpl_robo.signals.schema import PredictedLineupSignal, NewsSignal, PreseasonSignal

        players = [
            {
                "id": 136,
                "web_name": "Welbeck",
                "first_name": "Danny",
                "second_name": "Welbeck",
                "team_short": "BHA",
                "element_type": 4,
                "position": "FWD",
                "team_strength_home": 2,
            }
        ]
        title = "Transfer news: Welbeck joins Chelsea, Brentford complete record Sangaré deal"
        summary = (
            "35-year-old Brighton striker Welbeck seals £5m move. "
            "Chelsea have signed Brighton’s veteran striker Danny Welbeck."
        )
        hits = heuristic_signals_from_title(title, players, summary=summary)
        self.assertTrue(hits)
        self.assertEqual(hits[0].get("destination_hint"), "Chelsea")
        impact = infer_transfer_impact(
            team_short="BHA",
            position="FWD",
            element_type=4,
            evidence=summary,
            note=hits[0].get("note") or "",
            destination_hint=hits[0].get("destination_hint"),
            strength_home=2,
        )
        self.assertEqual(impact["tier"], "elite")
        self.assertEqual(impact["destination_team_short"], "CHE")
        self.assertEqual(impact["role"], "backup")
        self.assertLessEqual(impact["predicted_minutes"], 25)

        news = [
            {
                **hits[0],
                "player_id": 136,
                "transfer_inference": impact,
                "destination_hint": "Chelsea",
            }
        ]
        # Minimal CHE forwards for depth model.
        elements = [
            {
                "id": 165,
                "web_name": "João Pedro",
                "element_type": 4,
                "team": 8,
                "team_short": "CHE",
                "ep_next": 2.3,
                "minutes": 2658,
                "status": "a",
            },
            {
                "id": 999,
                "web_name": "Other",
                "element_type": 4,
                "team": 8,
                "team_short": "CHE",
                "ep_next": 1.4,
                "minutes": 200,
                "status": "a",
            },
        ]
        rows = derive_lineups_from_transfers(news, gameweek=1, elements=elements)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["team_short"], "CHE")
        self.assertLessEqual(rows[0]["predicted_minutes"], 25)
        self.assertEqual(rows[0]["source"], "transfer_lineup")

        bundle = SignalBundle(
            news=[
                NewsSignal(
                    player_id=136,
                    web_name="Welbeck",
                    kind="transfer",
                    severity=0.15,
                    note="joins Chelsea",
                )
            ],
            preseason=[
                PreseasonSignal(
                    player_id=136,
                    web_name="Welbeck",
                    role="backup",
                    minutes=20,
                    starts=0,
                    boost=0.05,
                    note="auto from transfer",
                )
            ],
            predicted_lineups=[
                PredictedLineupSignal(
                    player_id=136,
                    web_name="Welbeck",
                    team_short="CHE",
                    role="bench",
                    confidence=0.78,
                    predicted_minutes=20,
                    note="transfer_lineup",
                    source="transfer_lineup",
                )
            ],
        )
        factors = aggregate_player_signals(
            bundle, players_by_id={136: {"id": 136, "web_name": "Welbeck"}}
        )
        self.assertIn(136, factors)
        # Backup preseason * low transfer minutes must crush EV vs a neutral 1.0.
        self.assertLess(factors[136].minutes_mult, 0.45)

    def test_club_depth_new_arrival_behind_starter(self):
        from fpl_robo.signals.minutes_estimate import club_depth_minutes

        welbeck = {"id": 136, "web_name": "Welbeck", "element_type": 4, "team_short": "BHA"}
        elements = [
            {
                "id": 165,
                "web_name": "João Pedro",
                "element_type": 4,
                "team_short": "CHE",
                "ep_next": 2.3,
                "minutes": 2658,
                "status": "a",
            },
            {
                "id": 200,
                "web_name": "Delap",
                "element_type": 4,
                "team_short": "CHE",
                "ep_next": 1.8,
                "minutes": 1000,
                "status": "a",
            },
        ]
        hit = club_depth_minutes(welbeck, elements=elements, team_short="CHE", element_type=4)
        self.assertIsNotNone(hit)
        mins, role, conf, note = hit
        self.assertLessEqual(mins, 30)
        self.assertEqual(role, "bench")
        self.assertIn("João Pedro", note)

    def test_schema_parses_predicted_minutes(self):
        from fpl_robo.signals.schema import PredictedLineupSignal

        sig = PredictedLineupSignal.from_dict(
            {
                "player_id": 7,
                "web_name": "Mateta",
                "team_short": "CRY",
                "role": "bench",
                "confidence": 0.8,
                "predicted_minutes": 25,
            }
        )
        self.assertEqual(sig.predicted_minutes, 25)
        clipped = PredictedLineupSignal.from_dict(
            {"web_name": "X", "role": "start", "minutes": 120}
        )
        self.assertEqual(clipped.predicted_minutes, 90)

    def test_lineup_player_entry_and_minutes_map(self):
        from fpl_robo.signals.llm import _lineup_player_entry, interpret_lineup_article
        from unittest.mock import patch

        self.assertEqual(_lineup_player_entry("Salah"), ("Salah", None))
        self.assertEqual(
            _lineup_player_entry({"name": "Welbeck", "predicted_minutes": 60}),
            ("Welbeck", 60),
        )

        fake = {
            "lineups": [
                {
                    "team_short": "BHA",
                    "starters": [{"name": "Welbeck", "predicted_minutes": 55}],
                    "bench": ["Mitoma"],
                    "minutes": {"Mitoma": 20},
                    "confidence": 0.8,
                    "evidence": "team news",
                }
            ]
        }
        with patch("fpl_robo.signals.llm.chat_json", return_value=fake):
            rows = interpret_lineup_article(
                title="Brighton team news",
                summary="Welbeck expected to start but may be hooked",
                source="test",
                candidate_lines=["Welbeck (BHA FWD)", "Mitoma (BHA MID)"],
                gameweek=1,
            )
        by_name = {r["web_name"]: r for r in rows}
        self.assertEqual(by_name["Welbeck"]["role"], "start")
        self.assertEqual(by_name["Welbeck"]["predicted_minutes"], 55)
        self.assertEqual(by_name["Mitoma"]["role"], "bench")
        self.assertEqual(by_name["Mitoma"]["predicted_minutes"], 20)

    def test_owned_club_article_triggers_lineup_flag(self):
        from fpl_robo.signals.ingest import article_mentions_owned_club
        from fpl_robo.signals.scrape import NewsArticle

        players = {
            1: {
                "id": 1,
                "web_name": "Welbeck",
                "team_id": 3,
                "team_short": "BHA",
                "team_name": "Brighton",
            }
        }
        aliases = {3: ["Brighton", "BHA", "Brighton & Hove Albion"]}
        self.assertTrue(
            article_mentions_owned_club(
                "Brighton expect Welbeck to lead the line again",
                {"BHA"},
                aliases,
                players_by_id=players,
            )
        )
        self.assertFalse(
            article_mentions_owned_club(
                "Liverpool injury update on Salah",
                {"BHA"},
                aliases,
                players_by_id=players,
            )
        )
        art = NewsArticle(
            title="Crystal Palace team news: Mateta set for limited minutes",
            summary="Palace host the visitors",
            source="test",
            link="https://example.test/palace",
        )
        from fpl_robo.signals.ingest import prefer_owned_club_articles

        ordered = prefer_owned_club_articles(
            [
                NewsArticle(
                    title="Unrelated Serie A gossip",
                    summary="transfer chatter",
                    source="test",
                    link="https://example.test/a",
                ),
                art,
            ],
            {"CRY"},
            {14: ["Crystal Palace", "CRY", "Palace"]},
            players_by_id={
                2: {
                    "id": 2,
                    "web_name": "Mateta",
                    "team_id": 14,
                    "team_short": "CRY",
                    "team_name": "Crystal Palace",
                }
            },
        )
        self.assertIn("Palace", ordered[0].title)

    def test_interpret_articles_lineup_on_owned_club_daily(self):
        from unittest.mock import patch

        from fpl_robo.signals.ingest import interpret_articles
        from fpl_robo.signals.match import build_player_indexes
        from fpl_robo.signals.scrape import NewsArticle

        players = [
            {
                "id": 1,
                "web_name": "Welbeck",
                "first_name": "Danny",
                "second_name": "Welbeck",
                "team_id": 3,
                "team_short": "BHA",
                "team_name": "Brighton",
                "status": "a",
            },
            {
                "id": 2,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
                "team_id": 14,
                "team_short": "LIV",
                "team_name": "Liverpool",
                "status": "a",
            },
        ]
        indexes = build_player_indexes(players)
        aliases = {
            3: ["Brighton", "BHA", "Brighton & Hove Albion"],
            14: ["Liverpool", "LIV"],
        }
        art = NewsArticle(
            title="Brighton team news ahead of opening weekend",
            summary="Danny Welbeck is expected to start but may not complete 90 minutes.",
            source="test",
            link="https://example.test/bha",
        )
        lineup_rows = [
            {
                "web_name": "Welbeck",
                "team_short": "BHA",
                "role": "start",
                "confidence": 0.85,
                "predicted_minutes": 60,
                "evidence": "expected to start",
                "note": "predicted start",
                "source": "ollama_lineup",
            }
        ]
        with patch(
            "fpl_robo.signals.ingest.interpret_lineup_article",
            return_value=lineup_rows,
        ) as lineup_llm, patch(
            "fpl_robo.signals.ingest.interpret_article",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.heuristic_signals_from_title",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.looks_like_lineup_article",
            return_value=False,
        ):
            _news, lineups, _rej, _unres = interpret_articles(
                [art],
                indexes,
                aliases,
                extract_lineups=False,
                allow_llm=True,
                owned_ids={1},
                owned_team_shorts={"BHA"},
                owned_players=[players[0]],
                owned_lineup_on_daily=True,
                gameweek=1,
                min_confidence=0.65,
            )
        # Minutes wording + owned club still extracts; heuristics skip lineup LLM.
        lineup_llm.assert_not_called()
        self.assertEqual(len(lineups), 1)
        self.assertEqual(lineups[0]["web_name"], "Welbeck")
        self.assertEqual(lineups[0]["source"], "heuristic_lineup")
        self.assertGreaterEqual(int(lineups[0].get("predicted_minutes") or 0), 60)

    def test_daily_skips_lineup_llm_on_bare_club_gossip(self):
        from unittest.mock import patch

        from fpl_robo.signals.ingest import interpret_articles
        from fpl_robo.signals.match import build_player_indexes
        from fpl_robo.signals.scrape import NewsArticle

        players = [
            {
                "id": 1,
                "web_name": "Welbeck",
                "first_name": "Danny",
                "second_name": "Welbeck",
                "team_id": 3,
                "team_short": "BHA",
                "team_name": "Brighton",
                "status": "a",
            }
        ]
        indexes = build_player_indexes(players)
        aliases = {3: ["Brighton", "BHA", "Brighton & Hove Albion"]}
        art = NewsArticle(
            title="Brighton transfer latest: club open to midfield offers",
            summary="Officials remain calm about recruitment this summer.",
            source="test",
            link="https://example.test/gossip",
        )
        with patch(
            "fpl_robo.signals.ingest.interpret_lineup_article",
            return_value=[],
        ) as lineup_llm, patch(
            "fpl_robo.signals.ingest.interpret_article",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.heuristic_signals_from_title",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.looks_like_lineup_article",
            return_value=False,
        ):
            _news, lineups, _rej, _unres = interpret_articles(
                [art],
                indexes,
                aliases,
                extract_lineups=False,
                allow_llm=True,
                owned_ids={1},
                owned_team_shorts={"BHA"},
                owned_players=players,
                owned_lineup_on_daily=True,
                gameweek=1,
                min_confidence=0.65,
            )
        lineup_llm.assert_not_called()
        self.assertEqual(lineups, [])

    def test_lineup_llm_budget_caps_calls(self):
        from unittest.mock import patch

        from fpl_robo.config import SIGNAL_INGEST
        from fpl_robo.signals.ingest import interpret_articles
        from fpl_robo.signals.match import build_player_indexes
        from fpl_robo.signals.scrape import NewsArticle

        players = [
            {
                "id": 1,
                "web_name": "Welbeck",
                "first_name": "Danny",
                "second_name": "Welbeck",
                "team_id": 3,
                "team_short": "BHA",
                "team_name": "Brighton",
                "status": "a",
            }
        ]
        indexes = build_player_indexes(players)
        aliases = {3: ["Brighton", "BHA"]}
        arts = [
            NewsArticle(
                title=f"Brighton predicted XI piece {i}",
                summary="Starting XI and bench roles for Welbeck.",
                source="test",
                link=f"https://example.test/xi{i}",
            )
            for i in range(3)
        ]
        with patch.dict(
            SIGNAL_INGEST,
            {
                "lineup_llm_max_articles": 1,
                "skip_lineup_llm_when_heuristic": False,
            },
            clear=False,
        ), patch(
            "fpl_robo.signals.ingest.interpret_lineup_article",
            return_value=[],
        ) as lineup_llm, patch(
            "fpl_robo.signals.ingest.interpret_article",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.heuristic_signals_from_title",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.heuristic_lineup_signals_from_text",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.looks_like_lineup_article",
            return_value=True,
        ):
            interpret_articles(
                arts,
                indexes,
                aliases,
                extract_lineups=False,
                allow_llm=True,
                owned_ids={1},
                owned_team_shorts={"BHA"},
                owned_players=players,
                owned_lineup_on_daily=True,
                gameweek=1,
                min_confidence=0.65,
            )
        self.assertEqual(lineup_llm.call_count, 1)

    def test_lineup_heuristics_sufficient_helper(self):
        from fpl_robo.signals.ingest import lineup_heuristics_are_sufficient

        rows = [
            {
                "player_id": 1,
                "role": "start",
                "predicted_minutes": 80,
                "confidence": 0.8,
            }
        ]
        self.assertTrue(
            lineup_heuristics_are_sufficient(
                rows,
                title="Brighton team news",
                owned_ids={1},
                strong_xi_article=False,
            )
        )
        self.assertFalse(
            lineup_heuristics_are_sufficient(
                rows,
                title="Brighton predicted XI",
                owned_ids={1},
                strong_xi_article=True,
            )
        )
        self.assertFalse(
            lineup_heuristics_are_sufficient(
                [],
                title="Brighton team news",
                owned_ids={1},
                strong_xi_article=False,
            )
        )

    def test_build_squad_minutes_check(self):
        from fpl_robo.signals.ingest import build_squad_minutes_check

        owned = [
            {"id": 1, "web_name": "Welbeck", "team_short": "BHA"},
            {"id": 2, "web_name": "Mateta", "team_short": "CRY"},
            {"id": 3, "web_name": "Nmecha", "team_short": "LEE"},
        ]
        lineups = [
            {
                "player_id": 1,
                "web_name": "Welbeck",
                "role": "start",
                "predicted_minutes": 60,
                "confidence": 0.9,
                "gameweek": 1,
            },
            {
                "player_id": 2,
                "web_name": "Mateta",
                "role": "bench",
                "predicted_minutes": 20,
                "confidence": 0.75,
                "gameweek": 1,
            },
        ]
        check = build_squad_minutes_check(owned, lineups, gameweek=1)
        self.assertEqual(check["owned"], 3)
        self.assertEqual(check["covered"], 2)
        self.assertEqual(check["missing_count"], 1)
        self.assertEqual(check["missing"][0]["web_name"], "Nmecha")
        by_name = {p["web_name"]: p for p in check["players"]}
        self.assertTrue(by_name["Welbeck"]["covered"])
        self.assertEqual(by_name["Welbeck"]["predicted_minutes"], 60)
        self.assertFalse(by_name["Nmecha"]["covered"])

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

        minutes = NewsArticle(
            source="test",
            title="Mateta expected to start but may share minutes",
            summary="Palace depth options discussed ahead of the opener",
            link="https://example.test/mins",
        )
        self.assertTrue(looks_like_lineup_article(minutes))

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

    def test_heuristic_lineup_minutes_from_role_wording(self):
        from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text

        players = [
            {
                "id": 223,
                "web_name": "Mateta",
                "first_name": "Jean-Philippe",
                "second_name": "Mateta",
                "team_short": "CRY",
            },
            {
                "id": 136,
                "web_name": "Welbeck",
                "first_name": "Danny",
                "second_name": "Welbeck",
                "team_short": "BHA",
            },
            {
                "id": 208,
                "web_name": "Sarr",
                "first_name": "Ismaila",
                "second_name": "Sarr",
                "team_short": "CRY",
            },
        ]
        text = (
            "Mateta is expected to start but could see limited minutes. "
            "Welbeck looks unlikely to start and may come from the bench. "
            "Sarr arrives as a depth option to deepen the squad."
        )
        rows = heuristic_lineup_signals_from_text(
            text, players, prefer_ids={223, 136, 208}, gameweek=1
        )
        by_name = {r["web_name"]: r for r in rows}
        self.assertEqual(by_name["Mateta"]["role"], "start")
        self.assertGreaterEqual(by_name["Mateta"]["predicted_minutes"], 45)
        self.assertLessEqual(by_name["Mateta"]["predicted_minutes"], 80)
        self.assertEqual(by_name["Welbeck"]["role"], "bench")
        self.assertLessEqual(by_name["Welbeck"]["predicted_minutes"], 25)
        self.assertEqual(by_name["Sarr"]["role"], "bench")

    def test_soft_mention_for_named_owned_without_role(self):
        from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text

        players = [
            {
                "id": 1,
                "web_name": "Raya",
                "first_name": "David",
                "second_name": "Raya",
                "team_short": "ARS",
                "team_name": "Arsenal",
                "ep_next": 4.5,
                "minutes": 3000,
                "starts": 35,
                "status": "a",
            }
        ]
        rows = heuristic_lineup_signals_from_text(
            "Arsenal keepers coach praised Raya ahead of the opener.",
            players,
            prefer_ids={1},
            gameweek=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "heuristic_mention")
        self.assertGreaterEqual(rows[0]["predicted_minutes"], 80)
        self.assertEqual(rows[0]["role"], "start")

    def test_soft_mention_requires_club_hint(self):
        from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text

        players = [
            {
                "id": 1,
                "web_name": "Gabriel",
                "first_name": "Gabriel",
                "second_name": "Magalhães",
                "team_short": "ARS",
                "team_name": "Arsenal",
                "ep_next": 4.5,
                "minutes": 3000,
                "starts": 35,
                "status": "a",
            },
            {
                "id": 2,
                "web_name": "Gabriel",
                "first_name": "Gabriel",
                "second_name": "Other",
                "team_short": "MUN",
                "team_name": "Man Utd",
                "ep_next": 1.0,
                "minutes": 100,
                "starts": 2,
                "status": "a",
            },
        ]
        rows = heuristic_lineup_signals_from_text(
            "Man Utd latest: Gabriel makes debut to become club's youngest player",
            players,
            prefer_ids={1},
            gameweek=1,
        )
        # Arsenal Gabriel must not claim a Man Utd-only article.
        self.assertFalse(any(int(r["player_id"]) == 1 for r in rows))

    def test_named_owned_survives_candidate_cap(self):
        from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text

        sarr = {
            "id": 208,
            "web_name": "Sarr",
            "first_name": "Ismaila",
            "second_name": "Sarr",
            "team_short": "CRY",
            "ep_next": 3.0,
            "minutes": 2000,
            "starts": 25,
            "status": "a",
        }
        filler = [
            {
                "id": 1000 + i,
                "web_name": f"Fill{i}",
                "first_name": "F",
                "second_name": f"Fill{i}",
                "team_short": "NFO",
                "ep_next": 1.0,
                "minutes": 100,
                "starts": 1,
                "status": "a",
            }
            for i in range(50)
        ]
        # Simulate post-cap list that forgot Sarr, then re-prepend as ingest does.
        capped = filler[:40]
        named_owned = [sarr]
        candidates = named_owned + [p for p in capped if int(p["id"]) != 208]
        rows = heuristic_lineup_signals_from_text(
            "FPL pre-season: Four in four for Jesus + Sarr expected to start",
            candidates,
            prefer_ids={208},
            gameweek=1,
        )
        self.assertTrue(any(r["web_name"] == "Sarr" for r in rows))

    def test_soft_mention_for_named_owned_without_role(self):
        from fpl_robo.signals.heuristics import heuristic_lineup_signals_from_text

        players = [
            {
                "id": 1,
                "web_name": "Raya",
                "first_name": "David",
                "second_name": "Raya",
                "team_short": "ARS",
                "team_name": "Arsenal",
                "ep_next": 4.5,
                "minutes": 3000,
                "starts": 35,
                "status": "a",
            }
        ]
        rows = heuristic_lineup_signals_from_text(
            "Arsenal keepers coach praised Raya ahead of the opener.",
            players,
            prefer_ids={1},
            gameweek=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "heuristic_mention")
        self.assertGreaterEqual(rows[0]["predicted_minutes"], 80)
        self.assertEqual(rows[0]["role"], "start")

    def test_ensure_owned_minutes_coverage_fills_15(self):
        from fpl_robo.signals.minutes_estimate import (
            ensure_owned_minutes_coverage,
            implied_minutes_point,
        )
        from fpl_robo.signals.ingest import build_squad_minutes_check

        owned = [
            {
                "id": i,
                "web_name": f"P{i}",
                "team_short": "ARS",
                "ep_next": 2.5,
                "minutes": 2000,
                "starts": 25,
                "status": "a",
            }
            for i in range(1, 16)
        ]
        news = [
            {
                "player_id": 1,
                "web_name": "P1",
                "role": "bench",
                "predicted_minutes": 20,
                "confidence": 0.85,
                "source": "heuristic_lineup",
                "gameweek": 1,
            }
        ]
        filled = ensure_owned_minutes_coverage(owned, news, gameweek=1)
        check = build_squad_minutes_check(owned, filled, gameweek=1)
        self.assertTrue(check["complete"])
        self.assertEqual(check["covered"], 15)
        self.assertEqual(check["missing_count"], 0)
        self.assertEqual(check["news_covered"], 1)
        self.assertEqual(check["fpl_implied_covered"], 14)
        by_id = {int(r["player_id"]): r for r in filled if r.get("player_id")}
        self.assertEqual(by_id[1]["source"], "heuristic_lineup")
        self.assertEqual(by_id[1]["predicted_minutes"], 20)
        self.assertEqual(by_id[2]["source"], "fpl_implied")
        self.assertGreater(by_id[2]["predicted_minutes"], 0)

        mins, role, conf = implied_minutes_point(5.0, None, "a", 3000, starts=35)
        self.assertEqual(role, "start")
        self.assertGreaterEqual(mins, 80)
        self.assertGreaterEqual(conf, 0.65)


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


class ArticleBodyCleanupTests(unittest.TestCase):
    def test_rejects_sky_nav_chrome(self):
        from fpl_robo.signals.scrape import _body_looks_like_chrome

        chrome = (
            "Football F1 Cricket Watch Sky Sports Get Sky Sports Sky Sports App "
            "More Sports Scores All Teams Premier League Arsenal Aston Villa "
            "Women's Super League Kick It Out British South Asians in Football "
            * 3
        )
        self.assertTrue(_body_looks_like_chrome(chrome))
        self.assertFalse(
            _body_looks_like_chrome(
                "Joe Gomez is expected to miss the start of Liverpool's Premier League "
                "campaign through a muscular injury sustained in pre-season training."
            )
        )


class BlankAheadConfigTests(unittest.TestCase):
    def test_gated_blank_ahead_enabled(self):
        from fpl_robo.config import BLANK_AHEAD

        self.assertTrue(BLANK_AHEAD.get("enabled"))
        self.assertTrue(BLANK_AHEAD.get("only_if_owned_blank"))


class AbsoluteFwdMinutesTests(unittest.TestCase):
    def test_fwd_ep_bands_prefer_quality_over_fillers(self):
        from fpl_robo.signals.minutes_policy import (
            club_rank_play_prob,
            fwd_absolute_ep_play_prob,
        )

        # Quality lead strikers stay near full minutes.
        self.assertGreaterEqual(fwd_absolute_ep_play_prob(2.3), 0.95)
        self.assertGreaterEqual(fwd_absolute_ep_play_prob(2.1), 0.90)
        # Cheap club fillers are haircut even as club EP#1.
        self.assertLessEqual(fwd_absolute_ep_play_prob(2.0), 0.82)
        self.assertLess(
            club_rank_play_prob(rank=1, position="FWD", ep_next=2.0),
            club_rank_play_prob(rank=1, position="FWD", ep_next=2.3),
        )

    def test_bench_gk_weight_below_outfield(self):
        from fpl_robo.config import BENCH_COVER

        self.assertLess(
            float(BENCH_COVER["bench_gk_ev_weight"]),
            float(BENCH_COVER["bench_ev_weight"]),
        )

    def test_attack_floor_requires_quality_fwds(self):
        from fpl_robo.config import ATTACK_FLOOR
        from fpl_robo.optimize.attack_floor import apply_attack_floor_constraint
        import pulp

        self.assertTrue(ATTACK_FLOOR.get("enabled"))
        self.assertGreaterEqual(int(ATTACK_FLOOR["min_quality_fwds"]), 3)
        ids = [1, 2, 3, 4, 5]
        pos = {1: "FWD", 2: "FWD", 3: "FWD", 4: "FWD", 5: "MID"}
        # Only 2 quality → soft floor requires both
        prob = pulp.LpProblem("atk_floor_test", pulp.LpMaximize)
        in_squad = pulp.LpVariable.dicts("s", ids, cat="Binary")
        n = apply_attack_floor_constraint(
            prob,
            in_squad=in_squad,
            ids=ids,
            pos=pos,
            ep_next={1: 2.5, 2: 2.3, 3: 2.0, 4: 1.8, 5: 3.0},
        )
        self.assertEqual(n, 2)
        prob += pulp.lpSum(in_squad[i] for i in ids) == 3
        prob += 10 * in_squad[4] + 10 * in_squad[5] + in_squad[1] + in_squad[2] + in_squad[3]
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pulp.LpStatus[status], "Optimal")
        self.assertGreaterEqual(
            sum(1 for i in (1, 2) if (in_squad[i].value() or 0) > 0.5), 2
        )
        # 3 quality → constraint active at full need
        prob2 = pulp.LpProblem("atk_floor_test2", pulp.LpMaximize)
        in2 = pulp.LpVariable.dicts("s", ids, cat="Binary")
        ep2 = {1: 2.5, 2: 2.3, 3: 2.1, 4: 1.8, 5: 3.0}
        n2 = apply_attack_floor_constraint(
            prob2, in_squad=in2, ids=ids, pos=pos, ep_next=ep2
        )
        self.assertEqual(n2, 3)
        prob2 += pulp.lpSum(in2[i] for i in ids) == 3
        # Prefer cheap junk if unconstrained — EV says pick 4,5 and one quality
        # but floor forces 3 quality FWDs.
        prob2 += 10 * in2[4] + 10 * in2[5] + in2[1] + in2[2] + in2[3]
        status = prob2.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pulp.LpStatus[status], "Optimal")
        picked_quality = sum(1 for i in (1, 2, 3) if (in2[i].value() or 0) > 0.5)
        self.assertGreaterEqual(picked_quality, 3)
        self.assertLessEqual((in2[4].value() or 0), 0.5)


class TestIngestDeadlineBudget(unittest.TestCase):
    def test_build_budget_stops_before_execute_window(self):
        from datetime import datetime, timezone
        from unittest.mock import patch

        from fpl_robo.config import SEASON_RUNNER, SIGNAL_INGEST
        from fpl_robo.signals.ingest import build_ingest_budget

        now = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
        with patch.dict(
            SIGNAL_INGEST,
            {
                "ingest_execute_reserve_minutes": 20,
                "ingest_llm_max_seconds": 0,
                "ingest_llm_max_seconds_pre_deadline": 0,
            },
            clear=False,
        ), patch.dict(SEASON_RUNNER, {"execute_within_hours": 0.5}, clear=False):
            # 1.5h to deadline, execute at 0.5h, reserve 20m → stop at T-50m = now+40m
            budget = build_ingest_budget(
                hours_to_deadline=1.5,
                pre_deadline=True,
                now=now,
            )
        self.assertIsNotNone(budget.hard_stop_at)
        assert budget.hard_stop_at is not None
        self.assertEqual(
            (budget.hard_stop_at - now).total_seconds(),
            40 * 60,
        )
        self.assertFalse(budget.llm_disabled)
        self.assertTrue(budget.llm_allowed(now=now))
        self.assertFalse(budget.llm_allowed(now=budget.hard_stop_at))
        self.assertEqual(budget.stop_reason, "execute_reserve")

    def test_build_budget_disables_when_already_in_reserve(self):
        from datetime import datetime, timezone
        from unittest.mock import patch

        from fpl_robo.config import SEASON_RUNNER, SIGNAL_INGEST
        from fpl_robo.signals.ingest import build_ingest_budget

        now = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
        with patch.dict(
            SIGNAL_INGEST,
            {
                "ingest_execute_reserve_minutes": 20,
                "ingest_llm_max_seconds_pre_deadline": 0,
            },
            clear=False,
        ), patch.dict(SEASON_RUNNER, {"execute_within_hours": 0.5}, clear=False):
            # Only 40 minutes left (< execute 30m + reserve 20m)
            budget = build_ingest_budget(
                hours_to_deadline=0.4,
                pre_deadline=True,
                now=now,
            )
        self.assertTrue(budget.llm_disabled)
        self.assertEqual(budget.stop_reason, "execute_reserve")
        self.assertFalse(budget.llm_allowed(now=now))

    def test_interpret_skips_llm_when_budget_exhausted(self):
        from datetime import datetime, timezone
        from unittest.mock import patch

        from fpl_robo.signals.ingest import IngestBudget, interpret_articles
        from fpl_robo.signals.match import build_player_indexes
        from fpl_robo.signals.scrape import NewsArticle

        players = [
            {
                "id": 10,
                "web_name": "Salah",
                "first_name": "Mohamed",
                "second_name": "Salah",
                "team_id": 14,
                "team_short": "LIV",
                "team_name": "Liverpool",
                "status": "a",
            }
        ]
        indexes = build_player_indexes(players)
        aliases = {14: ["Liverpool", "LIV"]}
        art = NewsArticle(
            title="Salah fitness update ahead of matchday",
            summary="Latest on Mohamed Salah availability for Liverpool.",
            source="test",
            link="https://example.test/salah",
        )
        now = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
        budget = IngestBudget(
            started_at=now,
            hard_stop_at=now,
            llm_disabled=True,
            stop_reason="execute_reserve",
            binding_reason="execute_reserve",
        )
        with patch(
            "fpl_robo.signals.ingest.interpret_article",
            return_value=[],
        ) as news_llm, patch(
            "fpl_robo.signals.ingest.heuristic_signals_from_title",
            return_value=[],
        ), patch(
            "fpl_robo.signals.ingest.article_is_actionable",
            return_value=True,
        ):
            interpret_articles(
                [art],
                indexes,
                aliases,
                allow_llm=True,
                llm_budget=budget,
                min_confidence=0.65,
            )
        news_llm.assert_not_called()
        self.assertGreaterEqual(budget.articles_after_stop, 1)


if __name__ == "__main__":
    unittest.main()
