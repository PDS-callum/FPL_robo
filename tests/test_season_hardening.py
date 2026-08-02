"""Guards for season ops: quality alts, post-cooldown delta, feature stamp."""

from __future__ import annotations

import unittest

from fpl_robo.config import ATTACK_FLOOR, KG_ENRICH, SEASON_RUNNER, SIGNAL_INGEST
from fpl_robo.optimize.attack_quality import is_quality_fwd
from fpl_robo.predict.phase import phase_kg_enrich
from fpl_robo.runtime.season_runner import _live_optimiser_name, _preseason_execute_throttle


class AttackQualityAltTests(unittest.TestCase):
    def test_primary_ep_band(self):
        self.assertTrue(is_quality_fwd(ep_next=2.1))
        self.assertFalse(is_quality_fwd(ep_next=2.0, is_focal=False))

    def test_focal_alt_just_under_band(self):
        self.assertTrue(is_quality_fwd(ep_next=1.96, is_focal=True))
        self.assertFalse(is_quality_fwd(ep_next=1.80, is_focal=True))

    def test_form_and_minutes_alts(self):
        self.assertTrue(is_quality_fwd(ep_next=1.96, form=6.0))
        self.assertTrue(is_quality_fwd(ep_next=1.92, rolling_minutes=300))
        self.assertFalse(is_quality_fwd(ep_next=1.92, rolling_minutes=100))

    def test_unavailable_never_quality(self):
        self.assertFalse(is_quality_fwd(ep_next=4.0, status="u"))
        self.assertFalse(is_quality_fwd(ep_next=4.0, status="s", is_focal=True))


class PostCooldownThrottleTests(unittest.TestCase):
    def test_post_cooldown_requires_obj_delta(self):
        from unittest.mock import patch

        with patch.dict(
            "fpl_robo.runtime.season_runner.SEASON_RUNNER",
            {
                "preseason_post_cooldown_min_obj_delta": 3.0,
                "preseason_execute_min_hours_between": 0.0,
            },
            clear=False,
        ):
            decision = _preseason_execute_throttle(
                state={
                    "last_executed_at": "2026-08-01T01:00:00+00:00",
                    "last_unlimited_objective": 50.0,
                    "last_unlimited_optimiser": _live_optimiser_name(),
                },
                preseason=True,
                optimise_triggers=["bootstrap_change"],
                plan_quality_fwds=3,
                current_quality_fwds=3,
                transfers_made=2,
                owned_risk_sold=0,
                plan_objective=51.0,  # +1 < 3.0 bar
            )
        self.assertFalse(decision["allow"])
        self.assertIn("post_cooldown_delta", decision["reason"])

    def test_post_cooldown_allows_meaningful_lift(self):
        from unittest.mock import patch

        with patch.dict(
            "fpl_robo.runtime.season_runner.SEASON_RUNNER",
            {
                "preseason_post_cooldown_min_obj_delta": 3.0,
                "preseason_execute_min_hours_between": 0.0,
            },
            clear=False,
        ):
            decision = _preseason_execute_throttle(
                state={
                    "last_executed_at": "2026-08-01T01:00:00+00:00",
                    "last_unlimited_objective": 50.0,
                    "last_unlimited_optimiser": _live_optimiser_name(),
                },
                preseason=True,
                optimise_triggers=["bootstrap_change"],
                plan_quality_fwds=3,
                current_quality_fwds=3,
                transfers_made=2,
                owned_risk_sold=0,
                plan_objective=54.0,
            )
        self.assertTrue(decision["allow"])

    def test_transfer_cap_is_tight(self):
        self.assertLessEqual(
            int(SEASON_RUNNER.get("preseason_max_transfers_unless_repair") or 99), 2
        )
        decision = _preseason_execute_throttle(
            state={"last_executed_at": "2026-08-01T01:00:00+00:00"},
            preseason=True,
            optimise_triggers=["bootstrap_change"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
            transfers_made=3,
            owned_risk_sold=0,
            plan_objective=99.0,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("transfer_cap", decision["reason"])


class PhaseAndOpsConfigTests(unittest.TestCase):
    def test_feature_version(self):
        self.assertEqual(KG_ENRICH.get("feature_version"), "v15-vector-consensus")

    def test_in_season_def_restored(self):
        with phase_kg_enrich(preseason=False) as inn:
            self.assertEqual(float(inn["def_ev_scale"]), 1.0)
            self.assertGreater(float(inn["cs_boost_scale"]), 0.0)

    def test_attack_floor_alts_configured(self):
        self.assertTrue(ATTACK_FLOOR.get("allow_focal_alt"))
        self.assertLess(float(ATTACK_FLOOR["focal_min_ep"]), float(ATTACK_FLOOR["min_ep_next"]))

    def test_signal_confidence_raised(self):
        self.assertGreaterEqual(float(SIGNAL_INGEST["lineup_start_min_confidence"]), 0.75)
        self.assertGreaterEqual(float(SIGNAL_INGEST["ev_lineup_min_confidence"]), 0.78)

    def test_ops_bounds(self):
        self.assertLessEqual(int(SEASON_RUNNER["lock_stale_seconds"]), 900)
        self.assertLessEqual(int(SEASON_RUNNER["applied_moves_max_items"]), 40)
        self.assertGreaterEqual(float(SEASON_RUNNER["unlimited_min_obj_delta"]), 3.0)


if __name__ == "__main__":
    unittest.main()
