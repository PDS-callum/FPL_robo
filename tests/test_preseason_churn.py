"""Tests for preseason / unlimited churn guard."""

from __future__ import annotations

import unittest

from fpl_robo.runtime.deadline import is_preseason
from fpl_robo.runtime.season_runner import (
    _live_optimiser_name,
    _unlimited_churn_decision,
)


class PreseasonTests(unittest.TestCase):
    def test_is_preseason_when_no_finished_events(self):
        self.assertTrue(
            is_preseason({"events": [{"id": 1, "finished": False}, {"id": 2, "finished": False}]})
        )
        self.assertFalse(
            is_preseason({"events": [{"id": 1, "finished": True}, {"id": 2, "finished": False}]})
        )

    def test_preseason_respects_churn_guard_when_configured(self):
        # Config: preseason_skip_churn_guard=False — same GW needs obj delta.
        state = {
            "last_unlimited_gw": 1,
            "last_unlimited_objective": 50.0,
            "last_unlimited_optimiser": _live_optimiser_name(),
        }
        decision = _unlimited_churn_decision(
            state=state,
            gameweek=1,
            plan_objective=50.0,  # no improvement
            preseason=True,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("objective delta", decision["reason"])

    def test_preseason_attack_repair_still_allowed(self):
        state = {
            "last_unlimited_gw": 1,
            "last_unlimited_objective": 50.0,
            "last_unlimited_optimiser": _live_optimiser_name(),
        }
        decision = _unlimited_churn_decision(
            state=state,
            gameweek=1,
            plan_objective=40.0,
            preseason=True,
            plan_quality_fwds=3,
            current_quality_fwds=0,
        )
        self.assertTrue(decision["allow"])
        self.assertIn("attack_quality_repair", decision["reason"])

    def test_midseason_unlimited_still_gated(self):
        state = {
            "last_unlimited_gw": 10,
            "last_unlimited_objective": 50.0,
            "last_unlimited_optimiser": _live_optimiser_name(),
        }
        decision = _unlimited_churn_decision(
            state=state,
            gameweek=10,
            plan_objective=50.2,  # tiny improvement
            min_delta=1.0,
            preseason=False,
        )
        self.assertFalse(decision["allow"])

    def test_optimiser_switch_allows_first_write(self):
        state = {
            "last_unlimited_gw": 1,
            "last_unlimited_objective": 50.0,
            "last_unlimited_optimiser": "mip_pulp",
        }
        # Live is vector_team_sa — stale MIP obj must not gate.
        decision = _unlimited_churn_decision(
            state=state,
            gameweek=1,
            plan_objective=40.0,
            preseason=True,
        )
        self.assertTrue(decision["allow"])
        self.assertEqual(decision["reason"], "first_unlimited_write_this_gw")


if __name__ == "__main__":
    unittest.main()
