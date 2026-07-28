"""Tests for preseason / unlimited churn guard."""

from __future__ import annotations

import unittest

from fpl_robo.runtime.deadline import is_preseason
from fpl_robo.runtime.season_runner import _unlimited_churn_decision


class PreseasonTests(unittest.TestCase):
    def test_is_preseason_when_no_finished_events(self):
        self.assertTrue(
            is_preseason({"events": [{"id": 1, "finished": False}, {"id": 2, "finished": False}]})
        )
        self.assertFalse(
            is_preseason({"events": [{"id": 1, "finished": True}, {"id": 2, "finished": False}]})
        )

    def test_preseason_skips_churn_guard(self):
        state = {"last_unlimited_gw": 1, "last_unlimited_objective": 50.0}
        decision = _unlimited_churn_decision(
            state=state,
            gameweek=1,
            plan_objective=50.0,  # no improvement
            preseason=True,
        )
        self.assertTrue(decision["allow"])
        self.assertEqual(decision["reason"], "preseason_always_apply")

    def test_midseason_unlimited_still_gated(self):
        state = {"last_unlimited_gw": 10, "last_unlimited_objective": 50.0}
        decision = _unlimited_churn_decision(
            state=state,
            gameweek=10,
            plan_objective=50.2,  # tiny improvement
            min_delta=1.0,
            preseason=False,
        )
        self.assertFalse(decision["allow"])


if __name__ == "__main__":
    unittest.main()
