import unittest
from datetime import datetime, timezone

from fpl_robo.runtime.deadline import detect_dgw_teams, hours_to_deadline, next_deadline


class DeadlineTests(unittest.TestCase):
    def test_next_deadline_prefers_is_next(self):
        bootstrap = {
            "events": [
                {
                    "id": 1,
                    "name": "GW1",
                    "deadline_time": "2026-08-21T17:30:00Z",
                    "finished": False,
                    "is_next": True,
                    "is_current": False,
                },
                {
                    "id": 2,
                    "name": "GW2",
                    "deadline_time": "2026-08-28T17:30:00Z",
                    "finished": False,
                    "is_next": False,
                    "is_current": False,
                },
            ]
        }
        d = next_deadline(bootstrap)
        self.assertEqual(d["id"], 1)

    def test_detect_dgw_teams(self):
        fixtures = [
            {"event": 5, "team_h": 1, "team_a": 2, "finished": False},
            {"event": 5, "team_h": 1, "team_a": 3, "finished": False},
            {"event": 5, "team_h": 4, "team_a": 5, "finished": False},
        ]
        self.assertEqual(detect_dgw_teams(fixtures, 5), {1})

    def test_hours_to_deadline(self):
        now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
        deadline = datetime(2026, 8, 21, 18, 0, tzinfo=timezone.utc)
        self.assertEqual(hours_to_deadline(deadline, now=now), 6.0)


class ExecutePolicyTests(unittest.TestCase):
    def test_normal_mode_waits_for_window(self):
        from fpl_robo.runtime.season_runner import _can_auto_execute

        self.assertFalse(_can_auto_execute(hours_to_deadline=6.0, unlimited_transfers=False))
        self.assertTrue(_can_auto_execute(hours_to_deadline=0.4, unlimited_transfers=False))
        self.assertFalse(_can_auto_execute(hours_to_deadline=-0.1, unlimited_transfers=False))

    def test_unlimited_allows_early_execute(self):
        from fpl_robo.runtime.season_runner import _can_auto_execute

        self.assertTrue(_can_auto_execute(hours_to_deadline=600.0, unlimited_transfers=True))
        self.assertFalse(_can_auto_execute(hours_to_deadline=-1.0, unlimited_transfers=True))

    def test_unlimited_churn_allows_first_write(self):
        from fpl_robo.runtime.season_runner import _unlimited_churn_decision

        decision = _unlimited_churn_decision(
            state={},
            gameweek=1,
            plan_objective=30.0,
            locked_objective=29.5,
            min_delta=1.0,
        )
        self.assertTrue(decision["allow"])
        self.assertEqual(decision["reason"], "first_unlimited_write_this_gw")

    def test_unlimited_churn_blocks_small_delta(self):
        from fpl_robo.runtime.season_runner import _unlimited_churn_decision

        decision = _unlimited_churn_decision(
            state={"last_unlimited_gw": 1, "last_unlimited_objective": 30.0},
            gameweek=1,
            plan_objective=30.4,
            min_delta=1.0,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("objective delta", decision["reason"])

    def test_unlimited_churn_allows_large_delta(self):
        from fpl_robo.runtime.season_runner import _unlimited_churn_decision

        decision = _unlimited_churn_decision(
            state={"last_unlimited_gw": 1, "last_unlimited_objective": 30.0},
            gameweek=1,
            plan_objective=31.5,
            min_delta=1.0,
        )
        self.assertTrue(decision["allow"])
        self.assertEqual(decision["reason"], "objective_improved")
        self.assertEqual(decision["delta"], 1.5)

    def test_unlimited_churn_resets_each_gw(self):
        from fpl_robo.runtime.season_runner import _unlimited_churn_decision

        decision = _unlimited_churn_decision(
            state={"last_unlimited_gw": 1, "last_unlimited_objective": 40.0},
            gameweek=2,
            plan_objective=30.0,
            min_delta=1.0,
        )
        self.assertTrue(decision["allow"])
        self.assertEqual(decision["reason"], "first_unlimited_write_this_gw")


if __name__ == "__main__":
    unittest.main()
