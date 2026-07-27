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


if __name__ == "__main__":
    unittest.main()
