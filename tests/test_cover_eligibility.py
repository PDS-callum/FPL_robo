import unittest

from fpl_robo.predict.baseline import BaselineEVModel


class CoverEligibilityTests(unittest.TestCase):
    def test_dead_fodder_not_cover_eligible(self):
        model = BaselineEVModel()
        # Low EP enabler even with some minutes — not real auto-sub cover.
        self.assertFalse(
            model._cover_eligible(
                {
                    "status": "a",
                    "position": "MID",
                    "chance_of_playing_next_round": None,
                    "ep_next": 1.0,
                    "minutes": 1500,
                    "starts": 20,
                }
            )
        )
        self.assertFalse(
            model._cover_eligible(
                {
                    "status": "a",
                    "position": "MID",
                    "chance_of_playing_next_round": None,
                    "ep_next": 1.0,
                    "minutes": 0,
                    "starts": 0,
                }
            )
        )

    def test_rotation_player_is_cover_eligible(self):
        model = BaselineEVModel()
        # Meaningful EP + real involvement.
        self.assertTrue(
            model._cover_eligible(
                {
                    "status": "a",
                    "position": "FWD",
                    "chance_of_playing_next_round": 100,
                    "ep_next": 2.2,
                    "minutes": 1200,
                    "starts": 12,
                }
            )
        )
        # Preseason / new signing: EP only when no minutes sample yet.
        self.assertTrue(
            model._cover_eligible(
                {
                    "status": "a",
                    "position": "MID",
                    "chance_of_playing_next_round": None,
                    "ep_next": 2.0,
                    "minutes": 0,
                    "starts": 0,
                }
            )
        )
        # Strong minutes but tiny EP still fails (bench fodder pattern).
        self.assertFalse(
            model._cover_eligible(
                {
                    "status": "a",
                    "position": "MID",
                    "ep_next": 1.0,
                    "minutes": 1800,
                    "starts": 19,
                }
            )
        )

    def test_backup_gk_allowed(self):
        model = BaselineEVModel()
        self.assertTrue(
            model._cover_eligible(
                {
                    "status": "a",
                    "position": "GK",
                    "ep_next": 2.0,
                    "minutes": 0,
                    "starts": 0,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
