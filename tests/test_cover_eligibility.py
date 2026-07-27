import unittest

from fpl_robo.predict.baseline import BaselineEVModel


class CoverEligibilityTests(unittest.TestCase):
    def test_dead_fodder_not_cover_eligible(self):
        model = BaselineEVModel()
        self.assertFalse(
            model._cover_eligible(
                {
                    "status": "a",
                    "chance_of_playing_next_round": None,
                    "ep_next": 1.0,
                    "minutes": 0,
                }
            )
        )

    def test_rotation_player_is_cover_eligible(self):
        model = BaselineEVModel()
        self.assertTrue(
            model._cover_eligible(
                {
                    "status": "a",
                    "chance_of_playing_next_round": 100,
                    "ep_next": 2.2,
                    "minutes": 0,
                }
            )
        )
        self.assertTrue(
            model._cover_eligible(
                {
                    "status": "a",
                    "chance_of_playing_next_round": None,
                    "ep_next": 1.0,
                    "minutes": 180,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
