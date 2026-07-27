import unittest

from fpl_robo.data.team_updater import apply_initial_squad


class _FakeClient:
    def __init__(self) -> None:
        self._picks = [
            {
                "element": i,
                "purchase_price": 45,
                "selling_price": 45,
                "position": idx,
                "is_captain": i == 1,
                "is_vice_captain": i == 2,
            }
            for idx, i in enumerate(range(1, 16), start=1)
        ]
        self.transfer_calls = []

    def my_team(self, manager_id: int):
        return {"picks": list(self._picks)}

    def bootstrap_static(self):
        return {
            "elements": [{"id": i, "now_cost": 45, "element_type": 3} for i in range(1, 30)],
            "events": [{"id": 1, "is_next": True}],
        }

    def execute_transfers(self, payload):
        self.transfer_calls.append(payload)
        for t in payload.get("transfers") or []:
            for p in self._picks:
                if p["element"] == t["element_out"]:
                    p["element"] = t["element_in"]
                    p["purchase_price"] = t["purchase_price"]
                    p["selling_price"] = t["purchase_price"]
                    break
        return {"ok": True}

    def save_my_team(self, manager_id: int, picks, chip=None):
        self._picks = list(picks)
        return {"picks": list(self._picks), "chip": chip}


class ExecutePathTests(unittest.TestCase):
    def test_apply_initial_squad_persists_and_verifies(self):
        fake = _FakeClient()
        squad = [{"id": idx, "price": 4.5, "start": idx <= 11, "position": "MID", "ev": 2.0} for idx in range(1, 16)]
        out = apply_initial_squad(fake, manager_id=1687992, squad_rows=squad, captain_id=3)
        self.assertTrue(out["saved"])
        self.assertEqual(out["transfers_made"], 0)
        self.assertEqual(out["verified_count"], 15)
        self.assertEqual(out["captain_id"], 3)
        self.assertTrue(out["all_ids_verified"])

    def test_apply_initial_squad_runs_transfers(self):
        fake = _FakeClient()
        squad = [{"id": idx, "price": 4.5, "start": idx <= 11, "position": "MID", "ev": 2.0} for idx in range(1, 14)]
        squad.append({"id": 20, "price": 4.5, "start": False, "position": "MID", "ev": 2.0})
        squad.append({"id": 21, "price": 4.5, "start": False, "position": "MID", "ev": 2.0})
        out = apply_initial_squad(fake, manager_id=1687992, squad_rows=squad, captain_id=1, chip="3xc")
        self.assertTrue(out["saved"])
        self.assertEqual(out["transfers_made"], 2)
        self.assertEqual(out["chip"], "3xc")
        self.assertTrue(out["all_ids_verified"])
        self.assertEqual(len(fake.transfer_calls), 1)


if __name__ == "__main__":
    unittest.main()
