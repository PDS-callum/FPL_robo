"""Execute lineup verification and reviewed-plan freeze."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fpl_robo.data.team_updater import apply_weekly_plan
from fpl_robo.optimize.single_gw import SquadSolution
from fpl_robo.optimize.transfers import TransferPlan
from fpl_robo.runtime.season_runner import (
    _freeze_reviewed_execute_plan,
    _transfer_pair_key,
)


def _squad_row(pid: int, pos: str, *, start: bool, ev: float = 2.0) -> dict:
    return {
        "id": pid,
        "web_name": f"P{pid}",
        "position": pos,
        "price": 5.0,
        "ev": ev,
        "start": start,
        "captain": False,
        "vice_captain": False,
    }


class TransferKeyTests(unittest.TestCase):
    def test_pair_key_order_independent(self):
        a = [{"element_out": 1, "element_in": 2}]
        b = [{"out": 1, "in": 2}]
        self.assertEqual(_transfer_pair_key(a), _transfer_pair_key(b))


class FreezeReviewedPlanTests(unittest.TestCase):
    def test_freeze_reuses_reviewed_xi_when_transfers_match(self):
        reviewed_squad = (
            [_squad_row(i, "MID", start=True) for i in range(1, 12)]
            + [_squad_row(i, "DEF", start=False) for i in range(12, 16)]
        )
        # Flip one start flag on the "fresh" plan.
        fresh_squad = [dict(r) for r in reviewed_squad]
        fresh_squad[10]["start"] = False  # id 11
        fresh_squad[11]["start"] = True  # id 12

        prev = {
            "gameweek": 2,
            "transfer_plan": {
                "status": "Optimal",
                "objective": 30.0,
                "budget_left": 0.0,
                "free_transfers": 1,
                "transfers_made": 1,
                "hits": 0,
                "hit_cost": 0,
                "transfers": [{"element_out": 100, "element_in": 101}],
                "solution": {
                    "objective": 30.0,
                    "budget_used": 100.0,
                    "squad": reviewed_squad,
                    "captain_id": 1,
                    "captain_name": "P1",
                    "vice_captain_id": 2,
                    "vice_captain_name": "P2",
                },
            },
        }
        fresh = TransferPlan(
            status="Optimal",
            objective=31.0,
            budget_left=0.0,
            free_transfers=1,
            transfers_made=1,
            hits=0,
            hit_cost=0,
            transfers=[{"element_out": 100, "element_in": 101}],
            solution=SquadSolution(
                status="Optimal",
                objective=31.0,
                budget_used=100.0,
                gameweek=2,
                squad=fresh_squad,
                starting_xi=[r for r in fresh_squad if r["start"]],
                bench=[r for r in fresh_squad if not r["start"]],
                captain_id=1,
                captain_name="P1",
                vice_captain_id=2,
                vice_captain_name="P2",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(prev), encoding="utf-8")
            out = _freeze_reviewed_execute_plan(
                fresh,
                summary_path=path,
                gameweek=2,
                players_by_id={},
            )
        self.assertTrue((out.meta or {}).get("execute_freeze_reviewed_plan"))
        starts = {int(r["id"]) for r in out.solution.squad if r.get("start")}
        self.assertIn(11, starts)
        self.assertNotIn(12, starts)

    def test_freeze_skips_when_transfers_differ(self):
        squad = (
            [_squad_row(i, "MID", start=True) for i in range(1, 12)]
            + [_squad_row(i, "DEF", start=False) for i in range(12, 16)]
        )
        prev = {
            "gameweek": 2,
            "transfer_plan": {
                "status": "Optimal",
                "transfers": [{"element_out": 100, "element_in": 101}],
                "solution": {"squad": squad, "captain_id": 1},
            },
        }
        fresh = TransferPlan(
            status="Optimal",
            objective=31.0,
            budget_left=0.0,
            free_transfers=1,
            transfers_made=1,
            hits=0,
            hit_cost=0,
            transfers=[{"element_out": 100, "element_in": 999}],
            solution=SquadSolution(
                status="Optimal",
                objective=31.0,
                budget_used=100.0,
                gameweek=2,
                squad=squad,
                starting_xi=[r for r in squad if r["start"]],
                bench=[r for r in squad if not r["start"]],
                captain_id=1,
                captain_name="P1",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(prev), encoding="utf-8")
            out = _freeze_reviewed_execute_plan(
                fresh,
                summary_path=path,
                gameweek=2,
                players_by_id={},
            )
        self.assertFalse((out.meta or {}).get("execute_freeze_reviewed_plan"))


class LineupVerifyTests(unittest.TestCase):
    def test_mismatch_raises(self):
        squad = (
            [_squad_row(i, "MID", start=True, ev=3.0) for i in range(1, 12)]
            + [_squad_row(i, "DEF", start=False, ev=1.0) for i in range(12, 16)]
        )
        # Verified picks: id 12 started instead of id 11.
        verify_picks = []
        for idx, pid in enumerate(list(range(1, 11)) + [12] + [11, 13, 14, 15], start=1):
            verify_picks.append(
                {
                    "element": pid,
                    "position": idx,
                    "multiplier": 1 if idx <= 11 else 0,
                    "is_captain": pid == 1,
                    "is_vice_captain": pid == 2,
                    "purchase_price": 50,
                    "selling_price": 50,
                }
            )

        client = MagicMock()
        client.bootstrap_static.return_value = {
            "elements": [
                {"id": i, "now_cost": 50, "element_type": 3 if i < 12 else 2}
                for i in range(1, 16)
            ],
            "events": [{"id": 2, "is_next": True}],
        }
        client.my_team.side_effect = [
            {
                "picks": [
                    {
                        "element": i,
                        "selling_price": 50,
                        "purchase_price": 50,
                    }
                    for i in range(1, 16)
                ]
            },
            {"picks": verify_picks},
        ]
        client.save_my_team.return_value = {"picks": verify_picks}
        client.execute_transfers.return_value = {}

        with self.assertRaises(RuntimeError) as ctx:
            apply_weekly_plan(
                client,
                manager_id=1,
                squad_rows=squad,
                captain_id=1,
                event_id=2,
            )
        self.assertIn("Lineup save mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
