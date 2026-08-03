"""Regression guards for known preseason failure modes."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from fpl_robo.config import KG_ENRICH
from fpl_robo.predict.phase import phase_kg_enrich
from fpl_robo.runtime.season_runner import (
    _owned_risk_element_ids,
    _preseason_execute_throttle,
)


class PhaseEnrichTests(unittest.TestCase):
    def test_preseason_tight_def_in_season_relaxed(self):
        with phase_kg_enrich(preseason=True) as pre:
            self.assertLessEqual(float(pre["def_ev_scale"]), 0.85)
            self.assertEqual(float(pre["cs_boost_scale"]), 0.0)
            self.assertEqual(float(pre["def_ep_ev_cap_mult"]), 1.0)
        with phase_kg_enrich(preseason=False) as inn:
            self.assertEqual(float(inn["def_ev_scale"]), 1.0)
            self.assertGreater(float(inn["cs_boost_scale"]), 0.0)
            self.assertEqual(float(inn["def_ep_ev_cap_mult"]), 0.0)
        # Restored defaults after context.
        self.assertEqual(
            float(KG_ENRICH.get("def_ev_scale") or 0),
            float(KG_ENRICH.get("def_ev_scale_preseason") or 0.82),
        )


class OwnedRiskAndThrottleTests(unittest.TestCase):
    def test_owned_risk_from_status_and_chance(self):
        import pandas as pd

        pred = pd.DataFrame(
            [
                {"id": 1, "status": "a", "chance_of_playing_next_round": 100},
                {"id": 2, "status": "i", "chance_of_playing_next_round": None},
                {"id": 3, "status": "a", "chance_of_playing_next_round": 25},
            ]
        )
        risky = _owned_risk_element_ids([1, 2, 3], predictions=pred, factors={})
        self.assertEqual(risky, {2, 3})

    def test_owned_risk_repair_bypasses_cooldown(self):
        decision = _preseason_execute_throttle(
            state={"last_executed_at": "2026-08-02T10:59:00+00:00"},
            preseason=True,
            optimise_triggers=["new_signals"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
            transfers_made=2,
            owned_risk_sold=1,
        )
        self.assertTrue(decision["allow"])
        self.assertIn("owned_risk_repair", decision["reason"])

    def test_transfer_cap_blocks_large_reshuffle(self):
        decision = _preseason_execute_throttle(
            state={"last_executed_at": "2026-08-01T01:00:00+00:00"},
            preseason=True,
            optimise_triggers=["bootstrap_change"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
            transfers_made=10,
            owned_risk_sold=0,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("transfer_cap", decision["reason"])

    def test_pre_deadline_trigger_bypasses_signal_throttle(self):
        decision = _preseason_execute_throttle(
            state={},
            preseason=True,
            optimise_triggers=["after_ingest:pre_deadline"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
            transfers_made=2,
            owned_risk_sold=0,
        )
        self.assertTrue(decision["allow"])
        self.assertIn("pre_deadline_execute", decision["reason"])


class ShadowMipHoldGateTests(unittest.TestCase):
    def _sol(self):
        from fpl_robo.optimize.single_gw import SquadSolution

        return SquadSolution(
            status="Optimal",
            objective=10.0,
            budget_used=100.0,
            gameweek=1,
            squad=[],
            starting_xi=[],
            bench=[],
            captain_id=None,
            captain_name=None,
        )

    def test_forces_hold_when_mip_holds_and_ep_delta_weak(self):
        import pandas as pd
        from unittest.mock import patch

        from fpl_robo.optimize.transfers import TransferPlan
        from fpl_robo.runtime.season_runner import _apply_shadow_mip_hold_gate

        preds = pd.DataFrame(
            [
                {"id": 8, "ep_next": 2.58, "ev": 2.58, "position": "MID", "price": 6.0, "web_name": "Out"},
                {"id": 16, "ep_next": 2.55, "ev": 2.55, "position": "MID", "price": 6.0, "web_name": "In"},
            ]
        )
        plan = TransferPlan(
            status="Optimal",
            objective=40.0,
            transfers=[{"element_out": 8, "element_in": 16}],
            transfers_made=1,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=15,
            solution=self._sol(),
            meta={},
        )
        shadow = TransferPlan(
            status="Optimal",
            objective=41.0,
            transfers=[],
            transfers_made=0,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=15,
            solution=self._sol(),
        )
        hold = TransferPlan(
            status="Optimal",
            objective=39.0,
            transfers=[],
            transfers_made=0,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=15,
            solution=self._sol(),
            meta={"shadow_mip_hold_gate": {"applied": True}},
        )
        with patch(
            "fpl_robo.runtime.season_runner.hold_plan_from_current",
            return_value=hold,
        ) as mocked:
            out = _apply_shadow_mip_hold_gate(
                plan=plan,
                shadow=shadow,
                predictions=preds,
                current_ids=[8],
                owned_risk_ids=set(),
                bank=0.0,
                selling_prices={},
                free_transfers=15,
                team_fixtures=None,
                gameweek=1,
            )
        mocked.assert_called_once()
        self.assertEqual(out.transfers_made, 0)
        self.assertTrue((out.meta or {}).get("shadow_mip_hold_gate", {}).get("applied"))

    def test_allows_when_ep_delta_strong(self):
        import pandas as pd

        from fpl_robo.optimize.transfers import TransferPlan
        from fpl_robo.runtime.season_runner import _apply_shadow_mip_hold_gate

        preds = pd.DataFrame(
            [
                {"id": 1, "ep_next": 2.0, "ev": 2.0, "position": "MID", "price": 5.0, "web_name": "A"},
                {"id": 2, "ep_next": 4.0, "ev": 4.0, "position": "MID", "price": 5.0, "web_name": "B"},
            ]
        )
        plan = TransferPlan(
            status="Optimal",
            objective=10.0,
            transfers=[{"element_out": 1, "element_in": 2}],
            transfers_made=1,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=1,
            solution=self._sol(),
        )
        shadow = TransferPlan(
            status="Optimal",
            objective=9.0,
            transfers=[],
            transfers_made=0,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=1,
            solution=self._sol(),
        )
        out = _apply_shadow_mip_hold_gate(
            plan=plan,
            shadow=shadow,
            predictions=preds,
            current_ids=[1],
            owned_risk_ids=set(),
            bank=0.0,
            selling_prices={},
            free_transfers=1,
            team_fixtures=None,
            gameweek=1,
        )
        self.assertEqual(out.transfers_made, 1)

    def test_allows_owned_risk_repair(self):
        import pandas as pd

        from fpl_robo.optimize.transfers import TransferPlan
        from fpl_robo.runtime.season_runner import _apply_shadow_mip_hold_gate

        preds = pd.DataFrame(
            [
                {"id": 1, "ep_next": 3.0, "ev": 3.0, "position": "MID", "price": 5.0, "web_name": "A"},
                {"id": 2, "ep_next": 2.9, "ev": 2.9, "position": "MID", "price": 5.0, "web_name": "B"},
            ]
        )
        plan = TransferPlan(
            status="Optimal",
            objective=10.0,
            transfers=[{"element_out": 1, "element_in": 2}],
            transfers_made=1,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=1,
            solution=self._sol(),
        )
        shadow = TransferPlan(
            status="Optimal",
            objective=9.0,
            transfers=[],
            transfers_made=0,
            hits=0,
            hit_cost=0,
            budget_left=0.0,
            free_transfers=1,
            solution=self._sol(),
        )
        out = _apply_shadow_mip_hold_gate(
            plan=plan,
            shadow=shadow,
            predictions=preds,
            current_ids=[1],
            owned_risk_ids={1},
            bank=0.0,
            selling_prices={},
            free_transfers=1,
            team_fixtures=None,
            gameweek=1,
        )
        self.assertEqual(out.transfers_made, 1)


class RepairFtCapTests(unittest.TestCase):
    def test_owned_risk_widens_sa_neighbourhood(self):
        from fpl_robo.optimize.live_vector import _effective_ft_hits

        ft, hits = _effective_ft_hits(15, 2, unlimited=True, owned_risk_count=0)
        self.assertEqual(ft, 2)
        self.assertEqual(hits, 0)
        ft2, hits2 = _effective_ft_hits(15, 2, unlimited=True, owned_risk_count=4)
        self.assertEqual(ft2, 4)
        self.assertEqual(hits2, 0)
        ft3, _ = _effective_ft_hits(15, 2, unlimited=True, owned_risk_count=9)
        self.assertEqual(ft3, 5)  # capped by preseason_repair_max_transfers


class LockfileFlockTests(unittest.TestCase):
    def test_second_acquire_fails_while_held(self):
        import tempfile
        from pathlib import Path
        from fpl_robo.runtime.lockfile import RunLock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.lock"
            a = RunLock(path)
            b = RunLock(path)
            self.assertTrue(a.acquire(stale_seconds=3600))
            self.assertFalse(b.acquire(stale_seconds=3600))
            a.release()
            self.assertTrue(b.acquire(stale_seconds=3600))
            b.release()


if __name__ == "__main__":
    unittest.main()
