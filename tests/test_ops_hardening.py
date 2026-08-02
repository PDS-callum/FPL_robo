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
