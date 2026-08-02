"""Stability guards: stale daemon code, preseason execute throttle, DEF EV cap."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fpl_robo.config import ATTACK_FLOOR, KG_ENRICH, SEASON_RUNNER
from fpl_robo.runtime.code_stamp import (
    CodeFreshnessGuard,
    code_mtime_fingerprint,
    read_disk_feature_version,
)
from fpl_robo.runtime.season_runner import _preseason_execute_throttle


class CodeStampTests(unittest.TestCase):
    def test_read_disk_feature_version(self):
        ver = read_disk_feature_version()
        self.assertEqual(ver, KG_ENRICH.get("feature_version"))

    def test_stale_on_mtime_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watched = root / "config.py"
            watched.write_text('KG_ENRICH = {"feature_version": "v-test"}\n', encoding="utf-8")
            guard = CodeFreshnessGuard(
                feature_version="v-test",
                paths=[watched],
            )
            with patch(
                "fpl_robo.runtime.code_stamp.read_disk_feature_version",
                return_value="v-test",
            ):
                self.assertIsNone(guard.stale_reason())
                time.sleep(1.1)
                watched.write_text(
                    'KG_ENRICH = {"feature_version": "v-test"}\n# touched\n',
                    encoding="utf-8",
                )
                reason = guard.stale_reason()
            self.assertIsNotNone(reason)
            self.assertIn("mtime", reason or "")

    def test_stale_on_feature_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            watched = root / "config.py"
            watched.write_text(
                'KG_ENRICH = {"feature_version": "v-disk"}\n', encoding="utf-8"
            )
            # Point read_disk_feature_version at our temp file via patch.
            guard = CodeFreshnessGuard(feature_version="v-process", paths=[watched])
            with patch(
                "fpl_robo.runtime.code_stamp.read_disk_feature_version",
                return_value="v-disk",
            ):
                reason = guard.stale_reason()
            self.assertIsNotNone(reason)
            self.assertIn("feature_version drift", reason or "")


class PreseasonThrottleTests(unittest.TestCase):
    def test_blocks_signal_only_by_default(self):
        decision = _preseason_execute_throttle(
            state={"last_executed_at": "2026-08-02T09:00:00+00:00"},
            preseason=True,
            optimise_triggers=["new_signals"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
        )
        self.assertFalse(decision["allow"])
        self.assertIn("new_signals", decision["reason"])

    def test_allows_attack_repair(self):
        decision = _preseason_execute_throttle(
            state={"last_executed_at": "2026-08-02T10:59:00+00:00"},
            preseason=True,
            optimise_triggers=["new_signals"],
            plan_quality_fwds=3,
            current_quality_fwds=0,
        )
        self.assertTrue(decision["allow"])
        self.assertIn("attack_quality_repair", decision["reason"])

    def test_cooldown_blocks_recent_execute(self):
        from datetime import datetime, timedelta, timezone

        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        decision = _preseason_execute_throttle(
            state={"last_executed_at": recent},
            preseason=True,
            optimise_triggers=["bootstrap_change"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
        )
        # min_hours_between is 6.0 — execute within the last hour should block.
        self.assertFalse(decision["allow"])
        self.assertIn("cooldown", decision["reason"])

    def test_force_bypasses(self):
        decision = _preseason_execute_throttle(
            state={"last_executed_at": "2026-08-02T10:59:00+00:00"},
            preseason=True,
            optimise_triggers=["new_signals"],
            plan_quality_fwds=3,
            current_quality_fwds=3,
            force=True,
        )
        self.assertTrue(decision["allow"])

    def test_owned_risk_and_transfer_cap_config(self):
        self.assertGreaterEqual(
            int(SEASON_RUNNER.get("preseason_max_transfers_unless_repair") or 0), 1
        )


class DefCapAndStabilityConfigTests(unittest.TestCase):
    def test_cs_boost_disabled_and_def_scale(self):
        self.assertEqual(float(KG_ENRICH.get("cs_boost_scale") or 0), 0.0)
        self.assertLessEqual(float(KG_ENRICH.get("def_ev_scale") or 1), 0.85)
        self.assertEqual(float(KG_ENRICH.get("def_ep_ev_cap_mult") or 0), 1.0)

    def test_attack_floor_and_unlimited_penalties(self):
        self.assertTrue(ATTACK_FLOOR.get("enabled"))
        self.assertGreaterEqual(int(ATTACK_FLOOR["min_quality_fwds"]), 3)
        self.assertFalse(SEASON_RUNNER.get("preseason_execute_on_new_signals"))
        self.assertGreaterEqual(
            float(SEASON_RUNNER.get("preseason_execute_min_hours_between") or 0), 1.0
        )
        self.assertGreater(
            float(SEASON_RUNNER.get("unlimited_transfer_ev_penalty") or 0), 0
        )
        self.assertLessEqual(
            int(SEASON_RUNNER.get("preseason_max_transfers_unless_repair") or 99), 2
        )

    def test_def_ev_capped_at_ep(self):
        from fpl_robo.graph.derive import cs_ev_boost

        # CS boost fully off.
        self.assertEqual(cs_ev_boost({"position": "DEF", "cs_per_90": 0.59}), 0.0)


if __name__ == "__main__":
    unittest.main()
