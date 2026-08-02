"""Cadence: daily / pre-deadline ingest and event-driven optimise."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fpl_robo.runtime import cadence as cadence_mod
from fpl_robo.runtime.cadence import (
    bootstrap_fingerprint,
    decide_ingest,
    decide_optimise,
    first_kickoff_on_local_day,
)


def _bootstrap(deadline: datetime, gw: int = 1) -> dict:
    return {
        "events": [
            {
                "id": gw,
                "name": f"Gameweek {gw}",
                "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished": False,
                "is_next": True,
                "is_current": False,
            }
        ],
        "elements": [
            {
                "id": 1,
                "now_cost": 100,
                "status": "a",
                "chance_of_playing_next_round": 100,
                "chance_of_playing_this_round": None,
                "ep_next": "5.0",
                "news": "",
                "form": "0.0",
            }
        ],
    }


class CadenceIngestTests(unittest.TestCase):
    def test_daily_fires_after_local_time_once(self):
        # Fixed "now" mid-morning London after 08:30.
        now = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)  # 10:00 BST
        deadline = now + timedelta(days=10)
        with patch.object(cadence_mod, "load_ingest_stamp", return_value={}):
            with patch.dict(
                cadence_mod.SIGNAL_INGEST,
                {
                    "daily_ingest_hour": 8,
                    "daily_ingest_minute": 30,
                    "timezone": "Europe/London",
                    "pre_deadline_hours": 0.5,
                },
                clear=False,
            ):
                d = decide_ingest(
                    bootstrap=_bootstrap(deadline),
                    fixtures=[],
                    stamp={},
                    now=now,
                )
        self.assertTrue(d["run"])
        self.assertEqual(d["kind"], "daily")

        with patch.object(
            cadence_mod,
            "load_ingest_stamp",
            return_value={"daily_date": "2026-07-28"},
        ):
            d2 = decide_ingest(
                bootstrap=_bootstrap(deadline),
                fixtures=[],
                stamp={"daily_date": "2026-07-28"},
                now=now,
            )
        self.assertFalse(d2["run"])

        # Outside morning window → wait until tomorrow.
        late = datetime(2026, 7, 28, 20, 0, tzinfo=timezone.utc)
        d3 = decide_ingest(
            bootstrap=_bootstrap(deadline),
            fixtures=[],
            stamp={},
            now=late,
        )
        self.assertFalse(d3["run"])

    def test_no_pre_match_scrape(self):
        now = datetime(2026, 8, 16, 11, 0, tzinfo=timezone.utc)
        kick = now + timedelta(hours=1.5)
        deadline = now + timedelta(days=2)
        fixtures = [
            {
                "id": 1,
                "event": 1,
                "finished": False,
                "kickoff_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "team_h": 1,
                "team_a": 2,
            }
        ]
        with patch.dict(
            cadence_mod.SIGNAL_INGEST,
            {"pre_deadline_hours": 0.5, "timezone": "Europe/London"},
            clear=False,
        ):
            d = decide_ingest(
                bootstrap=_bootstrap(deadline),
                fixtures=fixtures,
                stamp={"daily_date": "2026-08-16"},
                now=now,
            )
        self.assertFalse(d["run"])
        self.assertNotEqual(d.get("kind"), "pre_match")

    def test_pre_deadline_half_hour(self):
        now = datetime(2026, 8, 15, 16, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(minutes=20)
        with patch.dict(
            cadence_mod.SIGNAL_INGEST,
            {"pre_deadline_hours": 0.5, "timezone": "Europe/London"},
            clear=False,
        ):
            d = decide_ingest(
                bootstrap=_bootstrap(deadline),
                fixtures=[],
                stamp={"daily_date": "2026-08-15"},
                now=now,
            )
        self.assertTrue(d["run"])
        self.assertEqual(d["kind"], "pre_deadline")

    def test_pre_deadline_not_yet_at_one_hour(self):
        now = datetime(2026, 8, 15, 16, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(hours=1.0)
        with patch.dict(
            cadence_mod.SIGNAL_INGEST,
            {"pre_deadline_hours": 0.5, "timezone": "Europe/London"},
            clear=False,
        ):
            d = decide_ingest(
                bootstrap=_bootstrap(deadline),
                fixtures=[],
                stamp={"daily_date": "2026-08-15"},
                now=now,
            )
        self.assertFalse(d["run"])

    def test_skip_daily_when_deadline_earlier_same_morning(self):
        # London morning after 08:30; deadline was 07:00 local (06:00 UTC in BST? use winter UTC).
        # Use a January date so London is UTC+0: deadline 07:00, daily 08:30, now 09:00.
        now = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
        deadline = datetime(2026, 1, 10, 7, 0, tzinfo=timezone.utc)
        with patch.dict(
            cadence_mod.SIGNAL_INGEST,
            {
                "daily_ingest_hour": 8,
                "daily_ingest_minute": 30,
                "timezone": "Europe/London",
                "pre_deadline_hours": 0.5,
                "daily_ingest_window_hours": 6.0,
            },
            clear=False,
        ):
            d = decide_ingest(
                bootstrap=_bootstrap(deadline),
                fixtures=[],
                stamp={},
                now=now,
            )
        self.assertFalse(d["run"])
        self.assertTrue(d.get("skip_daily_for_deadline"))
        self.assertEqual(d["reason"], "skip_daily_deadline_earlier_today")

    def test_daily_still_runs_when_deadline_later_same_day(self):
        # Deadline 18:00 London, daily 08:30, now 09:00 — still do daily.
        now = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
        deadline = datetime(2026, 1, 10, 18, 0, tzinfo=timezone.utc)
        with patch.dict(
            cadence_mod.SIGNAL_INGEST,
            {
                "daily_ingest_hour": 8,
                "daily_ingest_minute": 30,
                "timezone": "Europe/London",
                "pre_deadline_hours": 0.5,
                "daily_ingest_window_hours": 6.0,
            },
            clear=False,
        ):
            d = decide_ingest(
                bootstrap=_bootstrap(deadline),
                fixtures=[],
                stamp={},
                now=now,
            )
        self.assertTrue(d["run"])
        self.assertEqual(d["kind"], "daily")


class CadenceOptimiseTests(unittest.TestCase):
    def test_skips_when_unchanged(self):
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(days=5)
        bs = _bootstrap(deadline)
        fp = bootstrap_fingerprint(bs)
        state = {
            "cadence": {
                "last_optimise_gw": 1,
                "last_bootstrap_fp": fp,
                "last_signals_mtime": 1_000_000.0,
                "last_signals_fp": "abc123",
                "near_optimise_gw": 1,
                "final_optimise_gw": 1,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            sig = Path(tmp) / "signals.json"
            sig.write_text("{}", encoding="utf-8")
            with patch.object(cadence_mod, "signals_mtime", return_value=1_000_000.0):
                with patch.object(cadence_mod, "signals_fingerprint", return_value="abc123"):
                    d = decide_optimise(bootstrap=bs, state=state, stamp={}, now=now)
        self.assertFalse(d["run"])
        self.assertEqual(d["reason"], "no_new_data")

    def test_runs_on_signals_fingerprint_change(self):
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(days=5)
        bs = _bootstrap(deadline)
        fp = bootstrap_fingerprint(bs)
        state = {
            "cadence": {
                "last_optimise_gw": 1,
                "last_bootstrap_fp": fp,
                "last_signals_mtime": 1_000_000.0,
                "last_signals_fp": "old_fp",
            }
        }
        with patch.object(cadence_mod, "signals_mtime", return_value=1_000_000.0):
            with patch.object(cadence_mod, "signals_fingerprint", return_value="new_fp"):
                d = decide_optimise(bootstrap=bs, state=state, stamp={}, now=now)
        self.assertTrue(d["run"])
        self.assertIn("new_signals", d["triggers"])

    def test_skips_pending_ingest_when_content_unchanged(self):
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(days=5)
        bs = _bootstrap(deadline)
        fp = bootstrap_fingerprint(bs)
        state = {
            "cadence": {
                "last_optimise_gw": 1,
                "last_bootstrap_fp": fp,
                "last_signals_fp": "same",
                "last_signals_mtime": 1.0,
            }
        }
        with patch.object(cadence_mod, "signals_mtime", return_value=1.0):
            with patch.object(cadence_mod, "signals_fingerprint", return_value="same"):
                d = decide_optimise(
                    bootstrap=bs,
                    state=state,
                    stamp={"pending_optimise": "daily", "content_unchanged": True},
                    now=now,
                )
        self.assertFalse(d["run"])

    def test_runs_on_bootstrap_change(self):
        now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
        deadline = now + timedelta(days=5)
        bs = _bootstrap(deadline)
        state = {
            "cadence": {
                "last_optimise_gw": 1,
                "last_bootstrap_fp": "old",
                "last_signals_mtime": 1.0,
                "last_signals_fp": "same",
            }
        }
        with patch.object(cadence_mod, "signals_mtime", return_value=1.0):
            with patch.object(cadence_mod, "signals_fingerprint", return_value="same"):
                d = decide_optimise(bootstrap=bs, state=state, stamp={}, now=now)
        self.assertTrue(d["run"])
        self.assertIn("bootstrap_change", d["triggers"])

    def test_first_kickoff_helper(self):
        day = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)
        fixtures = [
            {
                "finished": False,
                "kickoff_time": "2026-08-16T14:00:00Z",
            },
            {
                "finished": False,
                "kickoff_time": "2026-08-16T11:30:00Z",
            },
            {
                "finished": False,
                "kickoff_time": "2026-08-17T15:00:00Z",
            },
        ]
        first = first_kickoff_on_local_day(fixtures, day=day)
        self.assertIsNotNone(first)
        self.assertEqual(first.hour, 11)
        self.assertEqual(first.minute, 30)


if __name__ == "__main__":
    unittest.main()
