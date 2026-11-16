import json
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from fpl_robo.revision import evidence, gates, ledger, loop
from fpl_robo.revision.gates import FileChange


class PathGateTests(unittest.TestCase):
    def test_code_and_tests_allowed(self):
        res = gates.check_paths(
            [FileChange("fpl_robo/optimize/x.py", "M", 10, 2), FileChange("tests/test_x.py", "A", 30, 0)]
        )
        self.assertTrue(res.ok, res.reasons)

    def test_protected_paths_rejected(self):
        for path in (
            "deploy/update.sh",
            "fpl_robo/revision/loop.py",
            "fpl_robo/auth.py",
            "revisions/ledger.json",
            "data/season_state.json",
            ".env",
            "requirements.txt",
        ):
            res = gates.check_paths([FileChange(path, "M", 1, 1)])
            self.assertFalse(res.ok, path)

    def test_outside_allowed_rejected(self):
        self.assertFalse(gates.check_paths([FileChange("setup.py", "A", 5, 0)]).ok)

    def test_report_dir_not_a_code_path(self):
        # The loop excludes its own report files before gating; anything else here is rejected.
        self.assertFalse(gates.check_paths([FileChange("revisions/2026-27/gw03.md", "A", 5, 0)]).ok)

    def test_deleting_tests_rejected(self):
        self.assertFalse(gates.check_paths([FileChange("tests/test_chips.py", "D", 0, 50)]).ok)

    def test_diff_size_cap(self):
        big = [FileChange("fpl_robo/a.py", "M", gates.MAX_CHANGED_LINES, 1)]
        self.assertFalse(gates.check_paths(big).ok)


class TestAndBacktestGateTests(unittest.TestCase):
    def test_existing_failures_tolerated(self):
        before = {"run": 10, "failed": ["a.T.test_1"]}
        after = {"run": 11, "failed": ["a.T.test_1"]}
        self.assertTrue(gates.check_tests(before, after).ok)

    def test_new_failure_rejected(self):
        before = {"run": 10, "failed": []}
        after = {"run": 10, "failed": ["a.T.test_2"]}
        self.assertFalse(gates.check_tests(before, after).ok)

    def test_fewer_tests_rejected(self):
        self.assertFalse(gates.check_tests({"run": 10, "failed": []}, {"run": 9, "failed": []}).ok)

    def test_runner_error_rejected(self):
        self.assertFalse(gates.check_tests({"run": 10, "failed": []}, {"run": 0, "failed": [], "error": "x"}).ok)

    def test_backtest_tolerance(self):
        self.assertTrue(gates.check_backtest(2600, 2600 - gates.BACKTEST_MAX_DROP).ok)
        self.assertFalse(gates.check_backtest(2600, 2600 - gates.BACKTEST_MAX_DROP - 1).ok)
        self.assertFalse(gates.check_backtest(2600, None).ok)


class StagedChangesTests(unittest.TestCase):
    def test_excludes_report_files_and_counts_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
            run("init", "-q")
            run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
            base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
            (repo / "fpl_robo").mkdir()
            (repo / "fpl_robo" / "a.py").write_text("x = 1\ny = 2\n")
            (repo / "revisions").mkdir()
            (repo / "revisions" / "gw01.md").write_text("# report\n")
            run("add", "-A")
            changes = gates.staged_changes(repo, base, exclude=("revisions/gw01.md",))
        self.assertEqual([(c.path, c.status, c.added) for c in changes], [("fpl_robo/a.py", "A", 2)])


class LedgerTests(unittest.TestCase):
    def test_upsert_replaces_same_gw_and_renders(self):
        e1 = {"season": "2026-27", "gw": 11, "status": "merged", "title": "Fix a|b", "live": {"delta_vs_average": -4}}
        e2 = dict(e1, status="reverted")
        entries = ledger.upsert(ledger.upsert([], e1), e2)
        self.assertEqual(len(entries), 1)
        md = ledger.render(entries)
        self.assertIn("**reverted**", md)
        self.assertIn("-4", md)
        self.assertIn("Fix a/b", md)  # pipes escaped so the table survives

    def test_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger.save(root, [{"season": "2026-27", "gw": 3, "status": "no_change"}])
            self.assertEqual(ledger.load(root)[0]["gw"], 3)
            self.assertTrue((root / ledger.LEDGER_MD).exists())


class EvidenceTests(unittest.TestCase):
    def test_finished_requires_data_checked(self):
        bs = {"events": [
            {"id": 1, "finished": True, "data_checked": True},
            {"id": 2, "finished": True, "data_checked": False},
            {"id": 3, "finished": False, "data_checked": False},
        ]}
        self.assertEqual(evidence.finished_gameweeks(bs), [1])

    def test_log_window_dedupes_problems(self):
        log_text = "\n".join([
            "2026-11-01 10:00:00,000 WARNING fpl_robo.x: before window 1",
            "2026-11-02 10:00:00,000 WARNING fpl_robo.x: timed out after 30s",
            "2026-11-02 11:00:00,000 WARNING fpl_robo.x: timed out after 45s",
            "2026-11-02 12:00:00,000 ERROR fpl_robo.season: Daemon cycle crashed; continuing",
            "Traceback (most recent call last):",
            "requests.exceptions.ConnectTimeout: host timed out",
            "2026-11-02 13:00:00,000 INFO fpl_robo.season: Optimiser=path_graph transfers=1",
            "2026-11-02 13:00:01,000 INFO fpl_robo.season: Cadence: nothing",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bot.log"
            path.write_text(log_text + "\n")
            decisions, problems, _ = evidence._log_window(
                path, datetime(2026, 11, 2), datetime(2026, 11, 3)
            )
        self.assertEqual(sum(problems.values()), 4)
        self.assertEqual(len(problems), 3)  # the two timeouts collapse into one pattern
        self.assertEqual(len(decisions), 1)


class LoopHelperTests(unittest.TestCase):
    def test_select_gameweek(self):
        bs = {"events": [{"id": i, "finished": True, "data_checked": True} for i in (1, 2, 3)]}
        self.assertEqual(loop.select_gameweek(bs, None), 3)
        self.assertEqual(loop.select_gameweek(bs, 2), 2)
        with self.assertRaises(loop.Skip):
            loop.select_gameweek(bs, 4)

    def test_report_title_strips_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.md"
            path.write_text(f"# {loop.SEASON} GW11: Guard bootstrap timeouts\n\nbody\n")
            self.assertEqual(loop.report_title(path, 11), "Guard bootstrap timeouts")
            self.assertEqual(loop.report_title(Path(tmp) / "missing.md", 11), "No changes")

    def test_scrubbed_env_drops_secrets(self):
        import os
        from unittest import mock

        env = {"PATH": "/bin", "FPL_PASSWORD": "x", "SMTP_PASSWORD": "y", "GH_TOKEN": "z", "HOME": "/h"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(loop.scrubbed_env(), {"PATH": "/bin", "HOME": "/h"})

    def test_prompt_template_formats(self):
        template = (Path(loop.__file__).parent / "prompt.md").read_text()
        text = template.format(
            season="2026-27", gw=11, evidence_path="e", report_path="r", inputs_dir="i", python="p",
            baseline_failed="none", baseline_backtest=2597, protected="deploy/", max_lines=600, backtest_drop=25,
        )
        self.assertIn("GW11", text)
        self.assertNotIn("{", text.replace("{season}", ""))  # nothing left unformatted
        json.dumps(text)


if __name__ == "__main__":
    unittest.main()
