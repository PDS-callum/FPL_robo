"""Hard gates a revision must pass before the loop merges it."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Paths the revision run may never change. Anything here needs a human PR.
PROTECTED_PREFIXES = (
    "deploy/",
    "fpl_robo/revision/",
    "revisions/ledger.json",
    "revisions/LEDGER.md",
    "revisions/README.md",
    ".github/",
    ".gitignore",
    ".env",
    "data/",
    "requirements.txt",
    "fpl_robo/auth.py",
    "fpl_robo/oidc.py",
    "fpl_robo/runtime/lockfile.py",
    "fpl_robo/notify/",
    "tools/capture_session.py",
)

# Where code changes are allowed at all.
ALLOWED_PREFIXES = ("fpl_robo/", "tests/", "tools/", "README.md")

MAX_CHANGED_LINES = 600
BACKTEST_MAX_DROP = 25  # season points on the 2025-26 regression backtest

BACKTEST_ARGS = [
    "-m",
    "fpl_robo.cli",
    "backtest-season",
    "--season",
    "2025-26",
    "--ev-mode",
    "ep_baseline",
    "--tiebreak",
    "--blank-ahead",
    "--max-hits",
    "2",
    "--no-chips",
]

_UNITTEST_RUNNER = r"""
import io, json, sys, unittest
suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
failed = [t.id() for t, _ in result.failures + result.errors]
failed += [t.id() for t in result.unexpectedSuccesses]
print("__RESULT__" + json.dumps({"run": result.testsRun, "failed": sorted(set(failed))}))
"""


@dataclass
class FileChange:
    path: str
    status: str  # git name-status letter: A, M, D, R...
    added: int = 0
    deleted: int = 0


@dataclass
class GateResult:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)


def is_protected(path: str) -> bool:
    return any(path == p.rstrip("/") or path.startswith(p) for p in PROTECTED_PREFIXES)


def is_allowed(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in ALLOWED_PREFIXES)


def check_paths(changes: list[FileChange]) -> GateResult:
    res = GateResult()
    for ch in changes:
        if is_protected(ch.path):
            res.fail(f"touches protected path {ch.path}")
        elif not is_allowed(ch.path):
            res.fail(f"touches path outside fpl_robo/, tests/, tools/: {ch.path}")
        if ch.status.startswith("D") and ch.path.startswith("tests/"):
            res.fail(f"deletes test file {ch.path}")
    total = sum(ch.added + ch.deleted for ch in changes)
    if total > MAX_CHANGED_LINES:
        res.fail(f"diff too large: {total} changed lines > {MAX_CHANGED_LINES}")
    return res


def check_tests(before: dict, after: dict) -> GateResult:
    res = GateResult()
    if after.get("error"):
        res.fail(f"test run failed to complete: {after['error']}")
        return res
    if int(after.get("run", 0)) < int(before.get("run", 0)):
        res.fail(f"test count dropped {before.get('run')} -> {after.get('run')}")
    new_failures = sorted(set(after.get("failed", [])) - set(before.get("failed", [])))
    if new_failures:
        res.fail("new failing tests: " + ", ".join(new_failures[:10]))
    return res


def check_backtest(before: int | None, after: int | None) -> GateResult:
    res = GateResult()
    if after is None:
        res.fail("backtest did not produce a result")
    elif before is not None and after < before - BACKTEST_MAX_DROP:
        res.fail(f"backtest regressed {before} -> {after} (max drop {BACKTEST_MAX_DROP})")
    return res


def staged_changes(worktree: Path, base: str, exclude: tuple[str, ...] = ()) -> list[FileChange]:
    """Changes in the index vs base, excluding the run's own report files."""
    status = _git(worktree, "diff", "--cached", "--name-status", "--no-renames", base)
    numstat = _git(worktree, "diff", "--cached", "--numstat", "--no-renames", base)
    counts: dict[str, tuple[int, int]] = {}
    for line in numstat.splitlines():
        added, deleted, path = line.split("\t", 2)
        counts[path] = (int(added) if added.isdigit() else 0, int(deleted) if deleted.isdigit() else 0)
    changes = []
    for line in status.splitlines():
        letter, path = line.split("\t", 1)
        if path in exclude:
            continue
        added, deleted = counts.get(path, (0, 0))
        changes.append(FileChange(path=path, status=letter, added=added, deleted=deleted))
    return changes


def run_tests(worktree: Path, python: str, timeout: int = 900) -> dict:
    try:
        out = subprocess.run(
            [python, "-c", _UNITTEST_RUNNER],
            cwd=worktree,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"run": 0, "failed": [], "error": f"timed out after {timeout}s"}
    for line in out.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"run": 0, "failed": [], "error": (out.stderr or out.stdout)[-500:]}


def run_backtest(worktree: Path, python: str, timeout: int = 600) -> int | None:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "bt.json"
        try:
            subprocess.run(
                [python, *BACKTEST_ARGS, "--report-file", str(report)],
                cwd=worktree,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        if not report.exists():
            return None
        return int(json.loads(report.read_text(encoding="utf-8")).get("total_points") or 0)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout
