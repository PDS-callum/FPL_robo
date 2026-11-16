"""Post-gameweek revision loop.

Run hourly by deploy/fpl-robo-revision.timer. Once per finished gameweek:

1. Preflight: not paused, live checkout clean and fast-forwardable, the GW is
   final, the next deadline is far enough away, and the GW was not done yet.
2. Fresh git worktree of origin/<branch>; baseline unit tests + backtest.
3. Evidence pack -> headless `claude -p` reviews and edits the worktree.
4. Gates: protected paths, diff size, no new failing tests, backtest drop.
5. Squash-merge a PR with the report, evidence and ledger row; tag it.
   A gate failure merges the report only and parks the code on a
   `-rejected` branch for a human to look at.

Usage:
  python -m fpl_robo.revision.loop [--gw N] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fpl_robo.config import DEFAULT_MANAGER_ID
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.revision import evidence, gates, ledger
from fpl_robo.runtime.deadline import hours_to_deadline, next_deadline

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "2026-2027"
SEASON = "2026-27"
REMOTE = "origin"
WORKTREE = ROOT.parent / "FPL_robo-revision"
LOG_DIR = ROOT / "logs" / "revision"
STATE_FILE = LOG_DIR / "state.json"
PAUSE_FILE = ROOT / "data" / "revision_loop.paused"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
INPUTS_DIR = ".revision_inputs"

MIN_HOURS_TO_DEADLINE = float(os.environ.get("REVISION_MIN_HOURS_TO_DEADLINE", "24"))
CLAUDE_TIMEOUT = int(os.environ.get("REVISION_CLAUDE_TIMEOUT", "3600"))
CLAUDE_MODEL = os.environ.get("REVISION_MODEL", "")
MAX_ATTEMPTS = 2

LABEL = "revision-loop"
LABEL_REJECTED = "revision-rejected"
AUTHOR_ENV = {
    "GIT_AUTHOR_NAME": "FPL Robo revision loop",
    "GIT_AUTHOR_EMAIL": "revision-loop@fpl-robo.local",
    "GIT_COMMITTER_NAME": "FPL Robo revision loop",
    "GIT_COMMITTER_EMAIL": "revision-loop@fpl-robo.local",
}
CLAUDE_TRAILER = "Co-Authored-By: Claude <noreply@anthropic.com>"

log = logging.getLogger("fpl_robo.revision")


class Skip(Exception):
    """Nothing to do this tick (not an error)."""


# --------------------------------------------------------------------------- helpers


def _run(args: list[str], cwd: Path | str, *, env: dict | None = None, check: bool = True) -> str:
    full_env = {**os.environ, **(env or {})}
    out = subprocess.run(args, cwd=cwd, env=full_env, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])} failed: {(out.stderr or out.stdout).strip()[-800:]}")
    return out.stdout.strip()


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return _run(["git", *args], cwd, env=AUTHOR_ENV, check=check)


def gh(*args: str) -> str:
    # Run outside any checkout so gh never switches local branches.
    return _run(["gh", *args, "--repo", repo_slug()], "/")


_SLUG: str | None = None


def repo_slug() -> str:
    global _SLUG
    if _SLUG is None:
        url = git(ROOT, "remote", "get-url", REMOTE)
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)", url)
        if not m:
            raise RuntimeError(f"cannot parse GitHub repo from {url}")
        _SLUG = m.group(1)
    return _SLUG


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"attempts": {}}


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def key_for(gw: int) -> str:
    return f"{SEASON}-gw{gw}"


def tag_for(gw: int) -> str:
    return f"revision-{SEASON}-gw{gw}"


def branch_for(gw: int) -> str:
    return f"revision/{SEASON}-gw{gw}"


def report_rel(gw: int) -> str:
    return f"revisions/{SEASON}/gw{gw:02d}.md"


def evidence_rel(gw: int) -> str:
    return f"revisions/{SEASON}/gw{gw:02d}-evidence.md"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def notify(subject: str, body: str) -> None:
    try:
        from fpl_robo.notify.email_alerts import send_email

        res = send_email(subject=subject, body=body)
        if not res.get("ok"):
            log.info("Email not sent: %s", res.get("reason"))
    except Exception as exc:  # notification is best-effort
        log.warning("Email failed: %s", exc)


# --------------------------------------------------------------------------- preflight


def preflight_live_checkout() -> None:
    """Merged revisions only deploy if update.sh can fast-forward the live tree."""
    dirty = git(ROOT, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise Skip(
            "live checkout has uncommitted changes to tracked files; update.sh could not "
            "deploy a merged revision. Commit or stash them first:\n" + dirty
        )
    head = git(ROOT, "rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", head, f"{REMOTE}/{BRANCH}"], cwd=ROOT)
    if ancestor.returncode != 0:
        raise Skip(f"live HEAD {head[:8]} is not an ancestor of {REMOTE}/{BRANCH}; cannot fast-forward")


def untracked_collisions(paths: list[str]) -> list[str]:
    """Paths a revision adds that exist untracked in the live tree (would block ff pull)."""
    return [p for p in paths if (ROOT / p).exists() and not git(ROOT, "ls-files", p)]


def ledger_on_remote() -> list[dict[str, Any]]:
    raw = git(ROOT, "show", f"{REMOTE}/{BRANCH}:{ledger.LEDGER_JSON}", check=False)
    try:
        return json.loads(raw) if raw else []
    except ValueError:
        return []


# --------------------------------------------------------------------------- worktree


def prepare_worktree(gw: int) -> tuple[Path, str]:
    if WORKTREE.exists():
        git(ROOT, "worktree", "remove", "--force", str(WORKTREE), check=False)
        shutil.rmtree(WORKTREE, ignore_errors=True)
    git(ROOT, "worktree", "prune")
    base = git(ROOT, "rev-parse", f"{REMOTE}/{BRANCH}")
    branch = branch_for(gw)
    git(ROOT, "branch", "-D", branch, check=False)
    git(ROOT, "worktree", "add", "-b", branch, str(WORKTREE), base)
    # Backtests read historical data from the (gitignored) cache.
    cache = ROOT / "data" / "cache"
    if cache.exists():
        shutil.copytree(cache, WORKTREE / "data" / "cache", dirs_exist_ok=True)
    return WORKTREE, base


def write_inputs(wt: Path) -> None:
    inputs = wt / INPUTS_DIR
    inputs.mkdir(exist_ok=True)
    for name in ("season_state.json", "last_run_summary.json"):
        src = ROOT / "data" / name
        if src.exists():
            shutil.copy2(src, inputs / name)


# --------------------------------------------------------------------------- claude


_SECRET_ENV = re.compile(r"^(FPL_|SMTP_|ALERT_|GH_|GITHUB_)|PASSWORD|SECRET|COOKIE|SESSION")


def scrubbed_env() -> dict[str, str]:
    """Environment for the review run without bot/GitHub credentials."""
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}


def run_claude(wt: Path, gw: int, baseline_tests: dict, baseline_bt: int | None) -> dict[str, Any]:
    template = (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8")
    prompt = template.format(
        season=SEASON,
        gw=gw,
        evidence_path=evidence_rel(gw),
        report_path=report_rel(gw),
        inputs_dir=INPUTS_DIR,
        python=PYTHON,
        baseline_failed=", ".join(baseline_tests.get("failed", [])) or "(none)",
        baseline_backtest=baseline_bt,
        protected=", ".join(gates.PROTECTED_PREFIXES),
        max_lines=gates.MAX_CHANGED_LINES,
        backtest_drop=gates.BACKTEST_MAX_DROP,
    )
    claude = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")
    allowed = [
        "Read", "Edit", "Write", "Glob", "Grep", "TodoWrite",
        f"Bash({PYTHON}:*)",
        "Bash(git diff:*)", "Bash(git status:*)", "Bash(git log:*)", "Bash(git show:*)",
        "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(grep:*)", "Bash(wc:*)",
    ]
    home = Path.home()
    denied = [
        "WebFetch", "WebSearch",
        f"Read(/{ROOT}/.env*)", f"Read(/{ROOT}/data/session.json)",
        f"Read(/{home}/.config/gh/**)", f"Read(/{home}/.ssh/**)", f"Read(/{home}/.git-credentials)",
    ]
    cmd = [
        claude, "-p", prompt,
        "--permission-mode", "dontAsk",
        "--allowedTools", *allowed,
        "--disallowedTools", *denied,
        "--strict-mcp-config",
        "--output-format", "stream-json", "--verbose",
    ]
    if CLAUDE_MODEL:
        cmd += ["--model", CLAUDE_MODEL]
    transcript = LOG_DIR / f"{key_for(gw)}.jsonl"
    log.info("Running claude (transcript %s)", transcript)
    started = datetime.now(timezone.utc)
    try:
        with transcript.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd,
                cwd=wt,
                env=scrubbed_env(),
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=CLAUDE_TIMEOUT,
                text=True,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    result: dict[str, Any] = {}
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if evt.get("type") == "result":
            result = evt
    return {
        "ok": rc == 0 and not result.get("is_error"),
        "returncode": rc,
        "minutes": round((datetime.now(timezone.utc) - started).total_seconds() / 60, 1),
        "num_turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd"),
        "transcript": str(transcript.relative_to(ROOT)),
    }


# --------------------------------------------------------------------------- reporting


def report_title(report: Path, gw: int) -> str:
    if report.exists():
        first = report.read_text(encoding="utf-8").splitlines()[:1]
        if first and first[0].startswith("#"):
            title = first[0].lstrip("# ").strip()
            title = re.sub(rf"^{re.escape(SEASON)}\s+GW{gw}\s*:\s*", "", title)
            if title:
                return title[:90]
    return "No changes"


def gates_section(
    *,
    verdict: str,
    reasons: list[str],
    changes: list[gates.FileChange],
    tests_before: dict,
    tests_after: dict | None,
    bt_before: int | None,
    bt_after: int | None,
    claude: dict[str, Any],
    rejected_url: str | None,
) -> str:
    lines = ["", "## Loop gates (written by the loop, not the model)", ""]
    lines.append(f"- Verdict: **{verdict}**")
    for r in reasons:
        lines.append(f"  - {r}")
    lines.append(
        f"- Files changed: {len(changes)}, lines changed: {sum(c.added + c.deleted for c in changes)}"
    )
    for c in changes:
        lines.append(f"  - `{c.status} {c.path}` (+{c.added} −{c.deleted})")
    after_failed = (tests_after or {}).get("failed", tests_before.get("failed", []))
    lines.append(
        f"- Unit tests: {tests_before.get('run')} run / {len(tests_before.get('failed', []))} failing before → "
        f"{(tests_after or tests_before).get('run')} run / {len(after_failed)} failing after"
    )
    lines.append(f"- Backtest 2025-26 total points: {bt_before} → {bt_after if bt_after is not None else bt_before}")
    lines.append(
        f"- Claude run: {claude.get('minutes')} min, {claude.get('num_turns')} turns, "
        f"cost {claude.get('cost_usd')}, transcript `{claude.get('transcript')}` (on the Pi)"
    )
    if rejected_url:
        lines.append(f"- Rejected code is parked on [{rejected_url.rsplit('/', 1)[-1]}]({rejected_url}) for review.")
    lines.append("")
    return "\n".join(lines)


def pr_body(report_text: str, gw: int, status: str) -> str:
    rollback = (
        f"\n---\n**Rollback:** `.venv/bin/python -m fpl_robo.revision.rollback --gw {gw}` "
        f"(reverts tag `{tag_for(gw)}`, keeps the report and ledger).\n"
        if status == "merged"
        else ""
    )
    return (
        f"Automated post-gameweek revision for {SEASON} GW{gw} (status: **{status}**).\n"
        f"Ledger: [`revisions/LEDGER.md`](../blob/{BRANCH}/revisions/LEDGER.md)\n\n"
        + report_text
        + rollback
        + "\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n"
    )


# --------------------------------------------------------------------------- main flow


def select_gameweek(bootstrap: dict[str, Any], requested: int | None) -> int:
    finished = evidence.finished_gameweeks(bootstrap)
    if requested is not None:
        if requested not in finished:
            raise Skip(f"GW{requested} is not finished and data-checked yet")
        return requested
    if not finished:
        raise Skip("no finished gameweek yet")
    return max(finished)


def run(args: argparse.Namespace) -> int:
    if PAUSE_FILE.exists() and not args.force:
        raise Skip(f"paused ({PAUSE_FILE.relative_to(ROOT)} exists)")

    git(ROOT, "fetch", "--quiet", REMOTE, BRANCH)
    if not args.dry_run:
        preflight_live_checkout()

    client = FPLClient()
    bootstrap = client.bootstrap_static()
    gw = select_gameweek(bootstrap, args.gw)
    key = key_for(gw)

    if any(e["season"] == SEASON and e["gw"] == gw for e in ledger_on_remote()) and not args.force:
        raise Skip(f"{key} already in the ledger")
    state = load_state()
    attempts = state["attempts"].get(key, {})
    if int(attempts.get("count", 0)) >= MAX_ATTEMPTS and not args.force:
        raise Skip(f"{key} already attempted {attempts['count']} times (last: {attempts.get('last_status')})")

    nd = next_deadline(bootstrap)
    hours = hours_to_deadline(nd["deadline_utc"]) if nd else None
    if hours is not None and hours < MIN_HOURS_TO_DEADLINE and not args.force:
        raise Skip(f"only {hours:.1f}h to the GW{nd.get('id')} deadline (< {MIN_HOURS_TO_DEADLINE}h)")

    log.info("Revision for %s starting (dry_run=%s)", key, args.dry_run)
    attempts = {"count": int(attempts.get("count", 0)) + 1, "last_at": datetime.now(timezone.utc).isoformat()}
    state["attempts"][key] = attempts
    save_state(state)

    wt, base = prepare_worktree(gw)
    write_inputs(wt)

    log.info("Baseline tests + backtest on %s", base[:8])
    tests_before = gates.run_tests(wt, PYTHON)
    bt_before = gates.run_backtest(wt, PYTHON)

    manager_id = int(os.environ.get("FPL_MANAGER_ID") or DEFAULT_MANAGER_ID)
    evidence_md, live = evidence.build(
        client=client,
        bootstrap=bootstrap,
        manager_id=manager_id,
        gw=gw,
        season=SEASON,
        log_path=ROOT / "logs" / "fpl_robo.log",
        season_state=_read_json(ROOT / "data" / "season_state.json"),
    )
    (wt / evidence_rel(gw)).parent.mkdir(parents=True, exist_ok=True)
    (wt / evidence_rel(gw)).write_text(evidence_md, encoding="utf-8")

    claude = run_claude(wt, gw, tests_before, bt_before)
    if not claude["ok"]:
        attempts["last_status"] = f"claude_failed rc={claude['returncode']}"
        save_state(state)
        raise RuntimeError(f"claude run failed (rc={claude['returncode']}); see {claude['transcript']}")

    report = wt / report_rel(gw)
    if not report.exists():
        report.write_text(f"# {SEASON} GW{gw}: No changes\n\nThe review run did not write a report.\n", encoding="utf-8")
    title = report_title(report, gw)

    shutil.rmtree(wt / INPUTS_DIR, ignore_errors=True)
    git(wt, "add", "-A")
    changes = gates.staged_changes(wt, base, exclude=(report_rel(gw), evidence_rel(gw)))

    verdict, reasons, tests_after, bt_after, rejected_url = "no_change", [], None, None, None
    if changes:
        check = gates.check_paths(changes)
        collisions = untracked_collisions([c.path for c in changes if c.status.startswith("A")])
        if collisions:
            check.fail("adds files that exist untracked in the live tree: " + ", ".join(collisions))
        if check.ok:
            tests_after = gates.run_tests(wt, PYTHON)
            check.reasons += gates.check_tests(tests_before, tests_after).reasons
            bt_after = gates.run_backtest(wt, PYTHON)
            check.reasons += gates.check_backtest(bt_before, bt_after).reasons
            check.ok = not check.reasons
        verdict, reasons = ("merged", []) if check.ok else ("rejected", check.reasons)
        log.info("Gates for %s: %s %s", key, verdict, "; ".join(reasons))

    if verdict == "rejected" and not args.dry_run:
        rejected_branch = f"{branch_for(gw)}-rejected"
        git(wt, "commit", "-q", "-m", f"revision({SEASON} GW{gw}) rejected candidate: {title}\n\n{CLAUDE_TRAILER}")
        git(wt, "push", "-q", "--force", REMOTE, f"HEAD:refs/heads/{rejected_branch}")
        rejected_url = f"https://github.com/{repo_slug()}/compare/{BRANCH}...{rejected_branch}"
        report_text = report.read_text(encoding="utf-8")
        git(wt, "reset", "-q", "--hard", base)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(report_text, encoding="utf-8")
        (wt / evidence_rel(gw)).write_text(evidence_md, encoding="utf-8")

    with report.open("a", encoding="utf-8") as fh:
        fh.write(
            gates_section(
                verdict=verdict,
                reasons=reasons,
                changes=changes,
                tests_before=tests_before,
                tests_after=tests_after,
                bt_before=bt_before,
                bt_after=bt_after,
                claude=claude,
                rejected_url=rejected_url,
            )
        )

    entry = {
        "season": SEASON,
        "gw": gw,
        "status": verdict,
        "title": title,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_commit": base,
        "report": f"{SEASON}/gw{gw:02d}.md",
        "evidence": f"{SEASON}/gw{gw:02d}-evidence.md",
        "files_changed": [c.path for c in changes] if verdict == "merged" else [],
        "lines_changed": sum(c.added + c.deleted for c in changes) if verdict == "merged" else 0,
        "gate_reasons": reasons,
        "rejected_branch": f"{branch_for(gw)}-rejected" if verdict == "rejected" else None,
        "tests": {
            "before_run": tests_before.get("run"),
            "before_failed": len(tests_before.get("failed", [])),
            "after_run": (tests_after or {}).get("run"),
            "after_failed": len((tests_after or {}).get("failed", [])) if tests_after else None,
        },
        "backtest": {"before": bt_before, "after": bt_after},
        "live": {k: live.get(k) for k in ("points", "net_points", "average", "delta_vs_average", "gw_rank", "overall_rank", "hits_cost")},
        "claude": {k: claude.get(k) for k in ("minutes", "num_turns", "cost_usd")},
        "tag": tag_for(gw) if verdict == "merged" else None,
        "pr": None,
        "pr_url": None,
    }

    if args.dry_run:
        out = LOG_DIR / f"{key}-dryrun.json"
        out.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        attempts["count"] -= 1  # dry runs do not use up attempts
        attempts["last_status"] = f"dry_run {verdict}"
        save_state(state)
        log.info("Dry run done: %s. Worktree left at %s, entry at %s", verdict, wt, out)
        return 0

    entries = ledger.upsert(ledger.load(wt), entry)
    ledger.save(wt, entries)
    git(wt, "add", "-A")
    commit_title = f"revision({SEASON} GW{gw}): {title}"
    git(wt, "commit", "-q", "-m", f"{commit_title}\n\nStatus: {verdict}. See {report_rel(gw)}.\n\n{CLAUDE_TRAILER}")
    git(wt, "push", "-q", "--force", "-u", REMOTE, branch_for(gw))

    for name, color, desc in (
        (LABEL, "5319e7", "Automated post-gameweek revision"),
        (LABEL_REJECTED, "d93f0b", "Revision code rejected by gates; needs a human"),
    ):
        gh("label", "create", name, "--color", color, "--description", desc, "--force")

    body_file = LOG_DIR / f"{key}-pr.md"
    body_file.write_text(pr_body(report.read_text(encoding="utf-8"), gw, verdict), encoding="utf-8")
    labels = [LABEL] + ([LABEL_REJECTED] if verdict == "rejected" else [])
    pr_url = gh(
        "pr", "create",
        "--base", BRANCH,
        "--head", branch_for(gw),
        "--title", commit_title,
        "--body-file", str(body_file),
        *sum((["--label", lb] for lb in labels), []),
    ).splitlines()[-1]
    pr_number = int(pr_url.rstrip("/").rsplit("/", 1)[-1])
    log.info("Opened %s", pr_url)

    entry["pr"], entry["pr_url"] = pr_number, pr_url
    ledger.save(wt, ledger.upsert(ledger.load(wt), entry))
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", f"revision({SEASON} GW{gw}): link PR #{pr_number} in ledger")
    git(wt, "push", "-q", REMOTE, branch_for(gw))

    gh(
        "pr", "merge", str(pr_number), "--squash",
        "--subject", f"{commit_title} (#{pr_number})",
        "--body", f"Status: {verdict}. Report: {report_rel(gw)}\n\n{CLAUDE_TRAILER}",
    )
    merge_sha = json.loads(gh("pr", "view", str(pr_number), "--json", "mergeCommit"))["mergeCommit"]["oid"]
    git(ROOT, "fetch", "--quiet", REMOTE, BRANCH)
    git(ROOT, "push", "-q", REMOTE, "--delete", branch_for(gw), check=False)
    if verdict == "merged":
        git(ROOT, "tag", "-f", "-a", tag_for(gw), merge_sha, "-m", f"{commit_title} (#{pr_number})")
        git(ROOT, "push", "-q", "--force", REMOTE, f"refs/tags/{tag_for(gw)}")

    attempts["last_status"] = verdict
    attempts["pr"] = pr_url
    save_state(state)
    git(ROOT, "worktree", "remove", "--force", str(wt), check=False)
    git(ROOT, "branch", "-D", branch_for(gw), check=False)

    summary = f"{SEASON} GW{gw}: {verdict} — {title}\n{pr_url}\n" + (
        "\nGate reasons:\n" + "\n".join(f"- {r}" for r in reasons) if reasons else ""
    )
    notify(f"[FPL Robo] Revision GW{gw}: {verdict}", summary)
    log.info("Revision done: %s", summary.replace("\n", " | "))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gw", type=int, help="Review this finished GW instead of the latest")
    parser.add_argument("--dry-run", action="store_true", help="Run review + gates, but do not push, open a PR or merge")
    parser.add_argument("--force", action="store_true", help="Ignore pause file, attempt limit, ledger and deadline guard")
    args = parser.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "loop.log")],
    )
    lock = (LOG_DIR / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("Another revision run is in progress")
        return 0
    try:
        return run(args)
    except Skip as skip:
        log.info("Skip: %s", skip)
        return 0
    except Exception as exc:
        log.exception("Revision loop failed")
        notify("[FPL Robo] Revision loop failed", f"{exc}\n\nSee {LOG_DIR / 'loop.log'}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
