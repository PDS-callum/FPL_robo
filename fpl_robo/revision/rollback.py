"""Roll back revision-loop changes.

Reverts the code of one or more merged revisions (by their tags) in a single
PR, keeps their reports, and marks them `reverted` in the ledger so the loop
never reapplies them.

Usage:
  python -m fpl_robo.revision.rollback --list
  python -m fpl_robo.revision.rollback --gw 11            # one revision
  python -m fpl_robo.revision.rollback --since 9          # GW9 onwards, newest first
  python -m fpl_robo.revision.rollback --gw 11 --no-merge # open the PR only
  add --pause to also stop the loop (data/revision_loop.paused)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone

from fpl_robo.revision import ledger
from fpl_robo.revision.loop import (
    BRANCH,
    CLAUDE_TRAILER,
    LABEL,
    PAUSE_FILE,
    REMOTE,
    ROOT,
    SEASON,
    gh,
    git,
)

WORKTREE = ROOT.parent / "FPL_robo-rollback"


def _list(entries: list[dict]) -> None:
    print(f"{'GW':>4}  {'status':<10} {'live Δ':>7}  {'tag':<28} title")
    for e in sorted(entries, key=lambda e: (e["season"], e["gw"])):
        delta = (e.get("live") or {}).get("delta_vs_average")
        print(
            f"{e['gw']:>4}  {e['status']:<10} {'' if delta is None else f'{delta:+g}':>7}  "
            f"{e.get('tag') or '-':<28} {e.get('title', '')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show the ledger")
    group.add_argument("--gw", type=int, help="Revert the revision made after this GW")
    group.add_argument("--since", type=int, help="Revert every merged revision from this GW onwards")
    parser.add_argument("--season", default=SEASON)
    parser.add_argument("--no-merge", action="store_true", help="Open the revert PR but do not merge it")
    parser.add_argument("--pause", action="store_true", help="Also pause the revision loop")
    args = parser.parse_args(argv)

    git(ROOT, "fetch", "--quiet", "--tags", REMOTE, BRANCH)
    remote_ledger = json.loads(git(ROOT, "show", f"{REMOTE}/{BRANCH}:{ledger.LEDGER_JSON}", check=False) or "[]")
    if args.list:
        _list(remote_ledger)
        return 0

    targets = [
        e
        for e in remote_ledger
        if e["season"] == args.season
        and e["status"] == "merged"
        and (e["gw"] == args.gw if args.gw is not None else e["gw"] >= args.since)
    ]
    if not targets:
        print("No merged revisions match.")
        return 1
    targets.sort(key=lambda e: e["gw"], reverse=True)
    gws = [e["gw"] for e in targets]
    label = f"GW{gws[0]}" if len(gws) == 1 else f"GW{min(gws)}–GW{max(gws)}"

    if WORKTREE.exists():
        git(ROOT, "worktree", "remove", "--force", str(WORKTREE), check=False)
        shutil.rmtree(WORKTREE, ignore_errors=True)
    git(ROOT, "worktree", "prune")
    branch = f"revert/{args.season}-{label.lower().replace('–', '-')}"
    git(ROOT, "branch", "-D", branch, check=False)
    git(ROOT, "worktree", "add", "-b", branch, str(WORKTREE), f"{REMOTE}/{BRANCH}")

    for e in targets:
        sha = git(ROOT, "rev-list", "-n", "1", e["tag"])
        print(f"Reverting GW{e['gw']} {e['tag']} ({sha[:8]}): {e.get('title')}")
        try:
            git(WORKTREE, "revert", "--no-commit", sha)
        except RuntimeError as exc:
            git(WORKTREE, "revert", "--abort", check=False)
            print(f"Revert of {e['tag']} conflicts with later changes; resolve by hand in a PR.\n{exc}")
            return 1
        # Keep the audit trail: reports, evidence and ledger stay as they are.
        git(WORKTREE, "checkout", "HEAD", "--", "revisions")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entries = ledger.load(WORKTREE)
    for e in entries:
        if e["season"] == args.season and e["gw"] in gws:
            e["status"] = "reverted"
            e["reverted_at"] = now
    ledger.save(WORKTREE, entries)
    git(WORKTREE, "add", "-A")
    title = f"revert({args.season} {label}): roll back revision-loop changes"
    tags = ", ".join(e["tag"] for e in targets)
    git(WORKTREE, "commit", "-q", "-m", f"{title}\n\nReverts {tags}.\n\n{CLAUDE_TRAILER}")
    git(WORKTREE, "push", "-q", "--force", "-u", REMOTE, branch)

    body = (
        f"Rolls back the code of {tags}. Reports and evidence are kept; the ledger marks them "
        "`reverted` so the loop will not reapply them.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n"
    )
    pr_url = gh("pr", "create", "--base", BRANCH, "--head", branch, "--title", title, "--body", body, "--label", LABEL).splitlines()[-1]
    print(f"Opened {pr_url}")
    if not args.no_merge:
        number = pr_url.rstrip("/").rsplit("/", 1)[-1]
        gh("pr", "merge", number, "--squash", "--subject", f"{title} (#{number})")
        git(ROOT, "push", "-q", REMOTE, "--delete", branch, check=False)
        print("Merged. update.sh deploys it within 15 minutes (or run deploy/update.sh).")
    git(ROOT, "worktree", "remove", "--force", str(WORKTREE), check=False)
    git(ROOT, "branch", "-D", branch, check=False)

    if args.pause:
        PAUSE_FILE.write_text(f"paused by rollback of {tags} at {now}\n", encoding="utf-8")
        print(f"Revision loop paused ({PAUSE_FILE}). Delete the file to resume.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
