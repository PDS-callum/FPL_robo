# Revision loop

After every finished gameweek, the Pi runs a headless Claude review of the bot
(`fpl_robo/revision/loop.py`, hourly via `deploy/fpl-robo-revision.timer`). It
looks for crashes, swallowed errors, logic mistakes and bad decision rules in that
gameweek's evidence, fixes up to three of them, and merges the result into
`2026-2027` as a single squash PR. `deploy/update.sh` then deploys it like any
other change.

## What gets recorded

| Where | What |
|---|---|
| [`LEDGER.md`](LEDGER.md) / `ledger.json` | One row per gameweek: live points vs average, revision status, backtest before→after, PR, tag |
| `<season>/gwNN.md` | The model's report (findings, changes, risks) plus a **Loop gates** section written by the loop |
| `<season>/gwNN-evidence.md` | Exactly what the model was shown |
| PRs labelled `revision-loop` | One squash commit per gameweek; `revision-rejected` marks gate failures |
| Tags `revision-<season>-gwNN` | The merge commit of each revision that changed code |
| `logs/revision/` on the Pi | `loop.log`, full Claude transcripts (`*.jsonl`), local attempt state |

## Gates (a revision must pass all of them to merge code)

- Only `fpl_robo/`, `tests/`, `tools/`, `README.md` change; never `deploy/`,
  `fpl_robo/revision/`, auth, locking, notify, `data/`, `requirements.txt`, the ledger.
- No test files deleted, test count not lower, no new failing tests.
- At most 600 changed lines.
- 2025-26 regression backtest drops by no more than 25 points.
- Does not add files that already exist untracked on the Pi (would block deploy).

On a gate failure, the report and ledger row still merge, and the code is parked
on `revision/<season>-gwNN-rejected` for you to review.

The loop only runs when the GW is finished and data-checked, the next deadline is
at least 24h away, and the Pi's checkout is clean and can fast-forward.

## Reading performance

The **Live** columns of a row are the score of the code *before* that row's
revision. A revision made after GW11 first plays GW12, so judge it by the rows
above it in the ledger (newer gameweeks).

## Rolling back

```bash
.venv/bin/python -m fpl_robo.revision.rollback --list            # ledger in the terminal
.venv/bin/python -m fpl_robo.revision.rollback --gw 11           # revert one revision
.venv/bin/python -m fpl_robo.revision.rollback --since 9 --pause # revert GW9 onwards and pause
```

A rollback is its own squash PR that reverts only code. Reports stay, and the
ledger marks the revisions `reverted` so the loop never reapplies them.

## Pausing and manual runs

```bash
touch data/revision_loop.paused                       # pause (rm to resume)
sudo systemctl disable --now fpl-robo-revision.timer  # stop entirely
.venv/bin/python -m fpl_robo.revision.loop --dry-run  # review + gates, no push/merge
.venv/bin/python -m fpl_robo.revision.loop --gw 11 --force
```
