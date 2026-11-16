"""Post-gameweek revision loop.

After each finished gameweek a headless Claude run reviews the bot (errors,
logic mistakes, bad decisions), makes small fixes on a branch, and the loop
merges them as a squash PR once they pass hard gates (protected paths, diff
size, unit tests, season backtest). Every run is recorded in revisions/ and
tagged so it can be audited and rolled back (see revisions/README.md).
"""
