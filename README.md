# FPL Robo — 2026/27

Fantasy Premier League bot built around a **knowledge graph** world model and
**mixed-integer programming** for squad decisions.

## Architecture (step 1)

```
FPL API  →  Knowledge Graph  →  Tabular EV model  →  Single-GW MIP
```

- **Data** — fetch bootstrap + fixtures from the official FPL API
- **Graph** — players, teams, fixtures, and gameweeks as a typed graph
- **Predict** — simple expected-points estimates from form / EP / fixture difficulty
- **Optimise** — PuLP MIP for a legal 15-man squad, starting XI, and captain

Team auto-update (live transfers) comes later.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m fpl_robo.cli show-team
python -m fpl_robo.cli build-squad
```

Default manager ID is `789800` (same entry as 2025/26). Optional flags:

```bash
python -m fpl_robo.cli show-team --manager-id 789800
python -m fpl_robo.cli build-squad --budget 100.0 --min-play-chance 50
```
