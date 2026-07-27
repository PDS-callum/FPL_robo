# FPL Robo — 2026/27

Fantasy Premier League bot built around a knowledge graph world model and mixed-integer programming for squad decisions.

## Core architecture

`FPL API + Signals -> Knowledge Graph -> EV model -> MIP (squad / transfers / chips)`

- Data: bootstrap + fixtures + manager + authenticated my-team
- Signals: local JSON for news / preseason friendlies / international fatigue
- Graph: players, teams, fixtures, gameweeks, signals
- Predict: baseline EV and preseason cold-start EV
- Optimise: squad, transfers (FT/hits), cover-eligible bench, chip timing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Default manager ID is `1687992`.

## Commands

```bash
python -m fpl_robo.cli show-team
python -m fpl_robo.cli build-squad
python -m fpl_robo.cli build-initial-team --horizon 6 --risk balanced
python -m fpl_robo.cli recommend-chips
python -m fpl_robo.cli backtest-chips --season 2025-26
python -m fpl_robo.cli run-week
python -m fpl_robo.cli run-week --execute --force
python -m fpl_robo.cli run-daemon --execute
python -m fpl_robo.cli run-dashboard
```

## Unattended Raspberry Pi (all season)

Goal: leave the Pi running; it checks regularly and only writes in the final **30 minutes** before deadline.

### One-time desktop auth

Password login is unreliable (OAuth). On a machine with a display, capture a real browser session:

```bash
pip install playwright
playwright install chromium
python -m fpl_robo.cli auth-capture
```

A Chromium window opens — complete the FPL/SSO login there. The script dumps
**all** browser state (cookies, storage, headers), writes `data/session.json`,
then probes several ways of replaying it with `requests`.

On that same machine, confirm before copying:

```bash
python -m fpl_robo.cli auth-check --load-session
```

Even if replay probes fail, copy the dump anyway — we can inspect it and wire
up whatever auth material FPL actually used.

Copy to Pi securely:

- `.env`
- `data/session.json`

Refresh with `auth-capture` again if the daemon starts failing auth.

### Preferred: systemd daemon

```bash
sudo cp deploy/fpl-robo.service /etc/systemd/system/
sudo cp deploy/fpl-robo-auth.service /etc/systemd/system/
sudo cp deploy/fpl-robo-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fpl-robo-auth.service
sudo systemctl enable --now fpl-robo.service
sudo systemctl enable --now fpl-robo-dashboard.service
journalctl -u fpl-robo-auth.service -f
journalctl -u fpl-robo.service -f
```

`fpl-robo-auth` refreshes the PingOne OIDC access token about 10 minutes before expiry so the season daemon stays logged in.

LAN dashboard (read-only): `http://<pi-ip>:8787` — planned transfers, XI, chips, deadline, auth, season state.

### Safe auto-updates from `2026-2027`

Daemons drain the current cycle on `SIGTERM`, then systemd restarts them on new code. Updates are applied only when idle (not holding `data/run.lock`, and more than 2 hours from deadline):

```bash
sudo cp deploy/fpl-robo-update.service deploy/fpl-robo-update.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fpl-robo-update.timer
```

Manual update:

```bash
./deploy/update.sh          # safe
./deploy/update.sh --force  # pull+restart even near deadline
```

Status / logs:

```bash
systemctl status fpl-robo.service fpl-robo-auth.service fpl-robo-update.timer
journalctl -u fpl-robo.service -f
journalctl -u fpl-robo-auth.service -f
tail -f logs/update.log
```

### Alternative: cron

See `deploy/crontab.example`.

### What `run-week` does

1. Lockfile + session auth
2. Load live squad / FT / bank from `my-team`
3. Predict EV, optimise transfers (max hits configurable)
4. Recommend chips (DGW-aware thresholds)
5. If `--execute` and within final 30 minutes of deadline: apply transfers + lineup + chip
6. Write `data/last_run_summary.json`, `data/season_state.json`, `logs/fpl_robo.log`

Near deadline (final 2 hours), polling speeds up to every 5 minutes so the 30-minute write window is not missed.

Dry-run anytime:

```bash
python -m fpl_robo.cli run-week
```

## Chips

EV-based chip recommendations for Triple Captain, Bench Boost, Free Hit, and Wildcard.

```bash
python -m fpl_robo.cli backtest-chips --season 2025-26
python -m fpl_robo.cli recommend-chips
```

## Preseason initial team

```bash
python -m fpl_robo.cli build-initial-team --risk balanced --horizon 6
python -m fpl_robo.cli build-initial-team --execute --load-session --save-session
```

Optional signals:

```bash
copy data\signals.example.json data\signals.json
python -m fpl_robo.cli build-initial-team --signals data/signals.json --risk balanced
```

## Safety behavior

- No writes unless `--execute` is set
- Daemon/cron only auto-executes in the final 30 minutes before deadline
- Same GW will not re-execute unless `--force`
- Process lock prevents overlapping runs
- Post-write verification via `my-team`
- Invalid/missing session exits non-zero and is logged (daemon keeps polling)
