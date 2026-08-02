# FPL Robo — 2026/27

Fantasy Premier League bot built around a knowledge graph world model and a vector-team simulated-annealing optimiser for squad decisions (PuLP MIP remains available as a fallback / backtest baseline).

## Core architecture

`FPL API + Signals -> Knowledge Graph -> EV model -> vector-team SA (squad / transfers / chips)`

- Data: bootstrap + fixtures + manager + authenticated my-team
- Signals: local JSON for news / preseason friendlies / international fatigue (stricter confidence/exclude gates)
- Graph: players, teams, fixtures, gameweeks, signals
- Predict: baseline EV / minutes-scaled / open EV; optional horizon (`*_h`) FDR+blank/DGW look-ahead
- Optimise: live path uses vector-team SA (`VECTOR_TEAM.enabled`, feature `v14-vector-team`); PuLP MIP still used for A/B and when vector is disabled
- Feature fingerprint: `KG_ENRICH["feature_version"]` (currently `v14-vector-team`) bumps daemon cadence when the live stack changes

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

Goal: leave the Pi running; it checks regularly and writes in the final **30 minutes** before deadline (or immediately while transfers are unlimited).

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

If auth fails, the Pi can email you (rate-limited to once per hour). Set in `.env`:

```bash
ALERT_EMAIL_TO=callumj321@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-gmail@gmail.com
SMTP_PASSWORD=your-gmail-app-password   # Google Account → App passwords
```

Test with:

```bash
python -m fpl_robo.cli test-alert
```

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
3. Attach lagged rolling minutes from finished `event/{gw}/live` (cached) for EV parity with backtests
4. Predict EV, optimise transfers with **vector-team SA** when `VECTOR_TEAM.enabled` (else PuLP MIP); max hits configurable; soft ITB / rising-player keep terms apply on the MIP path
5. Recommend chips (DGW-aware thresholds; FH gap relief when many owned players are on blanks). Live FH uses unconstrained rebuild XI EV − locked XI EV; WC uses a rolling sum of recent FH gaps; TC/BB use persisted weekly score history for percentiles.
6. If `--execute`: apply transfers + lineup when either (a) within final 30 minutes of deadline, or (b) FPL reports unlimited transfers (preseason / WC / FH) and the plan changed **and** (outside the final window) objective improved by ≥ `unlimited_min_obj_delta` vs the last unlimited write this GW **for the same optimiser**. Chips still wait for the final 30-minute window.
7. Write `data/last_run_summary.json`, `data/season_state.json`, `logs/fpl_robo.log`

Live `EV_MODE` is `ep_baseline` (with blank-ahead). Live transfer solver is **vector-team SA** (`feature_version=v14-vector-team`); set `VECTOR_TEAM["enabled"]=False` to revert to PuLP MIP. Horizon look-ahead (`*_h`) remains available but did **not** beat baseline on 2025-26 points after tuning (asymmetric DGW pull, blank fill, unknown=neutral); see `data/horizon_param_sweep.json`. A mild DGW xP fallback helps TC/BB land on doubles when historical xP is missing (`DGW_XP_FALLBACK`). Live transfer hit cap is `max_hits=2` after multi-season A/B (`data/multi_season_ev_ab.json`). Suite: vector ahead of MIP on 10/10 configs (`data/vector_team_eval_summary.json`).

Near deadline (final 2 hours), polling speeds up to every 5 minutes so the 30-minute write window is not missed. In that same window the daemon runs a **one-shot pre-deadline signal scrape** (injuries + predicted XIs from RSS/Ollama) before planning transfers.

Manual / cron helpers:

```bash
python -m fpl_robo.cli ingest-signals --apply                 # regular news scrape
python -m fpl_robo.cli ingest-signals --pre-deadline --apply  # only if within ~2h of deadline
python -m fpl_robo.cli ingest-signals --force-pre-deadline --apply  # test lineup mode anytime
```

Dry-run anytime:

```bash
python -m fpl_robo.cli run-week
```

## Chips

EV-based chip recommendations for Triple Captain, Bench Boost, Free Hit, and Wildcard.

Live `run-week` scores FH from the gap between an unconstrained optimal XI and the current locked XI, and WC from the rolling sum of recent FH gaps (stored in `data/season_state.json`). TC/BB use absolute thresholds plus percentile rank against that season's recorded weekly scores.

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
- Daemon/cron auto-executes in the final 30 minutes before deadline, and earlier whenever transfers are unlimited (with a minimum objective improvement between unlimited rewrites outside that window)
- Same GW will not re-execute unless `--force`
- Process lock prevents overlapping runs
- Post-write verification via `my-team`
- Invalid/missing session exits non-zero and is logged (daemon keeps polling)
