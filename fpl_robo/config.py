"""Project defaults for the 2026/27 season build."""

# FPL entry for the 2026/27 season
DEFAULT_MANAGER_ID = 1687992

# Preseason initial-team defaults
PRESEASON_HORIZON = 6
PRESEASON_GW_WEIGHTS = [0.35, 0.22, 0.17, 0.12, 0.08, 0.06]
PRESEASON_RISK_PROFILES = {
    "conservative": {
        "min_play_chance": 75,
        "ownership_weight": 0.03,
        "uncertainty_penalty": 0.24,
    },
    "balanced": {
        "min_play_chance": 60,
        "ownership_weight": 0.02,
        "uncertainty_penalty": 0.18,
    },
    "aggressive": {
        "min_play_chance": 40,
        "ownership_weight": 0.01,
        "uncertainty_penalty": 0.1,
    },
}

# Bench must be usable cover, not dead fodder.
# A player is cover-eligible if available and either has meaningful FPL EP
# or enough recent minutes to suggest they can actually appear.
BENCH_COVER = {
    "min_ep_next": 1.5,
    "min_minutes": 90,
    "min_chance": 50,
    "bench_ev_weight": 0.2,
}

# Raspberry Pi friendly defaults
DEFAULT_SESSION_PATH = "data/session.json"
DEFAULT_SUMMARY_PATH = "data/last_run_summary.json"
DEFAULT_SIGNALS_PATH = "data/signals.json"

# Chip decision defaults (EV units ≈ expected points)
CHIP_DEFAULTS = {
    "half_split_gw": 19,
    "tc_min_captain_ev": 6.0,
    "tc_min_percentile": 0.85,
    "bb_min_bench_ev": 8.0,
    "bb_min_percentile": 0.85,
    "fh_min_gap": 12.0,
    "wc_min_rolling_gap": 18.0,
    "force_within_weeks": 2,
    "rolling_gap_horizon": 3,
}

# Historical backtest defaults
CHIP_BACKTEST_SEASON = "2025-26"
CHIP_BACKTEST_CACHE = "data/cache/merged_gw_2025-26.csv"

# Unattended Raspberry Pi season runner
SEASON_RUNNER = {
    "log_file": "logs/fpl_robo.log",
    "lock_file": "data/run.lock",
    "state_file": "data/season_state.json",
    "lock_stale_seconds": 3600,
    "poll_seconds": 1800,  # normal cadence away from deadline
    "near_deadline_hours": 2.0,  # tighten polling inside this horizon
    "near_deadline_poll_seconds": 300,  # every 5 minutes near deadline
    "execute_within_hours": 0.5,  # write only in final 30 minutes before deadline
    "max_hits": 1,
}

# LAN dashboard (read-only view of bot state)
DASHBOARD = {
    "host": "0.0.0.0",
    "port": 8787,
    "summary_file": DEFAULT_SUMMARY_PATH,
    "state_file": "data/season_state.json",
    "session_path": DEFAULT_SESSION_PATH,
    "log_file": "logs/fpl_robo.log",
    "auth_log_file": "logs/fpl_robo_auth.log",
}

