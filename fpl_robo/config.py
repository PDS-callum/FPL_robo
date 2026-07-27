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

# Hardening gates for buys (non-owned players) when EV mode is ep_hardened / open_ev.
BUY_GATES = {
    "min_buy_ev": 2.0,
    "min_prior_minutes": 180,  # cumulative minutes before decision GW
    "min_rolling_minutes": 120,  # last-4 GW minutes before decision GW
    "require_minutes_or_rolling": True,  # pass if prior OR rolling meets floor
}

# Official EP haircut by lagged play probability (tuned on 2025-26: +63 vs baseline).
# EV = xP/ep_next * clip(rolling_minutes / divisor, floor, 1.0)
MINUTES_SCALED = {
    "divisor": 80.0,
    "floor": 0.30,
    "cold_play": 0.80,  # when no minutes history yet
}

# Multi-GW fixture look-ahead applied on top of base EV (not a multi-period MIP).
# Tuned on 2025-26: asymmetric look-ahead + filled blanks. No *_h mode beat
# ep_minutes_scaled on total points; keep horizon opt-in via EV_MODE suffix.
HORIZON_EV = {
    "enabled": False,  # on only when EV_MODE is a *_h mode
    "weights": [0.70, 0.20, 0.10],  # front-loaded
    "blank_mult": 0.20,  # this-GW blank haircut only
    "dgw_mult": 1.45,
    "fdr_weight": 0.06,
    "unknown_mult": 1.0,  # beyond schedule: neutral, not blank
    "asymmetric_lookahead": True,  # future blanks do not haircut; DGWs still pull
    "fh_blank_squad_threshold": 3,
    "fh_blank_gap_relief": 3.0,
}

# When historical/live xP is missing on double gameweeks, use mild rolling-points proxy
# so DGW players remain selectable (helps TC/BB land on doubles; small points tradeoff).
DGW_XP_FALLBACK = {
    "enabled": True,
    "ev_floor": 0.05,
    "rolling_points_mult": 1.4,
    "ev_cap": 12.0,
}

# Soft price / ITB terms in the transfer MIP (not a price predictor).
PRICE_AWARE = {
    "bank_weight": 0.05,
    "rising_sell_penalty": 0.25,
    "falling_buy_bonus": 0.10,
}

# Live / backtest EV mode:
# ep_baseline | ep_hardened | ep_minutes_scaled | open_ev
# (+ optional _h suffix for horizon layer, e.g. ep_minutes_scaled_h)
# 2025-26 compare-ev-models: ep_minutes_scaled beat horizon variants — keep without _h.
EV_MODE = "ep_minutes_scaled"

# Raspberry Pi friendly defaults
DEFAULT_SESSION_PATH = "data/session.json"
DEFAULT_SUMMARY_PATH = "data/last_run_summary.json"
DEFAULT_SIGNALS_PATH = "data/signals.json"

# News scrape + local LLM interpretation → data/signals.json
SIGNAL_INGEST = {
    "signals_path": DEFAULT_SIGNALS_PATH,
    "raw_cache_path": "data/cache/news_raw.json",
    "pre_deadline_stamp_path": "data/cache/pre_deadline_ingest.json",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "llama3.2:latest",
    "max_articles": 8,
    "pre_deadline_max_articles": 12,
    "max_article_chars": 900,
    "request_timeout_seconds": 20,
    "ollama_timeout_seconds": 180,
    "min_confidence": 0.65,
    "exclude_min_confidence": 0.8,
    "news_mult_floor": 0.45,
    "lineup_min_confidence": 0.65,
    "lineup_start_min_confidence": 0.7,
    "require_team_or_proximity": True,
    # Final scrape for team news / predicted XIs before GW lock.
    "pre_deadline_hours": 2.0,
    "lineup_start_minutes_mult": 1.12,
    "lineup_bench_minutes_mult": 0.88,
    "lineup_doubt_minutes_mult": 0.95,
    "rss_feeds": [
        {
            "name": "bbc_football",
            "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        },
        {
            "name": "sky_premier_league",
            "url": "https://www.skysports.com/rss/12040",
        },
        {
            "name": "guardian_football",
            "url": "https://www.theguardian.com/football/rss",
        },
    ],
    # Extra feeds pulled only in the pre-deadline lineup pass.
    "lineup_rss_feeds": [
        {
            "name": "ffs_feed",
            "url": "https://www.fantasyfootballscout.co.uk/feed/",
        },
    ],
}


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
    "execute_within_hours": 0.5,  # normal: write only in final 30 minutes before deadline
    # When FPL transfer limit is null (preseason / WC / FH), apply transfers as soon as planned.
    "execute_when_unlimited": True,
    # Outside the final window, only re-apply unlimited plans if objective improves by this much.
    "unlimited_min_obj_delta": 1.0,
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

