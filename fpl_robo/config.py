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

# Knowledge-graph enrichment → live EV context (matchup, DGW scale, form blend).
KG_ENRICH = {
    "enabled": True,
    "feature_version": "v5-playing-bench",  # bumps cadence fingerprint on feature ships
    "horizon": 3,  # GWs for congestion / lookahead bundles
    "matchup_weight": 0.55,  # vs classic FDR inside next_context_factor
    "fdr_weight": 0.45,
    "lookahead_blend": 0.22,  # mix next-GW context with mean of next `horizon` GWs
    "form_blend": 0.22,  # blend form/rolling_points into ep_next
    "xgi90_blend": 0.12,  # blend official EP toward xGI/90 involvement rate
    "xgi_boost_scale": 0.10,
    "xgi_boost_cap": 0.45,
    # Per-90 rates are meaningless on cameos (e.g. 2 mins → xGI90=3.6).
    "rate_feature_min_minutes": 180,
    "xgi90_hard_cap": 1.25,  # never feed EV a higher rate than this
    "xgi_involvement_ev_cap": 6.0,  # max points proxy used in the EP blend
    "congestion_penalty": 0.04,  # per fixture above 1/GW over horizon
    "news_risk": 0.12,  # haircut when news text present but chance null
    "news_risk_doubt": 0.05,
    "market_momentum_cap": 0.04,  # max EV mult from transfer net flow
    "price_momentum_cap": 0.03,  # max EV mult from cost_change_event
    "opp_form_weight": 0.06,  # haircut/boost vs opponent squad form
    "role_stability_floor": 0.92,  # min mult when rolling minutes are thin
    "set_piece_pen_boost": 0.08,  # primary penalty taker
    "set_piece_corner_boost": 0.04,  # primary corners / indirect FK
    "cs_boost_scale": 0.35,  # DEF/GK clean-sheet rate × matchup → additive EV
    "cs_boost_cap": 0.45,
    "rest_days_short": 3.0,  # below this → congestion haircut
    "rest_days_penalty": 0.05,
    "value_form_cap": 0.03,  # mild mult from FPL value_form
    "threat_focal_boost": 0.04,  # club's top threat share at position
    "write_player_faces": True,  # player→opponent FACES edges for next GW
    "write_set_piece_edges": True,
    "write_rival_edges": True,
}

# Bench must be usable auto-sub cover — players who actually play minutes.
# Dead £4.5 enablers leave you on 10 if someone blanks.
BENCH_COVER = {
    "min_ep_next": 1.8,  # official EP floor for outfield cover
    "min_minutes": 900,  # ~10 full matches of season evidence
    "min_starts": 8,  # alternative minutes evidence
    "require_ep_and_minutes": True,  # both EP and involvement (not OR)
    "allow_backup_gk": True,  # second GK is structural; outfield must play
    "min_chance": 50,
    "bench_ev_weight": 0.45,  # care more about bench EV in the MIP
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

# Historical ep_this repair for blank vaastav xP weeks (FPL-Core-Insights snapshots).
EP_HISTORY = {
    "enabled": True,
    "cache_path": "data/cache/ep_this_{season}.csv",
    "gw_blank_frac": 0.90,
}

# When historical xP is missing (vaastav often blanks whole GWs), fill EV from lagged
# open features so ep_* modes do not freeze mid-season. Live ep_next is usually present,
# so this mainly affects backtests. Prefer EP_HISTORY fill first; this is last resort.
# DGW rows get a mild extra pull.
MISSING_XP_FALLBACK = {
    "enabled": True,
    "ev_floor": 0.05,
    "ev_cap": 12.0,
    "dgw_mult": 1.4,
    # Treat a GW as dataset-blank when this fraction of rows have xP≈0.
    "gw_blank_frac": 0.90,
}

# Back-compat alias used by older call sites / docs.
DGW_XP_FALLBACK = {
    "enabled": True,
    "ev_floor": MISSING_XP_FALLBACK["ev_floor"],
    "rolling_points_mult": MISSING_XP_FALLBACK["dgw_mult"],
    "ev_cap": MISSING_XP_FALLBACK["ev_cap"],
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
# 2025-26 compare after Core Insights ep_this fill: ep_baseline wins.
EV_MODE = "ep_baseline"

# Raspberry Pi friendly defaults
DEFAULT_SESSION_PATH = "data/session.json"
DEFAULT_SUMMARY_PATH = "data/last_run_summary.json"
DEFAULT_SIGNALS_PATH = "data/signals.json"

# News scrape + local LLM interpretation → data/signals.json
SIGNAL_INGEST = {
    "signals_path": DEFAULT_SIGNALS_PATH,
    "raw_cache_path": "data/cache/news_raw.json",
    "pre_deadline_stamp_path": "data/cache/pre_deadline_ingest.json",
    "ingest_stamp_path": "data/cache/ingest_schedule.json",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "llama3.2:latest",
    # Daily pass needs enough slots for injury + transfer + preseason form.
    "max_articles": 20,
    "pre_deadline_max_articles": 24,
    "max_article_chars": 1400,
    "request_timeout_seconds": 20,
    "ollama_timeout_seconds": 180,
    # Fetch HTML body for thin RSS blurbs (allowlisted hosts only).
    "fetch_article_body": True,
    "article_body_max_chars": 2800,
    "article_body_max_fetches": 8,
    "article_body_timeout_seconds": 12,
    "article_body_hosts": [
        "fantasyfootballscout.co.uk",
        "www.fantasyfootballscout.co.uk",
        "bbc.co.uk",
        "www.bbc.co.uk",
        "skysports.com",
        "www.skysports.com",
        "theguardian.com",
        "www.theguardian.com",
    ],
    "min_confidence": 0.65,
    "exclude_min_confidence": 0.8,
    "news_mult_floor": 0.45,
    "lineup_min_confidence": 0.65,
    "lineup_start_min_confidence": 0.7,
    "require_team_or_proximity": True,
    # Keep prior auto news if a scrape yields nothing (avoid wiping real signals).
    "retain_auto_news_on_empty": True,
    "auto_news_max_age_days": 10,
    # Skip Ollama when headline heuristics already extracted clear signals.
    "skip_llm_when_heuristic": True,
    # Once daily in local time (catch morning press / team news).
    "timezone": "Europe/London",
    "daily_ingest_hour": 8,
    "daily_ingest_minute": 30,
    "daily_ingest_window_hours": 6.0,  # only run daily scrape in this morning window
    # Final scrape before first kickoff on a matchday, and before GW deadline.
    "pre_match_hours": 2.0,
    "pre_deadline_hours": 2.0,
    "lineup_start_minutes_mult": 1.12,
    "lineup_bench_minutes_mult": 0.88,
    "lineup_doubt_minutes_mult": 0.95,
    # Mild EV bump for confirmed club moves / strong preseason form.
    "transfer_form_boost": 0.06,
    "transfer_strong_boost": 0.08,
    "transfer_elite_boost": 0.10,
    "form_signal_boost": 0.08,
    "rss_feeds": [
        {
            "name": "bbc_premier_league",
            "url": "https://feeds.bbci.co.uk/sport/football/premier-league/rss.xml",
        },
        {
            "name": "bbc_football",
            "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        },
        {
            "name": "sky_premier_league",
            "url": "https://www.skysports.com/rss/12040",
        },
        {
            "name": "sky_transfer_centre",
            "url": "https://www.skysports.com/rss/11095",
        },
        {
            "name": "guardian_football",
            "url": "https://www.theguardian.com/football/rss",
        },
        {
            "name": "ffs_feed",
            "url": "https://www.fantasyfootballscout.co.uk/feed/",
        },
    ],
    # Extra feeds pulled only in the pre-deadline / pre-match lineup pass.
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
    # Schedule-aware threshold relief (applied in chips.schedule).
    "dgw_captain_tc_ev_relief": 1.0,  # captain club DGW → easier TC
    "dgw_bench_bb_ev_relief": 1.0,  # enough DGW bench → easier BB
    "dgw_bench_min_players": 2,
    "dgw_tc_min_percentile": 0.75,
    "dgw_bb_min_percentile": 0.75,
    "easy_fdr_max": 2.0,  # captain best FDR at/under this
    "easy_fdr_tc_relief": 0.5,
    "set_piece_tc_relief": 0.4,  # primary pen/corner captain → easier TC
    "set_piece_tc_min_score": 0.08,
    "strong_matchup_min": 1.05,  # captain next_matchup_factor at/over this
    "strong_matchup_tc_relief": 0.35,
    # Score chips on horizon-adjusted EV even when transfer EV_MODE has no _h suffix.
    "use_horizon_ev": True,
    # Suppress TC/BB when predicted lineups contradict the chip thesis.
    "lineup_gate": True,
    "bb_lineup_min_startish": 2,
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
    # Light wake to notice daily ingest / data changes; not a full optimise.
    "heartbeat_seconds": 900,  # 15 minutes
    "near_deadline_hours": 2.0,
    "near_deadline_poll_seconds": 300,  # every 5 minutes near deadline / match
    "execute_within_hours": 0.5,  # normal: write only in final 30 minutes before deadline
    # Legacy aliases used by older call sites / dashboard.
    "poll_seconds": 900,
    # When FPL transfer limit is null (preseason / WC / FH), apply transfers as soon as planned.
    "execute_when_unlimited": True,
    # Mid-season unlimited (WC/FH): only re-apply if objective improves by this much.
    # Preseason (no finished GWs yet) always applies the current best squad.
    "unlimited_min_obj_delta": 1.0,
    "preseason_skip_churn_guard": True,
    "max_hits": 1,
}

# Secondary objective for MIP: break ties among equal-EV plans.
# mode=lexicographic → maximise EV(+hits/bank/price) first, then quality among ties.
# mode=soft → old blended objective (quality can slightly override EV).
OPTIMISER_TIEBREAK = {
    "enabled": True,
    "mode": "lexicographic",
    "ev_tolerance": 1e-3,
    "quality_weight": 1.0,
    "transfer_penalty": 0.05,  # among equal-EV plans, prefer fewer transfers
    # Classic open features.
    "form_weight": 0.22,
    "xgi_weight": 0.22,
    "ict_weight": 0.12,
    "ownership_weight": 0.06,
    # Knowledge-graph context (among equal-EV plans).
    "context_weight": 0.16,
    "set_piece_weight": 0.10,
    "stability_weight": 0.08,
    "focal_weight": 0.04,
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
    "tunnel_url_file": "data/cloudflare_tunnel_url.txt",
    "metrics_history_file": "data/cache/dashboard_metrics.json",
    "metrics_min_interval_seconds": 900,  # snapshot at most every 15 minutes
    "metrics_max_points": 400,
}

