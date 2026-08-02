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
    "feature_version": "v16.1-chips-shadow",  # bumps cadence fingerprint on feature ships
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
    # Defaults = preseason-safe; phase_kg_enrich swaps in-season values.
    "cs_boost_scale": 0.0,
    "cs_boost_cap": 0.0,
    "def_ev_scale": 0.82,
    "def_ep_ev_cap_mult": 1.0,
    "cs_boost_scale_preseason": 0.0,
    "cs_boost_cap_preseason": 0.0,
    "def_ev_scale_preseason": 0.82,
    "def_ep_ev_cap_mult_preseason": 1.0,
    "cs_boost_scale_in_season": 0.25,
    "cs_boost_cap_in_season": 0.35,
    "def_ev_scale_in_season": 1.0,
    "def_ep_ev_cap_mult_in_season": 0.0,
    "saves_boost_scale": 0.08,  # GK saves/90 → mild additive EV
    "saves_boost_cap": 0.35,
    "rest_days_short": 3.0,  # below this → congestion haircut
    "rest_days_penalty": 0.05,
    "value_form_cap": 0.03,  # mild mult from FPL value_form
    "threat_focal_boost": 0.10,  # stronger lift for club×pos #1 attackers
    "fwd_ep_premium_high": 0.16,  # focal FWD with ep_next >= 2.25
    "fwd_ep_premium_mid": 0.08,  # focal FWD with ep_next >= 2.1
    "fwd_focal_min_ep": 2.1,  # club EP#1 alone is not enough below this
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
    # Soft bench EV in the MIP. Keep outfield cover meaningful, but do not
    # let a premium backup GK soak £1–2 that belongs in the attack.
    "bench_ev_weight": 0.18,
    "bench_gk_ev_weight": 0.06,
}

# Prevent draining attack into premium DEF when FPL ep_next for CBs is inflated.
# Primary: FWDs with official ep_next >= min_ep_next.
# Alternates (narrow): club focal / strong form / heavy minutes just under the band.
# If the eligible pool is thinner (injuries), require as many as are available.
ATTACK_FLOOR = {
    "enabled": True,
    "min_quality_fwds": 3,
    "min_ep_next": 2.1,
    "soft_when_pool_thin": True,
    "allow_focal_alt": True,
    "focal_min_ep": 1.95,
    "allow_form_alt": True,
    "min_form": 5.0,
    "form_min_ep": 1.95,
    "allow_minutes_alt": True,
    "min_rolling_minutes": 240,
    "minutes_min_ep": 1.9,
}

# Hardening gates for buys (non-owned players) when EV mode is ep_hardened / open_ev.
BUY_GATES = {
    "min_buy_ev": 2.0,
    "min_prior_minutes": 180,  # cumulative minutes before decision GW
    "min_rolling_minutes": 120,  # last-4 GW minutes before decision GW
    "require_minutes_or_rolling": True,  # pass if prior OR rolling meets floor
    "min_buy_chance": 50,  # chance_of_playing_next_round floor for non-owned
    "max_news_risk": 0.08,  # block buys with KG news_risk above this
    "min_signal_news_mult": 0.70,  # block buys with signal news_mult below this
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

# Narrow blank look-ahead: haircut players whose club blanks in upcoming GWs
# (not full horizon FDR). Live stays off until compare-blank-ahead wins.
# When only_if_owned_blank is True, the haircut applies only if the current
# squad already owns enough players heading into an upcoming blank.
BLANK_AHEAD = {
    "enabled": True,
    "lookahead": 2,
    # Multipliers for gw+1, gw+2, … (strongest of matching blanks is applied).
    "offset_mults": [0.82, 0.92],
    "only_if_owned_blank": True,
    "owned_blank_threshold": 1,
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
    "max_articles": 40,
    "pre_deadline_max_articles": 48,
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
    "lineup_min_confidence": 0.68,
    "lineup_start_min_confidence": 0.75,
    "require_team_or_proximity": True,
    # Keep prior auto news if a scrape yields nothing (avoid wiping real signals).
    "retain_auto_news_on_empty": True,
    "auto_news_max_age_days": 10,
    # Skip Ollama when headline heuristics already extracted clear signals.
    "skip_llm_when_heuristic": True,
    # Reject Ollama injury/suspension without FPL status align or name↔keyword proximity.
    "require_ollama_injury_proximity_or_status": True,
    # Once daily in local time (catch morning press / team news).
    "timezone": "Europe/London",
    "daily_ingest_hour": 8,
    "daily_ingest_minute": 30,
    "daily_ingest_window_hours": 6.0,  # only run daily scrape in this morning window
    # Final scrape before FPL GW deadline (when transfers/XI lock).
    # If deadline is earlier the same local day than daily_ingest, skip daily and only run this.
    "pre_deadline_hours": 0.5,
    "lineup_start_minutes_mult": 1.12,
    "lineup_bench_minutes_mult": 0.88,
    "lineup_doubt_minutes_mult": 0.95,
    # When lineup LLM returns predicted_minutes, map mins/90 → minutes_mult
    # (role mults are skipped for that signal to avoid double-counting).
    "lineup_minutes_floor": 0.20,
    "lineup_minutes_ceil": 1.15,
    # How strongly predicted_minutes move EV (1.0 = full mins/90 scaling).
    "lineup_minutes_weight": 0.92,
    "ev_lineup_min_confidence": 0.78,
    "ev_start_minutes_min_confidence": 0.84,
    "depth_fwd_rank2": 0.52,
    "depth_fwd_rank2_low": 0.38,
    "depth_fwd_rank3": 0.28,
    "depth_mid_rank2": 0.72,
    "depth_mid_rank3": 0.58,
    "depth_mid_rank4": 0.45,
    "transfer_backup_hist_scale": 0.45,
    # On daily ingest, also extract lineups/minutes for articles that mention
    # an owned-squad club (not only pre-deadline / obvious XI headlines).
    "owned_lineup_on_daily": True,
    # Fill every owned player with an FPL ep/status/minutes proxy when news
    # does not supply predicted_minutes (keeps coverage at 15/15).
    "fpl_implied_minutes_confidence": 0.70,
    "depth_minutes_confidence": 0.76,
    "transfer_lineup_confidence": 0.78,
    "prefer_depth_over_fpl_implied": True,
    "require_full_squad_minutes": True,
    # Preseason role → minutes_mult (rotation/backup are haircuts).
    # Starter tag is too noisy/asymmetric across the pool — treat as neutral.
    "preseason_starter_minutes_mult": 1.0,
    "preseason_rotation_minutes_mult": 0.78,
    "preseason_backup_minutes_mult": 0.55,
    # Mild EV bump for confirmed club moves / strong preseason form.
    "transfer_form_boost": 0.06,
    "transfer_strong_boost": 0.08,
    "transfer_elite_boost": 0.10,
    # Confirmed PL exits (sold abroad / non-PL destination).
    "transfer_exit_boost": -0.15,
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
    "lock_stale_seconds": 900,  # abandon stuck stamps faster (healthy runs refresh ts)
    # Light wake to notice daily ingest / data changes; not a full optimise.
    "heartbeat_seconds": 900,  # 15 minutes
    "near_deadline_hours": 1.0,  # poll more often in the hour before lock
    "near_deadline_poll_seconds": 300,  # every 5 minutes near deadline
    "execute_within_hours": 0.5,  # normal: write only in final 30 minutes before deadline
    # Legacy aliases used by older call sites / dashboard.
    "poll_seconds": 900,
    # When FPL transfer limit is null (preseason / WC / FH), apply transfers as soon as planned.
    "execute_when_unlimited": True,
    # Mid-season unlimited (WC/FH): only re-apply if objective improves by this much.
    "unlimited_min_obj_delta": 3.5,
    # Keep churn guard off in preseason — vector EV sanity + transfer cap already
    # limit thrash; unlimited FT should apply planned moves immediately.
    "preseason_skip_churn_guard": True,
    # Preseason: do not thrash-write on every new_signals ingest. Optimise+summary
    # still run; FPL writes need attack repair, owned-risk repair, or other triggers.
    "preseason_execute_on_new_signals": False,
    "preseason_execute_min_hours_between": 0.0,  # no time cool-down in preseason
    # After cool-down (non-repair), require this much objective lift vs last write.
    "preseason_post_cooldown_min_obj_delta": 0.0,
    # After cool-down, still cap how many transfers a single unlimited write may apply
    # unless repairing attack quality or selling an injured/doubtful owned player.
    "preseason_max_transfers_unless_repair": 2,
    # Owned player treated as risk if chance below this (or status i/d/s/u / exclude).
    "owned_risk_chance_below": 50,
    "owned_risk_news_mult_below": 0.75,
    # When FT are effectively unlimited, penalise churn in the MIP primary objective.
    "unlimited_transfer_ev_penalty": 0.18,
    "unlimited_keep_quality_fwd_bonus": 0.35,
    # Bound applied_moves history in season_state.json.
    "applied_moves_max_items": 40,
    "max_hits": 2,
    # Log MIP transfer plan beside live vector SA (never execute the shadow).
    "shadow_mip": True,
    # When FH/WC is about to execute, rebuild via anneal_squad (full 15).
    "chip_rebuild_on_fh_wc": True,
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
    "bps_weight": 0.10,  # bonus-points-system rate among equal-EV plans
}
# Live squad optimiser: vector-team SA (non-MIP). Set enabled=False to revert to PuLP.
VECTOR_TEAM = {
    "enabled": True,
    "horizon": 3,  # upcoming GWs for fixture-run features
    "ridge_alpha": 12.0,
    "min_ridge_rows": 120,  # below this, residual=0 (pure xP anchor)
    "fit_lookback_gws": 16,  # prior GWs used to fit residual ridge
    # Score = xP_anchor + residual_scale * ridge(vector). Residual predicts points−xP.
    "residual_mode": True,
    "residual_scale": 0.35,
    "residual_clip": 2.5,
    # Cold-start: fit residual ridge on this historical season when live hist is thin.
    "live_residual_season": "2025-26",
    "live_current_season": "2026-27",  # prefer this once enough GW points exist
    "prefer_current_season_residual": True,
    # Team geometry weights (kept light so they don't override xP rank).
    "portfolio_weight": 0.08,
    "attack_coverage_weight": 0.14,
    "minutes_reliability_weight": 0.35,  # stronger minutes edge vs xP-only XI
    "minutes_reliability_threshold": 0.80,  # play_prob below this is penalised
    "club_soft_weight": 0.08,  # was 0.20 — soft club diversity must not dump premiums
    "bench_weight": 0.12,
    "value_weight": 0.02,
    # Transfer sanity (live + SA): block geometry-only thrash / bench fodder buys.
    "require_cover_eligible": True,
    "exclude_unavailable": True,
    "require_buy_starts": True,  # each incoming player must make the XI
    "min_transfer_net_ev": 0.0,  # sum(ev_in − ev_out) across transfers
    "min_xi_ev_delta": 0.0,  # plan XI raw EV must not fall vs current XI
    "transfer_ev_penalty": 2.5,  # soft penalty weight inside SA for EV regressions
    "bench_upgrade_pass": True,  # post-SA: sell bench for higher-EV starters
    "enforce_attack_floor": True,  # hard ATTACK_FLOOR inside SA feasibility
    "enforce_buy_eligible": True,  # respect buy_eligible even on ep_baseline
    "min_play_chance": 50,  # align with MIP / BUY_GATES.min_buy_chance
    "price_aware": True,  # soft PRICE_AWARE keep/buy/bank terms in SA
    # Simulated annealing + local search
    "sa_iters_initial": 3500,
    "sa_iters_transfer": 1800,  # per consensus seed
    "sa_t0": 3.0,
    "sa_t_min": 0.015,
    "sa_seed": 42,
    "sa_restarts": 1,  # legacy neighbour diversification
    "sa_consensus_seeds": 3,  # independent SA runs
    "sa_consensus_min_agree": 2,  # require majority agreement to transfer
    "hillclimb_passes": 2,
    "hit_cost": 4,
    "max_random_init_tries": 400,
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

