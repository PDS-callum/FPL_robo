"""EV prediction modes for backtests and live decisions."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from fpl_robo.config import BENCH_COVER, BUY_GATES, MINUTES_SCALED
from fpl_robo.predict.horizon import mode_uses_horizon, strip_horizon_suffix

EvMode = Literal[
    "ep_baseline",
    "ep_hardened",
    "ep_minutes_scaled",
    "open_ev",
    "ep_baseline_h",
    "ep_hardened_h",
    "ep_minutes_scaled_h",
    "open_ev_h",
    "ep_baseline_run",
    "ep_minutes_scaled_run",
]
EV_MODES: tuple[EvMode, ...] = (
    "ep_baseline",
    "ep_hardened",
    "ep_minutes_scaled",
    "open_ev",
    "ep_baseline_h",
    "ep_hardened_h",
    "ep_minutes_scaled_h",
    "open_ev_h",
    "ep_baseline_run",
    "ep_minutes_scaled_run",
)


def attach_lagged_history(table: pd.DataFrame, *, window: int = 4) -> pd.DataFrame:
    """Add lagged rolling points/minutes (and optional open stats) with no same-GW leakage.

    For gameweek t, features use only gameweeks < t.
    """
    df = table.sort_values(["id", "gameweek"]).copy()
    g = df.groupby("id", sort=False)

    # Shift first so same-GW values never enter the rolling window.
    pts = g["total_points"].shift(1)
    mins = g["minutes"].shift(1)
    df["prior_minutes"] = mins.groupby(df["id"]).cumsum().fillna(0.0)
    df["rolling_points"] = (
        pts.groupby(df["id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_minutes"] = (
        mins.groupby(df["id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["rolling_points"] = df["rolling_points"].fillna(0.0)
    df["rolling_minutes"] = df["rolling_minutes"].fillna(0.0)

    if "price" in df.columns:
        prev_price = g["price"].shift(1)
        df["price_delta"] = (df["price"].astype(float) - prev_price.astype(float)).fillna(0.0)
    else:
        df["price_delta"] = 0.0

    for col in ("expected_goals", "expected_assists", "ict_index", "influence", "creativity", "threat"):
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        df[f"rolling_{col}"] = (
            shifted.groupby(df["id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"rolling_{col}"] = df[f"rolling_{col}"].fillna(0.0)

    return df


def play_prob_from_minutes(
    rolling_minutes: float,
    *,
    prior_minutes: float = 0.0,
    divisor: float | None = None,
    floor: float | None = None,
    cold_play: float | None = None,
    allow_season_proxy: bool = False,
) -> float:
    """Play-probability scalar used by ep_minutes_scaled."""
    div = float(MINUTES_SCALED["divisor"] if divisor is None else divisor)
    flo = float(MINUTES_SCALED["floor"] if floor is None else floor)
    cold = float(MINUTES_SCALED["cold_play"] if cold_play is None else cold_play)
    roll = float(rolling_minutes)
    prior = float(prior_minutes)
    if prior <= 0.0 and roll <= 0.0:
        return cold
    # GW1 / explicit cold-start only: prior is often season totals.
    if allow_season_proxy and roll <= 0.0 and prior > 0.0:
        season_div = float(MINUTES_SCALED.get("season_minutes_proxy_div") or 38.0)
        roll = min(90.0, prior / max(1.0, season_div))
    return float(min(1.0, max(flo, roll / div)))


def _season_proxy_allowed(rows: pd.DataFrame, mins: pd.Series) -> bool:
    """True only for opening GW or frames with no rolling history at all."""
    max_gw = int(MINUTES_SCALED.get("season_proxy_max_gameweek") or 1)
    if "gameweek" in rows.columns and rows["gameweek"].notna().any():
        try:
            gw0 = int(rows["gameweek"].dropna().iloc[0])
        except (TypeError, ValueError, IndexError):
            gw0 = None
        if gw0 is not None and gw0 <= max_gw:
            return True
        # Mid-season frame with a real GW: never inflate injured/low-roll players.
        return False
    # No GW column: only proxy when the whole frame lacks rolling minutes.
    return bool(len(mins) == 0 or float(mins.max()) <= 0.0)


def _play_prob_series(rows: pd.DataFrame) -> pd.Series:
    mins = rows["rolling_minutes"].astype(float) if "rolling_minutes" in rows.columns else 0.0
    prior = rows["prior_minutes"].astype(float) if "prior_minutes" in rows.columns else 0.0
    if isinstance(mins, pd.Series):
        mins = mins.fillna(0.0)
    else:
        mins = pd.Series(float(mins), index=rows.index)
    if isinstance(prior, pd.Series):
        prior = prior.fillna(0.0)
    else:
        prior = pd.Series(float(prior), index=rows.index)
    div = float(MINUTES_SCALED["divisor"])
    flo = float(MINUTES_SCALED["floor"])
    cold = float(MINUTES_SCALED["cold_play"])
    season_div = float(MINUTES_SCALED.get("season_minutes_proxy_div") or 38.0)
    if _season_proxy_allowed(rows, mins):
        proxy = (prior / max(1.0, season_div)).clip(lower=0.0, upper=90.0)
        effective = mins.where(~((mins <= 0.0) & (prior > 0.0)), proxy)
    else:
        effective = mins
    play = (effective / div).clip(lower=flo, upper=1.0)
    is_cold = (prior <= 0.0) & (mins <= 0.0)
    play = play.where(~is_cold, cold).fillna(cold)

    # Trusted lineup predicted_minutes override geometry play_prob.
    # (Do not also nudge via signal_minutes_mult — EV already applied that mult.)
    if "signal_predicted_minutes" in rows.columns:
        sig_mins = pd.to_numeric(rows["signal_predicted_minutes"], errors="coerce")
        override = sig_mins.notna()
        if bool(override.any()):
            from_sig = (sig_mins / 90.0).clip(lower=0.20, upper=1.15)
            play = play.where(~override, from_sig)

    # Recent transfer backups: haircut like baseline _minutes_factor.
    if "recent_transfer_backup" in rows.columns:
        from fpl_robo.config import SIGNAL_INGEST

        backup = rows["recent_transfer_backup"].fillna(False).astype(bool)
        if bool(backup.any()):
            scale = float(SIGNAL_INGEST.get("transfer_backup_hist_scale") or 0.45)
            play = play.where(~backup, (play * scale).clip(lower=0.20, upper=1.0))

    return play.astype(float)


def _minutes_gate_ok(row: pd.Series) -> bool:
    prior = float(row.get("prior_minutes") or 0.0)
    rolling = float(row.get("rolling_minutes") or 0.0)
    # Cold start / no history yet: do not block (EV floor still applies).
    if prior <= 0.0 and rolling <= 0.0:
        return True
    min_prior = float(BUY_GATES["min_prior_minutes"])
    min_roll = float(BUY_GATES["min_rolling_minutes"])
    if BUY_GATES.get("require_minutes_or_rolling", True):
        return prior >= min_prior or rolling >= min_roll
    return prior >= min_prior and rolling >= min_roll


def _n_fixtures_series(
    rows: pd.DataFrame,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
) -> pd.Series:
    """Per-row fixture count for the decision GW (defaults to 1)."""
    if "n_fixtures" in rows.columns:
        return rows["n_fixtures"].astype(float).fillna(1.0).clip(lower=1.0)
    out = pd.Series(1.0, index=rows.index, dtype=float)
    if team_fixtures is None or "gameweek" not in rows.columns or "team_id" not in rows.columns:
        return out
    if rows.empty or rows["gameweek"].isna().all():
        return out
    gw0 = int(rows["gameweek"].dropna().iloc[0])
    for idx, row in rows.iterrows():
        slot = (team_fixtures.get(int(row["team_id"])) or {}).get(gw0)
        out.at[idx] = float((slot or {}).get("n_fixtures") or 1.0)
    return out.clip(lower=1.0)


def _apply_missing_xp_fallback(
    rows: pd.DataFrame,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
) -> None:
    """Fill near-zero EV from lagged open features when historical xP is unusable.

    Two cases (both use lagged features only — never same-GW points/ICT):
    1. GW-wide blank xP (vaastav often zeros an entire round) → fill all near-zero EVs.
    2. Otherwise only DGW rows with missing EV (so doubles stay selectable).
    Individual xP=0 on a normal GW is left alone (often injured / out of contention).
    """
    from fpl_robo.config import MISSING_XP_FALLBACK

    if not MISSING_XP_FALLBACK.get("enabled", True) or rows.empty:
        return
    floor = float(MISSING_XP_FALLBACK.get("ev_floor") or 0.05)
    cap = float(MISSING_XP_FALLBACK.get("ev_cap") or 12.0)
    dgw_mult = float(MISSING_XP_FALLBACK.get("dgw_mult") or 1.4)
    blank_frac = float(MISSING_XP_FALLBACK.get("gw_blank_frac") or 0.90)

    xp = rows["xP"].astype(float) if "xP" in rows.columns else rows["ev"].astype(float)
    gw_blank = float((xp <= floor).mean()) >= blank_frac
    nfix = _n_fixtures_series(rows, team_fixtures)
    if gw_blank:
        missing = rows["ev"].astype(float) <= floor
    else:
        missing = (nfix >= 2.0) & (rows["ev"].astype(float) <= floor)
    if not missing.any():
        return

    proxy = _open_ev_from_row(rows).astype(float)
    boost = nfix.ge(2.0).map({True: dgw_mult, False: 1.0}).astype(float)
    proxy = (proxy * boost).clip(lower=0.0, upper=cap)
    rows.loc[missing, "ev"] = proxy.loc[missing]


def build_predictions(
    gw_df: pd.DataFrame,
    *,
    mode: EvMode = "ep_baseline",
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    blank_ahead: bool | None = None,
    fixture_run: bool | None = None,
    current_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Build a decision-time prediction frame for one gameweek.

    Expects `gw_df` rows for a single GW, ideally after `attach_lagged_history`
    has been applied to the full season table.
    Horizon (`*_h`) modes reweight EV using upcoming fixture counts when provided.
    `blank_ahead` haircuts players whose club blanks in the next few GWs
    (defaults to config BLANK_AHEAD.enabled). When gated (config default),
    that haircut only runs if `current_ids` already own upcoming blanks.
    """
    from fpl_robo.config import BLANK_AHEAD
    from fpl_robo.predict.horizon import apply_blank_ahead_ev, apply_horizon_ev

    rows = gw_df.copy()
    base_mode = strip_horizon_suffix(mode)
    if "team_id" not in rows.columns:
        rows["team_id"] = pd.factorize(rows["team"])[0] + 1

    if "prior_minutes" not in rows.columns:
        rows["prior_minutes"] = 0.0
    if "rolling_minutes" not in rows.columns:
        rows["rolling_minutes"] = 0.0
    if "cost_change_event" not in rows.columns:
        if "price_delta" in rows.columns:
            # Tenths of a million → approximate FPL cost_change_event units.
            rows["cost_change_event"] = (rows["price_delta"].astype(float) * 10.0).round()
        else:
            rows["cost_change_event"] = 0

    if base_mode in ("ep_baseline", "ep_hardened"):
        rows["ev"] = rows["xP"].astype(float).clip(lower=0.0)
    elif base_mode == "ep_minutes_scaled":
        xp = rows["xP"].astype(float).clip(lower=0.0)
        rows["ev"] = (xp * _play_prob_series(rows)).clip(lower=0.0)
    elif base_mode == "open_ev":
        rows["ev"] = _open_ev_from_row(rows)
    else:
        raise ValueError(f"Unknown EV mode: {mode}")

    # Align historical xP with live ep_next so ATTACK_FLOOR / quality gates apply
    # the same way in backtests as in live BaselineEVModel predictions.
    if "ep_next" not in rows.columns:
        if "xP" in rows.columns:
            rows["ep_next"] = rows["xP"].astype(float).clip(lower=0.0)
        else:
            rows["ep_next"] = rows["ev"].astype(float).clip(lower=0.0)

    # Historical merged_gw often blanks xP for whole GWs. Fill those (and DGW gaps)
    # from lagged open features only — never same-GW points/ICT.
    if base_mode != "open_ev":
        _apply_missing_xp_fallback(rows, team_fixtures=team_fixtures)

    rows["status"] = "a"
    rows["chance_of_playing_next_round"] = 100

    minutes_ok = rows.apply(_minutes_gate_ok, axis=1)
    cold = (rows["prior_minutes"].astype(float) <= 0) & (rows["rolling_minutes"].astype(float) <= 0)
    if base_mode in ("ep_baseline", "ep_minutes_scaled"):
        # Soft cover on post-haircut EV (minutes_scaled) or raw EP (baseline).
        rows["cover_eligible"] = rows["ev"] >= float(BENCH_COVER["min_ep_next"])
        rows["buy_eligible"] = True
    elif base_mode == "ep_hardened":
        # Stricter: meaningful EV and prior involvement evidence.
        rows["cover_eligible"] = (rows["ev"] >= float(BUY_GATES["min_buy_ev"])) & minutes_ok
        rows["buy_eligible"] = rows["cover_eligible"]
    else:  # open_ev
        # After history exists, require open EV + minutes evidence.
        # Cold start: allow lower EV so initial squads can form without EP.
        open_cover_floor = rows["ev"] >= 1.5
        cold_cover = cold & (rows["ev"] >= 0.05)
        rows["cover_eligible"] = (open_cover_floor | cold_cover) & minutes_ok
        rows["buy_eligible"] = (
            ((rows["ev"] >= float(BUY_GATES["min_buy_ev"])) | cold_cover) & minutes_ok
        )

    gw = None
    if "gameweek" in rows.columns and len(rows) and not rows["gameweek"].isna().all():
        gw = int(rows["gameweek"].dropna().iloc[0])

    if mode_uses_horizon(mode) and team_fixtures is not None and gw is not None:
        rows = apply_horizon_ev(rows, start_gw=gw, team_fixtures=team_fixtures)

    # Fixture-run planning (symmetric FDR). Explicit *_run modes always apply;
    # `fixture_run` override wins over FIXTURE_RUN.enabled for A/B backtests.
    from fpl_robo.config import FIXTURE_RUN
    from fpl_robo.predict.fixture_run import apply_fixture_run_ev
    from fpl_robo.predict.horizon import mode_uses_fixture_run

    apply_run = False
    if team_fixtures is not None and gw is not None and not mode_uses_horizon(mode):
        if fixture_run is True or mode_uses_fixture_run(mode):
            apply_run = True
        elif fixture_run is False:
            apply_run = False
        elif bool(FIXTURE_RUN.get("enabled")):
            apply_run = True
    if apply_run:
        rows = apply_fixture_run_ev(rows, start_gw=gw, team_fixtures=team_fixtures)

    use_blank = bool(BLANK_AHEAD.get("enabled")) if blank_ahead is None else bool(blank_ahead)
    if use_blank and team_fixtures is not None and gw is not None:
        rows = apply_blank_ahead_ev(
            rows,
            start_gw=gw,
            team_fixtures=team_fixtures,
            current_ids=current_ids,
        )

    keep = [
        "id",
        "web_name",
        "position",
        "team_id",
        "team",
        "price",
        "ev",
        "status",
        "chance_of_playing_next_round",
        "cover_eligible",
        "buy_eligible",
        "prior_minutes",
        "rolling_minutes",
        "gameweek",
        "total_points",
        "minutes",
        "xP",
        "ep_next",
        "cost_change_event",
        "ev_gw0",
        "horizon_factor",
        "fixture_run_factor",
        "blank_ahead_factor",
        "blank_ahead_triggered",
        "n_fixtures",
        # Secondary features for optimiser tie-breaks (optional).
        "form",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "selected",
        "selected_by_percent",
        "rolling_points",
        "rolling_expected_goals",
        "rolling_expected_assists",
        "rolling_ict_index",
    ]
    for c in keep:
        if c not in rows.columns:
            if c in ("xP", "ev_gw0", "horizon_factor", "fixture_run_factor", "ep_next"):
                rows[c] = 0.0
            elif c == "blank_ahead_factor":
                rows[c] = 1.0
            elif c == "blank_ahead_triggered":
                rows[c] = False
            elif c == "team":
                continue
            elif c in {"web_name", "position"}:
                rows[c] = ""
            elif c in {
                "form",
                "ict_index",
                "expected_goals",
                "expected_assists",
                "selected",
                "selected_by_percent",
                "rolling_points",
                "rolling_expected_goals",
                "rolling_expected_assists",
                "rolling_ict_index",
            }:
                continue
            else:
                rows[c] = 0
    present = [c for c in keep if c in rows.columns]
    return rows[present]


def _open_ev_from_row(rows: pd.DataFrame) -> pd.Series:
    """Open EV: lagged points/minutes + weak price prior (+ optional xG/ICT)."""
    pts = rows["rolling_points"].astype(float) if "rolling_points" in rows.columns else 0.0
    mins = rows["rolling_minutes"].astype(float) if "rolling_minutes" in rows.columns else 0.0
    price = rows["price"].astype(float)

    # Involvement scalar in [0.35, 1.15]
    involv = (0.35 + (mins / 90.0).clip(lower=0.0, upper=1.0) * 0.80).astype(float)
    # Price prior: centre around 6.0m
    price_prior = (0.15 * (price - 4.5).clip(lower=0.0, upper=8.0)).astype(float)

    xg = rows["rolling_expected_goals"].astype(float) if "rolling_expected_goals" in rows.columns else 0.0
    xa = (
        rows["rolling_expected_assists"].astype(float)
        if "rolling_expected_assists" in rows.columns
        else 0.0
    )
    ict = rows["rolling_ict_index"].astype(float) if "rolling_ict_index" in rows.columns else 0.0

    open_bonus = 2.5 * xg + 1.5 * xa + 0.02 * ict
    ev = (pts * involv + price_prior + open_bonus).clip(lower=0.0)
    # Cold start (no history): tiny floor from price only so MIP can still fill early GWs.
    cold = (pts.isna() if hasattr(pts, "isna") else False) | (
        (pts == 0) & (mins == 0)
    )
    if hasattr(cold, "astype"):
        ev = ev.where(~cold, (0.05 * price).clip(lower=0.05, upper=1.0))
    return ev.astype(float)


def open_ev_from_player_history(
    *,
    rolling_points: float,
    rolling_minutes: float,
    price: float,
    rolling_xg: float = 0.0,
    rolling_xa: float = 0.0,
    rolling_ict: float = 0.0,
) -> float:
    """Live helper matching open_ev formula for a single player."""
    involv = 0.35 + min(1.0, max(0.0, rolling_minutes / 90.0)) * 0.80
    price_prior = 0.15 * min(8.0, max(0.0, price - 4.5))
    open_bonus = 2.5 * rolling_xg + 1.5 * rolling_xa + 0.02 * rolling_ict
    if rolling_points <= 0 and rolling_minutes <= 0:
        return float(min(1.0, max(0.05, 0.05 * price)))
    return float(max(0.0, rolling_points * involv + price_prior + open_bonus))


def select_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick best mode: total_points, then points_vs_average, then fewer hits_cost."""

    def key(r: dict[str, Any]) -> tuple:
        return (
            float(r.get("total_points") or 0.0),
            float(r.get("points_vs_average") or -1e18),
            -float(r.get("hits_cost") or 0.0),
        )

    return max(results, key=key)


def transfer_gate_kwargs(mode: EvMode) -> dict[str, Any]:
    """Kwargs for optimise_transfers buy gates."""
    base = strip_horizon_suffix(mode)
    if base in ("ep_baseline", "ep_minutes_scaled"):
        return {"enforce_buy_eligible": False, "min_buy_ev": None}
    return {
        "enforce_buy_eligible": True,
        "min_buy_ev": float(BUY_GATES["min_buy_ev"]),
    }


def make_live_ev_model(mode: EvMode | None = None):
    """Return the live predictor for the configured (or given) EV mode."""
    from fpl_robo.config import EV_MODE
    from fpl_robo.predict.baseline import BaselineEVModel
    from fpl_robo.predict.open_ev import OpenEVModel

    resolved: str = mode or EV_MODE
    base = strip_horizon_suffix(resolved)
    if base == "open_ev":
        return OpenEVModel()
    if base == "ep_hardened":
        return BaselineEVModel(hardened=True, mode="ep_hardened")
    if base == "ep_minutes_scaled":
        return BaselineEVModel(hardened=False, mode="ep_minutes_scaled")
    return BaselineEVModel(hardened=False, mode="ep_baseline")
