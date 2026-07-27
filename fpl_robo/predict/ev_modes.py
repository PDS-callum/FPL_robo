"""EV prediction modes for backtests and live decisions."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from fpl_robo.config import BENCH_COVER, BUY_GATES, MINUTES_SCALED

EvMode = Literal["ep_baseline", "ep_hardened", "ep_minutes_scaled", "open_ev"]
EV_MODES: tuple[EvMode, ...] = ("ep_baseline", "ep_hardened", "ep_minutes_scaled", "open_ev")


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
) -> float:
    """Play-probability scalar used by ep_minutes_scaled."""
    div = float(MINUTES_SCALED["divisor"] if divisor is None else divisor)
    flo = float(MINUTES_SCALED["floor"] if floor is None else floor)
    cold = float(MINUTES_SCALED["cold_play"] if cold_play is None else cold_play)
    if float(prior_minutes) <= 0.0 and float(rolling_minutes) <= 0.0:
        return cold
    return float(min(1.0, max(flo, float(rolling_minutes) / div)))


def _play_prob_series(rows: pd.DataFrame) -> pd.Series:
    mins = rows["rolling_minutes"].astype(float) if "rolling_minutes" in rows.columns else 0.0
    prior = rows["prior_minutes"].astype(float) if "prior_minutes" in rows.columns else 0.0
    div = float(MINUTES_SCALED["divisor"])
    flo = float(MINUTES_SCALED["floor"])
    cold = float(MINUTES_SCALED["cold_play"])
    play = (mins / div).clip(lower=flo, upper=1.0)
    is_cold = (prior <= 0.0) & (mins <= 0.0)
    return play.where(~is_cold, cold)


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


def build_predictions(
    gw_df: pd.DataFrame,
    *,
    mode: EvMode = "ep_baseline",
) -> pd.DataFrame:
    """Build a decision-time prediction frame for one gameweek.

    Expects `gw_df` rows for a single GW, ideally after `attach_lagged_history`
    has been applied to the full season table.
    """
    rows = gw_df.copy()
    if "team_id" not in rows.columns:
        rows["team_id"] = pd.factorize(rows["team"])[0] + 1

    if "prior_minutes" not in rows.columns:
        rows["prior_minutes"] = 0.0
    if "rolling_minutes" not in rows.columns:
        rows["rolling_minutes"] = 0.0

    if mode in ("ep_baseline", "ep_hardened"):
        rows["ev"] = rows["xP"].astype(float).clip(lower=0.0)
    elif mode == "ep_minutes_scaled":
        xp = rows["xP"].astype(float).clip(lower=0.0)
        rows["ev"] = (xp * _play_prob_series(rows)).clip(lower=0.0)
    elif mode == "open_ev":
        rows["ev"] = _open_ev_from_row(rows)
    else:
        raise ValueError(f"Unknown EV mode: {mode}")

    rows["status"] = "a"
    rows["chance_of_playing_next_round"] = 100

    minutes_ok = rows.apply(_minutes_gate_ok, axis=1)
    cold = (rows["prior_minutes"].astype(float) <= 0) & (rows["rolling_minutes"].astype(float) <= 0)
    if mode in ("ep_baseline", "ep_minutes_scaled"):
        # Soft cover on post-haircut EV (minutes_scaled) or raw EP (baseline).
        rows["cover_eligible"] = rows["ev"] >= float(BENCH_COVER["min_ep_next"])
        rows["buy_eligible"] = True
    elif mode == "ep_hardened":
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

    keep = [
        "id",
        "web_name",
        "position",
        "team_id",
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
    ]
    for c in keep:
        if c not in rows.columns:
            if c == "xP":
                rows[c] = 0.0
            elif c in ("web_name", "position"):
                rows[c] = ""
            else:
                rows[c] = 0
    return rows[keep]


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
    if mode in ("ep_baseline", "ep_minutes_scaled"):
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

    resolved: EvMode = mode or EV_MODE  # type: ignore[assignment]
    if resolved == "open_ev":
        return OpenEVModel()
    if resolved == "ep_hardened":
        return BaselineEVModel(hardened=True, mode="ep_hardened")
    if resolved == "ep_minutes_scaled":
        return BaselineEVModel(hardened=False, mode="ep_minutes_scaled")
    return BaselineEVModel(hardened=False, mode="ep_baseline")
