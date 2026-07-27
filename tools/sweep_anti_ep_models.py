"""Backtest models aimed at baseline EP failure modes + open EP reconstruction.

Writes data/ev_anti_ep_models.json. Does not change live EV_MODE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import fpl_robo.backtest.season as season_mod
import fpl_robo.predict.ev_modes as ev_modes
from fpl_robo.backtest.season import run_season_backtest
from fpl_robo.config import BENCH_COVER, BUY_GATES
from fpl_robo.predict.ev_modes import select_winner

GOAL_PTS = {"GK": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
CS_PTS = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
ASSIST_PTS = 3.0


def _attach_extra(table: pd.DataFrame, *, window: int = 4) -> pd.DataFrame:
    """Lagged features beyond the default attach_lagged_history set."""
    df = ev_modes.attach_lagged_history(table, window=window)
    g = df.groupby("id", sort=False)
    extras = {
        "starts": "sum",
        "bonus": "sum",
        "bps": "sum",
        "clean_sheets": "sum",
        "expected_goals_conceded": "sum",
        "expected_goal_involvements": "sum",
        "saves": "sum",
        "goals_scored": "sum",
        "assists": "sum",
    }
    for col in extras:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        df[f"rolling_{col}"] = (
            shifted.groupby(df["id"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df[f"rolling_{col}"] = df[f"rolling_{col}"].fillna(0.0)
    # Start rate proxy from minutes if starts missing.
    if "rolling_starts" not in df.columns:
        df["rolling_starts"] = (df["rolling_minutes"] / 90.0).clip(0, 1)
    return df


def _finalize(
    rows: pd.DataFrame,
    *,
    cover_floor: float = 1.5,
    enforce_buy: bool = False,
    buy_floor: float | None = None,
) -> pd.DataFrame:
    rows = rows.copy()
    rows["status"] = "a"
    rows["chance_of_playing_next_round"] = 100
    if "team_id" not in rows.columns:
        rows["team_id"] = pd.factorize(rows["team"])[0] + 1
    if "prior_minutes" not in rows.columns:
        rows["prior_minutes"] = 0.0
    if "rolling_minutes" not in rows.columns:
        rows["rolling_minutes"] = 0.0

    minutes_ok = rows.apply(ev_modes._minutes_gate_ok, axis=1)
    cold = (rows["prior_minutes"].astype(float) <= 0) & (rows["rolling_minutes"].astype(float) <= 0)
    rows["cover_eligible"] = rows["ev"] >= cover_floor
    if enforce_buy:
        bf = float(BUY_GATES["min_buy_ev"] if buy_floor is None else buy_floor)
        rows["buy_eligible"] = (rows["ev"] >= bf) & minutes_ok
        rows["cover_eligible"] = rows["buy_eligible"] | (cold & (rows["ev"] >= 0.05))
    else:
        rows["buy_eligible"] = True

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
            rows[c] = 0.0 if c not in ("web_name", "position") else ""
    return rows[keep]


def model_ep_minutes_scaled(gw: pd.DataFrame) -> pd.DataFrame:
    """Haircut xP by recent play probability — baseline's main failure mode."""
    rows = gw.copy()
    xp = rows["xP"].astype(float).clip(lower=0.0)
    play = (rows["rolling_minutes"].astype(float) / 80.0).clip(lower=0.20, upper=1.0)
    cold = (rows["prior_minutes"].astype(float) <= 0) & (rows["rolling_minutes"].astype(float) <= 0)
    play = play.where(~cold, 0.75)  # mild prior preseason
    rows["ev"] = (xp * play).clip(lower=0.0)
    return _finalize(rows)


def model_ep_shrink_to_form(gw: pd.DataFrame) -> pd.DataFrame:
    """Shrink xP toward lagged rolling points when EP and form disagree."""
    rows = gw.copy()
    xp = rows["xP"].astype(float).clip(lower=0.0)
    form = rows["rolling_points"].astype(float).clip(lower=0.0)
    # More weight on form when minutes evidence exists; else trust xP.
    mins = rows["rolling_minutes"].astype(float)
    form_w = (0.25 + 0.50 * (mins / 90.0).clip(0, 1)).clip(0.25, 0.75)
    rows["ev"] = ((1.0 - form_w) * xp + form_w * form).clip(lower=0.0)
    return _finalize(rows)


def model_ep_reliability(gw: pd.DataFrame) -> pd.DataFrame:
    """Trust xP only when recent minutes support it; else blend heavily to form."""
    rows = gw.copy()
    xp = rows["xP"].astype(float).clip(lower=0.0)
    form = rows["rolling_points"].astype(float).clip(lower=0.0)
    mins = rows["rolling_minutes"].astype(float)
    prior = rows["prior_minutes"].astype(float)
    reliable = (mins >= 60) | (prior >= 270)
    # Unreliable: 30% xP / 70% form, and multiply by soft play factor.
    play = (mins / 75.0).clip(0.15, 1.0)
    blended = 0.30 * xp + 0.70 * form
    rows["ev"] = pd.Series(0.0, index=rows.index)
    rows.loc[reliable, "ev"] = xp[reliable]
    rows.loc[~reliable, "ev"] = (blended[~reliable] * play[~reliable]).clip(lower=0.0)
    cold = (prior <= 0) & (mins <= 0)
    rows.loc[cold, "ev"] = xp[cold] * 0.7
    return _finalize(rows)


def model_ep_anti_hype(gw: pd.DataFrame) -> pd.DataFrame:
    """Aggressive discount when high xP meets thin minutes history."""
    rows = gw.copy()
    xp = rows["xP"].astype(float).clip(lower=0.0)
    prior = rows["prior_minutes"].astype(float)
    rollm = rows["rolling_minutes"].astype(float)
    # Hype score: high EP with low involvement.
    thin = (prior < 180) & (rollm < 90)
    very_thin = (prior < 90) & (rollm < 45)
    ev = xp.copy()
    ev = ev.where(~thin, xp * 0.55 + rows["rolling_points"].astype(float) * 0.45)
    ev = ev.where(~very_thin, xp * 0.30 + rows["rolling_points"].astype(float) * 0.40)
    # Extra haircut for very thin + expensive.
    expensive = rows["price"].astype(float) >= 6.5
    ev = ev.where(~(very_thin & expensive), ev * 0.75)
    rows["ev"] = ev.clip(lower=0.0)
    return _finalize(rows, enforce_buy=True, buy_floor=1.8)


def model_recon_ep(gw: pd.DataFrame) -> pd.DataFrame:
    """Open reconstruction of an EP-like metric from lagged xG/xA/minutes/CS."""
    rows = gw.copy()
    pos = rows["position"].astype(str)
    mins = rows["rolling_minutes"].astype(float)
    play60 = (mins / 90.0).clip(0.0, 1.0)
    play_any = (mins / 45.0).clip(0.0, 1.0)

    # Appearance points: ~2 if 60+, ~1 if cameo.
    appear = play60 * 2.0 + (play_any - play60).clip(lower=0.0) * 1.0

    xg = rows["rolling_expected_goals"].astype(float) if "rolling_expected_goals" in rows.columns else 0.0
    xa = rows["rolling_expected_assists"].astype(float) if "rolling_expected_assists" in rows.columns else 0.0
    xgi = (
        rows["rolling_expected_goal_involvements"].astype(float)
        if "rolling_expected_goal_involvements" in rows.columns
        else (xg + xa if hasattr(xg, "__add__") else 0.0)
    )
    goal_pts = pos.map(GOAL_PTS).fillna(4.0).astype(float)
    attack = xg * goal_pts + xa * ASSIST_PTS
    # Prefer explicit xG/xA; fall back to xGI split if needed.
    if isinstance(xgi, pd.Series):
        attack = attack.where((xg + xa) > 0, xgi * 4.0)

    # Clean sheet proxy from lagged CS rate * CS points.
    cs_rate = rows["rolling_clean_sheets"].astype(float) if "rolling_clean_sheets" in rows.columns else 0.0
    # Soften with xGC if present (lower xGC → higher CS chance).
    if "rolling_expected_goals_conceded" in rows.columns:
        xgc = rows["rolling_expected_goals_conceded"].astype(float).clip(lower=0.0)
        cs_from_xgc = (1.0 / (1.0 + xgc)).clip(0.05, 0.55)
        cs_prob = (0.5 * cs_rate + 0.5 * cs_from_xgc) * play60
    else:
        cs_prob = cs_rate * play60
    cs = cs_prob * pos.map(CS_PTS).fillna(0.0).astype(float)

    # Bonus / ICT crumbs.
    bonus = rows["rolling_bonus"].astype(float) if "rolling_bonus" in rows.columns else 0.0
    ict = rows["rolling_ict_index"].astype(float) if "rolling_ict_index" in rows.columns else 0.0
    extras = 0.6 * bonus + 0.01 * ict

    # GK saves crumb.
    saves = rows["rolling_saves"].astype(float) if "rolling_saves" in rows.columns else 0.0
    save_pts = (saves / 3.0) * (pos == "GK").astype(float)

    # Weak price prior for cold start / unknown quantity.
    price = rows["price"].astype(float)
    price_prior = 0.08 * (price - 4.5).clip(0, 8)

    ev = appear + attack + cs + extras + save_pts + price_prior
    cold = (rows["prior_minutes"].astype(float) <= 0) & (mins <= 0)
    ev = ev.where(~cold, (0.08 * price).clip(0.1, 1.8))
    rows["ev"] = ev.clip(lower=0.0)
    return _finalize(rows, cover_floor=1.2, enforce_buy=True, buy_floor=1.5)


def model_recon_ep_plus_xp_anchor(gw: pd.DataFrame) -> pd.DataFrame:
    """Mostly reconstructed EP, lightly anchored to official xP when minutes are solid."""
    recon = model_recon_ep(gw)
    rows = gw.copy()
    xp = rows["xP"].astype(float).clip(lower=0.0)
    mins = rows["rolling_minutes"].astype(float)
    trust_xp = (mins / 90.0).clip(0.0, 0.35)  # at most 35% xP when full starter
    rows["ev"] = ((1.0 - trust_xp) * recon["ev"].astype(float) + trust_xp * xp).clip(lower=0.0)
    return _finalize(rows, cover_floor=1.2, enforce_buy=True, buy_floor=1.5)


def model_ep_capped_by_involvement(gw: pd.DataFrame) -> pd.DataFrame:
    """Cap official xP by a minutes-implied ceiling (can't outrun involvement)."""
    rows = gw.copy()
    xp = rows["xP"].astype(float).clip(lower=0.0)
    mins = rows["rolling_minutes"].astype(float)
    form = rows["rolling_points"].astype(float)
    # Ceiling: recent form + small upside, scaled by play time.
    ceiling = (form + 1.25) * (0.35 + 0.65 * (mins / 90.0).clip(0, 1))
    cold = (rows["prior_minutes"].astype(float) <= 0) & (mins <= 0)
    ceiling = ceiling.where(~cold, xp)  # no cap preseason
    rows["ev"] = pd.concat([xp, ceiling], axis=1).min(axis=1).clip(lower=0.0)
    return _finalize(rows)


def model_ppm_quality(gw: pd.DataFrame) -> pd.DataFrame:
    """Lagged points-per-match * play prob + mild price quality — no xP."""
    rows = gw.copy()
    form = rows["rolling_points"].astype(float)
    mins = rows["rolling_minutes"].astype(float)
    play = (mins / 85.0).clip(0.2, 1.0)
    xg = rows["rolling_expected_goals"].astype(float) if "rolling_expected_goals" in rows.columns else 0.0
    xa = rows["rolling_expected_assists"].astype(float) if "rolling_expected_assists" in rows.columns else 0.0
    price = rows["price"].astype(float)
    ev = form * play + 2.0 * xg + 1.5 * xa + 0.1 * (price - 4.5).clip(0, 8)
    cold = (rows["prior_minutes"].astype(float) <= 0) & (mins <= 0)
    ev = ev.where(~cold, (0.1 * price).clip(0.1, 2.0))
    rows["ev"] = ev.clip(lower=0.0)
    return _finalize(rows, cover_floor=1.2, enforce_buy=True, buy_floor=1.5)


MODELS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "ep_minutes_scaled": model_ep_minutes_scaled,
    "ep_shrink_to_form": model_ep_shrink_to_form,
    "ep_reliability": model_ep_reliability,
    "ep_anti_hype": model_ep_anti_hype,
    "ep_capped_by_involvement": model_ep_capped_by_involvement,
    "recon_ep": model_recon_ep,
    "recon_ep_xp_anchor": model_recon_ep_plus_xp_anchor,
    "ppm_quality": model_ppm_quality,
}


def run_model(label: str, builder: Callable[[pd.DataFrame], pd.DataFrame], *, max_hits: int = 1) -> dict[str, Any]:
    print(f"--- {label} ---", flush=True)
    old_attach = season_mod.attach_lagged_history
    old_build = season_mod.build_predictions
    old_pf = season_mod._prediction_frame
    old_gates = season_mod._transfer_gate_kwargs

    season_mod.attach_lagged_history = _attach_extra  # type: ignore[assignment]

    def build(gw_df, *, mode="ep_baseline"):  # noqa: ARG001
        return builder(gw_df)

    season_mod.build_predictions = build  # type: ignore[assignment]
    season_mod._prediction_frame = lambda gw_df, *, mode="ep_baseline": builder(gw_df)  # type: ignore[assignment]

    # Use hardened-style transfer gates when model sets buy_eligible meaningfully.
    def gates(mode="ep_baseline"):  # noqa: ARG001
        # Inspect builder name via label
        if label in {"ep_anti_hype", "recon_ep", "recon_ep_xp_anchor", "ppm_quality"}:
            return {"enforce_buy_eligible": True, "min_buy_ev": 1.5}
        return {"enforce_buy_eligible": False, "min_buy_ev": None}

    season_mod._transfer_gate_kwargs = gates  # type: ignore[assignment]

    try:
        # ev_mode string only affects notes/gates via our patched gates; use ep_baseline stub.
        result = run_season_backtest(ev_mode="ep_baseline", max_hits=max_hits)
    finally:
        season_mod.attach_lagged_history = old_attach  # type: ignore[assignment]
        season_mod.build_predictions = old_build  # type: ignore[assignment]
        season_mod._prediction_frame = old_pf  # type: ignore[assignment]
        season_mod._transfer_gate_kwargs = old_gates  # type: ignore[assignment]

    row = {
        "label": label,
        "total_points": result.total_points,
        "points_vs_average": result.points_vs_average,
        "hits_cost": result.hits_cost,
        "transfers_made": result.transfers_made,
        "max_hits": max_hits,
    }
    print(
        f"  → {result.total_points} | vs {result.points_vs_average} | "
        f"hits -{result.hits_cost} | xfers {result.transfers_made}",
        flush=True,
    )
    return row


def main() -> None:
    rows: list[dict[str, Any]] = []
    # Control
    print("--- baseline_control ---", flush=True)
    base = run_season_backtest(ev_mode="ep_baseline", max_hits=1)
    rows.append(
        {
            "label": "baseline_control",
            "total_points": base.total_points,
            "points_vs_average": base.points_vs_average,
            "hits_cost": base.hits_cost,
            "transfers_made": base.transfers_made,
            "max_hits": 1,
        }
    )
    print(f"  → {base.total_points}", flush=True)

    for label, builder in MODELS.items():
        rows.append(run_model(label, builder))

    # Best anti-hype / minutes-scaled also with max_hits=2
    for label in ("ep_minutes_scaled", "ep_anti_hype", "recon_ep", "ep_reliability"):
        rows.append(run_model(f"{label}_hits2", MODELS[label], max_hits=2))

    baseline_pts = next(r["total_points"] for r in rows if r["label"] == "baseline_control")
    ranked = sorted(
        rows,
        key=lambda r: (
            float(r["total_points"]),
            float(r["points_vs_average"] or -1e18),
            -float(r["hits_cost"]),
        ),
        reverse=True,
    )
    winner = select_winner(rows)
    report = {
        "season": "2025-26",
        "baseline_points": baseline_pts,
        "winner_label": winner["label"],
        "notes": {
            "ep_minutes_scaled": "xP * play_prob from lagged minutes (attacks EP overtrust of non-starters)",
            "ep_shrink_to_form": "Blend xP toward lagged rolling points",
            "ep_reliability": "Full xP only if recent starter; else form-heavy blend",
            "ep_anti_hype": "Haircut high-xP thin-minutes (and expensive) names",
            "ep_capped_by_involvement": "Cap xP by form+minutes ceiling",
            "recon_ep": "Open EP reconstruction: appear + xG/xA + CS + bonus (no xP)",
            "recon_ep_xp_anchor": "recon_ep with up to 35% official xP when starting",
            "ppm_quality": "Lagged points*play + xG/xA (no xP)",
        },
        "ranked": [
            {
                **{k: r[k] for k in ("label", "total_points", "points_vs_average", "hits_cost", "transfers_made", "max_hits")},
                "delta_vs_baseline": r["total_points"] - baseline_pts,
            }
            for r in ranked
        ],
        "results": rows,
    }
    out = Path("data/ev_anti_ep_models.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RANKING ===")
    print(f"{'label':<32} {'pts':>6} {'Δ':>6} {'vs_avg':>8} {'hits':>5}")
    for r in ranked:
        print(
            f"{r['label']:<32} {r['total_points']:>6} {r['total_points'] - baseline_pts:>+6} "
            f"{str(r['points_vs_average']):>8} {r['hits_cost']:>5}"
        )
    print(f"\nBest: {winner['label']} ({winner['total_points']})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
