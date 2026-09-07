"""Backtest chip timing heuristics on historical merged GW data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

from fpl_robo.chips.decide import recommend_chips
from fpl_robo.chips.inventory import ChipInventory, ChipWindow, half_window
from fpl_robo.chips.scoring import score_chip_opportunities
from fpl_robo.config import (
    BENCH_COVER,
    CHIP_BACKTEST_CACHE,
    CHIP_BACKTEST_SEASON,
    CHIP_DEFAULTS,
)
from fpl_robo.optimize.single_gw import optimise_single_gw

VAASTAV_MERGED_GW = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/gws/merged_gw.csv"
)


@dataclass
class ChipBacktestResult:
    season: str
    half_split_gw: int
    triple_captain: dict[str, Any]
    bench_boost: dict[str, Any]
    free_hit: dict[str, Any]
    wildcard: dict[str, Any]
    weekly: list[dict[str, Any]]
    verdict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_merged_gw(
    season: str = CHIP_BACKTEST_SEASON,
    cache_path: str = CHIP_BACKTEST_CACHE,
) -> pd.DataFrame:
    """Load/cache vaastav merged_gw.csv for a season."""
    cache = Path(cache_path)
    if cache.exists():
        return pd.read_csv(cache)

    url = VAASTAV_MERGED_GW.format(season=season)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed github raw URL
        cache.write_bytes(resp.read())
    return pd.read_csv(cache)


def prepare_gw_table(
    merged: pd.DataFrame,
    *,
    season: str = CHIP_BACKTEST_SEASON,
    fill_missing_xp: bool = True,
) -> pd.DataFrame:
    """Aggregate DGW rows to one row per player-gameweek."""
    df = merged.copy()
    df["price"] = df["value"].astype(float) / 10.0
    df["position"] = df["position"].astype(str).str.upper()
    # Map any legacy labels
    df["position"] = df["position"].replace({"GKP": "GK", "FWD": "FWD"})

    agg: dict[str, tuple[str, str]] = {
        "xP": ("xP", "sum"),
        "total_points": ("total_points", "sum"),
        "minutes": ("minutes", "sum"),
        "price": ("price", "first"),
        "web_name": ("name", "first"),
        "selected": ("selected", "max"),
    }
    optional = {
        "expected_goals": "sum",
        "expected_assists": "sum",
        "expected_goal_involvements": "sum",
        "expected_goals_conceded": "sum",
        "ict_index": "mean",
        "influence": "mean",
        "creativity": "mean",
        "threat": "mean",
        "starts": "max",
        "bonus": "sum",
        "bps": "sum",
        "clean_sheets": "max",
        "saves": "sum",
        "goals_scored": "sum",
        "assists": "sum",
    }
    for col, how in optional.items():
        if col in df.columns:
            agg[col] = (col, how)
    if "fixture" in df.columns:
        agg["n_fixtures"] = ("fixture", "nunique")
    elif "opponent_team" in df.columns:
        agg["n_fixtures"] = ("opponent_team", "nunique")

    grouped = (
        df.groupby(["element", "GW", "position", "team"], as_index=False)
        .agg(**{k: v for k, v in agg.items()})
        .rename(columns={"element": "id", "GW": "gameweek", "web_name": "web_name"})
    )
    # .agg with named tuples via ** might name web_name oddly if source is "name"
    if "web_name" not in grouped.columns and "name" in grouped.columns:
        grouped = grouped.rename(columns={"name": "web_name"})
    # Prefer official FPL team ids (aligns with fixtures.csv FDR). Fall back to
    # stable name factorisation when teams.csv is missing.
    try:
        from fpl_robo.predict.fixture_run import fpl_team_id_map_from_merged

        name_map = fpl_team_id_map_from_merged(grouped, season)
        if len(name_map) >= max(1, int(grouped["team"].nunique() * 0.8)):
            grouped["team_id"] = grouped["team"].map(lambda t: name_map.get(str(t)))
            if grouped["team_id"].isna().any():
                # Fill any gaps with high synthetic ids so uniqueness holds.
                next_id = int(max((i for i in name_map.values()), default=20)) + 1
                for idx in grouped.index[grouped["team_id"].isna()]:
                    grouped.at[idx, "team_id"] = next_id
                    next_id += 1
            grouped["team_id"] = grouped["team_id"].astype(int)
        else:
            raise ValueError("incomplete FPL team map")
    except Exception:  # noqa: BLE001
        team_codes = sorted(str(t) for t in grouped["team"].unique())
        team_map = {t: i + 1 for i, t in enumerate(team_codes)}
        grouped["team_id"] = grouped["team"].map(lambda t: team_map[str(t)])
    if "n_fixtures" not in grouped.columns:
        grouped["n_fixtures"] = 1
    if fill_missing_xp:
        from fpl_robo.data.ep_history import fill_blank_gw_xp

        grouped = fill_blank_gw_xp(grouped, season=season)
    return grouped


def _prediction_frame(gw_df: pd.DataFrame) -> pd.DataFrame:
    rows = gw_df.copy()
    rows["ev"] = rows["xP"].astype(float).clip(lower=0.0)
    rows["status"] = "a"
    rows["chance_of_playing_next_round"] = 100
    rows["cover_eligible"] = (rows["ev"] >= 1.5) | (rows["minutes"] >= 90)
    rows["team_id"] = pd.factorize(rows["team"])[0] + 1
    return rows[
        [
            "id",
            "web_name",
            "position",
            "team_id",
            "price",
            "ev",
            "status",
            "chance_of_playing_next_round",
            "cover_eligible",
            "gameweek",
            "total_points",
            "minutes",
            "xP",
        ]
    ]


def _actual_points_map(gw_df: pd.DataFrame) -> dict[int, float]:
    return {int(r["id"]): float(r["total_points"]) for _, r in gw_df.iterrows()}


def run_chip_backtest(
    season: str = CHIP_BACKTEST_SEASON,
    half_split_gw: int = CHIP_DEFAULTS["half_split_gw"],
    cache_path: str = CHIP_BACKTEST_CACHE,
) -> ChipBacktestResult:
    """Backtest TC/BB/FH/WC opportunity timing on historical xP → actual points."""
    merged = load_merged_gw(season=season, cache_path=cache_path)
    table = prepare_gw_table(merged)
    gameweeks = sorted(int(g) for g in table["gameweek"].unique())

    weekly: list[dict[str, Any]] = []
    prev_squad_ids: list[int] | None = None
    rolling_gaps: list[float] = []

    for gw in gameweeks:
        gw_df = table[table["gameweek"] == gw].copy()
        preds = _prediction_frame(gw_df)
        sol = optimise_single_gw(
            preds,
            budget=100.0,
            min_play_chance=None,
            require_cover_eligible=True,
            bench_ev_weight=BENCH_COVER["bench_ev_weight"],
        )
        if sol.status != "Optimal":
            weekly.append({"gameweek": gw, "status": sol.status})
            continue

        actual = _actual_points_map(gw_df)
        xi_ids = [int(p["id"]) for p in sol.starting_xi]
        bench_ids = [int(p["id"]) for p in sol.bench]
        cap_id = int(sol.captain_id) if sol.captain_id is not None else xi_ids[0]

        # Predicted opportunity scores from reconstructed squad.
        scores = score_chip_opportunities(
            starting_xi=sol.starting_xi,
            bench=sol.bench,
            captain_id=cap_id,
        )

        # TC timing signal: premium haul potential (not only current squad captain).
        premiums = preds[preds["price"] >= 9.0]
        if premiums.empty:
            premiums = preds.nlargest(20, "ev")
        best_prem = premiums.sort_values("ev", ascending=False).iloc[0]
        prem_id = int(best_prem["id"])
        pred_tc = round(2.0 * float(best_prem["ev"]), 3)
        # Blend squad-captain view with premium-haul view.
        pred_tc = round(0.45 * scores.triple_captain + 0.55 * pred_tc, 3)

        # BB timing signal: only count cover-like bench projections.
        cover_bench = [p for p in sol.bench if float(p.get("ev") or 0) >= 1.5]
        pred_bb = round(float(sum(float(p.get("ev") or 0) for p in cover_bench)), 3)
        if len(cover_bench) < 4:
            # Penalise incomplete cover benches.
            pred_bb = round(pred_bb * (len(cover_bench) / 4.0), 3)

        # Actual realized values for the chosen premium / bench targets.
        tc_actual = 2.0 * float(actual.get(prem_id, 0.0))
        bb_actual = float(sum(actual.get(i, 0.0) for i in bench_ids))
        xi_actual = float(sum(actual.get(i, 0.0) for i in xi_ids))

        # FH: compare optimal XI actual vs previous squad XI actual (if known)
        fh_actual = 0.0
        fh_pred = 0.0
        if prev_squad_ids is not None:
            # Keep previous 15, pick best legal-ish XI by predicted EV among owned.
            owned = preds[preds["id"].isin(prev_squad_ids)].copy()
            if len(owned) >= 11:
                # Approximate locked XI: top EV by position bounds is heavy; use top 11 EV owned.
                locked = owned.sort_values("ev", ascending=False).head(11)
                locked_xi_ev = float(locked["ev"].sum())
                locked_ids = [int(x) for x in locked["id"].tolist()]
                locked_actual = float(sum(actual.get(i, 0.0) for i in locked_ids))
                fh_pred = max(0.0, scores.xi_ev - locked_xi_ev)
                fh_actual = max(0.0, xi_actual - locked_actual)
        scores.free_hit = round(fh_pred, 3)

        rolling_gaps.append(fh_pred)
        horizon = int(CHIP_DEFAULTS["rolling_gap_horizon"])
        rolling = float(sum(rolling_gaps[-horizon:]))
        scores.wildcard = round(rolling, 3)

        weekly.append(
            {
                "gameweek": gw,
                "status": "Optimal",
                "captain_id": cap_id,
                "captain_name": sol.captain_name,
                "tc_target_id": prem_id,
                "pred_tc": pred_tc,
                "pred_bb": pred_bb,
                "pred_fh": scores.free_hit,
                "pred_wc": scores.wildcard,
                "actual_tc": round(tc_actual, 3),
                "actual_bb": round(bb_actual, 3),
                "actual_fh": round(fh_actual, 3),
                "actual_xi": round(xi_actual, 3),
                "xi_ids": xi_ids,
                "bench_ids": bench_ids,
            }
        )
        prev_squad_ids = [int(p["id"]) for p in sol.squad]

    usable = [w for w in weekly if w.get("status") == "Optimal"]
    tc_report = _evaluate_chip_timing(
        usable,
        pred_key="pred_tc",
        actual_key="actual_tc",
        half_split_gw=half_split_gw,
        chip="triple_captain",
    )
    bb_report = _evaluate_chip_timing(
        usable,
        pred_key="pred_bb",
        actual_key="actual_bb",
        half_split_gw=half_split_gw,
        chip="bench_boost",
    )
    fh_report = _evaluate_threshold_hits(
        usable,
        pred_key="pred_fh",
        actual_key="actual_fh",
        threshold=float(CHIP_DEFAULTS["fh_min_gap"]),
        chip="free_hit",
    )
    wc_report = _evaluate_threshold_hits(
        usable,
        pred_key="pred_wc",
        actual_key="actual_fh",  # proxy: WC should target persistent FH-like gaps
        threshold=float(CHIP_DEFAULTS["wc_min_rolling_gap"]),
        chip="wildcard",
    )

    verdict = _competence_verdict(tc_report, bb_report, fh_report)
    return ChipBacktestResult(
        season=season,
        half_split_gw=half_split_gw,
        triple_captain=tc_report,
        bench_boost=bb_report,
        free_hit=fh_report,
        wildcard=wc_report,
        weekly=usable,
        verdict=verdict,
    )


def _evaluate_chip_timing(
    weekly: list[dict[str, Any]],
    *,
    pred_key: str,
    actual_key: str,
    half_split_gw: int,
    chip: str,
) -> dict[str, Any]:
    halves = {
        "first_half": [w for w in weekly if int(w["gameweek"]) <= half_split_gw],
        "second_half": [w for w in weekly if int(w["gameweek"]) > half_split_gw],
    }
    out: dict[str, Any] = {"chip": chip, "halves": {}}
    capture_rates: list[float] = []
    top_quartile_hits = 0
    half_count = 0

    for half_name, rows in halves.items():
        if not rows:
            continue
        half_count += 1
        # Greedy: pick the GW with best predicted score in the half.
        recommended = max(rows, key=lambda r: float(r[pred_key]))
        # If the top predicted week is a clear outlier, keep it; otherwise choose
        # among the top-3 predicted weeks the one with best predicted score density
        # (still no actual leakage).
        ranked_pred = sorted(rows, key=lambda r: float(r[pred_key]), reverse=True)
        top_k = ranked_pred[: min(3, len(ranked_pred))]
        recommended = top_k[0]
        oracle = max(rows, key=lambda r: float(r[actual_key]))
        mean_actual = float(sum(float(r[actual_key]) for r in rows) / len(rows))
        # Prefer top predicted week unless the 2nd/3rd is within 5% and later in
        # the window (optional diversification) — keep greedy for stability.
        rec_actual = float(recommended[actual_key])
        ora_actual = float(oracle[actual_key])
        capture = 0.0 if ora_actual <= 0 else rec_actual / ora_actual
        capture_rates.append(capture)

        ranked = sorted(rows, key=lambda r: float(r[actual_key]), reverse=True)
        rec_rank = next(i for i, r in enumerate(ranked, start=1) if r["gameweek"] == recommended["gameweek"])
        top_q = max(1, int(round(0.25 * len(rows))))
        in_top_q = rec_rank <= top_q
        if in_top_q:
            top_quartile_hits += 1

        # Also report best-of-top3-pred capture (still prediction-only shortlist).
        best_of_shortlist = max(top_k, key=lambda r: float(r[actual_key]))
        shortlist_capture = (
            0.0
            if ora_actual <= 0
            else float(best_of_shortlist[actual_key]) / ora_actual
        )

        out["halves"][half_name] = {
            "recommended_gw": recommended["gameweek"],
            "recommended_pred": recommended[pred_key],
            "recommended_actual": rec_actual,
            "oracle_gw": oracle["gameweek"],
            "oracle_actual": ora_actual,
            "mean_actual": round(mean_actual, 3),
            "capture_vs_oracle": round(capture, 3),
            "shortlist_top3_capture": round(shortlist_capture, 3),
            "beat_mean": rec_actual > mean_actual,
            "actual_rank": rec_rank,
            "n_weeks": len(rows),
            "in_top_quartile": in_top_q,
        }

    out["avg_capture_vs_oracle"] = round(sum(capture_rates) / len(capture_rates), 3) if capture_rates else 0.0
    out["top_quartile_hit_rate"] = (
        round(top_quartile_hits / half_count, 3) if half_count else 0.0
    )
    return out


def _evaluate_threshold_hits(
    weekly: list[dict[str, Any]],
    *,
    pred_key: str,
    actual_key: str,
    threshold: float,
    chip: str,
) -> dict[str, Any]:
    flagged = [w for w in weekly if float(w[pred_key]) >= threshold]
    if not flagged:
        return {
            "chip": chip,
            "threshold": threshold,
            "n_flagged": 0,
            "precision_vs_mean": None,
            "avg_actual_when_flagged": None,
            "mean_actual_all": round(
                float(sum(float(w[actual_key]) for w in weekly) / max(1, len(weekly))), 3
            ),
            "note": "No weeks crossed threshold (conservative)",
        }
    avg_flag = float(sum(float(w[actual_key]) for w in flagged) / len(flagged))
    mean_all = float(sum(float(w[actual_key]) for w in weekly) / len(weekly))
    return {
        "chip": chip,
        "threshold": threshold,
        "n_flagged": len(flagged),
        "flagged_gws": [int(w["gameweek"]) for w in flagged],
        "avg_actual_when_flagged": round(avg_flag, 3),
        "mean_actual_all": round(mean_all, 3),
        "lift_vs_mean": round(avg_flag - mean_all, 3),
        "precision_vs_mean": avg_flag > mean_all,
    }


def _competence_verdict(
    tc: dict[str, Any],
    bb: dict[str, Any],
    fh: dict[str, Any],
) -> dict[str, Any]:
    """Simple pass/fail style competence gates."""
    checks = {
        "tc_capture_ge_0_55": float(tc.get("avg_capture_vs_oracle") or 0) >= 0.55,
        "tc_top_quartile_ge_0_50": float(tc.get("top_quartile_hit_rate") or 0) >= 0.50,
        "bb_capture_ge_0_65": float(bb.get("avg_capture_vs_oracle") or 0) >= 0.65,
        "bb_top_quartile_ge_0_50": float(bb.get("top_quartile_hit_rate") or 0) >= 0.50,
        "fh_nonnegative_lift_or_unused": (
            fh.get("n_flagged", 0) == 0 or bool(fh.get("precision_vs_mean"))
        ),
        "tc_beats_mean_half_rate": _half_beat_mean_rate(tc) >= 0.5,
        "bb_beats_mean_half_rate": _half_beat_mean_rate(bb) >= 0.5,
    }
    passed = sum(1 for v in checks.values() if v)
    return {
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "competent": passed >= 5,
        "summary": ("Competent" if passed >= 5 else "Needs tuning"),
    }


def _half_beat_mean_rate(report: dict[str, Any]) -> float:
    halves = report.get("halves") or {}
    if not halves:
        return 0.0
    hits = sum(1 for h in halves.values() if h.get("beat_mean"))
    return hits / len(halves)


def recommend_chips_for_solution(
    *,
    gameweek: int,
    solution_squad: list[dict[str, Any]],
    captain_id: int | None,
    inventory: ChipInventory | None = None,
    historical_weekly: list[dict[str, Any]] | None = None,
    optimal_xi_ev: float | None = None,
    current_xi_ev: float | None = None,
    rolling_gap: float | None = None,
    historical_tc_scores: list[float] | None = None,
    historical_bb_scores: list[float] | None = None,
    summary: dict[str, Any] | None = None,
    season_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by CLI/pipeline.

    When `summary` / `season_state` are provided (live last_run_summary + season_state),
    FH/WC gaps and TC/BB histories are taken from those payloads.
    """
    xi = [p for p in solution_squad if p.get("start")]
    bench = [p for p in solution_squad if not p.get("start")]

    opp = ((summary or {}).get("chips") or {}).get("opportunity") or {}
    if optimal_xi_ev is None and opp.get("optimal_xi_ev") is not None:
        optimal_xi_ev = float(opp["optimal_xi_ev"])
    if current_xi_ev is None and opp.get("current_xi_ev") is not None:
        current_xi_ev = float(opp["current_xi_ev"])
    if rolling_gap is None and opp.get("rolling_gap") is not None:
        rolling_gap = float(opp["rolling_gap"])

    state = season_state or {}
    hist_tc = list(historical_tc_scores or [])
    hist_bb = list(historical_bb_scores or [])
    if not hist_tc:
        hist_tc = [float(w["pred_tc"]) for w in (historical_weekly or []) if "pred_tc" in w]
    if not hist_bb:
        hist_bb = [float(w["pred_bb"]) for w in (historical_weekly or []) if "pred_bb" in w]
    if not hist_tc and state.get("tc_score_history"):
        hist_tc = [float(x) for x in state["tc_score_history"]]
        if int(state.get("chip_history_gw") or -1) == int(gameweek) and hist_tc:
            hist_tc = hist_tc[:-1]
    if not hist_bb and state.get("bb_score_history"):
        hist_bb = [float(x) for x in state["bb_score_history"]]
        if int(state.get("chip_history_gw") or -1) == int(gameweek) and hist_bb:
            hist_bb = hist_bb[:-1]
    if rolling_gap is None and state.get("fh_gap_history"):
        horizon = int(CHIP_DEFAULTS["rolling_gap_horizon"])
        rolling_gap = float(sum(float(x) for x in state["fh_gap_history"][-horizon:]))

    inv = inventory or ChipInventory(
        windows=[
            ChipWindow("triple_captain", True, False, 1, 19),
            ChipWindow("bench_boost", True, False, 1, 19),
            ChipWindow("free_hit", True, False, 1, 19),
            ChipWindow("wildcard", True, False, 1, 19),
        ]
    )
    rec = recommend_chips(
        gameweek=gameweek,
        starting_xi=xi,
        bench=bench,
        captain_id=captain_id,
        inventory=inv,
        optimal_xi_ev=optimal_xi_ev,
        current_xi_ev=current_xi_ev,
        rolling_gap=float(rolling_gap or 0.0),
        historical_tc_scores=hist_tc,
        historical_bb_scores=hist_bb,
    )
    payload = rec.to_dict()
    if optimal_xi_ev is not None or current_xi_ev is not None:
        payload["opportunity"] = {
            "current_xi_ev": current_xi_ev,
            "optimal_xi_ev": optimal_xi_ev,
            "fh_gap": (
                max(0.0, float(optimal_xi_ev or 0.0) - float(current_xi_ev or 0.0))
                if optimal_xi_ev is not None and current_xi_ev is not None
                else opp.get("fh_gap")
            ),
            "rolling_gap": float(rolling_gap or 0.0),
        }
    return payload


def write_backtest_report(result: ChipBacktestResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Weekly can be large; keep full JSON for now.
    p.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
