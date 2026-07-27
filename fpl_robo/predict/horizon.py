"""Horizon-weighted EV: FDR + blank/DGW look-ahead over upcoming gameweeks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.config import HORIZON_EV
from fpl_robo.graph.schema import EdgeType, NodeType
from fpl_robo.graph.store import KnowledgeGraph


def fixture_scale(
    *,
    n_fixtures: int,
    fdr: float | None = None,
    blank_mult: float | None = None,
    dgw_mult: float | None = None,
    fdr_weight: float | None = None,
) -> float:
    """Scale factor for one GW contribution given fixture count and FDR."""
    blank = float(HORIZON_EV["blank_mult"] if blank_mult is None else blank_mult)
    dgw = float(HORIZON_EV["dgw_mult"] if dgw_mult is None else dgw_mult)
    weight = float(HORIZON_EV["fdr_weight"] if fdr_weight is None else fdr_weight)
    if int(n_fixtures) <= 0:
        return blank
    difficulty = 3.0 if fdr is None else float(fdr)
    fdr_adj = 1.0 + weight * (3.0 - difficulty)
    if int(n_fixtures) >= 2:
        return dgw * fdr_adj
    return fdr_adj


def team_fixtures_from_kg(
    kg: KnowledgeGraph,
    *,
    start_gw: int,
    horizon: int,
) -> dict[int, dict[int, dict[str, float]]]:
    """Map team_id -> gw -> {n_fixtures, best_fdr} from FACES edges."""
    end_gw = int(start_gw) + int(horizon) - 1
    out: dict[int, dict[int, dict[str, float]]] = {}
    for u, _, _, data in kg.g.edges(keys=True, data=True):
        if data.get("edge_type") != EdgeType.FACES.value:
            continue
        if u[0] != NodeType.TEAM.value:
            continue
        gw = data.get("gameweek")
        if gw is None:
            continue
        gw = int(gw)
        if gw < int(start_gw) or gw > end_gw:
            continue
        team_id = int(u[1])
        team_map = out.setdefault(team_id, {})
        slot = team_map.setdefault(gw, {"n_fixtures": 0.0, "best_fdr": None})
        slot["n_fixtures"] = float(slot["n_fixtures"]) + 1.0
        diff = data.get("difficulty")
        if diff is not None:
            d = float(diff)
            prev = slot["best_fdr"]
            if prev is None or d < float(prev):
                slot["best_fdr"] = d
    return out


def team_fixtures_from_merged(
    table: pd.DataFrame,
    *,
    team_col: str = "team_id",
    gw_col: str = "gameweek",
    fill_known_blanks: bool = True,
) -> dict[int, dict[int, dict[str, float]]]:
    """Build team×GW fixture counts from historical merged GW rows.

    Uses `n_fixtures` when present (from prepare_gw_table), else distinct fixture /
    opponent columns. FDR is left unknown (neutral) for backtests without difficulty.

    When `fill_known_blanks` is True, every team gets an explicit slot for every GW
    that appears in the table (missing = 0 fixtures), so look-ahead can distinguish
    true blanks from out-of-schedule unknowns.
    """
    df = table.copy()
    if team_col not in df.columns and "team" in df.columns:
        codes = sorted(str(t) for t in df["team"].unique())
        mapping = {t: i + 1 for i, t in enumerate(codes)}
        df[team_col] = df["team"].map(lambda t: mapping[str(t)])
    if gw_col not in df.columns and "round" in df.columns:
        gw_col = "round"

    out: dict[int, dict[int, dict[str, float]]] = {}
    group_cols = [team_col, gw_col]
    for (tid, gw), grp in df.groupby(group_cols, sort=False):
        if "n_fixtures" in grp.columns:
            n = int(grp["n_fixtures"].max())
        elif "fixture" in grp.columns:
            n = int(grp["fixture"].nunique())
        elif "opponent_team" in grp.columns:
            n = int(grp["opponent_team"].nunique())
        else:
            n = 1 if len(grp) else 0
        out.setdefault(int(tid), {})[int(gw)] = {
            "n_fixtures": float(n),
            "best_fdr": None,
        }

    if fill_known_blanks and out:
        all_teams = sorted(out.keys())
        all_gws = sorted({gw for gws in out.values() for gw in gws})
        for tid in all_teams:
            team_map = out.setdefault(tid, {})
            for gw in all_gws:
                if gw not in team_map:
                    team_map[gw] = {"n_fixtures": 0.0, "best_fdr": None}
    return out


def apply_horizon_ev(
    predictions: pd.DataFrame,
    *,
    start_gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    weights: list[float] | None = None,
    blank_mult: float | None = None,
    dgw_mult: float | None = None,
    fdr_weight: float | None = None,
    unknown_mult: float | None = None,
    asymmetric_lookahead: bool | None = None,
) -> pd.DataFrame:
    """Replace `ev` with horizon-weighted EV; keep `ev_gw0` and `horizon_factor`.

    Explicit n_fixtures=0 → blank haircut. Missing slot (beyond schedule) → neutral
    `unknown_mult` (default 1.0), not blank — otherwise late-season EV is crushed.

    When `asymmetric_lookahead` is True (default from config), future GWs (offset>0)
    only contribute upside (DGW / easy FDR). Future blanks do not haircut this week's
    haul players — that was destroying season points in backtests.
    """
    df = predictions.copy()
    if df.empty or "ev" not in df.columns:
        return df
    w = list(HORIZON_EV["weights"] if weights is None else weights)
    if not w:
        return df
    unk = float(
        HORIZON_EV.get("unknown_mult", 1.0) if unknown_mult is None else unknown_mult
    )
    asym = bool(
        HORIZON_EV.get("asymmetric_lookahead", True)
        if asymmetric_lookahead is None
        else asymmetric_lookahead
    )

    base = df["ev"].astype(float)
    df["ev_gw0"] = base
    team_ids = df["team_id"].astype(int) if "team_id" in df.columns else None
    if team_ids is None:
        return df

    horizon_vals: list[float] = []
    factors: list[float] = []
    for idx, base_ev in enumerate(base.tolist()):
        tid = int(team_ids.iloc[idx])
        weighted = 0.0
        scale_sum = 0.0
        for offset, weight in enumerate(w):
            gw = int(start_gw) + offset
            slot = (team_fixtures.get(tid) or {}).get(gw)
            if slot is None:
                scale = unk
            else:
                n_fix = int(float(slot.get("n_fixtures") or 0))
                fdr = slot.get("best_fdr")
                fdr = float(fdr) if fdr is not None else None
                scale = fixture_scale(
                    n_fixtures=n_fix,
                    fdr=fdr,
                    blank_mult=blank_mult,
                    dgw_mult=dgw_mult,
                    fdr_weight=fdr_weight,
                )
            if asym and offset > 0:
                scale = max(1.0, float(scale))
            weighted += float(weight) * float(base_ev) * scale
            scale_sum += float(weight) * scale
        horizon_vals.append(max(0.0, weighted))
        factors.append(round(scale_sum, 4) if base_ev else 0.0)

    df["ev"] = horizon_vals
    df["horizon_factor"] = factors
    return df.sort_values("ev", ascending=False).reset_index(drop=True)


def count_blank_owned(
    *,
    current_ids: list[int],
    predictions: pd.DataFrame,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    gameweek: int,
) -> int:
    """How many owned players have 0 fixtures in `gameweek`."""
    if predictions.empty or not current_ids:
        return 0
    by_id = {int(r["id"]): r for _, r in predictions.iterrows()}
    n_blank = 0
    for pid in current_ids:
        row = by_id.get(int(pid))
        if row is None:
            continue
        tid = int(row.get("team_id") or 0)
        slot = (team_fixtures.get(tid) or {}).get(int(gameweek))
        n_fix = int(float((slot or {}).get("n_fixtures") or 0))
        if n_fix <= 0:
            n_blank += 1
    return n_blank


def mode_uses_horizon(mode: str | None) -> bool:
    """True for explicit *_h modes (horizon baked into the mode name)."""
    return str(mode or "").endswith("_h")


def strip_horizon_suffix(mode: str) -> str:
    text = str(mode)
    if text.endswith("_h"):
        return text[:-2]
    return text
