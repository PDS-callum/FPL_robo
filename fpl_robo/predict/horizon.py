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


def horizon_bundle_factor(
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    team_id: int,
    start_gw: int,
    *,
    horizon: int | None = None,
    weights: list[float] | None = None,
) -> float:
    """Weighted future-GW scale (offsets ≥ 1). Asym: blanks don't haircut; DGWs pull.

    Returns ~1.0 for all-SGW, >1 when upcoming DGWs. Used as soft SA geometry only —
    does not rewrite prediction EV (`ep_baseline` stays intact).
    """
    from fpl_robo.config import VECTOR_TEAM

    h = int(horizon if horizon is not None else VECTOR_TEAM.get("horizon") or 3)
    w = list(weights if weights is not None else (HORIZON_EV.get("weights") or [0.70, 0.20, 0.10]))
    if h <= 1 or not w:
        return 1.0
    asym = bool(HORIZON_EV.get("asymmetric_lookahead", True))
    unk = float(HORIZON_EV.get("unknown_mult") or 1.0)
    team_map = (team_fixtures or {}).get(int(team_id)) or {}
    # Skip offset 0 — this GW already in live EV / KG scale.
    acc = 0.0
    wsum = 0.0
    for offset in range(1, h):
        weight = float(w[offset]) if offset < len(w) else float(w[-1]) * 0.5
        gw = int(start_gw) + offset
        slot = team_map.get(gw)
        if slot is None:
            scale = unk
        else:
            n_fix = int(float(slot.get("n_fixtures") or 0))
            fdr = slot.get("best_fdr")
            fdr = float(fdr) if fdr is not None else None
            # FDR already soft in portfolio run_ease; bundle focuses on DGW/blank count.
            scale = fixture_scale(n_fixtures=n_fix, fdr=fdr, fdr_weight=0.0)
        if asym:
            scale = max(1.0, float(scale))
        acc += weight * float(scale)
        wsum += weight
    if wsum <= 1e-9:
        return 1.0
    return float(acc / wsum)


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
    # Explicit blanks for known teams across the horizon (missing ≠ unknown).
    gws = list(range(int(start_gw), end_gw + 1))
    for tid, team_map in out.items():
        for gw in gws:
            if gw not in team_map:
                team_map[gw] = {"n_fixtures": 0.0, "best_fdr": None}
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


def count_upcoming_blank_owned(
    *,
    current_ids: list[int],
    predictions: pd.DataFrame,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    start_gw: int,
    lookahead: int | None = None,
) -> int:
    """How many owned players have a club blank in the next `lookahead` GWs."""
    from fpl_robo.config import BLANK_AHEAD

    if predictions.empty or not current_ids:
        return 0
    look = int(BLANK_AHEAD["lookahead"] if lookahead is None else lookahead)
    if look <= 0:
        return 0
    by_id = {int(r["id"]): r for _, r in predictions.iterrows()}
    n = 0
    for pid in current_ids:
        row = by_id.get(int(pid))
        if row is None:
            continue
        tid = int(row.get("team_id") or 0)
        team_map = team_fixtures.get(tid) or {}
        for offset in range(1, look + 1):
            slot = team_map.get(int(start_gw) + offset)
            if slot is None:
                continue
            if int(float(slot.get("n_fixtures") or 0)) <= 0:
                n += 1
                break
    return n


def apply_blank_ahead_ev(
    predictions: pd.DataFrame,
    *,
    start_gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    lookahead: int | None = None,
    offset_mults: list[float] | None = None,
    current_ids: list[int] | None = None,
    only_if_owned_blank: bool | None = None,
    owned_blank_threshold: int | None = None,
) -> pd.DataFrame:
    """Haircut EV for players whose club blanks in an upcoming GW.

    Only future blanks (offset ≥ 1). This-week blanks are left to normal EV /
    Free Hit logic. Missing fixture slots are ignored (not treated as blanks).
    When several upcoming blanks match, the strongest (lowest) multiplier wins.

    When `only_if_owned_blank` is True (config default), the haircut is skipped
    unless the current squad already owns enough upcoming-blank players.
    """
    from fpl_robo.config import BLANK_AHEAD

    df = predictions.copy()
    if df.empty or "ev" not in df.columns or "team_id" not in df.columns:
        return df

    gated = (
        bool(BLANK_AHEAD.get("only_if_owned_blank", True))
        if only_if_owned_blank is None
        else bool(only_if_owned_blank)
    )
    threshold = int(
        BLANK_AHEAD.get("owned_blank_threshold", 1)
        if owned_blank_threshold is None
        else owned_blank_threshold
    )
    owned_upcoming = 0
    triggered = False
    if gated:
        if not current_ids:
            df["blank_ahead_factor"] = 1.0
            df["blank_ahead_triggered"] = False
            return df
        owned_upcoming = count_upcoming_blank_owned(
            current_ids=current_ids,
            predictions=df,
            team_fixtures=team_fixtures,
            start_gw=start_gw,
            lookahead=lookahead,
        )
        if owned_upcoming < threshold:
            df["blank_ahead_factor"] = 1.0
            df["blank_ahead_triggered"] = False
            df.attrs["blank_ahead_owned_upcoming"] = owned_upcoming
            return df
        triggered = True

    look = int(BLANK_AHEAD["lookahead"] if lookahead is None else lookahead)
    mults = list(
        BLANK_AHEAD["offset_mults"] if offset_mults is None else offset_mults
    )
    if look <= 0 or not mults:
        df["blank_ahead_factor"] = 1.0
        df["blank_ahead_triggered"] = False
        return df

    factors: list[float] = []
    for tid in df["team_id"].astype(int).tolist():
        factor = 1.0
        team_map = team_fixtures.get(int(tid)) or {}
        for offset in range(1, look + 1):
            slot = team_map.get(int(start_gw) + offset)
            if slot is None:
                continue
            if int(float(slot.get("n_fixtures") or 0)) > 0:
                continue
            m = float(mults[min(offset - 1, len(mults) - 1)])
            factor = min(factor, m)
        factors.append(factor)

    df["blank_ahead_factor"] = factors
    df["blank_ahead_triggered"] = triggered if gated else True
    df.attrs["blank_ahead_owned_upcoming"] = owned_upcoming
    df["ev"] = (df["ev"].astype(float) * df["blank_ahead_factor"]).clip(lower=0.0)
    return df


def mode_uses_horizon(mode: str | None) -> bool:
    """True for explicit *_h modes (horizon baked into the mode name)."""
    return str(mode or "").endswith("_h")


def strip_horizon_suffix(mode: str) -> str:
    text = str(mode)
    if text.endswith("_h"):
        return text[:-2]
    return text
