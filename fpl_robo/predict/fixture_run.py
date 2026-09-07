"""Multi-GW fixture-run EV: plan transfers around upcoming FDR stretches.

Unlike the legacy asymmetric horizon rewrite (future blanks ignored; FDR weak),
this module:

1. Attaches real FPL FDR into backtest fixture maps (from vaastav fixtures.csv)
2. Reweights player EV by a *symmetric* fixture-run scale so hard runs (Isak into
   a tough stretch) lose EV and easy runs gain it — even when recent form is hot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fpl_robo.config import FIXTURE_RUN, HORIZON_EV
from fpl_robo.predict.horizon import fixture_scale


def load_season_fixtures(season: str, *, cache_dir: str | Path = "data/cache") -> pd.DataFrame:
    """Load cached vaastav fixtures.csv (includes team_*_difficulty)."""
    path = Path(cache_dir) / f"fixtures_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download from vaastav Fantasy-Premier-League data/{season}/fixtures.csv"
        )
    return pd.read_csv(path)


def team_fixtures_with_fdr(
    season: str,
    *,
    cache_dir: str | Path = "data/cache",
    fill_blanks: bool = True,
) -> dict[int, dict[int, dict[str, float]]]:
    """Build team_id → gw → {n_fixtures, best_fdr} using official FPL difficulties.

    `team_h_difficulty` is the FDR for the *home* side; `team_a_difficulty` for away.
    """
    fx = load_season_fixtures(season, cache_dir=cache_dir)
    out: dict[int, dict[int, dict[str, float]]] = {}
    for _, row in fx.iterrows():
        gw = row.get("event")
        if gw is None or (isinstance(gw, float) and pd.isna(gw)):
            continue
        gw = int(gw)
        for side, diff_key in (
            ("team_h", "team_h_difficulty"),
            ("team_a", "team_a_difficulty"),
        ):
            tid = int(row[side])
            diff = row.get(diff_key)
            fdr = float(diff) if diff is not None and not pd.isna(diff) else None
            slot = out.setdefault(tid, {}).setdefault(
                gw, {"n_fixtures": 0.0, "best_fdr": None}
            )
            slot["n_fixtures"] = float(slot["n_fixtures"]) + 1.0
            if fdr is not None:
                prev = slot["best_fdr"]
                if prev is None or fdr < float(prev):
                    slot["best_fdr"] = fdr

    if fill_blanks and out:
        all_gws = sorted({gw for gws in out.values() for gw in gws})
        for tid in list(out.keys()):
            for gw in all_gws:
                out[tid].setdefault(gw, {"n_fixtures": 0.0, "best_fdr": None})
    return out


def merge_fixture_maps(
    primary: dict[int, dict[int, dict[str, float]]],
    fdr_source: dict[int, dict[int, dict[str, float]]],
) -> dict[int, dict[int, dict[str, float]]]:
    """Copy FDR from `fdr_source` onto `primary` slots (matched by team_id + gw).

    When team_id schemes differ (name-factorized vs FPL ids), callers should build
    fixtures directly via `team_fixtures_with_fdr` and remap predictions' team_id
    to FPL ids instead.
    """
    out: dict[int, dict[int, dict[str, float]]] = {
        tid: {gw: dict(slot) for gw, slot in gws.items()}
        for tid, gws in primary.items()
    }
    for tid, gws in fdr_source.items():
        for gw, slot in gws.items():
            dst = out.setdefault(int(tid), {}).setdefault(
                int(gw), {"n_fixtures": 0.0, "best_fdr": None}
            )
            if slot.get("best_fdr") is not None:
                dst["best_fdr"] = float(slot["best_fdr"])
            if float(slot.get("n_fixtures") or 0) > float(dst.get("n_fixtures") or 0):
                dst["n_fixtures"] = float(slot["n_fixtures"])
    return out


def fpl_team_id_map_from_merged(
    merged: pd.DataFrame,
    season: str,
    *,
    cache_dir: str | Path = "data/cache",
) -> dict[str, int]:
    """Map vaastav team name → FPL team id using teams.csv short/full names."""
    path = Path(cache_dir) / f"teams_{season}.csv"
    teams = pd.read_csv(path)
    by_name = {str(r["name"]).strip().lower(): int(r["id"]) for _, r in teams.iterrows()}
    by_short = {
        str(r["short_name"]).strip().lower(): int(r["id"]) for _, r in teams.iterrows()
    }
    # Common aliases in merged_gw.
    aliases = {
        "man utd": "man utd",
        "manchester united": "man utd",
        "man city": "man city",
        "manchester city": "man city",
        "spurs": "spurs",
        "tottenham": "spurs",
        "nottingham forest": "nfo",
        "nott'm forest": "nfo",
        "wolves": "wolves",
    }
    short_by_lower = {str(r["short_name"]).strip().lower(): int(r["id"]) for _, r in teams.iterrows()}
    name_to_short = {
        str(r["name"]).strip().lower(): str(r["short_name"]).strip().lower()
        for _, r in teams.iterrows()
    }

    out: dict[str, int] = {}
    for raw in merged["team"].astype(str).unique():
        key = raw.strip().lower()
        if key in by_name:
            out[raw] = by_name[key]
            continue
        if key in by_short:
            out[raw] = by_short[key]
            continue
        # fuzzy: substring
        hit = None
        for name, tid in by_name.items():
            if key in name or name in key:
                hit = tid
                break
        if hit is not None:
            out[raw] = hit
            continue
        # last resort: leave unmapped
    return out


def remap_predictions_team_ids(
    predictions: pd.DataFrame,
    *,
    name_to_fpl_id: dict[str, int],
) -> pd.DataFrame:
    """Ensure predictions.team_id uses FPL ids so fixture FDR maps align."""
    df = predictions.copy()
    if "team" not in df.columns:
        return df
    mapped = df["team"].astype(str).map(name_to_fpl_id)
    if mapped.notna().any():
        df["team_id"] = mapped.fillna(df.get("team_id")).astype(int)
    return df


def apply_fixture_run_ev(
    predictions: pd.DataFrame,
    *,
    start_gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Replace `ev` with fixture-run weighted EV (symmetric FDR look-ahead).

    Stores `ev_gw0` (pre-run) and `fixture_run_factor` (effective scale sum).
    Optional `blend_gw0` mixes raw GW0 EV back in so form is not abandoned.
    Optional `factor_min` / `factor_max` clamp the effective run multiplier.
    """
    cfg = {**FIXTURE_RUN, **(config or {})}
    df = predictions.copy()
    if df.empty or "ev" not in df.columns or "team_id" not in df.columns:
        return df

    weights = list(cfg.get("weights") or HORIZON_EV.get("weights") or [0.55, 0.25, 0.12, 0.08])
    fdr_w = float(cfg.get("fdr_weight") or 0.18)
    blank_mult = float(cfg.get("blank_mult") or HORIZON_EV.get("blank_mult") or 0.20)
    dgw_mult = float(cfg.get("dgw_mult") or HORIZON_EV.get("dgw_mult") or 1.45)
    unk = float(cfg.get("unknown_mult") or 1.0)
    # Symmetric: future hard FDR haircuts (unlike asymmetric horizon).
    symmetric = bool(cfg.get("symmetric_fdr", True))
    power = float(cfg.get("fixture_power") or 1.0)
    blend_gw0 = float(cfg.get("blend_gw0") or 0.0)
    factor_min = cfg.get("factor_min")
    factor_max = cfg.get("factor_max")

    base = df["ev"].astype(float)
    df["ev_gw0"] = base
    team_ids = df["team_id"].astype(int)

    horizon_vals: list[float] = []
    factors: list[float] = []
    for idx, base_ev in enumerate(base.tolist()):
        tid = int(team_ids.iloc[idx])
        weighted = 0.0
        scale_sum = 0.0
        for offset, weight in enumerate(weights):
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
                    fdr_weight=fdr_w,
                )
            if not symmetric and offset > 0:
                scale = max(1.0, float(scale))
            scale = float(scale) ** power
            weighted += float(weight) * float(base_ev) * scale
            scale_sum += float(weight) * scale
        run_ev = max(0.0, weighted)
        fac = round(scale_sum, 4) if base_ev else 0.0
        if factor_min is not None or factor_max is not None:
            lo = float(factor_min) if factor_min is not None else 0.0
            hi = float(factor_max) if factor_max is not None else 10.0
            fac = float(min(hi, max(lo, fac)))
            run_ev = max(0.0, float(base_ev) * fac)
        if blend_gw0 > 0.0:
            run_ev = (1.0 - blend_gw0) * run_ev + blend_gw0 * float(base_ev)
        horizon_vals.append(max(0.0, run_ev))
        factors.append(fac)

    df["ev"] = horizon_vals
    df["fixture_run_factor"] = factors
    df["horizon_factor"] = factors
    return df.sort_values("ev", ascending=False).reset_index(drop=True)
