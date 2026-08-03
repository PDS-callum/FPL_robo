"""Player feature vectors and ridge map → expected GW points (no leakage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_robo.config import VECTOR_TEAM
from fpl_robo.predict.ev_modes import _play_prob_series
from fpl_robo.predict.horizon import horizon_bundle_factor

# Fixed feature layout (z-scored per decision GW pool, then ridge / hand weights).
FEATURE_NAMES: tuple[str, ...] = (
    "rolling_points",
    "rolling_xgi",
    "rolling_ict",
    "play_prob",
    "rolling_minutes",  # stronger minutes edge (raw rolling mins / 90)
    "n_fixtures",
    "fixture_ease",  # 1 = easy (low FDR), 0 = hard / unknown neutral
    "is_home",  # +1 home, -1 away, 0 unknown
    "run_ease",  # mean fixture_ease over next horizon GWs
    "run_n_fixtures",  # mean n_fixtures over horizon
    "price",
    "set_piece_score",
    "is_focal_attacker",
    "club_pos_threat",  # 1 / rank (or threat share) within club×pos
    "pos_gk",
    "pos_def",
    "pos_mid",
    "pos_fwd",
)

# Hand-weight fallback when ridge sample is thin.
# In residual mode these should be near-zero (prefer pure xP anchor).
HAND_WEIGHTS = np.zeros(len(FEATURE_NAMES), dtype=float)
HAND_BIAS = 0.0


@dataclass
class RidgeModel:
    """Position-agnostic ridge on z-scored vectors (bias in coef[0])."""

    coef: np.ndarray  # shape (1 + n_features,)
    feature_mean: np.ndarray
    feature_std: np.ndarray
    fitted: bool
    n_train: int
    source: str = "hand"

    def predict_raw(
        self, X: np.ndarray, positions: np.ndarray | list[str] | None = None
    ) -> np.ndarray:
        """Predict from already-built (unscaled) feature matrix."""
        _ = positions  # accepted for PosRidgeEnsemble API parity
        Z = (X - self.feature_mean) / self.feature_std
        ones = np.ones((Z.shape[0], 1), dtype=float)
        return (np.hstack([ones, Z]) @ self.coef).astype(float)


@dataclass
class PosRidgeEnsemble:
    """Separate residual ridges for GK / DEF / ATT (MID+FWD)."""

    by_group: dict[str, RidgeModel]
    fallback: RidgeModel
    fitted: bool
    n_train: int
    source: str = "pos_ridge"

    def predict_raw(
        self, X: np.ndarray, positions: np.ndarray | list[str] | None = None
    ) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if positions is None or len(positions) != len(X):
            return self.fallback.predict_raw(X)
        pos = np.asarray([str(p).upper() for p in positions])
        out = np.zeros(len(X), dtype=float)
        group_of = _position_group_map()
        groups = np.array([group_of.get(p, "ATT") for p in pos])
        for g in sorted(set(groups.tolist())):
            mask = groups == g
            if not mask.any():
                continue
            model = self.by_group.get(g) or self.fallback
            out[mask] = model.predict_raw(X[mask])
        return out


def _position_group_map() -> dict[str, str]:
    raw = VECTOR_TEAM.get("position_residual_groups") or {
        "GK": ["GK"],
        "DEF": ["DEF"],
        "ATT": ["MID", "FWD"],
    }
    out: dict[str, str] = {}
    for group, positions in raw.items():
        for p in positions:
            out[str(p).upper()] = str(group)
    return out


def blend_ridge_models(
    prior: RidgeModel, current: RidgeModel, *, weight_current: float
) -> RidgeModel:
    """Linear blend of two same-dimension ridge models."""
    w = float(np.clip(weight_current, 0.0, 1.0))
    if prior.coef.shape != current.coef.shape:
        return current if w >= 0.5 else prior
    return RidgeModel(
        coef=(1.0 - w) * prior.coef + w * current.coef,
        feature_mean=(1.0 - w) * prior.feature_mean + w * current.feature_mean,
        feature_std=(1.0 - w) * prior.feature_std + w * current.feature_std,
        fitted=bool(prior.fitted or current.fitted),
        n_train=int(max(prior.n_train, current.n_train)),
        source=f"blend:{getattr(prior, 'source', 'prior')}+{getattr(current, 'source', 'cur')}:{w:.2f}",
    )


def blend_residual_models(
    prior: RidgeModel | PosRidgeEnsemble,
    current: RidgeModel | PosRidgeEnsemble,
    *,
    weight_current: float,
) -> RidgeModel | PosRidgeEnsemble:
    """Blend plain or position-ensemble residual models."""
    w = float(np.clip(weight_current, 0.0, 1.0))
    if isinstance(prior, PosRidgeEnsemble) and isinstance(current, PosRidgeEnsemble):
        groups = sorted(set(prior.by_group) | set(current.by_group))
        blended = {
            g: blend_ridge_models(
                prior.by_group.get(g) or prior.fallback,
                current.by_group.get(g) or current.fallback,
                weight_current=w,
            )
            for g in groups
        }
        fb = blend_ridge_models(prior.fallback, current.fallback, weight_current=w)
        return PosRidgeEnsemble(
            by_group=blended,
            fallback=fb,
            fitted=bool(prior.fitted or current.fitted),
            n_train=int(max(prior.n_train, current.n_train)),
            source=f"pos_blend:{w:.2f}",
        )
    if isinstance(prior, RidgeModel) and isinstance(current, RidgeModel):
        return blend_ridge_models(prior, current, weight_current=w)
    return current if w >= 0.5 else prior



def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _fixture_slot(
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None,
    team_id: int,
    gw: int,
) -> dict[str, float]:
    if not team_fixtures:
        return {"n_fixtures": 1.0, "best_fdr": 3.0, "is_home": 0.0}
    slot = (team_fixtures.get(int(team_id)) or {}).get(int(gw)) or {}
    return {
        "n_fixtures": float(slot.get("n_fixtures") if slot.get("n_fixtures") is not None else 1.0),
        "best_fdr": float(slot["best_fdr"]) if slot.get("best_fdr") is not None else 3.0,
        "is_home": float(slot.get("is_home") or 0.0),
    }


def _ease_from_fdr(fdr: float) -> float:
    # FDR 1..5 → ease 1..0 centred on 3.
    return float(np.clip(1.0 + 0.25 * (3.0 - float(fdr)), 0.0, 1.5))


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    horizon: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build (n, d) feature matrix and an annotated frame with vector metadata."""
    h = int(horizon if horizon is not None else VECTOR_TEAM["horizon"])
    rows = df.copy()
    if "team_id" not in rows.columns and "team" in rows.columns:
        rows["team_id"] = pd.factorize(rows["team"])[0] + 1

    rolling_pts = _col(rows, "rolling_points")
    xg = _col(rows, "rolling_expected_goals", 0.0)
    xa = _col(rows, "rolling_expected_assists", 0.0)
    rolling_xgi = xg + xa
    if "rolling_xgi" in rows.columns:
        rolling_xgi = _col(rows, "rolling_xgi")
    rolling_ict = _col(rows, "rolling_ict_index", 0.0)
    if rolling_ict.sum() == 0 and "rolling_ict" in rows.columns:
        rolling_ict = _col(rows, "rolling_ict")

    if "rolling_minutes" not in rows.columns:
        rows["rolling_minutes"] = 0.0
    if "prior_minutes" not in rows.columns:
        rows["prior_minutes"] = 0.0
    play_prob = _play_prob_series(rows)

    gw0 = int(rows["gameweek"].dropna().iloc[0]) if "gameweek" in rows.columns and len(rows) else 1
    tids = rows["team_id"].astype(int).to_numpy() if "team_id" in rows.columns else np.zeros(len(rows), dtype=int)
    gws = rows["gameweek"].astype(int).to_numpy() if "gameweek" in rows.columns else np.full(len(rows), gw0)

    # Cache team×gw fixture features (shared across squad mates).
    fixture_cache: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}

    def _cached(tid: int, gw: int) -> tuple[float, float, float, float, float]:
        key = (tid, gw)
        if key in fixture_cache:
            return fixture_cache[key]
        slot = _fixture_slot(team_fixtures, tid, gw)
        nf = max(0.0, float(slot["n_fixtures"]))
        ea = _ease_from_fdr(float(slot["best_fdr"]))
        eases, ns = [], []
        for k in range(h):
            s = _fixture_slot(team_fixtures, tid, gw + k)
            eases.append(_ease_from_fdr(float(s["best_fdr"])) if float(s["n_fixtures"]) > 0 else 0.15)
            ns.append(float(s["n_fixtures"]))
        re = float(np.mean(eases)) if eases else 0.5
        rn = float(np.mean(ns)) if ns else 1.0
        ih_slot = float(slot.get("is_home") or 0.0)
        fixture_cache[key] = (nf, ea, ih_slot, re, rn)
        return fixture_cache[key]

    n_fix = np.ones(len(rows), dtype=float)
    ease = np.full(len(rows), 0.5, dtype=float)
    is_home = np.zeros(len(rows), dtype=float)
    run_ease = np.full(len(rows), 0.5, dtype=float)
    run_n = np.ones(len(rows), dtype=float)

    for i in range(len(rows)):
        nf, ea, ih_slot, re, rn = _cached(int(tids[i]), int(gws[i]))
        n_fix[i] = nf
        ease[i] = ea
        run_ease[i] = re
        run_n[i] = rn
        r = rows.iloc[i]
        if "was_home" in rows.columns and pd.notna(r.get("was_home")):
            is_home[i] = 1.0 if bool(r.get("was_home")) else -1.0
        elif "is_home" in rows.columns and pd.notna(r.get("is_home")):
            ih = float(r.get("is_home"))
            is_home[i] = ih if ih != 0 else 0.0
        else:
            is_home[i] = ih_slot

    pos = rows["position"].astype(str).str.upper() if "position" in rows.columns else pd.Series("", index=rows.index)
    price = _col(rows, "price", 4.5).clip(lower=4.0)
    roll_mins = _col(rows, "rolling_minutes", 0.0).clip(lower=0.0) / 90.0
    set_piece = _col(rows, "set_piece_score", 0.0).clip(lower=0.0)
    focal = _col(rows, "is_focal_attacker", 0.0).clip(lower=0.0, upper=1.0)
    # Rival / focal depth: prefer threat share; else invert club_pos_ep_rank.
    if "club_pos_threat_share" in rows.columns:
        club_threat = _col(rows, "club_pos_threat_share", 0.0).clip(lower=0.0, upper=1.0)
    elif "club_pos_ep_rank" in rows.columns:
        rank = _col(rows, "club_pos_ep_rank", 9.0).clip(lower=1.0)
        club_threat = (1.0 / rank).clip(lower=0.0, upper=1.0)
    else:
        club_threat = focal.copy()

    X = np.column_stack(
        [
            rolling_pts.to_numpy(dtype=float),
            rolling_xgi.to_numpy(dtype=float),
            rolling_ict.to_numpy(dtype=float),
            play_prob.to_numpy(dtype=float),
            roll_mins.to_numpy(dtype=float),
            n_fix,
            ease,
            is_home,
            run_ease,
            run_n,
            price.to_numpy(dtype=float),
            set_piece.to_numpy(dtype=float),
            focal.to_numpy(dtype=float),
            club_threat.to_numpy(dtype=float),
            (pos == "GK").astype(float).to_numpy(),
            (pos == "DEF").astype(float).to_numpy(),
            (pos == "MID").astype(float).to_numpy(),
            (pos == "FWD").astype(float).to_numpy(),
        ]
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    meta = rows.copy()
    meta["play_prob"] = play_prob
    meta["fixture_ease"] = ease
    meta["run_ease"] = run_ease
    meta["run_n_fixtures"] = run_n
    meta["vec_n_fixtures"] = n_fix
    bundles = [
        horizon_bundle_factor(team_fixtures, int(tids[i]), int(gws[i]), horizon=h)
        for i in range(len(rows))
    ]
    meta["horizon_bundle"] = bundles
    return X, meta


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float | None = None,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    min_rows: int | None = None,
) -> RidgeModel:
    """Closed-form ridge: (Z'Z + αI) β = Z'y with intercept column unpenalised."""
    a = float(VECTOR_TEAM["ridge_alpha"] if alpha is None else alpha)
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if feature_mean is None:
        feature_mean = X.mean(axis=0)
    if feature_std is None:
        feature_std = X.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    Z = (X - feature_mean) / feature_std
    n, d = Z.shape
    need = int(VECTOR_TEAM["min_ridge_rows"] if min_rows is None else min_rows)
    if n < need:
        coef = np.concatenate([[HAND_BIAS], HAND_WEIGHTS.copy()])
        return RidgeModel(
            coef=coef,
            feature_mean=feature_mean,
            feature_std=feature_std,
            fitted=False,
            n_train=n,
            source="hand_thin",
        )

    A = np.hstack([np.ones((n, 1)), Z])
    # Penalty on non-intercept coefficients only.
    pen = np.eye(d + 1) * a
    pen[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + pen, A.T @ y)
    return RidgeModel(
        coef=coef,
        feature_mean=feature_mean,
        feature_std=feature_std,
        fitted=True,
        n_train=n,
        source="ridge",
    )


def hand_model_from_pool(X: np.ndarray) -> RidgeModel:
    mean = X.mean(axis=0) if len(X) else np.zeros(len(FEATURE_NAMES))
    std = X.std(axis=0) if len(X) else np.ones(len(FEATURE_NAMES))
    std = np.where(std < 1e-6, 1.0, std)
    return RidgeModel(
        coef=np.concatenate([[HAND_BIAS], HAND_WEIGHTS.copy()]),
        feature_mean=mean,
        feature_std=std,
        fitted=False,
        n_train=len(X),
        source="hand",
    )


def fit_ridge_from_history(
    table: pd.DataFrame,
    *,
    before_gw: int,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    lookback: int | None = None,
) -> RidgeModel | PosRidgeEnsemble:
    """Fit ridge on prior GWs.

    Residual mode (default): target = total_points − xP (vector corrects the published
    EP). Absolute mode: target = total_points. Features are lagged only.
    """
    lb = int(lookback if lookback is not None else VECTOR_TEAM["fit_lookback_gws"])
    lo = max(1, int(before_gw) - lb)
    hist = table[(table["gameweek"] >= lo) & (table["gameweek"] < int(before_gw))].copy()
    if hist.empty or "total_points" not in hist.columns:
        return hand_model_from_pool(np.zeros((0, len(FEATURE_NAMES))))
    X, meta = build_feature_matrix(hist, team_fixtures=team_fixtures)
    y_pts = hist["total_points"].astype(float).to_numpy()
    if VECTOR_TEAM.get("residual_mode", True) and "xP" in hist.columns:
        xp = hist["xP"].astype(float).clip(lower=0.0).to_numpy()
        # Only use residual labels when xP is informative for most rows.
        if float(np.mean(xp > 0.05)) >= 0.25:
            y = y_pts - xp
        else:
            y = y_pts
    else:
        y = y_pts

    fallback = fit_ridge(X, y)
    if not bool(VECTOR_TEAM.get("position_residual", True)):
        return fallback

    pos = (
        meta["position"].astype(str).str.upper().to_numpy()
        if "position" in meta.columns
        else np.array([""] * len(meta))
    )
    group_of = _position_group_map()
    min_pos = int(VECTOR_TEAM.get("min_ridge_rows_pos") or 40)
    by_group: dict[str, RidgeModel] = {}
    for group in sorted(set(group_of.values())):
        mask = np.array([group_of.get(str(p).upper(), "ATT") == group for p in pos])
        n = int(mask.sum())
        if n < min_pos:
            continue
        by_group[group] = fit_ridge(X[mask], y[mask], min_rows=min_pos)
        by_group[group].source = f"pos:{group}"

    if not by_group:
        return fallback
    fitted_any = any(m.fitted for m in by_group.values()) or fallback.fitted
    return PosRidgeEnsemble(
        by_group=by_group,
        fallback=fallback,
        fitted=bool(fitted_any),
        n_train=int(fallback.n_train),
        source="pos_ridge",
    )


def attach_vector_ev(
    gw_df: pd.DataFrame,
    model: RidgeModel | PosRidgeEnsemble,
    *,
    team_fixtures: dict[int, dict[int, dict[str, float]]] | None = None,
    xp_anchor: pd.Series | np.ndarray | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Annotate a GW frame with feature vectors and `base_points` / `ev`.

    When residual_mode is on and an xP (or blank-ahead-adjusted) anchor is available,
    base_points = anchor + residual_scale * clip(ridge(vector)).
    Unfitted models use the anchor alone (competitive cold start).
    """
    X, meta = build_feature_matrix(gw_df, team_fixtures=team_fixtures, horizon=horizon)
    pos = (
        meta["position"].astype(str).str.upper().to_numpy()
        if "position" in meta.columns
        else None
    )
    residual = model.predict_raw(X, positions=pos)

    if xp_anchor is not None:
        anchor = pd.Series(xp_anchor, index=meta.index).astype(float).fillna(0.0).clip(lower=0.0)
    elif "xP" in meta.columns:
        anchor = meta["xP"].astype(float).fillna(0.0).clip(lower=0.0)
    elif "ev" in meta.columns:
        anchor = meta["ev"].astype(float).fillna(0.0).clip(lower=0.0)
    else:
        anchor = None

    clip = float(VECTOR_TEAM.get("residual_clip") or 2.5)
    scale = float(VECTOR_TEAM.get("residual_scale") or 0.35)
    residual_c = np.clip(residual, -clip, clip)

    if anchor is not None and VECTOR_TEAM.get("residual_mode", True):
        if model.fitted:
            base = anchor.to_numpy(dtype=float) + scale * residual_c
        else:
            base = anchor.to_numpy(dtype=float)
    elif anchor is not None:
        # Absolute blend: mostly anchor, light model.
        raw = np.clip(residual, 0.0, 20.0)
        base = 0.85 * anchor.to_numpy(dtype=float) + 0.15 * raw
    else:
        base = np.clip(residual, 0.0, 20.0)

    base = np.clip(base, 0.0, 20.0)
    out = meta.copy()
    if anchor is not None:
        out["ep_anchor"] = anchor.to_numpy(dtype=float)
    else:
        out["ep_anchor"] = base
    out["residual"] = residual_c
    out["residual_scaled"] = (
        scale * residual_c
        if (anchor is not None and VECTOR_TEAM.get("residual_mode", True) and model.fitted)
        else np.zeros(len(out), dtype=float)
    )
    out["base_points"] = base
    out["ev"] = base
    for i, name in enumerate(FEATURE_NAMES):
        out[f"v_{name}"] = X[:, i]
    if "cover_eligible" not in out.columns:
        out["cover_eligible"] = out["base_points"] >= 0.5
    if "buy_eligible" not in out.columns:
        out["buy_eligible"] = True
    if "status" not in out.columns:
        out["status"] = "a"
    if "chance_of_playing_next_round" not in out.columns:
        out["chance_of_playing_next_round"] = 100
    return out


def vector_dict_from_row(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    get = row.get if isinstance(row, dict) else row.get
    return {name: float(get(f"v_{name}") or 0.0) for name in FEATURE_NAMES}
