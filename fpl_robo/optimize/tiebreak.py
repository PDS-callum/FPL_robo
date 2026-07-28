"""Secondary quality scores for optimiser tie-breaking."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.config import OPTIMISER_TIEBREAK


def _col(df: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def raw_quality_components(df: pd.DataFrame) -> pd.DataFrame:
    """Build unnormalised form / xGI / ICT / ownership / KG context components."""
    from fpl_robo.config import KG_ENRICH

    form = _col(df, "rolling_points", "form")

    # Prefer rate / rolling involvement over prior-season cumulative xG.
    # Zero out absurd per-90 rates when season minutes are known and below the floor.
    # If minutes are absent, trust upstream (derive already nulls thin-sample xgi90).
    min_mins = float(KG_ENRICH.get("rate_feature_min_minutes") or 180.0)
    hard_cap = float(KG_ENRICH.get("xgi90_hard_cap") or 1.25)
    has_minutes = any(
        c in df.columns for c in ("minutes", "prior_minutes", "season_minutes")
    )
    if "xgi90" in df.columns and float(pd.to_numeric(df["xgi90"], errors="coerce").fillna(0).sum()) > 0:
        xgi = _col(df, "xgi90").clip(upper=hard_cap)
        if has_minutes:
            minutes = _col(df, "minutes", "prior_minutes", "season_minutes", default=0.0)
            xgi = xgi.where(minutes >= min_mins, 0.0)
    elif "rolling_xgi" in df.columns and float(
        pd.to_numeric(df["rolling_xgi"], errors="coerce").fillna(0).sum()
    ) > 0:
        xgi = _col(df, "rolling_xgi")
        if has_minutes:
            minutes = _col(df, "minutes", "prior_minutes", "season_minutes", default=0.0)
            xgi = xgi.where(minutes >= min_mins, 0.0)
    else:
        xg = _col(df, "rolling_expected_goals", "rolling_xg", "expected_goals")
        xa = _col(df, "rolling_expected_assists", "rolling_xa", "expected_assists")
        # Season totals → soft per-GW proxy so preseason does not dominate.
        xgi = (xg + xa) / 38.0

    ict = _col(df, "rolling_ict_index", "rolling_ict", "ict_index")
    if "selected_by_percent" in df.columns:
        own = _col(df, "selected_by_percent") / 100.0
    else:
        selected = _col(df, "selected")
        max_sel = float(selected.max()) if len(selected) else 0.0
        own = selected / max_sel if max_sel > 0 else selected * 0.0

    context = _col(df, "context_factor", "next_matchup_factor", default=1.0)
    set_piece = _col(df, "set_piece_score", default=0.0)
    stability = _col(df, "role_stability", "starts_rate", default=1.0)
    focal = _col(df, "is_focal_attacker", default=0.0)

    return pd.DataFrame(
        {
            "q_form": form,
            "q_xgi": xgi,
            "q_ict": ict,
            "q_own": own,
            "q_context": context,
            "q_set_piece": set_piece,
            "q_stability": stability,
            "q_focal": focal,
        },
        index=df.index,
    )


def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    std = float(s.std(ddof=0) or 0.0)
    if std < 1e-9:
        return s * 0.0
    z = (s - float(s.mean())) / std
    return z.clip(-3.0, 3.0) / 3.0  # roughly [-1, 1]


def attach_quality_scores(
    players: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add a `quality` column (~[-1,1]) from form / xGI / ICT / ownership / KG."""
    cfg = dict(OPTIMISER_TIEBREAK)
    if config:
        cfg.update(config)
    df = players.copy()
    parts = raw_quality_components(df)
    quality = (
        float(cfg.get("form_weight") or 0.0) * _zscore(parts["q_form"])
        + float(cfg.get("xgi_weight") or 0.0) * _zscore(parts["q_xgi"])
        + float(cfg.get("ict_weight") or 0.0) * _zscore(parts["q_ict"])
        + float(cfg.get("ownership_weight") or 0.0) * _zscore(parts["q_own"])
        + float(cfg.get("context_weight") or 0.0) * _zscore(parts["q_context"])
        + float(cfg.get("set_piece_weight") or 0.0) * _zscore(parts["q_set_piece"])
        + float(cfg.get("stability_weight") or 0.0) * _zscore(parts["q_stability"])
        + float(cfg.get("focal_weight") or 0.0) * _zscore(parts["q_focal"])
    )
    df["quality"] = quality.astype(float).fillna(0.0)
    for col in parts.columns:
        df[col] = parts[col].astype(float).fillna(0.0)
    return df


def tiebreak_enabled(config: dict[str, Any] | None = None) -> bool:
    cfg = dict(OPTIMISER_TIEBREAK)
    if config:
        cfg.update(config)
    return bool(cfg.get("enabled"))


def tiebreak_mode(config: dict[str, Any] | None = None) -> str:
    cfg = dict(OPTIMISER_TIEBREAK)
    if config:
        cfg.update(config)
    mode = str(cfg.get("mode") or "lexicographic").strip().lower()
    return mode if mode in {"lexicographic", "soft"} else "lexicographic"


def solve_lexicographic(
    prob: Any,
    primary: Any,
    secondary: Any | None,
    *,
    tolerance: float = 1e-3,
    msg: bool = False,
) -> int:
    """Maximise `primary`, then `secondary` among solutions within `tolerance`."""
    import pulp

    solver = pulp.PULP_CBC_CMD(msg=msg)
    prob.setObjective(primary)
    status_code = prob.solve(solver)
    if pulp.LpStatus[status_code] != "Optimal" or secondary is None:
        return status_code
    opt = float(pulp.value(primary) or 0.0)
    prob += primary >= opt - float(tolerance), "_lex_keep_primary"
    prob.setObjective(secondary)
    return prob.solve(solver)
