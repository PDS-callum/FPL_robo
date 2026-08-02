"""Join enriched KG player attrs onto prediction frames for optimisation / chips."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.graph.store import KnowledgeGraph

# Columns the optimiser / chips / dashboard benefit from beyond core EV.
KG_PRED_COLUMNS: tuple[str, ...] = (
    "rolling_points",
    "rolling_minutes",
    "rolling_xg",
    "rolling_xa",
    "rolling_xgi",
    "rolling_ict",
    "xgi90",
    "xgi90_raw",
    "minutes",
    "xgi",
    "set_piece_score",
    "set_piece_factor",
    "role_stability",
    "starts_rate",
    "is_focal_attacker",
    "next_n_fixtures",
    "next_matchup_factor",
    "next_fixture_scale",
    "next_fdr",
    "context_factor",
    "rest_days",
    "ict_share",
    "club_pos_threat_share",
    "cs_per_90",
    "saves_per_90",
    "bps",
    "bonus",
    "lineup_role",
    "lineup_confidence",
)


def enrich_predictions_from_kg(
    predictions: pd.DataFrame,
    kg: KnowledgeGraph,
) -> pd.DataFrame:
    """Left-join useful KG player attrs onto the predictions DataFrame.

    Does not overwrite existing non-null prediction columns (EV model wins).
    """
    if predictions is None or predictions.empty:
        return predictions
    df = predictions.copy()
    if "id" not in df.columns:
        return df

    by_id: dict[int, dict[str, Any]] = {}
    for p in kg.players():
        try:
            by_id[int(p["id"])] = p
        except (KeyError, TypeError, ValueError):
            continue

    kg_rows: list[dict[str, Any]] = []
    for pid in df["id"]:
        try:
            player = by_id.get(int(pid)) or {}
        except (TypeError, ValueError):
            player = {}
        kg_rows.append({col: player.get(col) for col in KG_PRED_COLUMNS})
    kg_df = pd.DataFrame(kg_rows, index=df.index)

    for col in KG_PRED_COLUMNS:
        if col not in df.columns:
            df[col] = kg_df[col]
        else:
            df[col] = df[col].combine_first(kg_df[col])

    if "rolling_ict_index" not in df.columns and "rolling_ict" in df.columns:
        df["rolling_ict_index"] = df["rolling_ict"]
    if "rolling_expected_goals" not in df.columns and "rolling_xg" in df.columns:
        df["rolling_expected_goals"] = df["rolling_xg"]
    if "rolling_expected_assists" not in df.columns and "rolling_xa" in df.columns:
        df["rolling_expected_assists"] = df["rolling_xa"]
    return df


def prediction_row_extras(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """KG-ish fields to carry into optimiser squad/XI solution rows."""
    get = row.get if isinstance(row, dict) else row.get
    out: dict[str, Any] = {}
    for col in (
        "context_factor",
        "next_fdr",
        "next_matchup_factor",
        "next_n_fixtures",
        "set_piece_score",
        "set_piece_factor",
        "xgi90",
        "role_stability",
        "is_focal_attacker",
        "rest_days",
        "quality",
        "ep_next",
        "cover_eligible",
        "status",
        "chance_of_playing_next_round",
        "lineup_role",
        "lineup_confidence",
    ):
        val = get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if col == "is_focal_attacker":
            out[col] = bool(val)
        elif col == "cover_eligible":
            out[col] = bool(val)
        elif col in {"status", "lineup_role"}:
            out[col] = str(val)
        else:
            try:
                out[col] = float(val)
            except (TypeError, ValueError):
                out[col] = val
    return out
