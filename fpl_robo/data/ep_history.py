"""Historical FPL expected points (ep_this) from FPL-Core-Insights snapshots.

Vaastav merged_gw often blanks xP for whole gameweeks (scraper gaps after weekly
updates stopped). olbauday/FPL-Core-Insights keeps per-GW playerstats with ep_this
for all 38 weeks — join on FPL element id to repair blank rounds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from fpl_robo.config import CHIP_BACKTEST_SEASON, EP_HISTORY

logger = logging.getLogger(__name__)

CORE_INSIGHTS_PLAYERSTATS = (
    "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/"
    "data/{season_dir}/By%20Gameweek/GW{gw}/playerstats.csv"
)


def season_to_core_insights_dir(season: str) -> str:
    """Map `2025-26` → `2025-2026` (Core Insights folder naming)."""
    s = str(season).strip()
    if len(s) == 7 and s[4] == "-" and s[5:].isdigit() and len(s[5:]) == 2:
        yy = int(s[5:])
        return f"{s[:4]}-20{yy:02d}"
    return s


def ep_history_cache_path(season: str = CHIP_BACKTEST_SEASON) -> Path:
    pattern = EP_HISTORY.get("cache_path") or "data/cache/ep_this_{season}.csv"
    return Path(str(pattern).format(season=season))


def load_ep_history(
    season: str = CHIP_BACKTEST_SEASON,
    *,
    force_refresh: bool = False,
    max_gw: int = 38,
) -> pd.DataFrame:
    """Load cached (element_id, gameweek, ep_this, ep_next) or download snapshots."""
    cache = ep_history_cache_path(season)
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache)

    season_dir = season_to_core_insights_dir(season)
    frames: list[pd.DataFrame] = []
    for gw in range(1, max_gw + 1):
        url = CORE_INSIGHTS_PLAYERSTATS.format(season_dir=season_dir, gw=gw)
        try:
            with urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed github raw URL
                raw = pd.read_csv(resp)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            logger.warning("ep_history: skip GW%s (%s)", gw, exc)
            continue
        if "id" not in raw.columns or "ep_this" not in raw.columns:
            logger.warning("ep_history: GW%s missing id/ep_this columns", gw)
            continue
        frame = pd.DataFrame(
            {
                "element": raw["id"].astype(int),
                "gameweek": int(gw),
                "ep_this": pd.to_numeric(raw["ep_this"], errors="coerce").fillna(0.0),
                "ep_next": pd.to_numeric(raw.get("ep_next"), errors="coerce").fillna(0.0)
                if "ep_next" in raw.columns
                else 0.0,
            }
        )
        frames.append(frame)

    if not frames:
        empty = pd.DataFrame(columns=["element", "gameweek", "ep_this", "ep_next"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(cache, index=False)
        return empty

    out = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    logger.info("ep_history: cached %s rows → %s", len(out), cache)
    return out


def fill_blank_gw_xp(
    table: pd.DataFrame,
    *,
    season: str = CHIP_BACKTEST_SEASON,
    blank_frac: float | None = None,
    ep_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replace xP on GW-wide blank rounds with Core Insights ep_this (by element id).

    Expects a prepared table with columns: id, gameweek, xP.
    Leaves non-blank GWs and individual legitimate xP=0 rows untouched.
    """
    if not EP_HISTORY.get("enabled", True):
        return table
    if table.empty or "xP" not in table.columns or "gameweek" not in table.columns:
        return table
    if "id" not in table.columns:
        return table

    frac = float(
        blank_frac
        if blank_frac is not None
        else EP_HISTORY.get("gw_blank_frac", 0.90)
    )
    df = table.copy()
    blank_gws: list[int] = []
    for gw, grp in df.groupby("gameweek"):
        gxp = pd.to_numeric(grp["xP"], errors="coerce").fillna(0.0)
        if float((gxp <= 0.05).mean()) >= frac:
            blank_gws.append(int(gw))
    if not blank_gws:
        return df

    hist = ep_source if ep_source is not None else load_ep_history(season)
    if hist.empty:
        return df

    lookup = hist[hist["gameweek"].isin(blank_gws)][["element", "gameweek", "ep_this"]].copy()
    lookup = lookup.rename(columns={"element": "id"})
    merged = df.merge(lookup, on=["id", "gameweek"], how="left", suffixes=("", "_ci"))
    fill_mask = merged["gameweek"].isin(blank_gws) & merged["ep_this"].notna()
    if fill_mask.any():
        merged.loc[fill_mask, "xP"] = merged.loc[fill_mask, "ep_this"].astype(float)
        merged["xp_filled"] = fill_mask
    else:
        merged["xp_filled"] = False
    merged = merged.drop(columns=["ep_this"], errors="ignore")
    n_filled = int(fill_mask.sum())
    logger.info(
        "ep_history: filled xP on %s blank GW(s) for %s player-rows",
        sorted(blank_gws),
        n_filled,
    )
    return merged
