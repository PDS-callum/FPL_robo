"""Shared Hub → bot soft priors (form heat, differentials, fixture ease helpers).

These are continuous, clamped multipliers — not hard gates from Hub leaderboards.
Market / price momentum already live in KG enrich; blank_ahead covers BGW haircuts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_robo.config import HUB_PRIORS


def form_heat(form: float, ppg: float) -> float:
    """Signed [-1, 1]: form above / below season PPG (Hub 'heating / cooling')."""
    delta = float(form or 0.0) - float(ppg or 0.0)
    # ±2.0 form points ≈ full signal.
    return max(-1.0, min(1.0, delta / 2.0))


def differential_score(
    ownership: float,
    form: float,
    *,
    own_ceiling: float = 8.0,
    form_floor: float = 4.0,
) -> float:
    """[0, 1] Hub-style differential: low ownership + strong form."""
    own = float(ownership or 0.0)
    frm = float(form or 0.0)
    if own >= own_ceiling or frm < form_floor:
        return 0.0
    own_term = (own_ceiling - own) / own_ceiling
    form_term = min(1.0, (frm - form_floor) / 3.0)
    return max(0.0, min(1.0, 0.55 * own_term + 0.45 * form_term))


def fdr_cell_ease(fdr: float | None) -> float | None:
    """Single-fixture ease in the Hub sense (higher = easier)."""
    if fdr is None:
        return None
    return max(0.0, 6.0 - float(fdr))


def team_run_ease(cells: list[dict[str, Any]], *, horizon: int | None = None) -> float | None:
    """Mean ease over FDR cells (blank-aware)."""
    slice_ = cells if horizon is None else cells[: int(horizon)]
    vals = []
    for c in slice_ or []:
        if c.get("blank") or c.get("fdr") is None:
            continue
        e = fdr_cell_ease(c.get("fdr"))
        if e is not None:
            vals.append(e)
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def blank_double_by_gw(
    *,
    fixtures_by_gw: dict[int, list[dict[str, Any]]],
    team_shorts: dict[int, str],
    gws: list[int],
) -> list[dict[str, Any]]:
    """Per-GW blank / double lists for Hub display and shared diagnostics.

    `fixtures_by_gw` values are match dicts with home/away id keys
    (`home.id` / `away.id` or `team_h` / `team_a`).
    """
    out: list[dict[str, Any]] = []
    for gw in gws:
        counts: dict[int, int] = {tid: 0 for tid in team_shorts}
        for m in fixtures_by_gw.get(int(gw)) or []:
            home = m.get("home") if isinstance(m.get("home"), dict) else None
            away = m.get("away") if isinstance(m.get("away"), dict) else None
            hid = home.get("id") if home else m.get("team_h")
            aid = away.get("id") if away else m.get("team_a")
            if hid is not None:
                counts[int(hid)] = counts.get(int(hid), 0) + 1
            if aid is not None:
                counts[int(aid)] = counts.get(int(aid), 0) + 1
        blanks = sorted(
            team_shorts[tid]
            for tid, n in counts.items()
            if n == 0 and tid in team_shorts
        )
        doubles = sorted(
            team_shorts[tid]
            for tid, n in counts.items()
            if n >= 2 and tid in team_shorts
        )
        out.append({"gameweek": int(gw), "blanks": blanks, "doubles": doubles})
    return out


def hub_prior_multiplier(
    *,
    form: float = 0.0,
    ppg: float = 0.0,
    ownership: float = 0.0,
    status: str = "a",
    chance: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """Clamped EV multiplier from Hub-style soft priors (not market/blank)."""
    c = cfg if cfg is not None else HUB_PRIORS
    if not bool(c.get("enabled", True)):
        return 1.0, {"form_heat": 0.0, "differential": 0.0}

    # Doubtful / unavailable: do not boost differentials or form heat.
    blocked = (status or "a") != "a" or (
        chance is not None and float(chance) < float(c.get("min_chance") or 75)
    )

    heat = form_heat(form, ppg)
    heat_cap = float(c.get("form_heat_cap") or 0.0)
    heat_term = 0.0 if blocked else heat_cap * heat

    diff = 0.0 if blocked else differential_score(
        ownership,
        form,
        own_ceiling=float(c.get("diff_own_ceiling") or 8.0),
        form_floor=float(c.get("diff_form_floor") or 4.0),
    )
    diff_cap = float(c.get("differential_cap") or 0.0)
    diff_term = diff_cap * diff

    raw = 1.0 + heat_term + diff_term
    lo = float(c.get("factor_min") or 0.96)
    hi = float(c.get("factor_max") or 1.04)
    mult = max(lo, min(hi, raw))
    return mult, {
        "form_heat": round(heat, 4),
        "differential": round(diff, 4),
        "raw": round(raw, 4),
    }


def apply_hub_priors(
    predictions: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Multiply prediction EV by Hub soft priors; stamp `hub_prior_mult`."""
    c = cfg if cfg is not None else HUB_PRIORS
    df = predictions.copy()
    if not bool(c.get("enabled", True)) or df.empty or "ev" not in df.columns:
        if "hub_prior_mult" not in df.columns:
            df["hub_prior_mult"] = 1.0
        return df

    mults: list[float] = []
    for _, row in df.iterrows():
        try:
            form = float(row.get("form") or 0.0)
        except (TypeError, ValueError):
            form = 0.0
        ppg_raw = row.get("points_per_game")
        if ppg_raw is None:
            ppg_raw = row.get("ppg")
        try:
            ppg = float(ppg_raw) if ppg_raw is not None and ppg_raw == ppg_raw else None
        except (TypeError, ValueError):
            ppg = None
        # Form heat needs PPG; without it force heat=0 by pairing form with itself.
        heat_form = form
        heat_ppg = float(ppg) if ppg is not None and float(ppg) > 0 else form
        try:
            own = float(row.get("selected_by_percent") or row.get("own") or 0.0)
        except (TypeError, ValueError):
            own = 0.0
        chance = row.get("chance_of_playing_next_round")
        if chance is None:
            chance = row.get("chance")
        try:
            chance_f = float(chance) if chance is not None and chance == chance else None
        except (TypeError, ValueError):
            chance_f = None
        m, _meta = hub_prior_multiplier(
            form=heat_form,
            ppg=heat_ppg,
            ownership=own,
            status=str(row.get("status") or "a"),
            chance=chance_f,
            cfg=c,
        )
        mults.append(m)

    df["hub_prior_mult"] = mults
    df["ev"] = (df["ev"].astype(float) * df["hub_prior_mult"]).clip(lower=0.0)
    return df
