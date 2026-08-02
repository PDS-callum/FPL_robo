"""Shared attack-quality predicate (MIP floor + execute gates).

Quality is primarily official ep_next, with narrow alternate paths for
lead strikers whose EP sits just under the band (focal / form / minutes).
"""

from __future__ import annotations

from typing import Any, Mapping

from fpl_robo.config import ATTACK_FLOOR


def _cfg(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**ATTACK_FLOOR, **(overrides or {})}


def is_quality_fwd(
    *,
    ep_next: float | None,
    status: str | None = None,
    is_focal: bool = False,
    form: float | None = None,
    rolling_minutes: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """True if this FWD counts toward the attack floor / quality gates."""
    c = _cfg(cfg)
    st = str(status or "a").lower()
    if st in {"u", "s"}:
        return False
    try:
        ep = float(ep_next or 0.0)
    except (TypeError, ValueError):
        ep = 0.0
    try:
        min_ep = float(c.get("min_ep_next") or 2.1)
    except (TypeError, ValueError):
        min_ep = 2.1
    if ep >= min_ep:
        return True

    # Narrow alternates: must still clear a soft EP floor (not Welbeck-class noise).
    if bool(c.get("allow_focal_alt", True)) and is_focal:
        try:
            focal_floor = float(c.get("focal_min_ep") or 1.95)
        except (TypeError, ValueError):
            focal_floor = 1.95
        if ep >= focal_floor:
            return True

    if bool(c.get("allow_form_alt", True)) and form is not None:
        try:
            form_v = float(form)
            form_need = float(c.get("min_form") or 5.0)
            form_floor = float(c.get("form_min_ep") or 1.95)
        except (TypeError, ValueError):
            form_v, form_need, form_floor = 0.0, 5.0, 1.95
        if form_v >= form_need and ep >= form_floor:
            return True

    if bool(c.get("allow_minutes_alt", True)) and rolling_minutes is not None:
        try:
            mins = float(rolling_minutes)
            mins_need = float(c.get("min_rolling_minutes") or 240)
            mins_floor = float(c.get("minutes_min_ep") or 1.9)
        except (TypeError, ValueError):
            mins, mins_need, mins_floor = 0.0, 240.0, 1.9
        if mins >= mins_need and ep >= mins_floor:
            return True

    return False


def quality_fwd_ids(
    ids: list[Any],
    *,
    pos: Mapping[Any, str],
    ep_next: Mapping[Any, float],
    status: Mapping[Any, str] | None = None,
    is_focal: Mapping[Any, bool] | None = None,
    form: Mapping[Any, float] | None = None,
    rolling_minutes: Mapping[Any, float] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[Any]:
    """Return element ids that are quality FWDs under ATTACK_FLOOR."""
    out: list[Any] = []
    for i in ids:
        if str(pos.get(i) or "").upper() != "FWD":
            continue
        if is_quality_fwd(
            ep_next=float(ep_next.get(i) or 0.0),
            status=(status or {}).get(i),
            is_focal=bool((is_focal or {}).get(i)),
            form=(form or {}).get(i) if form is not None else None,
            rolling_minutes=(
                (rolling_minutes or {}).get(i) if rolling_minutes is not None else None
            ),
            cfg=cfg,
        ):
            out.append(i)
    return out


def alt_maps_from_frame(df: Any) -> dict[str, dict[Any, Any]]:
    """Build optional quality-alt maps from a predictions DataFrame."""
    focal: dict[Any, bool] = {}
    form: dict[Any, float] = {}
    rolling: dict[Any, float] = {}
    if df is None or not hasattr(df, "iterrows"):
        return {"is_focal": focal, "form": form, "rolling_minutes": rolling}
    try:
        cols = set(df.columns)
    except Exception:
        cols = set()
    for _, r in df.iterrows():
        try:
            pid = int(r["id"])
        except (TypeError, ValueError, KeyError):
            continue
        if "is_focal_attacker" in cols:
            focal[pid] = bool(r.get("is_focal_attacker"))
        if "form" in cols and r.get("form") is not None:
            try:
                form[pid] = float(r.get("form"))
            except (TypeError, ValueError):
                pass
        if "rolling_minutes" in cols and r.get("rolling_minutes") is not None:
            try:
                rolling[pid] = float(r.get("rolling_minutes"))
            except (TypeError, ValueError):
                pass
    return {"is_focal": focal, "form": form, "rolling_minutes": rolling}
