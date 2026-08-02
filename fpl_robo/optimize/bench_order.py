"""Bench auto-sub ordering for FPL pick positions 12–15."""

from __future__ import annotations

from typing import Any


_POS_RANK = {"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _pos(row: dict[str, Any]) -> str:
    return str(row.get("position") or "").upper()


def _ev(row: dict[str, Any]) -> float:
    try:
        return float(row.get("ev") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _ep(row: dict[str, Any]) -> float:
    try:
        return float(row.get("ep_next") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _cover(row: dict[str, Any]) -> float:
    return 1.0 if bool(row.get("cover_eligible", True)) else 0.0


def is_gk(row: dict[str, Any]) -> bool:
    return _pos(row) in {"GK", "GKP"}


def order_starters_for_fpl(starters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legal XI payload order: GK → DEF → MID → FWD, EV desc within position."""
    return sorted(
        starters,
        key=lambda r: (_POS_RANK.get(_pos(r), 9), -_ev(r)),
    )


def order_bench_for_autosub(bench: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """FPL auto-sub priority: outfield by EV (cover, then ep), backup GK last."""
    outfield = [r for r in bench if not is_gk(r)]
    gks = [r for r in bench if is_gk(r)]
    outfield.sort(key=lambda r: (-_ev(r), -_cover(r), -_ep(r), str(r.get("web_name") or "")))
    gks.sort(key=lambda r: (-_ev(r), str(r.get("web_name") or "")))
    return outfield + gks


def order_squad_for_fpl(squad_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Starters (formation order) then autosub-priority bench."""
    starters = order_starters_for_fpl([r for r in squad_rows if r.get("start")])
    bench = order_bench_for_autosub([r for r in squad_rows if not r.get("start")])
    return starters + bench
