"""Phase-aware KG_ENRICH overrides (preseason vs in-season)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from fpl_robo.config import KG_ENRICH


_PHASE_KEYS = (
    "def_ev_scale",
    "def_ep_ev_cap_mult",
    "cs_boost_scale",
    "cs_boost_cap",
    "form_blend",
    "form_boost_scale",
)


@contextmanager
def phase_kg_enrich(
    *,
    preseason: bool,
    gameweek: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Temporarily apply preseason / early-season / in-season EV settings."""
    saved = {k: KG_ENRICH.get(k) for k in _PHASE_KEYS}
    try:
        if preseason:
            KG_ENRICH["def_ev_scale"] = float(
                KG_ENRICH.get("def_ev_scale_preseason", KG_ENRICH.get("def_ev_scale") or 0.82)
            )
            KG_ENRICH["def_ep_ev_cap_mult"] = float(
                KG_ENRICH.get("def_ep_ev_cap_mult_preseason", 1.0)
            )
            KG_ENRICH["cs_boost_scale"] = float(KG_ENRICH.get("cs_boost_scale_preseason", 0.0))
            KG_ENRICH["cs_boost_cap"] = float(KG_ENRICH.get("cs_boost_cap_preseason", 0.0))
        else:
            KG_ENRICH["def_ev_scale"] = float(KG_ENRICH.get("def_ev_scale_in_season", 1.0))
            KG_ENRICH["def_ep_ev_cap_mult"] = float(
                KG_ENRICH.get("def_ep_ev_cap_mult_in_season", 0.0)
            )
            KG_ENRICH["cs_boost_scale"] = float(KG_ENRICH.get("cs_boost_scale_in_season", 0.25))
            KG_ENRICH["cs_boost_cap"] = float(KG_ENRICH.get("cs_boost_cap_in_season", 0.35))

        until = int(KG_ENRICH.get("form_blend_early_until_gw") or 0)
        early = (
            (not preseason)
            and gameweek is not None
            and until > 0
            and int(gameweek) <= until
        )
        if early:
            KG_ENRICH["form_blend"] = float(KG_ENRICH.get("form_blend_early") or 0.0)
            KG_ENRICH["form_boost_scale"] = float(
                KG_ENRICH.get("form_boost_scale_early") or 0.0
            )
        else:
            # Keep configured form_blend; ensure boost scale is active.
            if saved.get("form_blend") is not None:
                KG_ENRICH["form_blend"] = float(saved["form_blend"])
            KG_ENRICH["form_boost_scale"] = float(
                saved.get("form_boost_scale")
                if saved.get("form_boost_scale") is not None
                else 1.0
            )
        yield {k: KG_ENRICH.get(k) for k in _PHASE_KEYS}
    finally:
        for k, v in saved.items():
            if v is None:
                KG_ENRICH.pop(k, None)
            else:
                KG_ENRICH[k] = v
