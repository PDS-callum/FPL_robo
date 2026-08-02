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
)


@contextmanager
def phase_kg_enrich(*, preseason: bool) -> Iterator[dict[str, Any]]:
    """Temporarily apply preseason or in-season DEF/CS settings."""
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
        yield {k: KG_ENRICH.get(k) for k in _PHASE_KEYS}
    finally:
        for k, v in saved.items():
            if v is None:
                KG_ENRICH.pop(k, None)
            else:
                KG_ENRICH[k] = v
