"""Show minutes-related signals for the live squad."""

from __future__ import annotations

import json
from pathlib import Path

from fpl_robo.config import PRESEASON_GW_WEIGHTS, PRESEASON_RISK_PROFILES
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.predict.baseline import PreseasonEVModel

LIVE = [
    (1, True, ""),
    (4, True, "V"),
    (387, True, ""),
    (356, True, ""),
    (229, True, ""),
    (200, True, ""),
    (426, True, "C"),
    (428, True, ""),
    (480, True, ""),
    (208, True, ""),
    (346, True, ""),
    (109, False, ""),
    (127, False, ""),
    (248, False, ""),
    (249, False, ""),
]


def implied_minutes(ep_next, chance, status, hist_minutes) -> str:
    """Rough next-GW minutes band from available signals (not a formal model)."""
    st = (status or "a").lower()
    if st in {"u", "s"}:
        return "0"
    if st == "i":
        return "0-20"
    if chance is not None and float(chance) <= 25:
        return "0-20"
    if chance is not None and float(chance) <= 50:
        return "20-60"
    ep = float(ep_next) if ep_next is not None else 0.0
    mins = int(hist_minutes or 0)
    # Preseason: EP is the strongest public proxy for expected involvement.
    if ep >= 4.0 or mins >= 2500:
        return "75-90"
    if ep >= 3.0 or mins >= 1500:
        return "60-90"
    if ep >= 2.0 or mins >= 600:
        return "45-75"
    if ep >= 1.5 or mins >= 90:
        return "20-60"
    return "0-20"


def main() -> None:
    client = FPLClient()
    bs = client.bootstrap_static()
    teams = {t["id"]: t["short_name"] for t in bs["teams"]}
    elements = {int(e["id"]): e for e in bs["elements"]}

    kg = build_knowledge_graph(client=client)
    cfg = PRESEASON_RISK_PROFILES["balanced"]
    model = PreseasonEVModel(
        gameweek_weights=PRESEASON_GW_WEIGHTS,
        ownership_weight=cfg["ownership_weight"],
        uncertainty_penalty=cfg["uncertainty_penalty"],
    )
    preds = model.predict(kg)
    by_id = {int(r["id"]): r for _, r in preds.iterrows()}

    rows = []
    print(f"{'Role':<6} {'Pos':<3} {'Name':<16} {'Club':<3} {'Pred mins':<9} {'EP':>4} {'Chance':>6} {'Hist mins':>9} {'Starts':>6} {'m_fac':>5}")
    for pid, start, flag in LIVE:
        el = elements[pid]
        pr = by_id[pid]
        role = "XI" if start else "Bench"
        if flag:
            role = flag
        chance = el.get("chance_of_playing_next_round")
        ep = el.get("ep_next")
        hist = int(el.get("minutes") or 0)
        pred = implied_minutes(ep, chance, el.get("status"), hist)
        row = {
            "role": role,
            "position": pr["position"],
            "web_name": pr["web_name"],
            "team_short": teams.get(el["team"]),
            "predicted_minutes_band": pred,
            "ep_next": None if ep is None else float(ep),
            "chance_of_playing_next_round": chance,
            "season_minutes": hist,
            "starts": int(el.get("starts") or 0),
            "minutes_factor": float(pr.get("minutes_factor") or 0),
            "status": el.get("status"),
        }
        rows.append(row)
        chance_s = "-" if chance is None else str(chance)
        print(
            f"{role:<6} {row['position']:<3} {row['web_name']:<16} {row['team_short']:<3} "
            f"{pred:<9} {float(ep or 0):4.1f} {chance_s:>6} {hist:>9} {row['starts']:>6} "
            f"{row['minutes_factor']:5.2f}"
        )

    Path("data/live_team_minutes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nNote: FPL does not publish official predicted minutes; bands are inferred from EP/availability/history.")


if __name__ == "__main__":
    main()
