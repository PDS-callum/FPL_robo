"""Repredict EV for the currently selected live squad."""

from __future__ import annotations

import json
from pathlib import Path

from fpl_robo.config import (
    BENCH_COVER,
    DEFAULT_MANAGER_ID,
    PRESEASON_GW_WEIGHTS,
    PRESEASON_RISK_PROFILES,
)
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.predict.baseline import PreseasonEVModel


# Verified live picks after apply (slot order).
LIVE_PICKS = [
    {"id": 1, "start": True, "captain": False, "vice": False},
    {"id": 4, "start": True, "captain": False, "vice": True},
    {"id": 387, "start": True, "captain": False, "vice": False},
    {"id": 356, "start": True, "captain": False, "vice": False},
    {"id": 229, "start": True, "captain": False, "vice": False},
    {"id": 200, "start": True, "captain": False, "vice": False},
    {"id": 426, "start": True, "captain": True, "vice": False},
    {"id": 428, "start": True, "captain": False, "vice": False},
    {"id": 480, "start": True, "captain": False, "vice": False},
    {"id": 208, "start": True, "captain": False, "vice": False},
    {"id": 346, "start": True, "captain": False, "vice": False},
    {"id": 109, "start": False, "captain": False, "vice": False},
    {"id": 127, "start": False, "captain": False, "vice": False},
    {"id": 248, "start": False, "captain": False, "vice": False},
    {"id": 249, "start": False, "captain": False, "vice": False},
]


def main() -> None:
    kg = build_knowledge_graph()
    cfg = PRESEASON_RISK_PROFILES["balanced"]
    model = PreseasonEVModel(
        gameweek_weights=PRESEASON_GW_WEIGHTS,
        ownership_weight=cfg["ownership_weight"],
        uncertainty_penalty=cfg["uncertainty_penalty"],
    )
    preds = model.predict(kg)
    by_id = {int(r["id"]): r for _, r in preds.iterrows()}

    rows = []
    for pick in LIVE_PICKS:
        p = by_id[int(pick["id"])]
        rows.append(
            {
                "id": int(p["id"]),
                "web_name": p["web_name"],
                "position": p["position"],
                "team_short": p["team_short"],
                "price": float(p["price"]),
                "ev": float(p["ev"]),
                "cover_eligible": bool(p["cover_eligible"]),
                "start": bool(pick["start"]),
                "captain": bool(pick["captain"]),
                "vice": bool(pick["vice"]),
            }
        )

    xi = [r for r in rows if r["start"]]
    bench = [r for r in rows if not r["start"]]
    xi_ev = sum(r["ev"] for r in xi)
    captain_extra = next(r["ev"] for r in xi if r["captain"])
    bench_ev = sum(r["ev"] for r in bench)
    objective = xi_ev + captain_extra + BENCH_COVER["bench_ev_weight"] * bench_ev
    budget = round(sum(r["price"] for r in rows), 1)

    out = {
        "manager_id": DEFAULT_MANAGER_ID,
        "gameweek": int(preds["gameweek"].iloc[0]),
        "budget_used": budget,
        "xi_ev": round(xi_ev, 3),
        "captain_extra_ev": round(captain_extra, 3),
        "bench_ev": round(bench_ev, 3),
        "objective_like": round(objective, 3),
        "all_bench_cover_eligible": all(r["cover_eligible"] for r in bench),
        "starting_xi": xi,
        "bench": bench,
    }
    Path("data/live_team_repredict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"GW{out['gameweek']} live repredict | budget {budget}m")
    print(f"XI EV {out['xi_ev']} + captain {out['captain_extra_ev']} | bench EV {out['bench_ev']}")
    print(f"Objective-like {out['objective_like']} | all bench coverable: {out['all_bench_cover_eligible']}")
    print("\nStarting XI:")
    for r in xi:
        flag = " (C)" if r["captain"] else (" (V)" if r["vice"] else "")
        print(
            f"  {r['position']:<3} {r['web_name']:<16} {r['team_short']:<3} "
            f"{r['price']:.1f}m  EV {r['ev']:.2f}{flag}"
        )
    print("\nBench:")
    for r in bench:
        cover = "cover" if r["cover_eligible"] else "DEAD"
        print(
            f"  {r['position']:<3} {r['web_name']:<16} {r['team_short']:<3} "
            f"{r['price']:.1f}m  EV {r['ev']:.2f} [{cover}]"
        )


if __name__ == "__main__":
    main()
