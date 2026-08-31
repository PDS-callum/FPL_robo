"""Team write operations for execute flows."""

from __future__ import annotations

from typing import Any

from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.optimize.captaincy import pick_vice_captain_id


def apply_initial_squad(
    client: FPLClient,
    manager_id: int,
    squad_rows: list[dict[str, Any]],
    *,
    captain_id: int | None,
    event_id: int | None = None,
    chip: str | None = None,
) -> dict[str, Any]:
    """Transfer to the target 15 and persist lineup/captain via my-team.

    Expects 15 rows with `id`, `price`, and `start` flags.
    `chip` should be an FPL API chip name: 3xc / bboost / freehit / wildcard.
    """
    return apply_weekly_plan(
        client=client,
        manager_id=manager_id,
        squad_rows=squad_rows,
        captain_id=captain_id,
        event_id=event_id,
        chip=chip,
    )


def apply_weekly_plan(
    client: FPLClient,
    manager_id: int,
    squad_rows: list[dict[str, Any]],
    *,
    captain_id: int | None,
    event_id: int | None = None,
    chip: str | None = None,
) -> dict[str, Any]:
    """Apply transfers + lineup (+ optional chip) for a weekly run."""
    if len(squad_rows) != 15:
        raise ValueError("Expected exactly 15 squad rows")

    # FPL expects a legal formation order: GK, DEFs, MIDs, FWDs, then bench.
    # Starters must be grouped by position; outfield bench order is preserved from
    # the optimiser (autosub priority), with bench GK first when present.
    pos_rank = {"GK": 0, "GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    starters = sorted(
        [r for r in squad_rows if r.get("start")],
        key=lambda r: (pos_rank.get(str(r.get("position") or ""), 9), -float(r.get("ev") or 0)),
    )
    bench_rows = [r for r in squad_rows if not r.get("start")]
    gk_bench = [
        r
        for r in bench_rows
        if str(r.get("position") or "").upper() in {"GK", "GKP"}
    ]
    out_bench = [
        r
        for r in bench_rows
        if str(r.get("position") or "").upper() not in {"GK", "GKP"}
    ]
    bench = gk_bench + out_bench
    if len(starters) != 11 or len(bench) != 4:
        raise ValueError("Squad rows must contain 11 starters and 4 bench players")

    bootstrap = client.bootstrap_static()
    prices = {
        int(el["id"]): int(el.get("now_cost") or 0) for el in (bootstrap.get("elements") or [])
    }
    element_types = {
        int(el["id"]): int(el.get("element_type") or 0) for el in (bootstrap.get("elements") or [])
    }

    current = client.my_team(manager_id)
    current_picks = list(current.get("picks") or [])
    current_by_element = {
        int(p["element"]): p for p in current_picks if p.get("element") is not None
    }
    target_ids = [int(r["id"]) for r in squad_rows]
    target_set = set(target_ids)
    current_ids = set(current_by_element)

    outs = list(current_ids - target_set)
    inns = list(target_set - current_ids)
    if len(outs) != len(inns):
        raise ValueError(
            f"Transfer mismatch: {len(outs)} outs vs {len(inns)} inns"
        )

    outs_by_type: dict[int, list[int]] = {}
    inns_by_type: dict[int, list[int]] = {}
    for pid in outs:
        outs_by_type.setdefault(element_types.get(pid, 0), []).append(pid)
    for pid in inns:
        inns_by_type.setdefault(element_types.get(pid, 0), []).append(pid)

    transfers: list[dict[str, int]] = []
    for etype in sorted(set(outs_by_type) | set(inns_by_type)):
        o_list = outs_by_type.get(etype, [])
        i_list = inns_by_type.get(etype, [])
        if len(o_list) != len(i_list):
            raise ValueError(f"Position-type transfer mismatch for type {etype}")
        for element_out, element_in in zip(o_list, i_list):
            out_pick = current_by_element[element_out]
            in_price = prices.get(element_in)
            if in_price is None:
                row = next(r for r in squad_rows if int(r["id"]) == element_in)
                in_price = int(round(float(row["price"]) * 10))
            transfers.append(
                {
                    "element_out": element_out,
                    "element_in": element_in,
                    "purchase_price": int(in_price),
                    "selling_price": int(
                        out_pick.get("selling_price")
                        or out_pick.get("purchase_price")
                        or prices.get(element_out)
                        or 0
                    ),
                }
            )

    # Wildcard / Free Hit travel on the transfers payload.
    transfer_chip = chip if chip in {"wildcard", "freehit"} else None
    lineup_chip = chip if chip in {"3xc", "bboost"} else None

    transfer_resp: dict[str, Any] = {}
    if transfers or transfer_chip:
        resolved_event = event_id
        if resolved_event is None:
            resolved_event = 1
            for ev in bootstrap.get("events") or []:
                if ev.get("is_next") or ev.get("is_current"):
                    resolved_event = int(ev["id"])
                    break
        transfer_resp = client.execute_transfers(
            {
                "transfers": transfers,
                "chip": transfer_chip,
                "entry": manager_id,
                "event": resolved_event,
            }
        )
        current = client.my_team(manager_id)
        current_by_element = {
            int(p["element"]): p
            for p in (current.get("picks") or [])
            if p.get("element") is not None
        }

    ordered = starters + bench
    vice_id = pick_vice_captain_id(squad_rows, captain_id)
    captain_mult = 3 if lineup_chip == "3xc" else 2

    picks: list[dict[str, Any]] = []
    for idx, row in enumerate(ordered, start=1):
        existing = current_by_element.get(int(row["id"]), {})
        is_start = idx <= 11
        is_captain = row["id"] == captain_id
        if lineup_chip == "bboost":
            multiplier = captain_mult if is_captain else 1
        else:
            multiplier = (captain_mult if is_captain else 1) if is_start else 0
        picks.append(
            {
                "element": int(row["id"]),
                "position": idx,
                "multiplier": multiplier,
                "is_captain": is_captain,
                "is_vice_captain": row["id"] == vice_id,
                "purchase_price": int(
                    existing.get("purchase_price")
                    or prices.get(int(row["id"]))
                    or round(float(row["price"]) * 10)
                ),
                "selling_price": int(
                    existing.get("selling_price")
                    or existing.get("purchase_price")
                    or prices.get(int(row["id"]))
                    or round(float(row["price"]) * 10)
                ),
            }
        )

    save_resp = client.save_my_team(manager_id, picks=picks, chip=lineup_chip)
    verify = client.my_team(manager_id)
    verify_picks = verify.get("picks") or []
    verify_ids = {int(p.get("element")) for p in verify_picks if p.get("element") is not None}
    planned_start = {int(r["id"]) for r in squad_rows if r.get("start")}
    planned_bench = {int(r["id"]) for r in squad_rows if not r.get("start")}
    verified_start = {
        int(p["element"])
        for p in verify_picks
        if p.get("element") is not None and int(p.get("multiplier") or 0) > 0
    }
    verified_bench = {
        int(p["element"])
        for p in verify_picks
        if p.get("element") is not None and int(p.get("multiplier") or 0) == 0
    }
    lineup_ok = planned_start == verified_start and planned_bench == verified_bench
    if not lineup_ok:
        raise RuntimeError(
            "Lineup save mismatch: FPL start/bench does not match planned XI. "
            f"missing_start={sorted(planned_start - verified_start)} "
            f"extra_start={sorted(verified_start - planned_start)} "
            f"missing_bench={sorted(planned_bench - verified_bench)} "
            f"extra_bench={sorted(verified_bench - planned_bench)}"
        )
    verify_captain = next(
        (int(p["element"]) for p in verify_picks if p.get("is_captain") and p.get("element") is not None),
        None,
    )
    if captain_id is not None and verify_captain is not None and int(captain_id) != int(verify_captain):
        raise RuntimeError(
            f"Captain save mismatch: planned={captain_id} verified={verify_captain}"
        )

    return {
        "saved": True,
        "lineup_verified": True,
        "transfers_made": len(transfers),
        "transfers": transfers,
        "chip": chip,
        "transfer_response_keys": sorted(transfer_resp.keys()) if isinstance(transfer_resp, dict) else [],
        "saved_count": len(save_resp.get("picks") or []),
        "verified_count": len(verify_picks),
        "planned_xi": sorted(planned_start),
        "verified_xi": sorted(verified_start),
        "planned_bench": sorted(planned_bench),
        "verified_bench": sorted(verified_bench),
        "captain_id": verify_captain,
        "vice_captain_id": next(
            (p.get("element") for p in verify_picks if p.get("is_vice_captain")), None
        ),
        "all_ids_verified": all(int(r["id"]) in verify_ids for r in squad_rows),
    }
