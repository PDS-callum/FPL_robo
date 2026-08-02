"""Full-season backtest: initial squad + weekly transfers scored on actual points."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

from fpl_robo.chips.backtest import load_merged_gw, prepare_gw_table
from fpl_robo.chips.decide import recommend_chips
from fpl_robo.chips.inventory import ChipInventory, ChipWindow
from fpl_robo.chips.schedule import chip_schedule_adjustments
from fpl_robo.config import (
    BENCH_COVER,
    CHIP_BACKTEST_CACHE,
    CHIP_BACKTEST_SEASON,
    CHIP_DEFAULTS,
    HORIZON_EV,
)
from fpl_robo.optimize.single_gw import optimise_single_gw
from fpl_robo.optimize.transfers import optimise_transfers
from fpl_robo.predict.ev_modes import (
    EvMode,
    attach_lagged_history,
    build_predictions,
    transfer_gate_kwargs,
)
from fpl_robo.predict.horizon import count_blank_owned, team_fixtures_from_merged

GW_SUMMARY_URL = (
    "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/"
    "data/2025-2026/gameweek_summaries.csv"
)
GW_SUMMARY_CACHE = "data/cache/gw_summaries_2025-26.csv"


@dataclass
class SeasonBacktestResult:
    season: str
    total_points: int
    average_manager_points: float | None
    points_vs_average: float | None
    hits_cost: int
    transfers_made: int
    chips_played: dict[str, list[int]]
    weekly: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prediction_frame(
    gw_df: pd.DataFrame,
    *,
    mode: EvMode = "ep_baseline",
    team_fixtures: dict | None = None,
    blank_ahead: bool | None = None,
    current_ids: list[int] | None = None,
) -> pd.DataFrame:
    return build_predictions(
        gw_df,
        mode=mode,
        team_fixtures=team_fixtures,
        blank_ahead=blank_ahead,
        current_ids=current_ids,
    )


def _transfer_gate_kwargs(mode: EvMode) -> dict[str, Any]:
    return transfer_gate_kwargs(mode)


def _load_average_scores(cache_path: str = GW_SUMMARY_CACHE) -> dict[int, float]:
    cache = Path(cache_path)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(GW_SUMMARY_URL, timeout=60) as resp:  # noqa: S310
            cache.write_bytes(resp.read())
    df = pd.read_csv(cache)
    out: dict[int, float] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("average_entry_score")):
            continue
        out[int(row["id"])] = float(row["average_entry_score"])
    return out


def _score_xi(
    *,
    starting_xi: list[dict[str, Any]],
    bench: list[dict[str, Any]],
    captain_id: int | None,
    points: dict[int, float],
    minutes: dict[int, float],
    chip: str | None,
) -> dict[str, Any]:
    """Score XI with captain multiplier and a simple 0-minute autosub heuristic."""
    from fpl_robo.optimize.bench_order import order_bench_for_autosub

    xi = list(starting_xi)
    bench_q = order_bench_for_autosub(list(bench))
    # Simple autosubs: replace 0-minute starters with first bench player who played
    # (outfield-first order matches live FPL priority; formation-agnostic).
    for i, starter in enumerate(list(xi)):
        sid = int(starter["id"])
        if minutes.get(sid, 0) > 0:
            continue
        for j, b in enumerate(bench_q):
            bid = int(b["id"])
            if minutes.get(bid, 0) <= 0:
                continue
            xi[i] = b
            bench_q.pop(j)
            break

    total = 0.0
    breakdown = []
    for p in xi:
        pid = int(p["id"])
        pts = float(points.get(pid, 0.0))
        mult = 1
        if captain_id is not None and pid == int(captain_id):
            mult = 3 if chip == "3xc" else 2
        scored = pts * mult
        total += scored
        breakdown.append(
            {
                "id": pid,
                "web_name": p.get("web_name"),
                "points": pts,
                "multiplier": mult,
                "scored": scored,
            }
        )

    if chip == "bboost":
        for p in bench_q:
            pid = int(p["id"])
            pts = float(points.get(pid, 0.0))
            total += pts
            breakdown.append(
                {
                    "id": pid,
                    "web_name": p.get("web_name"),
                    "points": pts,
                    "multiplier": 1,
                    "scored": pts,
                    "bench_boost": True,
                }
            )

    return {"points": round(total, 1), "breakdown": breakdown}


def _default_inventory(half_split: int = 19) -> ChipInventory:
    return ChipInventory(
        half_split_gw=half_split,
        windows=[
            ChipWindow("triple_captain", True, False, 1, half_split),
            ChipWindow("bench_boost", True, False, 1, half_split),
            ChipWindow("free_hit", True, False, 1, half_split),
            ChipWindow("wildcard", True, False, 1, half_split),
            ChipWindow("triple_captain", True, False, half_split + 1, 38),
            ChipWindow("bench_boost", True, False, half_split + 1, 38),
            ChipWindow("free_hit", True, False, half_split + 1, 38),
            ChipWindow("wildcard", True, False, half_split + 1, 38),
        ],
    )


def _mark_chip_used(inv: ChipInventory, chip: str, gameweek: int) -> None:
    for w in inv.windows:
        if w.name != chip:
            continue
        if w.start_event is not None and gameweek < w.start_event:
            continue
        if w.stop_event is not None and gameweek > w.stop_event:
            continue
        if w.available:
            w.available = False
            w.used = True
            w.played_by_entry = [gameweek]
            return


def _count_quality_fwds_in_preds(element_ids: list[int], preds: pd.DataFrame) -> int:
    """Count quality FWDs in a backtest squad using shared ATTACK_FLOOR predicate."""
    from fpl_robo.optimize.attack_quality import is_quality_fwd

    by_id = {int(r["id"]): r for _, r in preds.iterrows()}
    n = 0
    for eid in element_ids:
        row = by_id.get(int(eid))
        if row is None:
            continue
        if str(row.get("position") or "").upper() != "FWD":
            continue
        ep = row.get("ep_next")
        if ep is None:
            ep = row.get("xP", row.get("ev"))
        form = None
        if row.get("form") is not None:
            try:
                form = float(row.get("form"))
            except (TypeError, ValueError):
                form = None
        rolling = None
        if row.get("rolling_minutes") is not None:
            try:
                rolling = float(row.get("rolling_minutes"))
            except (TypeError, ValueError):
                rolling = None
        if is_quality_fwd(
            ep_next=ep,
            status=row.get("status"),
            is_focal=bool(row.get("is_focal_attacker")),
            form=form,
            rolling_minutes=rolling,
        ):
            n += 1
    return n


def run_season_backtest(
    season: str = CHIP_BACKTEST_SEASON,
    *,
    max_hits: int = 1,
    use_chips: bool = True,
    half_split_gw: int = CHIP_DEFAULTS["half_split_gw"],
    cache_path: str = CHIP_BACKTEST_CACHE,
    ev_mode: EvMode = "ep_baseline",
    use_tiebreak: bool = False,
    blank_ahead: bool | None = None,
) -> SeasonBacktestResult:
    """Simulate a full season with our initial-team + transfer + chip logic."""
    merged = load_merged_gw(season=season, cache_path=cache_path)
    table = attach_lagged_history(prepare_gw_table(merged))
    team_fixtures = team_fixtures_from_merged(table)
    averages = _load_average_scores()
    gameweeks = sorted(int(g) for g in table["gameweek"].unique())
    gate_kwargs = _transfer_gate_kwargs(ev_mode)
    tb_kwargs = {"use_tiebreak": bool(use_tiebreak)}

    weekly: list[dict[str, Any]] = []
    current_ids: list[int] = []
    bank = 0.0
    selling_prices: dict[int, float] = {}
    free_transfers = 0
    total_points = 0.0
    hits_cost = 0
    transfers_made_total = 0
    chips_played: dict[str, list[int]] = {
        "triple_captain": [],
        "bench_boost": [],
        "free_hit": [],
        "wildcard": [],
    }
    inventory = _default_inventory(half_split_gw)
    hist_tc: list[float] = []
    hist_bb: list[float] = []
    last_starting_xi: list[dict[str, Any]] = []
    last_bench: list[dict[str, Any]] = []
    last_captain_id: int | None = None
    last_squad: list[dict[str, Any]] = []

    for gw in gameweeks:
        gw_df = table[table["gameweek"] == gw].copy()
        preds = _prediction_frame(
            gw_df,
            mode=ev_mode,
            team_fixtures=team_fixtures,
            blank_ahead=blank_ahead,
            current_ids=current_ids or None,
        )
        points = {int(r["id"]): float(r["total_points"]) for _, r in gw_df.iterrows()}
        minutes = {int(r["id"]): float(r["minutes"]) for _, r in gw_df.iterrows()}
        price_map = {int(r["id"]): float(r["price"]) for _, r in gw_df.iterrows()}

        if gw == gameweeks[0] or not current_ids:
            sol = optimise_single_gw(
                preds,
                budget=100.0,
                min_play_chance=None,
                require_cover_eligible=True,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                **tb_kwargs,
            )
            if sol.status != "Optimal":
                weekly.append({"gameweek": gw, "status": sol.status, "points": 0})
                continue
            plan_transfers: list[dict[str, int]] = []
            transfers_made = 0
            hits = 0
            hit_cost = 0
            objective = sol.objective
            squad = sol.squad
            captain_id = sol.captain_id
            bank = round(100.0 - float(sol.budget_used), 1)
            free_transfers = 1  # after GW1 deadline / for GW2
            fallback_used = False
        else:
            # Refresh selling prices from this GW's listed values.
            selling_prices = {pid: float(price_map.get(pid, selling_prices.get(pid, 4.5))) for pid in current_ids}
            plan = optimise_transfers(
                preds,
                current_ids=current_ids,
                bank=bank,
                selling_prices=selling_prices,
                free_transfers=free_transfers,
                max_hits=max_hits,
                min_play_chance=None,
                require_cover_eligible=True,
                bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                **gate_kwargs,
                **tb_kwargs,
            )
            if plan.status != "Optimal":
                # Fallback: keep squad, only reselect XI/captain from owned players.
                owned_preds = preds[preds["id"].isin(current_ids)].copy()
                if len(owned_preds) < 15:
                    for pid in current_ids:
                        if int(pid) in set(owned_preds["id"].astype(int)):
                            continue
                        owned_preds = pd.concat(
                            [
                                owned_preds,
                                pd.DataFrame(
                                    [
                                        {
                                            "id": int(pid),
                                            "web_name": str(pid),
                                            "position": "MID",
                                            "team_id": 1,
                                            "price": float(selling_prices.get(int(pid), 4.5)),
                                            "ev": 0.01,
                                            "status": "a",
                                            "chance_of_playing_next_round": 100,
                                            "cover_eligible": True,
                                            "gameweek": gw,
                                            "total_points": 0,
                                            "minutes": 0,
                                            "xP": 0,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
                # Force all owned eligible for lineup solve.
                owned_preds["cover_eligible"] = True
                owned_preds["ev"] = owned_preds["ev"].astype(float).clip(lower=0.01)
                lineup = optimise_single_gw(
                    owned_preds,
                    budget=1000.0,
                    min_play_chance=None,
                    require_cover_eligible=False,
                    bench_ev_weight=BENCH_COVER["bench_ev_weight"],
                    **tb_kwargs,
                )
                if lineup.status != "Optimal":
                    # Last resort: hold previous XI/bench and score actual points.
                    if not last_starting_xi:
                        weekly.append(
                            {
                                "gameweek": gw,
                                "status": plan.status,
                                "points": 0,
                                "free_transfers": free_transfers,
                                "fallback_failed": True,
                            }
                        )
                        free_transfers = min(5, free_transfers + 1)
                        continue
                    chip = None
                    scored = _score_xi(
                        starting_xi=last_starting_xi,
                        bench=last_bench,
                        captain_id=last_captain_id,
                        points=points,
                        minutes=minutes,
                        chip=chip,
                    )
                    gw_points = float(scored["points"])
                    total_points += gw_points
                    avg = averages.get(gw)
                    free_transfers = min(5, free_transfers + 1)
                    weekly.append(
                        {
                            "gameweek": gw,
                            "status": "Held",
                            "points": round(gw_points, 1),
                            "raw_xi_points": scored["points"],
                            "hit_cost": 0,
                            "hits": 0,
                            "transfers_made": 0,
                            "transfers": [],
                            "free_transfers_next": free_transfers,
                            "bank": bank,
                            "objective": None,
                            "captain_id": last_captain_id,
                            "captain_name": None,
                            "chip": chip,
                            "fallback_lineup_only": True,
                            "average_entry_score": avg,
                            "delta_vs_average": None if avg is None else round(gw_points - avg, 1),
                            "squad": last_squad,
                        }
                    )
                    continue
                sol = lineup
                plan_transfers = []
                transfers_made = 0
                hits = 0
                hit_cost = 0
                objective = lineup.objective
                squad = lineup.squad
                captain_id = lineup.captain_id
                free_transfers = min(5, free_transfers + 1)
                fallback_used = True
            else:
                fallback_used = False
                sol = plan.solution
                plan_transfers = plan.transfers
                transfers_made = plan.transfers_made
                hits = plan.hits
                hit_cost = plan.hit_cost
                objective = plan.objective
                squad = sol.squad
                captain_id = sol.captain_id
                bank = plan.budget_left
                transfers_made_total += transfers_made
                hits_cost += hit_cost
                free_transfers = min(5, max(0, free_transfers - transfers_made) + 1)

        chip = None
        chip_rec = None
        if use_chips:
            n_blank = 0
            dgw_teams: set[int] = set()
            if team_fixtures:
                for tid, by_gw in team_fixtures.items():
                    slot = by_gw.get(int(gw)) or {}
                    if float(slot.get("n_fixtures") or 0) >= 2:
                        dgw_teams.add(int(tid))
            if current_ids:
                n_blank = count_blank_owned(
                    current_ids=current_ids,
                    predictions=preds,
                    team_fixtures=team_fixtures,
                    gameweek=gw,
                )
            chip_cfg, _schedule = chip_schedule_adjustments(
                gameweek=gw,
                starting_xi=sol.starting_xi,
                bench=sol.bench,
                captain_id=captain_id,
                dgw_teams=dgw_teams,
                team_fixtures=team_fixtures,
                blank_owned=n_blank,
            )
            chip_rec = recommend_chips(
                gameweek=gw,
                starting_xi=sol.starting_xi,
                bench=sol.bench,
                captain_id=captain_id,
                inventory=inventory,
                historical_tc_scores=hist_tc,
                historical_bb_scores=hist_bb,
                config=chip_cfg or None,
            )
            hist_tc.append(float(chip_rec.scores["triple_captain"]))
            hist_bb.append(float(chip_rec.scores["bench_boost"]))
            play = chip_rec.play_now
            # Prefer TC then BB for scoring chips in this simplified sim.
            if play.get("triple_captain"):
                chip = "3xc"
                _mark_chip_used(inventory, "triple_captain", gw)
                chips_played["triple_captain"].append(gw)
            elif play.get("bench_boost"):
                chip = "bboost"
                _mark_chip_used(inventory, "bench_boost", gw)
                chips_played["bench_boost"].append(gw)
            elif play.get("wildcard"):
                _mark_chip_used(inventory, "wildcard", gw)
                chips_played["wildcard"].append(gw)
            elif play.get("free_hit"):
                _mark_chip_used(inventory, "free_hit", gw)
                chips_played["free_hit"].append(gw)

        scored = _score_xi(
            starting_xi=sol.starting_xi,
            bench=sol.bench,
            captain_id=captain_id,
            points=points,
            minutes=minutes,
            chip=chip,
        )
        gw_points = float(scored["points"]) - float(hit_cost)
        total_points += gw_points
        avg = averages.get(gw)

        current_ids = [int(p["id"]) for p in squad]
        selling_prices = {pid: float(price_map.get(pid, 4.5)) for pid in current_ids}
        last_starting_xi = list(sol.starting_xi)
        last_bench = list(sol.bench)
        last_captain_id = captain_id
        last_squad = list(squad)

        blank_owned = 0
        blank_owned_next = 0
        if team_fixtures and current_ids:
            blank_owned = count_blank_owned(
                current_ids=current_ids,
                predictions=preds,
                team_fixtures=team_fixtures,
                gameweek=gw,
            )
            blank_owned_next = count_blank_owned(
                current_ids=current_ids,
                predictions=preds,
                team_fixtures=team_fixtures,
                gameweek=int(gw) + 1,
            )

        weekly.append(
            {
                "gameweek": gw,
                "status": "Optimal",
                "points": round(gw_points, 1),
                "raw_xi_points": scored["points"],
                "hit_cost": hit_cost,
                "hits": hits if gw != gameweeks[0] else 0,
                "transfers_made": transfers_made if gw != gameweeks[0] else 0,
                "transfers": plan_transfers if gw != gameweeks[0] else [],
                "free_transfers_next": free_transfers,
                "bank": bank,
                "objective": objective,
                "captain_id": captain_id,
                "captain_name": sol.captain_name,
                "chip": chip,
                "fallback_lineup_only": bool(fallback_used) if gw != gameweeks[0] else False,
                "average_entry_score": avg,
                "delta_vs_average": None if avg is None else round(gw_points - avg, 1),
                "blank_owned": blank_owned,
                "blank_owned_next": blank_owned_next,
                "blank_ahead_triggered": bool(
                    preds["blank_ahead_triggered"].iloc[0]
                    if "blank_ahead_triggered" in preds.columns and len(preds)
                    else False
                ),
                "squad": [
                    {
                        "id": int(p["id"]),
                        "web_name": p.get("web_name"),
                        "position": p.get("position"),
                        "start": bool(p.get("start")),
                        "captain": bool(p.get("captain")),
                        "ev": float(p.get("ev") or 0),
                    }
                    for p in squad
                ],
                "quality_fwds": _count_quality_fwds_in_preds(
                    [int(p["id"]) for p in squad], preds
                ),
            }
        )

    scored_weeks = [w for w in weekly if w.get("status") in {"Optimal", "Held"}]
    avg_total = None
    if averages:
        # Sum averages for weeks we scored; missing GW1 avg is common.
        avg_total = float(sum(averages.get(int(w["gameweek"]), 0.0) for w in scored_weeks))
        # If some weeks missing from averages, scale note in summary.
    vs_avg = None if avg_total is None else round(float(total_points) - avg_total, 1)

    best = max(scored_weeks, key=lambda w: float(w["points"])) if scored_weeks else None
    worst = min(scored_weeks, key=lambda w: float(w["points"])) if scored_weeks else None
    beats = [w for w in scored_weeks if w.get("delta_vs_average") is not None and w["delta_vs_average"] > 0]

    mode_notes = {
        "ep_baseline": "EV = historical xP; soft cover only.",
        "ep_hardened": "EV = historical xP; hard buy/cover gates on minutes + EV floor.",
        "ep_minutes_scaled": "EV = xP * play_prob from lagged minutes (tuned haircut for thin involvement).",
        "open_ev": "EV from lagged rolling points/minutes/price (+ xG/ICT when present); no xP.",
    }
    blank_weeks = [w for w in scored_weeks if int(w.get("blank_owned") or 0) > 0]
    blank_ahead_note = (
        "blank_ahead=on"
        if blank_ahead is True
        else ("blank_ahead=off" if blank_ahead is False else "blank_ahead=config")
    )
    from fpl_robo.config import ATTACK_FLOOR

    need_q = int(ATTACK_FLOOR.get("min_quality_fwds") or 3)
    q_counts = [int(w.get("quality_fwds") or 0) for w in scored_weeks if "quality_fwds" in w]
    summary = {
        "ev_mode": ev_mode,
        "blank_ahead": blank_ahead,
        "use_tiebreak": bool(use_tiebreak),
        "gameweeks_simulated": len(scored_weeks),
        "mean_gw_points": round(float(total_points) / max(1, len(scored_weeks)), 2),
        "best_gw": None
        if best is None
        else {"gameweek": best["gameweek"], "points": best["points"], "chip": best.get("chip")},
        "worst_gw": None
        if worst is None
        else {"gameweek": worst["gameweek"], "points": worst["points"], "chip": worst.get("chip")},
        "weeks_beat_average": len(beats),
        "weeks_with_average": sum(1 for w in scored_weeks if w.get("average_entry_score") is not None),
        "blank_weeks": len(blank_weeks),
        "mean_blank_owned_on_blank_weeks": (
            round(
                sum(int(w.get("blank_owned") or 0) for w in blank_weeks) / len(blank_weeks),
                2,
            )
            if blank_weeks
            else 0.0
        ),
        "points_on_blank_weeks": round(
            sum(float(w.get("points") or 0) for w in blank_weeks), 1
        ),
        "mean_blank_owned_next": round(
            sum(int(w.get("blank_owned_next") or 0) for w in scored_weeks)
            / max(1, len(scored_weeks)),
            2,
        ),
        "blank_ahead_trigger_weeks": sum(
            1 for w in scored_weeks if w.get("blank_ahead_triggered")
        ),
        "mean_quality_fwds": (
            round(sum(q_counts) / len(q_counts), 2) if q_counts else None
        ),
        "weeks_below_attack_floor": sum(1 for q in q_counts if q < need_q),
        "notes": [
            mode_notes.get(str(ev_mode).removesuffix("_h"), mode_notes.get(ev_mode, ev_mode)),
            blank_ahead_note,
            "Selling prices approximated by same-GW listed value.",
            "Autosubs are formation-agnostic (optimistic vs strict FPL rules).",
            "FH/WC are tracked when recommended but not fully resimulated as one-week rebuilds.",
            "Live KG enrich / news signals are not in this harness.",
        ],
    }

    return SeasonBacktestResult(
        season=season,
        total_points=int(round(total_points)),
        average_manager_points=None if avg_total is None else round(avg_total, 1),
        points_vs_average=vs_avg,
        hits_cost=int(hits_cost),
        transfers_made=int(transfers_made_total),
        chips_played=chips_played,
        weekly=weekly,
        summary=summary,
    )


def write_season_report(result: SeasonBacktestResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
