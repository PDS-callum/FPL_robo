"""Command-line entrypoints for FPL Robo."""

from __future__ import annotations

import argparse
import json
import sys

from fpl_robo.config import DEFAULT_MANAGER_ID
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.pipeline import build_squad, show_manager


def _print_manager(manager) -> None:
    print(f"Manager: {manager.name} ({manager.manager_id})")
    print(f"Team: {manager.team_name}")
    print(f"Gameweek picks: {manager.gameweek}")
    print(f"Bank: {manager.bank:.1f}m | Squad value: {manager.team_value:.1f}m")
    print(f"Points: {manager.overall_points} | Rank: {manager.overall_rank}")
    if not manager.picks:
        print("No picks available yet for this season.")
        return

    kg = build_knowledge_graph()
    players = {p["id"]: p for p in kg.players()}
    print("\nCurrent squad:")
    for pick in sorted(manager.picks, key=lambda p: p.get("position", 99)):
        pid = int(pick["element"])
        info = players.get(pid, {})
        role = []
        if pick.get("is_captain"):
            role.append("C")
        if pick.get("is_vice_captain"):
            role.append("V")
        role_s = f" ({'/'.join(role)})" if role else ""
        print(
            f"  {pick.get('position', '?'):>2}. "
            f"{info.get('position', '?'):<3} "
            f"{info.get('web_name', pid):<16} "
            f"{float(info.get('price') or 0):.1f}m{role_s}"
        )


def _print_squad(result) -> None:
    if result.manager is not None:
        m = result.manager
        print(
            f"Manager: {m.name} ({m.manager_id}) | "
            f"{m.team_name} | bank {m.bank:.1f}m"
        )
    sol = result.solution
    print(f"Knowledge graph: {result.graph_summary}")
    print(f"Gameweek: {result.gameweek}")
    print(f"Solver status: {sol.status}")
    if sol.status != "Optimal":
        return

    pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    print(f"Objective (XI EV + captain): {sol.objective:.2f}")
    print(f"Budget used: {sol.budget_used:.1f}m")
    print(f"Captain: {sol.captain_name} ({sol.captain_id})")
    print("\nStarting XI:")
    for p in sorted(sol.starting_xi, key=lambda r: (pos_order.get(r["position"], 9), -r["ev"])):
        flag = " (C)" if p["captain"] else ""
        print(
            f"  {p['position']:<3} {p['web_name']:<16} "
            f"{p['price']:.1f}m  EV {p['ev']:.2f}{flag}"
        )
    print("\nBench:")
    for p in sorted(sol.bench, key=lambda r: (pos_order.get(r["position"], 9), -r["ev"])):
        print(
            f"  {p['position']:<3} {p['web_name']:<16} "
            f"{p['price']:.1f}m  EV {p['ev']:.2f}"
        )


def cmd_build_squad(args: argparse.Namespace) -> int:
    result = build_squad(
        budget=args.budget,
        min_play_chance=None if args.min_play_chance < 0 else args.min_play_chance,
        manager_id=args.manager_id,
    )
    if args.json:
        payload = {
            "manager": None
            if result.manager is None
            else {
                "manager_id": result.manager.manager_id,
                "name": result.manager.name,
                "team_name": result.manager.team_name,
                "bank": result.manager.bank,
                "team_value": result.manager.team_value,
                "player_ids": result.manager.player_ids,
            },
            "graph_summary": result.graph_summary,
            "gameweek": result.gameweek,
            "top_ev": result.top_ev,
            "solution": result.solution.to_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_squad(result)
    return 0 if result.solution.status == "Optimal" else 1


def cmd_show_team(args: argparse.Namespace) -> int:
    manager = show_manager(manager_id=args.manager_id)
    if args.json:
        print(
            json.dumps(
                {
                    "manager_id": manager.manager_id,
                    "name": manager.name,
                    "team_name": manager.team_name,
                    "bank": manager.bank,
                    "team_value": manager.team_value,
                    "overall_points": manager.overall_points,
                    "overall_rank": manager.overall_rank,
                    "gameweek": manager.gameweek,
                    "player_ids": manager.player_ids,
                    "picks": manager.picks,
                },
                indent=2,
            )
        )
    else:
        _print_manager(manager)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpl_robo", description="FPL Robo 2026/27")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-squad", help="Build an optimal single-GW squad from scratch")
    build.add_argument(
        "--manager-id",
        type=int,
        default=DEFAULT_MANAGER_ID,
        help=f"FPL manager ID (default: {DEFAULT_MANAGER_ID})",
    )
    build.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget in millions (default: manager team value + bank, else 100.0)",
    )
    build.add_argument(
        "--min-play-chance",
        type=int,
        default=50,
        help="Min chance_of_playing_next_round (use -1 to disable)",
    )
    build.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    build.set_defaults(func=cmd_build_squad)

    show = sub.add_parser("show-team", help="Show the configured manager's current team")
    show.add_argument(
        "--manager-id",
        type=int,
        default=DEFAULT_MANAGER_ID,
        help=f"FPL manager ID (default: {DEFAULT_MANAGER_ID})",
    )
    show.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    show.set_defaults(func=cmd_show_team)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
