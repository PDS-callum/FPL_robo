"""Command-line entrypoints for FPL Robo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fpl_robo.auth import auth_check, ensure_authenticated, load_auth_config
from fpl_robo.chips.backtest import (
    recommend_chips_for_solution,
    run_chip_backtest,
    write_backtest_report,
)
from fpl_robo.chips.inventory import ChipInventory, ChipWindow, parse_chip_inventory
from fpl_robo.config import (
    CHIP_BACKTEST_SEASON,
    DASHBOARD,
    DEFAULT_MANAGER_ID,
    DEFAULT_SESSION_PATH,
    DEFAULT_SIGNALS_PATH,
    DEFAULT_SUMMARY_PATH,
    EV_MODE,
    PRESEASON_HORIZON,
    SIGNAL_INGEST,
)
from fpl_robo.data.fpl_client import FPLClient
from fpl_robo.graph.builder import build_knowledge_graph
from fpl_robo.pipeline import (
    build_initial_team,
    build_squad,
    initial_team_result_to_dict,
    show_manager,
)


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
        print(f"Manager: {m.name} ({m.manager_id}) | {m.team_name} | bank {m.bank:.1f}m")
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
        print(f"  {p['position']:<3} {p['web_name']:<16} {p['price']:.1f}m  EV {p['ev']:.2f}{flag}")
    print("\nBench:")
    for p in sorted(sol.bench, key=lambda r: (pos_order.get(r["position"], 9), -r["ev"])):
        print(f"  {p['position']:<3} {p['web_name']:<16} {p['price']:.1f}m  EV {p['ev']:.2f}")


def _print_initial_team(result) -> None:
    sol = result.solution
    print(f"Manager: {result.manager.name} ({result.manager.manager_id})")
    print(
        f"Initial-team mode: horizon={result.horizon}, risk={result.risk}, budget={result.budget:.1f}m"
    )
    print(f"Graph: {result.graph_summary} | Gameweek: {result.gameweek}")
    if result.signals:
        s = result.signals
        print(
            f"Signals: {s.get('total', 0)} loaded "
            f"(news={s.get('news', 0)}, preseason={s.get('preseason', 0)}, "
            f"intl={s.get('international', 0)}; matched={s.get('matched_players', 0)})"
        )
    print(f"Solver status: {sol.status}")
    if sol.status != "Optimal":
        return

    print(f"Objective: {sol.objective:.2f}")
    print(f"Captain: {sol.captain_name} ({sol.captain_id})")
    print("\nTop preseason EV candidates:")
    for row in result.top_ev:
        sig = " *" if row.get("signal_applied") else ""
        print(
            f"  {row['position']:<3} {row['web_name']:<16} {row['team_short']:<4} "
            f"{row['price']:.1f}m EV {row['ev']:.2f}{sig}"
        )

    if result.execution:
        print("\nExecution:")
        print(json.dumps(result.execution, indent=2))


def _write_summary(path: str, payload: dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def cmd_build_initial_team(args: argparse.Namespace) -> int:
    client = FPLClient()
    if args.execute or args.load_session or args.save_session:
        cfg = load_auth_config()
        auth_result = ensure_authenticated(
            client,
            cfg,
            load_session=args.load_session,
            save_session=args.save_session,
        )
        if not auth_result.get("ok"):
            print(f"Auth failed: {auth_result.get('reason')}")
            return 1

    result = build_initial_team(
        manager_id=args.manager_id,
        budget=args.budget,
        horizon=args.horizon,
        risk=args.risk,
        execute=args.execute,
        signals_path=args.signals,
        client=client,
    )

    payload = initial_team_result_to_dict(result)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_initial_team(result)

    if args.summary_file:
        _write_summary(args.summary_file, payload)

    if args.execute:
        ok = bool(result.execution and result.execution.get("saved"))
        return 0 if ok else 1
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


def cmd_auth_check(args: argparse.Namespace) -> int:
    result = auth_check(
        load_session=args.load_session,
        save_session=args.save_session,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if not result.get("ok"):
            print(f"Auth check failed: {result.get('reason')}")
            return 1
        print(f"Auth check passed via: {result.get('method')}")
        print(f"Logged-in entry: {result.get('entry_id')}")
        print(f"Expected manager id: {result.get('expected_manager_id')}")
        print(f"Matches expected manager id: {result.get('matches_expected_manager_id')}")
        print(f"Session path: {result.get('session_path')}")
    return 0


def cmd_auth_refresh(args: argparse.Namespace) -> int:
    from fpl_robo.oidc import refresh_session_file

    result = refresh_session_file(args.session_path, force=args.force)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if not result.get("ok"):
            print(f"Auth refresh failed: {result.get('reason')}")
            return 1
        if result.get("refreshed"):
            print(f"Refreshed OK; entry={result.get('entry_id')} expires_at={result.get('expires_at')}")
        else:
            print(
                f"Token still valid; seconds_remaining={result.get('seconds_remaining'):.0f} "
                f"expires_at={result.get('expires_at')}"
            )
    return 0


def cmd_auth_refresh_daemon(args: argparse.Namespace) -> int:
    from fpl_robo.oidc import run_refresh_daemon

    run_refresh_daemon(
        session_path=args.session_path,
        skew_seconds=args.skew_seconds,
        poll_seconds=args.poll_seconds,
    )
    return 0


def cmd_auth_capture(args: argparse.Namespace) -> int:
    """Delegate to headed Playwright OAuth capture helper."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "tools" / "capture_session.py"
    spec = importlib.util.spec_from_file_location("capture_session", script)
    if spec is None or spec.loader is None:
        print(f"Could not load capture helper: {script}")
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.capture(
        session_path=args.session_path,
        timeout_seconds=args.timeout,
        url=args.url,
    )


def cmd_recommend_chips(args: argparse.Namespace) -> int:
    """Recommend chips for the latest optimised / live squad summary."""
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"Missing summary file: {summary_path}. Run build-initial-team first.")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sol = summary.get("solution") or {}
    squad = sol.get("squad") or []
    if len(squad) != 15:
        print("Summary does not contain a 15-man squad.")
        return 1

    gameweek = int(summary.get("gameweek") or sol.get("gameweek") or 1)
    inventory = ChipInventory(
        windows=[
            ChipWindow("triple_captain", True, False, 1, 19),
            ChipWindow("bench_boost", True, False, 1, 19),
            ChipWindow("free_hit", True, False, 1, 19),
            ChipWindow("wildcard", True, False, 1, 19),
            ChipWindow("triple_captain", True, False, 20, 38),
            ChipWindow("bench_boost", True, False, 20, 38),
            ChipWindow("free_hit", True, False, 20, 38),
            ChipWindow("wildcard", True, False, 20, 38),
        ]
    )

    # Optional: enrich inventory from authenticated my-team.
    if args.load_session:
        client = FPLClient()
        cfg = load_auth_config()
        auth_result = ensure_authenticated(
            client, cfg, load_session=True, save_session=args.save_session
        )
        if auth_result.get("ok"):
            try:
                inventory = parse_chip_inventory(client.my_team(args.manager_id))
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not load live chip inventory ({exc})")

    payload = recommend_chips_for_solution(
        gameweek=gameweek,
        solution_squad=squad,
        captain_id=sol.get("captain_id"),
        inventory=inventory,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Chip recommendations for GW{payload['gameweek']}")
        for chip, play in payload["play_now"].items():
            score = payload["scores"].get(chip, 0.0)
            print(f"  {chip:<15} {'PLAY' if play else 'hold':<4}  score={score:.2f}")
            print(f"    {payload['reasons'].get(chip)}")
    return 0


def cmd_backtest_chips(args: argparse.Namespace) -> int:
    print(f"Running chip backtest for season {args.season} (this can take a few minutes)...")
    result = run_chip_backtest(season=args.season, half_split_gw=args.half_split)
    write_backtest_report(result, args.report_file)
    payload = {
        "season": result.season,
        "verdict": result.verdict,
        "triple_captain": {
            k: v
            for k, v in result.triple_captain.items()
            if k in {"avg_capture_vs_oracle", "top_quartile_hit_rate", "halves", "chip"}
        },
        "bench_boost": {
            k: v
            for k, v in result.bench_boost.items()
            if k in {"avg_capture_vs_oracle", "top_quartile_hit_rate", "halves", "chip"}
        },
        "free_hit": result.free_hit,
        "wildcard": {k: v for k, v in result.wildcard.items() if k != "flagged_gws"}
        | {"flagged_gws": (result.wildcard.get("flagged_gws") or [])[:8]},
        "report_file": args.report_file,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        v = result.verdict
        print(f"Season {result.season} chip backtest: {v['summary']} ({v['passed']}/{v['total']} checks)")
        print(
            f"TC capture vs oracle: {result.triple_captain.get('avg_capture_vs_oracle')} | "
            f"top-quartile hit rate: {result.triple_captain.get('top_quartile_hit_rate')}"
        )
        for half, info in (result.triple_captain.get("halves") or {}).items():
            print(
                f"  TC {half}: rec GW{info['recommended_gw']} "
                f"(actual {info['recommended_actual']}) vs oracle GW{info['oracle_gw']} "
                f"({info['oracle_actual']})"
            )
        print(
            f"BB capture vs oracle: {result.bench_boost.get('avg_capture_vs_oracle')} | "
            f"top-quartile hit rate: {result.bench_boost.get('top_quartile_hit_rate')}"
        )
        for half, info in (result.bench_boost.get("halves") or {}).items():
            print(
                f"  BB {half}: rec GW{info['recommended_gw']} "
                f"(actual {info['recommended_actual']}) vs oracle GW{info['oracle_gw']} "
                f"({info['oracle_actual']})"
            )
        fh = result.free_hit
        print(
            f"FH flagged weeks: {fh.get('n_flagged')} | "
            f"lift vs mean: {fh.get('lift_vs_mean')} | "
            f"precision_vs_mean: {fh.get('precision_vs_mean')}"
        )
        print(f"Full report: {args.report_file}")
    return 0 if result.verdict.get("competent") else 1


def cmd_backtest_season(args: argparse.Namespace) -> int:
    from fpl_robo.backtest.season import run_season_backtest, write_season_report

    print(
        f"Running full-season backtest for {args.season} "
        f"(ev_mode={args.ev_mode}; this can take a few minutes)..."
    )
    result = run_season_backtest(
        season=args.season,
        max_hits=args.max_hits,
        use_chips=not args.no_chips,
        ev_mode=args.ev_mode,
    )
    write_season_report(result, args.report_file)
    payload = {
        "season": result.season,
        "total_points": result.total_points,
        "average_manager_points": result.average_manager_points,
        "points_vs_average": result.points_vs_average,
        "hits_cost": result.hits_cost,
        "transfers_made": result.transfers_made,
        "chips_played": result.chips_played,
        "summary": result.summary,
        "weekly_tail": result.weekly[-5:],
        "report_file": args.report_file,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Season {result.season} [{args.ev_mode}]")
        print(f"Total points: {result.total_points}")
        if result.average_manager_points is not None:
            print(
                f"Average manager (sum of GW averages): {result.average_manager_points} | "
                f"delta: {result.points_vs_average:+}"
            )
        print(f"Transfers: {result.transfers_made} | Hit cost: -{result.hits_cost}")
        print(f"Chips played: {result.chips_played}")
        s = result.summary
        print(
            f"Mean GW: {s.get('mean_gw_points')} | "
            f"Beat average in {s.get('weeks_beat_average')}/{s.get('weeks_with_average')} weeks"
        )
        if s.get("best_gw"):
            print(f"Best GW: {s['best_gw']}")
        if s.get("worst_gw"):
            print(f"Worst GW: {s['worst_gw']}")
        print("Recent weeks:")
        for w in result.weekly[-8:]:
            if w.get("status") != "Optimal":
                continue
            delta = w.get("delta_vs_average")
            delta_s = "" if delta is None else f" ({delta:+} vs avg)"
            chip = f" chip={w['chip']}" if w.get("chip") else ""
            print(
                f"  GW{w['gameweek']:>2}: {w['points']:>5} pts | "
                f"tr={w.get('transfers_made', 0)} hits={w.get('hits', 0)}{chip}{delta_s}"
            )
        print(f"Full report: {args.report_file}")
    return 0



def cmd_compare_ev_models(args: argparse.Namespace) -> int:
    from fpl_robo.backtest.season import run_season_backtest
    from fpl_robo.predict.ev_modes import EV_MODES, select_winner

    modes = list(args.modes) if args.modes else list(EV_MODES)
    rows: list[dict] = []
    for mode in modes:
        print(f"=== Running {mode} on {args.season} ===")
        result = run_season_backtest(
            season=args.season,
            max_hits=args.max_hits,
            use_chips=not args.no_chips,
            ev_mode=mode,
        )
        row = {
            "ev_mode": mode,
            "total_points": result.total_points,
            "average_manager_points": result.average_manager_points,
            "points_vs_average": result.points_vs_average,
            "hits_cost": result.hits_cost,
            "transfers_made": result.transfers_made,
            "chips_played": result.chips_played,
            "summary": result.summary,
        }
        rows.append(row)
        print(
            f"{mode}: {result.total_points} pts | vs avg {result.points_vs_average} | "
            f"hits -{result.hits_cost} | transfers {result.transfers_made}"
        )

    winner = select_winner(rows)
    report = {
        "season": args.season,
        "selection_rule": "highest total_points; then points_vs_average; then fewer hits_cost",
        "results": rows,
        "winner": winner["ev_mode"],
        "winner_metrics": {
            "total_points": winner["total_points"],
            "points_vs_average": winner["points_vs_average"],
            "hits_cost": winner["hits_cost"],
            "transfers_made": winner["transfers_made"],
        },
        "live_ev_mode_config": EV_MODE,
    }
    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print()
        print(f"{'mode':<14} {'pts':>6} {'vs_avg':>8} {'hits':>6} {'xfers':>6}")
        for r in rows:
            print(
                f"{r['ev_mode']:<14} {r['total_points']:>6} "
                f"{str(r['points_vs_average']):>8} {r['hits_cost']:>6} {r['transfers_made']:>6}"
            )
        print(f"\nWinner: {winner['ev_mode']}")
        print(f"Report: {out}")
        if EV_MODE != winner["ev_mode"]:
            print(
                f"Note: config EV_MODE is currently '{EV_MODE}'. "
                f"Update to '{winner['ev_mode']}' to go live."
            )
    return 0


def cmd_run_week(args: argparse.Namespace) -> int:
    from fpl_robo.runtime.season_runner import run_week

    result = run_week(
        manager_id=args.manager_id,
        execute=args.execute,
        force=args.force,
        max_hits=args.max_hits,
        signals_path=args.signals,
        summary_file=args.summary_file,
        load_session=True,
        save_session=True,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"run-week: ok={result.ok} skipped={result.skipped} "
            f"gw={result.gameweek} hours_to_deadline={result.hours_to_deadline}"
        )
        if result.reason:
            print(f"reason: {result.reason}")
        if result.transfer_plan:
            tp = result.transfer_plan
            print(
                f"transfers={tp.get('transfers_made')} hits={tp.get('hits')} "
                f"obj={tp.get('objective')} status={tp.get('status')}"
            )
            for t in tp.get("transfers") or []:
                print(f"  out {t.get('element_out')} -> in {t.get('element_in')}")
        if result.chips:
            play = result.chips.get("play_now") or {}
            print("chips:", ", ".join(f"{k}={'PLAY' if v else 'hold'}" for k, v in play.items()))
        if result.execution:
            print("execution:", json.dumps(result.execution, indent=2))
    if result.skipped:
        return 0
    return 0 if result.ok else 1


def cmd_run_daemon(args: argparse.Namespace) -> int:
    from fpl_robo.runtime.season_runner import run_daemon

    run_daemon(
        poll_seconds=args.poll_seconds,
        execute=args.execute,
        manager_id=args.manager_id,
    )
    return 0


def cmd_run_dashboard(args: argparse.Namespace) -> int:
    from fpl_robo.dashboard import run_dashboard

    run_dashboard(host=args.host, port=args.port, manager_id=args.manager_id)
    return 0


def cmd_ingest_signals(args: argparse.Namespace) -> int:
    from fpl_robo.signals.ingest import ingest_signals
    from fpl_robo.signals.pre_deadline import maybe_run_pre_deadline_ingest

    if args.pre_deadline or args.force_pre_deadline:
        wrapped = maybe_run_pre_deadline_ingest(
            apply=args.apply,
            force=bool(args.force_pre_deadline),
        )
        if wrapped.get("skipped") and not args.force_pre_deadline:
            print(
                f"Pre-deadline ingest skipped: {wrapped.get('reason')} "
                f"(gw={wrapped.get('gameweek')} hours={wrapped.get('hours_to_deadline')})"
            )
            if args.json:
                print(json.dumps(wrapped, indent=2))
            return 0
        result_payload = wrapped.get("ingest") or {}
        # Fall through to pretty-print via a lightweight shim if ingest ran.
        from fpl_robo.signals.ingest import IngestResult

        result = IngestResult(
            ok=bool(wrapped.get("ok")),
            dry_run=not args.apply,
            applied=bool((result_payload or {}).get("applied")),
            articles=int((result_payload or {}).get("articles") or 0),
            proposed_news=list((result_payload or {}).get("proposed_news") or []),
            proposed_lineups=list((result_payload or {}).get("proposed_lineups") or []),
            rejected=list((result_payload or {}).get("rejected") or []),
            unresolved=list((result_payload or {}).get("unresolved") or []),
            feed_errors=list((result_payload or {}).get("feed_errors") or []),
            signals_path=(result_payload or {}).get("signals_path"),
            raw_cache_path=(result_payload or {}).get("raw_cache_path"),
            reason=(result_payload or {}).get("reason") or wrapped.get("reason"),
            ollama_model=(result_payload or {}).get("ollama_model"),
            pre_deadline=True,
            gameweek=wrapped.get("gameweek"),
        )
    else:
        result = ingest_signals(
            dry_run=not args.apply,
            apply=args.apply,
            signals_path=args.signals,
            max_articles=args.max_articles,
            model=args.model,
            ollama_url=args.ollama_url,
            skip_llm=args.skip_llm,
            pre_deadline=False,
        )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if result.ok else 1

    mode = "pre-deadline" if result.pre_deadline else "normal"
    print(f"Mode: {mode}" + (f" GW{result.gameweek}" if result.gameweek else ""))
    print(f"Articles fetched: {result.articles}")
    if result.feed_errors:
        print(f"Feed errors: {len(result.feed_errors)}")
        for err in result.feed_errors[:5]:
            print(f"  - {err.get('source')}: {err.get('error')}")
    if result.reason:
        print(result.reason)
    print(f"Proposed news signals: {len(result.proposed_news)}")
    for sig in result.proposed_news[:20]:
        flag = " EXCLUDE" if sig.get("exclude") else ""
        conf = sig.get("confidence")
        conf_s = f" conf={conf}" if conf is not None else ""
        team = f" {sig.get('team_short')}" if sig.get("team_short") else ""
        print(
            f"  - {sig.get('first_name') or ''} {sig.get('web_name')}{team} ({sig.get('player_id')}) "
            f"{sig.get('type')} sev={sig.get('severity')}{conf_s}{flag}"
        )
        if sig.get("note"):
            print(f"      {sig.get('note')}")
        if sig.get("evidence"):
            print(f"      evidence: {sig.get('evidence')}")
    print(f"Proposed predicted lineups: {len(result.proposed_lineups)}")
    for sig in result.proposed_lineups[:25]:
        team = f" {sig.get('team_short')}" if sig.get("team_short") else ""
        print(
            f"  - {sig.get('first_name') or ''} {sig.get('web_name')}{team} "
            f"role={sig.get('role')} conf={sig.get('confidence')}"
        )
    if result.rejected:
        print(f"Rejected by KG/evidence: {len(result.rejected)}")
        for item in result.rejected[:10]:
            print(
                f"  - {item.get('web_name')} {item.get('type') or item.get('role')} "
                f"reason={item.get('reason')} conf={item.get('confidence')}"
            )
    if result.unresolved:
        print(f"Unresolved: {len(result.unresolved)}")
        for item in result.unresolved[:10]:
            print(f"  - {item}")
    if result.raw_cache_path:
        print(f"Raw cache: {result.raw_cache_path}")
    if result.applied:
        print(f"Wrote: {result.signals_path}")
    elif not args.apply and result.ok and not args.skip_llm:
        print("Dry-run only. Re-run with --apply to write data/signals.json")
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fpl_robo", description="FPL Robo 2026/27")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-squad", help="Build an optimal single-GW squad from scratch")
    build.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    build.add_argument("--budget", type=float, default=None)
    build.add_argument("--min-play-chance", type=int, default=50)
    build.add_argument("--json", action="store_true")
    build.set_defaults(func=cmd_build_squad)

    init = sub.add_parser("build-initial-team", help="Build preseason initial squad")
    init.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    init.add_argument("--budget", type=float, default=100.0)
    init.add_argument("--horizon", type=int, default=PRESEASON_HORIZON)
    init.add_argument(
        "--risk",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
    )
    init.add_argument("--execute", action="store_true", help="Apply picks to FPL")
    init.add_argument(
        "--signals",
        default=DEFAULT_SIGNALS_PATH,
        help="Path to preseason/news/international signals JSON",
    )
    init.add_argument("--load-session", action="store_true", help="Load saved session before auth")
    init.add_argument("--save-session", action="store_true", help="Save session after successful auth")
    init.add_argument("--summary-file", default=DEFAULT_SUMMARY_PATH)
    init.add_argument("--json", action="store_true")
    init.set_defaults(func=cmd_build_initial_team)

    show = sub.add_parser("show-team", help="Show the configured manager's current team")
    show.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show_team)

    auth = sub.add_parser("auth-check", help="Check FPL login from .env credentials")
    auth.add_argument("--load-session", action="store_true", help="Try saved session first")
    auth.add_argument("--save-session", action="store_true", help="Save session after successful auth")
    auth.add_argument("--json", action="store_true")
    auth.set_defaults(func=cmd_auth_check)

    refresh = sub.add_parser("auth-refresh", help="Refresh PingOne OIDC access token in session.json")
    refresh.add_argument("--session-path", default=DEFAULT_SESSION_PATH)
    refresh.add_argument("--force", action="store_true", help="Refresh even if token is still valid")
    refresh.add_argument("--json", action="store_true")
    refresh.set_defaults(func=cmd_auth_refresh)

    refresh_daemon = sub.add_parser(
        "auth-refresh-daemon",
        help="Background loop: refresh OIDC token before it expires",
    )
    refresh_daemon.add_argument("--session-path", default=DEFAULT_SESSION_PATH)
    refresh_daemon.add_argument(
        "--skew-seconds",
        type=int,
        default=600,
        help="Refresh this many seconds before expiry (default 600)",
    )
    refresh_daemon.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Minimum seconds between refresh checks",
    )
    refresh_daemon.set_defaults(func=cmd_auth_refresh_daemon)

    capture = sub.add_parser(
        "auth-capture",
        help="Open headed browser, complete OAuth, save data/session.json",
    )
    capture.add_argument("--session-path", default=DEFAULT_SESSION_PATH)
    capture.add_argument("--timeout", type=int, default=300)
    capture.add_argument("--url", default="https://fantasy.premierleague.com/")
    capture.set_defaults(func=cmd_auth_capture)

    chips = sub.add_parser("recommend-chips", help="Recommend chip usage for current summary squad")
    chips.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    chips.add_argument("--summary-file", default=DEFAULT_SUMMARY_PATH)
    chips.add_argument("--load-session", action="store_true")
    chips.add_argument("--save-session", action="store_true")
    chips.add_argument("--json", action="store_true")
    chips.set_defaults(func=cmd_recommend_chips)

    bt = sub.add_parser("backtest-chips", help="Backtest chip timing on historical season data")
    bt.add_argument("--season", default=CHIP_BACKTEST_SEASON)
    bt.add_argument("--half-split", type=int, default=19)
    bt.add_argument("--report-file", default="data/chip_backtest_report.json")
    bt.add_argument("--json", action="store_true")
    bt.set_defaults(func=cmd_backtest_chips)

    season_bt = sub.add_parser("backtest-season", help="Full-season points backtest on historical data")
    season_bt.add_argument("--season", default=CHIP_BACKTEST_SEASON)
    season_bt.add_argument("--max-hits", type=int, default=1)
    season_bt.add_argument("--no-chips", action="store_true", help="Disable chip plays in simulation")
    season_bt.add_argument(
        "--ev-mode",
        default="ep_baseline",
        choices=["ep_baseline", "ep_hardened", "ep_minutes_scaled", "open_ev"],
        help="EV prediction mode",
    )
    season_bt.add_argument("--report-file", default="data/season_backtest_report.json")
    season_bt.add_argument("--json", action="store_true")
    season_bt.set_defaults(func=cmd_backtest_season)

    cmp = sub.add_parser(
        "compare-ev-models",
        help="Backtest EV modes (baseline / hardened / minutes_scaled / open) and pick a winner",
    )
    cmp.add_argument("--season", default=CHIP_BACKTEST_SEASON)
    cmp.add_argument("--max-hits", type=int, default=1)
    cmp.add_argument("--no-chips", action="store_true")
    cmp.add_argument(
        "--modes",
        nargs="+",
        choices=["ep_baseline", "ep_hardened", "ep_minutes_scaled", "open_ev"],
        default=None,
    )
    cmp.add_argument("--report-file", default="data/ev_model_comparison.json")
    cmp.add_argument("--json", action="store_true")
    cmp.set_defaults(func=cmd_compare_ev_models)

    week = sub.add_parser("run-week", help="Weekly season cycle: transfers + chips (+ optional execute)")
    week.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    week.add_argument("--execute", action="store_true", help="Apply transfers/lineup/chip if in window")
    week.add_argument("--force", action="store_true", help="Ignore deadline window / already-run guards")
    week.add_argument("--max-hits", type=int, default=None)
    week.add_argument("--signals", default=DEFAULT_SIGNALS_PATH)
    week.add_argument("--summary-file", default=DEFAULT_SUMMARY_PATH)
    week.add_argument("--json", action="store_true")
    week.set_defaults(func=cmd_run_week)

    daemon = sub.add_parser("run-daemon", help="Unattended loop for Raspberry Pi season operation")
    daemon.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    daemon.add_argument("--execute", action="store_true", help="Allow executes inside deadline window")
    daemon.add_argument("--poll-seconds", type=int, default=None)
    daemon.set_defaults(func=cmd_run_daemon)

    dash = sub.add_parser("run-dashboard", help="LAN web dashboard for bot status and plans")
    dash.add_argument("--host", default=DASHBOARD["host"])
    dash.add_argument("--port", type=int, default=DASHBOARD["port"])
    dash.add_argument("--manager-id", type=int, default=DEFAULT_MANAGER_ID)
    dash.set_defaults(func=cmd_run_dashboard)

    ingest = sub.add_parser(
        "ingest-signals",
        help="Scrape football RSS and interpret with local Ollama into signals.json",
    )
    ingest.add_argument("--signals", default=SIGNAL_INGEST["signals_path"])
    ingest.add_argument("--max-articles", type=int, default=SIGNAL_INGEST["max_articles"])
    ingest.add_argument("--model", default=SIGNAL_INGEST["ollama_model"])
    ingest.add_argument("--ollama-url", default=SIGNAL_INGEST["ollama_url"])
    ingest.add_argument(
        "--apply",
        action="store_true",
        help="Write interpreted news into signals.json (default is dry-run)",
    )
    ingest.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only fetch/cache RSS articles; do not call Ollama",
    )
    ingest.add_argument(
        "--pre-deadline",
        action="store_true",
        help="Run only if within ~2h of the next deadline (lineup-focused scrape)",
    )
    ingest.add_argument(
        "--force-pre-deadline",
        action="store_true",
        help="Force lineup-focused pre-deadline scrape regardless of deadline window",
    )
    ingest.add_argument("--json", action="store_true")
    ingest.set_defaults(func=cmd_ingest_signals)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
