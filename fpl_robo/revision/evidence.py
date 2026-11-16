"""Build the evidence pack the revision run reviews for one finished gameweek."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fpl_robo.data.fpl_client import FPLClient

_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ (\w+) (\S+): (.*)$")
_EXC_LINE = re.compile(r"^([\w.]+(?:Error|Exception|Timeout|Interrupt)):\s*(.*)$")
_DECISION = re.compile(
    r"Optimiser=|Path plan|Shadow MIP|Weekly run complete|[Ee]xecut|[Cc]hip|[Cc]aptain|"
    r"Transfer|Hub priors|Lineup|Daemon cycle: ok=False"
)
_NUMBERS = re.compile(r"\d+(?:\.\d+)?")

MAX_DECISION_LINES = 60
MAX_DISTINCT_PROBLEMS = 40


def finished_gameweeks(bootstrap: dict[str, Any]) -> list[int]:
    """Gameweeks whose points are final (finished and data_checked)."""
    return [
        int(ev["id"])
        for ev in bootstrap.get("events") or []
        if ev.get("finished") and ev.get("data_checked")
    ]


def _event(bootstrap: dict[str, Any], gw: int) -> dict[str, Any]:
    for ev in bootstrap.get("events") or []:
        if int(ev.get("id", 0)) == gw:
            return ev
    return {}


def _deadline_local(ev: dict[str, Any]) -> datetime | None:
    raw = ev.get("deadline_time")
    if not raw:
        return None
    utc = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    # The bot log uses naive local time.
    return utc.astimezone().replace(tzinfo=None)


def live_result(client: FPLClient, bootstrap: dict[str, Any], manager_id: int, gw: int) -> dict[str, Any]:
    ev = _event(bootstrap, gw)
    history = client.entry_history(manager_id)
    row = next((h for h in history.get("current") or [] if int(h.get("event", 0)) == gw), {})
    points = row.get("points")
    hits = row.get("event_transfers_cost") or 0
    average = ev.get("average_entry_score")
    net = (points - hits) if points is not None else None
    return {
        "points": points,
        "hits_cost": hits,
        "net_points": net,
        "average": average,
        "delta_vs_average": (net - average) if net is not None and average is not None else None,
        "highest": ev.get("highest_score"),
        "gw_rank": row.get("rank"),
        "overall_rank": row.get("overall_rank"),
        "transfers": row.get("event_transfers"),
        "bench_points": row.get("points_on_bench"),
        "chips": [c for c in history.get("chips") or [] if int(c.get("event", 0)) == gw],
    }


def _picks_table(client: FPLClient, bootstrap: dict[str, Any], manager_id: int, gw: int) -> list[str]:
    names = {int(e["id"]): e for e in bootstrap.get("elements") or []}
    teams = {int(t["id"]): t.get("short_name") for t in bootstrap.get("teams") or []}
    positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    picks = client.entry_picks(manager_id, gw)
    live = {int(e["id"]): e.get("stats") or {} for e in client.event_live(gw).get("elements") or []}
    lines = [
        f"Active chip: {picks.get('active_chip') or 'none'}",
        "",
        "| Slot | Player | Team | Pos | Mins | Pts | Mult | Counted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in picks.get("picks") or []:
        el = names.get(int(p["element"]), {})
        stats = live.get(int(p["element"]), {})
        pts = int(stats.get("total_points") or 0)
        mult = int(p.get("multiplier") or 0)
        tag = " (C)" if p.get("is_captain") else (" (VC)" if p.get("is_vice_captain") else "")
        lines.append(
            f"| {p.get('position')} | {el.get('web_name', p['element'])}{tag} | {teams.get(el.get('team'), '?')} "
            f"| {positions.get(el.get('element_type'), '?')} | {stats.get('minutes', 0)} | {pts} | {mult} | {pts * mult} |"
        )
    subs = picks.get("automatic_subs") or []
    if subs:
        lines.append("")
        lines.append(
            "Automatic subs: "
            + ", ".join(
                f"{names.get(s['element_out'], {}).get('web_name', s['element_out'])} → "
                f"{names.get(s['element_in'], {}).get('web_name', s['element_in'])}"
                for s in subs
            )
        )
    return lines


def _log_window(log_path: Path, start: datetime, end: datetime) -> tuple[list[str], Counter, dict[str, str]]:
    """Decision lines plus de-duplicated warnings/errors/exceptions in [start, end]."""
    decisions: list[str] = []
    problems: Counter = Counter()
    examples: dict[str, str] = {}
    if not log_path.exists():
        return decisions, problems, examples
    in_window = False
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = _LOG_TS.match(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                in_window = start <= ts <= end
                if not in_window:
                    continue
                level, msg = m.group(2), m.group(4)
                if level in ("WARNING", "ERROR", "CRITICAL"):
                    key = f"{level} {m.group(3)}: {_NUMBERS.sub('#', msg)[:160]}"
                    problems[key] += 1
                    examples.setdefault(key, line[:400])
                elif _DECISION.search(msg):
                    decisions.append(line[:400])
                continue
            if in_window:
                exc = _EXC_LINE.match(line.strip())
                if exc:
                    key = f"EXCEPTION {exc.group(1)}: {_NUMBERS.sub('#', exc.group(2))[:160]}"
                    problems[key] += 1
                    examples.setdefault(key, line.strip()[:400])
    return decisions, problems, examples


def build(
    *,
    client: FPLClient,
    bootstrap: dict[str, Any],
    manager_id: int,
    gw: int,
    season: str,
    log_path: Path,
    season_state: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return (markdown evidence, live result dict)."""
    ev = _event(bootstrap, gw)
    prev = _event(bootstrap, gw - 1)
    nxt = _event(bootstrap, gw + 1)
    deadline = _deadline_local(ev)
    window_start = _deadline_local(prev) or (deadline - timedelta(days=7) if deadline else None)
    window_end = _deadline_local(nxt) or datetime.now()
    window_end = min(window_end, datetime.now())

    live = live_result(client, bootstrap, manager_id, gw)
    out = [
        f"# Evidence: {season} GW{gw}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by the revision loop.",
        "",
        "## Result",
        "",
        f"- Points: {live['points']} (hits −{live['hits_cost']}, net {live['net_points']})",
        f"- GW average: {live['average']}, highest: {live['highest']}, Δ vs average: {live['delta_vs_average']}",
        f"- GW rank: {live['gw_rank']}, overall rank: {live['overall_rank']}",
        f"- Transfers: {live['transfers']}, bench points: {live['bench_points']}",
        f"- Chips played: {', '.join(c.get('name', '?') for c in live['chips']) or 'none'}",
        "",
        "## Team that played",
        "",
    ]
    try:
        out += _picks_table(client, bootstrap, manager_id, gw)
    except Exception as exc:  # evidence is best-effort; the review still runs
        out.append(f"(could not load picks: {exc})")

    moves = [m for m in season_state.get("applied_moves") or [] if int(m.get("gameweek", 0)) == gw]
    out += ["", "## Transfers the bot applied for this GW", ""]
    out += [
        f"- {m.get('out_name')} ({m.get('out_team')}, EV {m.get('out_ev')}) → "
        f"{m.get('in_name')} ({m.get('in_team')}, EV {m.get('in_ev')})"
        for m in moves
    ] or ["- none"]

    if window_start and window_end:
        decisions, problems, examples = _log_window(log_path, window_start, window_end)
        out += [
            "",
            f"## Bot log {window_start:%Y-%m-%d %H:%M} → {window_end:%Y-%m-%d %H:%M} (local)",
            "",
            f"### Warnings, errors and exceptions ({sum(problems.values())} total, {len(problems)} distinct)",
            "",
        ]
        if problems:
            out += ["| Count | Pattern | First example |", "|---|---|---|"]
            for key, count in problems.most_common(MAX_DISTINCT_PROBLEMS):
                out.append(f"| {count} | `{key.replace('|', '/')}` | `{examples[key].replace('|', '/')}` |")
        else:
            out.append("None.")
        out += ["", f"### Decision log (last {MAX_DECISION_LINES} lines)", "", "```"]
        out += decisions[-MAX_DECISION_LINES:] or ["(none)"]
        out += ["```"]
    out.append("")
    return "\n".join(out), live
