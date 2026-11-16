"""revisions/ledger.json (source of truth) and its rendered LEDGER.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEDGER_JSON = Path("revisions/ledger.json")
LEDGER_MD = Path("revisions/LEDGER.md")

STATUS_LABEL = {
    "merged": "merged",
    "no_change": "no change",
    "rejected": "rejected (gates)",
    "reverted": "**reverted**",
}


def load(root: Path) -> list[dict[str, Any]]:
    path = root / LEDGER_JSON
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save(root: Path, entries: list[dict[str, Any]]) -> None:
    entries = sorted(entries, key=lambda e: (e["season"], e["gw"]))
    (root / LEDGER_JSON).parent.mkdir(parents=True, exist_ok=True)
    (root / LEDGER_JSON).write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    (root / LEDGER_MD).write_text(render(entries), encoding="utf-8")


def upsert(entries: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    out = [e for e in entries if not (e["season"] == entry["season"] and e["gw"] == entry["gw"])]
    out.append(entry)
    return out


def _fmt(value: Any) -> str:
    return "–" if value is None or value == "" else str(value)


def _signed(value: Any) -> str:
    if value is None:
        return "–"
    return f"{value:+g}"


def render(entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Revision ledger",
        "",
        "Generated from `ledger.json` by the revision loop; do not edit by hand.",
        "One row per finished gameweek. **Live** columns are how the bot scored that",
        "gameweek, i.e. the code *before* that row's revision. The revision first plays",
        "the following gameweek, so judge a revision by the rows above it (newer GWs).",
        "",
        "| Season | GW | Live pts | Avg | Δ vs avg | Overall rank | Revision | Status | Backtest before→after | Lines | PR | Tag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in sorted(entries, key=lambda e: (e["season"], e["gw"]), reverse=True):
        live = e.get("live") or {}
        bt = e.get("backtest") or {}
        pr = e.get("pr")
        pr_cell = f"[#{pr}]({e.get('pr_url')})" if pr else "–"
        lines.append(
            "| {season} | {gw} | {pts} | {avg} | {delta} | {rank} | [{title}]({report}) | {status} | {btb}→{bta} | {lines_} | {pr} | {tag} |".format(
                season=e["season"],
                gw=e["gw"],
                pts=_fmt(live.get("points")),
                avg=_fmt(live.get("average")),
                delta=_signed(live.get("delta_vs_average")),
                rank=_fmt(live.get("overall_rank")),
                title=(e.get("title") or "").replace("|", "/"),
                report=e.get("report", ""),
                status=STATUS_LABEL.get(e.get("status", ""), e.get("status", "")),
                btb=_fmt(bt.get("before")),
                bta=_fmt(bt.get("after")),
                lines_=_fmt(e.get("lines_changed")),
                pr=pr_cell,
                tag=f"`{e['tag']}`" if e.get("tag") else "–",
            )
        )
    lines.append("")
    return "\n".join(lines)
