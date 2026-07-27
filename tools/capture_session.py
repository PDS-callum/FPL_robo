#!/usr/bin/env python3
"""Capture an FPL authenticated session via a headed browser (OAuth-friendly).

Run on a machine with a display (desktop/laptop), then copy data/session.json
to the Pi.

  pip install playwright
  playwright install chromium
  python tools/capture_session.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpl_robo.auth import load_auth_config  # noqa: E402
from fpl_robo.config import DEFAULT_SESSION_PATH  # noqa: E402


def _cookies_to_session_payload(cookies: list[dict]) -> dict:
    """Flatten browser cookies into the format FPLClient.load_session expects."""
    # Prefer fantasy / premierleague cookies when names collide.
    ranked: dict[str, tuple[int, str]] = {}
    for c in cookies:
        name = c.get("name")
        value = c.get("value")
        if not name or value is None:
            continue
        domain = str(c.get("domain") or "").lstrip(".").lower()
        if "fantasy.premierleague.com" in domain:
            rank = 0
        elif domain.endswith("premierleague.com"):
            rank = 1
        else:
            rank = 2
        prev = ranked.get(name)
        if prev is None or rank < prev[0]:
            ranked[name] = (rank, str(value))
    return {
        "cookies": {name: value for name, (_, value) in ranked.items()},
        "headers": {},
    }


def capture(*, session_path: str, timeout_seconds: int, url: str) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is required for headed OAuth capture:")
        print("  pip install playwright")
        print("  playwright install chromium")
        return 1

    cfg = load_auth_config()
    out = Path(session_path or cfg.session_path or DEFAULT_SESSION_PATH)
    print(f"Opening headed Chromium. Log in to FPL in the window.")
    print(f"Waiting up to {timeout_seconds}s for /api/me/ to return a player…")
    print(f"Will write: {out}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")

        deadline = time.time() + timeout_seconds
        me = None
        while time.time() < deadline:
            try:
                me = page.evaluate(
                    """async () => {
                        const r = await fetch('/api/me/', { credentials: 'include' });
                        if (!r.ok) return { _status: r.status };
                        return await r.json();
                    }"""
                )
            except Exception as exc:  # noqa: BLE001 — keep polling while user logs in
                me = {"_error": str(exc)}

            player = (me or {}).get("player") if isinstance(me, dict) else None
            entry = (player or {}).get("entry") if isinstance(player, dict) else None
            if entry is not None:
                break
            time.sleep(1.5)
        else:
            browser.close()
            print("Timed out waiting for authenticated /api/me/.")
            print("Last payload:", json.dumps(me, indent=2)[:500])
            return 1

        # Brief settle for late-set auth cookies after hydration.
        time.sleep(2.0)
        cookies = context.cookies()
        payload = _cookies_to_session_payload(cookies)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        browser.close()

    entry = ((me or {}).get("player") or {}).get("entry")
    names = sorted((payload.get("cookies") or {}).keys())
    print(f"Session saved for entry {entry} → {out}")
    print(f"Cookie names ({len(names)}): {', '.join(names)}")
    if "pl_profile" not in names and "sessionid" not in names:
        print("Warning: neither pl_profile nor sessionid found; auth may still fail.")
    print("Copy this file to the Pi if you captured on another machine, then run:")
    print("  python -m fpl_robo.cli auth-check --load-session")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture FPL session via headed browser OAuth")
    parser.add_argument("--session-path", default=DEFAULT_SESSION_PATH)
    parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for login")
    parser.add_argument("--url", default="https://fantasy.premierleague.com/")
    args = parser.parse_args(argv)
    return capture(
        session_path=args.session_path,
        timeout_seconds=args.timeout,
        url=args.url,
    )


if __name__ == "__main__":
    raise SystemExit(main())
