#!/usr/bin/env python3
"""Capture an FPL authenticated session via a headed browser (OAuth-friendly).

Run on a machine with a display (desktop/laptop). The script only writes
data/session.json after verifying the session works with requests (same path
the Pi bot uses). Then copy that file to the Pi.

  pip install playwright
  playwright install chromium
  python -m fpl_robo.cli auth-capture
  python -m fpl_robo.cli auth-check --load-session
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpl_robo.auth import load_auth_config  # noqa: E402
from fpl_robo.config import DEFAULT_SESSION_PATH  # noqa: E402
from fpl_robo.data.fpl_client import FPLClient  # noqa: E402


def _flatten_cookies(cookies: list[dict]) -> dict[str, str]:
    """Prefer fantasy/premierleague cookies when names collide."""
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
    return {name: value for name, (_, value) in ranked.items()}


def _cdp_all_cookies(context, page) -> list[dict]:
    """Pull every cookie Chromium knows about, including partitioned ones."""
    try:
        client = context.new_cdp_session(page)
        result = client.send("Network.getAllCookies")
        cookies = result.get("cookies") or []
        return [
            {
                "name": c.get("name"),
                "value": c.get("value"),
                "domain": c.get("domain"),
                "path": c.get("path"),
                "expires": c.get("expires"),
                "httpOnly": c.get("httpOnly"),
                "secure": c.get("secure"),
                "sameSite": c.get("sameSite"),
                "partitionKey": c.get("partitionKey"),
            }
            for c in cookies
            if c.get("name") and c.get("value") is not None
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"CDP cookie dump failed ({exc}); falling back to context.cookies()", flush=True)
        return context.cookies()


def _verify_with_requests(payload: dict) -> dict:
    """Prove the saved session works the same way the Pi bot will use it."""
    client = FPLClient()
    tmp = Path("/tmp/fpl_robo_session_verify.json")
    # Windows-friendly temp path fallback
    try:
        tmp.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        tmp = Path.cwd() / "data" / "_session_verify_tmp.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload), encoding="utf-8")
    if not client.load_session(str(tmp)):
        return {"ok": False, "reason": "load_session returned False"}
    try:
        me = client.me()
    except requests.RequestException as exc:
        return {"ok": False, "reason": f"me/ request failed: {exc}"}
    player = me.get("player") if isinstance(me, dict) else None
    entry = (player or {}).get("entry") if isinstance(player, dict) else None
    if entry is None:
        return {"ok": False, "reason": "me/ returned no player.entry", "me": me}
    return {"ok": True, "entry_id": entry, "me": me}


def capture(*, session_path: str, timeout_seconds: int, url: str) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is required for headed OAuth capture:", flush=True)
        print("  pip install playwright", flush=True)
        print("  playwright install chromium", flush=True)
        return 1

    cfg = load_auth_config()
    out = Path(session_path or cfg.session_path or DEFAULT_SESSION_PATH)
    print("Opening headed Chromium. Log in to FPL in the window.", flush=True)
    print(f"Waiting up to {timeout_seconds}s for authenticated /api/me/…", flush=True)
    print(f"Will write only after requests verification passes: {out}", flush=True)

    with sync_playwright() as p:
        print("Launching Chromium…", flush=True)
        browser = p.chromium.launch(headless=False)
        print("Chromium launched. Opening FPL…", flush=True)
        context = browser.new_context()
        page = context.new_page()
        observed_headers: dict[str, str] = {}

        def _capture_headers(req) -> None:
            req_url = req.url
            if "fantasy.premierleague.com/api/" not in req_url:
                return
            try:
                hdrs = req.headers
            except Exception:  # noqa: BLE001
                return
            for key in ("authorization", "cookie", "x-csrftoken"):
                val = hdrs.get(key)
                if isinstance(val, str) and val.strip():
                    observed_headers[key] = val.strip()

        page.on("request", _capture_headers)
        page.goto(url, wait_until="domcontentloaded")
        print("FPL loaded. Complete login in the browser window…", flush=True)

        deadline = time.time() + timeout_seconds
        me = None
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                resp = page.request.get(
                    "https://fantasy.premierleague.com/api/me/",
                    timeout=15000,
                )
                me = resp.json()
                page.evaluate(
                    """async () => {
                        await fetch('/api/me/', { credentials: 'include' });
                    }"""
                )
            except Exception as exc:  # noqa: BLE001
                me = {"_error": str(exc)}

            player = (me or {}).get("player") if isinstance(me, dict) else None
            entry = (player or {}).get("entry") if isinstance(player, dict) else None
            if entry is not None:
                break
            remaining = max(0, int(deadline - time.time()))
            status = "player=null"
            if isinstance(me, dict) and me.get("_error"):
                status = f"error={me['_error']}"
            elif isinstance(me, dict) and me.get("_status"):
                status = f"status={me['_status']}"
            print(f"[{attempt}] waiting for login ({status}) — {remaining}s left", flush=True)
            time.sleep(1.5)
        else:
            browser.close()
            print("Timed out waiting for authenticated /api/me/.", flush=True)
            print("Last payload:", json.dumps(me, indent=2)[:500], flush=True)
            return 1

        entry = ((me or {}).get("player") or {}).get("entry")
        print(f"Browser login detected for entry {entry}. Settling cookies…", flush=True)
        time.sleep(2.0)

        try:
            print("Probing /my-team and private APIs…", flush=True)
            page.goto(
                "https://fantasy.premierleague.com/my-team",
                wait_until="domcontentloaded",
                timeout=30000,
            )
            page.request.get(
                f"https://fantasy.premierleague.com/api/my-team/{entry}/",
                timeout=15000,
            )
            page.request.get(
                "https://fantasy.premierleague.com/api/me/",
                timeout=15000,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Post-login probe warning: {exc}", flush=True)

        time.sleep(1.0)
        print("Dumping cookies via CDP…", flush=True)
        raw_cookies = _cdp_all_cookies(context, page)
        flat = _flatten_cookies(raw_cookies)
        payload = {
            "cookies": flat,
            "cookie_records": raw_cookies,
            "headers": observed_headers,
            "storage_state": context.storage_state(),
        }

        print("Verifying session with requests (Pi bot path)…", flush=True)
        verify = _verify_with_requests(payload)
        if not verify.get("ok"):
            browser.close()
            print("VERIFICATION FAILED — not writing a reusable session.", flush=True)
            print("Reason:", verify.get("reason"), flush=True)
            print("Cookie names:", ", ".join(sorted(flat.keys())), flush=True)
            print(
                "Observed headers:",
                ", ".join(sorted(observed_headers.keys())) or "(none)",
                flush=True,
            )
            if "cookie" in observed_headers:
                names = [
                    part.split("=", 1)[0].strip()
                    for part in observed_headers["cookie"].split(";")
                    if part.strip()
                ]
                print("Cookie header names:", ", ".join(names), flush=True)
            if verify.get("me"):
                print(json.dumps(verify.get("me"), indent=2)[:400], flush=True)
            print(
                "\nFPL authenticated the browser UI without transferable API cookies.",
                flush=True,
            )
            return 1

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        browser.close()

    names = sorted((payload.get("cookies") or {}).keys())
    print(f"Session verified + saved for entry {verify['entry_id']} → {out}", flush=True)
    print(f"Cookie names ({len(names)}): {', '.join(names)}", flush=True)
    if payload.get("headers"):
        print(
            "Captured auth headers:",
            ", ".join(sorted(payload["headers"].keys())),
            flush=True,
        )
    print("On this laptop, confirm with:", flush=True)
    print("  python -m fpl_robo.cli auth-check --load-session", flush=True)
    print("Then copy data/session.json to the Pi.", flush=True)
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
