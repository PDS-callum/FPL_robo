#!/usr/bin/env python3
"""Capture full FPL browser state, then probe how to replay it with requests.

Philosophy: dump everything first (cookies, storage, request headers), write it
out, then try multiple replay strategies. Do not gate capture on knowing the
exact cookie names up front.

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
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpl_robo.auth import load_auth_config  # noqa: E402
from fpl_robo.config import DEFAULT_SESSION_PATH  # noqa: E402
from fpl_robo.data.fpl_client import FPLClient  # noqa: E402

API_ME = "https://fantasy.premierleague.com/api/me/"
API_BASE = "https://fantasy.premierleague.com"


def _flatten_cookies(cookies: list[dict]) -> dict[str, str]:
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


def _cookie_header_from_records(records: list[dict]) -> str:
    # Preserve order; last write wins for duplicate names via flatten.
    flat = _flatten_cookies(records)
    return "; ".join(f"{k}={v}" for k, v in flat.items())


def _dump_browser_state(context, page) -> dict[str, Any]:
    """Copy everything we can from the live authenticated browser."""
    client = context.new_cdp_session(page)
    try:
        client.send("Network.enable")
    except Exception:  # noqa: BLE001
        pass

    all_cookies: list[dict] = []
    try:
        all_cookies = (client.send("Network.getAllCookies") or {}).get("cookies") or []
    except Exception as exc:  # noqa: BLE001
        print(f"CDP getAllCookies failed: {exc}", flush=True)
        all_cookies = context.cookies()

    url_cookies: list[dict] = []
    try:
        url_cookies = (
            client.send(
                "Network.getCookies",
                {
                    "urls": [
                        API_ME,
                        f"{API_BASE}/",
                        f"{API_BASE}/my-team",
                        "https://account.premierleague.com/",
                        "https://users.premierleague.com/",
                    ]
                },
            )
            or {}
        ).get("cookies") or []
    except Exception as exc:  # noqa: BLE001
        print(f"CDP getCookies(urls) failed: {exc}", flush=True)

    storage_state = context.storage_state()

    web_storage = page.evaluate(
        """() => {
            const dump = (store) => {
              const out = {};
              try {
                for (let i = 0; i < store.length; i++) {
                  const k = store.key(i);
                  out[k] = store.getItem(k);
                }
              } catch (e) {
                return {_error: String(e)};
              }
              return out;
            };
            return {
              localStorage: dump(window.localStorage),
              sessionStorage: dump(window.sessionStorage),
              documentCookie: document.cookie,
              href: location.href,
            };
        }"""
    )

    # Force a same-origin fetch and read Set-Cookie is impossible from JS,
    # but we can still observe the response body and document state.
    fetch_probe = page.evaluate(
        """async () => {
            const r = await fetch('/api/me/', { credentials: 'include' });
            let body = null;
            try { body = await r.json(); } catch (e) { body = {_parse_error: String(e)}; }
            return { status: r.status, ok: r.ok, body, url: r.url };
        }"""
    )

    cookie_header = _cookie_header_from_records(url_cookies or all_cookies)
    flat = _flatten_cookies(url_cookies or all_cookies)

    return {
        "cookies": flat,
        "cookie_records": all_cookies,
        "url_cookie_records": url_cookies,
        "headers": {"cookie": cookie_header} if cookie_header else {},
        "storage_state": storage_state,
        "web_storage": web_storage,
        "fetch_probe": fetch_probe,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _probe_replay_strategies(payload: dict) -> list[dict[str, Any]]:
    """Try several ways of feeding the dump into requests; report what works."""
    results: list[dict[str, Any]] = []

    def _check(name: str, session: requests.Session) -> dict[str, Any]:
        try:
            r = session.get(API_ME, timeout=30)
            body = r.json() if r.content else {}
            entry = ((body.get("player") or {}) if isinstance(body, dict) else {}).get("entry")
            return {
                "strategy": name,
                "ok": entry is not None,
                "status": r.status_code,
                "entry_id": entry,
                "player_null": isinstance(body, dict) and body.get("player") is None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"strategy": name, "ok": False, "error": str(exc)}

    # 1) FPLClient.load_session path (what the Pi uses)
    client = FPLClient()
    tmp = Path.cwd() / "data" / "_session_verify_tmp.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    if client.load_session(str(tmp)):
        results.append(_check("fpl_client.load_session", client.session))
    else:
        results.append({"strategy": "fpl_client.load_session", "ok": False, "error": "load failed"})

    # 2) Raw Cookie header only
    s = requests.Session()
    s.headers.update({"User-Agent": "FPL-Robo/0.1", "Cookie": (payload.get("headers") or {}).get("cookie", "")})
    results.append(_check("raw_cookie_header", s))

    # 3) Flat cookie jar
    s = requests.Session()
    s.headers.update({"User-Agent": "FPL-Robo/0.1"})
    s.cookies.update(payload.get("cookies") or {})
    results.append(_check("flat_cookie_jar", s))

    # 4) Domain-aware cookie records
    s = requests.Session()
    s.headers.update({"User-Agent": "FPL-Robo/0.1"})
    for rec in payload.get("cookie_records") or []:
        if not isinstance(rec, dict) or not rec.get("name"):
            continue
        try:
            s.cookies.set(
                str(rec["name"]),
                str(rec.get("value") or ""),
                domain=str(rec.get("domain") or ""),
                path=str(rec.get("path") or "/"),
            )
        except Exception:  # noqa: BLE001
            continue
    results.append(_check("domain_cookie_records", s))

    # 5) URL-scoped cookies only + Cookie header
    s = requests.Session()
    hdr = _cookie_header_from_records(payload.get("url_cookie_records") or [])
    s.headers.update({"User-Agent": "FPL-Robo/0.1", "Cookie": hdr})
    results.append(_check("url_scoped_cookie_header", s))

    # 6) OIDC access_token from localStorage (modern FPL / PingOne)
    from fpl_robo.data.fpl_client import _extract_oidc_access_token

    token = _extract_oidc_access_token(payload)
    s = requests.Session()
    s.headers.update({"User-Agent": "FPL-Robo/0.1"})
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    results.append(_check("oidc_bearer_access_token", s))

    return results


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
    print("Strategy: dump ALL browser state, then probe how to replay it.", flush=True)
    print(f"Dump path: {out}", flush=True)

    with sync_playwright() as p:
        print("Launching Chromium…", flush=True)
        browser = p.chromium.launch(headless=False)
        print("Chromium launched. Opening FPL…", flush=True)
        context = browser.new_context()
        page = context.new_page()
        observed_headers: dict[str, str] = {}

        def _capture_headers(req) -> None:
            if "fantasy.premierleague.com/api/" not in req.url:
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
                # Use in-page fetch (true browser credentials). page.request can
                # miss partitioned/auth cookies that the page itself sends.
                me = page.evaluate(
                    """async () => {
                        const r = await fetch('/api/me/', { credentials: 'include' });
                        if (!r.ok) return { _status: r.status };
                        return await r.json();
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
        print(f"Browser login detected for entry {entry}. Dumping everything…", flush=True)
        time.sleep(1.0)
        try:
            page.goto(f"{API_BASE}/my-team", wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:  # noqa: BLE001
            print(f"my-team navigation warning: {exc}", flush=True)

        payload = _dump_browser_state(context, page)
        if observed_headers:
            payload.setdefault("headers", {}).update(observed_headers)
            # Prefer a Cookie header actually observed on an API request.
            if "cookie" in observed_headers:
                payload["headers"]["cookie"] = observed_headers["cookie"]

        payload["browser_me"] = me
        payload["entry_id"] = entry

        print("Probing replay strategies with requests…", flush=True)
        probes = _probe_replay_strategies(payload)
        payload["replay_probes"] = probes
        for probe in probes:
            mark = "OK" if probe.get("ok") else "FAIL"
            extra = probe.get("entry_id") or probe.get("error") or probe.get("status")
            print(f"  [{mark}] {probe.get('strategy')}: {extra}", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        browser.close()

    winners = [p for p in probes if p.get("ok")]
    print(f"\nFull dump written → {out}", flush=True)
    print(f"Cookie names: {', '.join(sorted((payload.get('cookies') or {}).keys()))}", flush=True)
    if winners:
        print(f"Working replay strategy: {winners[0]['strategy']} (entry {winners[0].get('entry_id')})", flush=True)
        print("Confirm with: python -m fpl_robo.cli auth-check --load-session", flush=True)
        return 0

    print(
        "No requests replay strategy worked yet — but the full dump is saved.",
        flush=True,
    )
    print("Copy data/session.json here and we can inspect what the browser had.", flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture full FPL browser state and probe replay")
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
