"""Official FPL API client."""

from __future__ import annotations

import json
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

import requests
from requests.cookies import RequestsCookieJar, cookiejar_from_dict

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Client for public and authenticated FPL endpoints."""

    def __init__(self, base_url: str = FPL_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "FPL-Robo/0.1 (2026-27; local research bot)",
            }
        )

    def get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def bootstrap_static(self) -> dict[str, Any]:
        return self.get_json("bootstrap-static/")

    def fixtures(self) -> list[dict[str, Any]]:
        data = self.get_json("fixtures/")
        if not isinstance(data, list):
            raise TypeError("Unexpected fixtures payload")
        return data

    def entry(self, manager_id: int) -> dict[str, Any]:
        return self.get_json(f"entry/{manager_id}/")

    def entry_history(self, manager_id: int) -> dict[str, Any]:
        return self.get_json(f"entry/{manager_id}/history/")

    def entry_picks(self, manager_id: int, gameweek: int) -> dict[str, Any]:
        return self.get_json(f"entry/{manager_id}/event/{gameweek}/picks/")

    def entry_picks_latest(
        self, manager_id: int, gameweek: int | None = None, lookback: int = 3
    ) -> tuple[int, dict[str, Any]] | None:
        if gameweek is None:
            bootstrap = self.bootstrap_static()
            gameweek = _resolve_current_gameweek(bootstrap)
        if gameweek is None:
            return None
        for offset in range(0, lookback):
            gw = gameweek - offset
            if gw < 1:
                break
            try:
                return gw, self.entry_picks(manager_id, gw)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise
        return None

    def me(self) -> dict[str, Any]:
        return self.get_json("me/")

    def my_team(self, manager_id: int) -> dict[str, Any]:
        return self.get_json(f"my-team/{manager_id}/")

    def save_my_team(self, manager_id: int, picks: list[dict[str, Any]], chip: str | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/my-team/{manager_id}/"
        payload = {"picks": picks, "chip": chip}
        headers = {"content-type": "application/json"}
        response = self.session.post(url, json=payload, timeout=self.timeout, headers=headers)
        response.raise_for_status()
        data = response.json() if response.content else {}
        if not isinstance(data, dict):
            raise TypeError("Unexpected my-team response payload")
        return data

    def execute_transfers(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/transfers/"
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        if not response.content:
            return {}
        data = response.json()
        if not isinstance(data, dict):
            return {"raw": data}
        return data

    def authenticate(self, email: str, password: str) -> bool:
        try:
            # Legacy flow - may fail in some regions/environments; session loading is preferred.
            login_url = "https://users.premierleague.com/accounts/login/"
            self.session.get(login_url, timeout=self.timeout)
            csrf_token = self.session.cookies.get("csrftoken")
            if not csrf_token:
                return False

            payload = {
                "login": email,
                "password": password,
                "redirect_uri": "https://fantasy.premierleague.com/",
                "app": "plfpl-web",
            }
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://fantasy.premierleague.com/",
                "X-CSRFToken": csrf_token,
                "Origin": "https://fantasy.premierleague.com",
            }
            response = self.session.post(
                login_url,
                data=payload,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
            )
            if response.status_code != 200:
                return False
            _ = self.me()
            return True
        except requests.RequestException:
            return False

    def save_session(self, path: str) -> None:
        from fpl_robo.runtime.atomic import atomic_write_json

        payload = {
            "cookies": requests.utils.dict_from_cookiejar(self.session.cookies),
            "headers": dict(self.session.headers),
        }
        atomic_write_json(path, payload)

    def load_session(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False

        loaded = False

        cookies = data.get("cookies")
        if isinstance(cookies, dict) and cookies:
            jar: RequestsCookieJar = cookiejar_from_dict(cookies)
            self.session.cookies.update(jar)
            loaded = True

        # Full cookie records (domain/path) from CDP / Playwright storage.
        records = data.get("cookie_records")
        if isinstance(records, list):
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                name = rec.get("name")
                value = rec.get("value")
                if not name or value is None:
                    continue
                self.session.cookies.set(
                    str(name),
                    str(value),
                    domain=str(rec.get("domain") or ""),
                    path=str(rec.get("path") or "/"),
                )
                loaded = True

        storage = data.get("storage_state")
        if isinstance(storage, dict):
            for rec in storage.get("cookies") or []:
                if not isinstance(rec, dict):
                    continue
                name = rec.get("name")
                value = rec.get("value")
                if not name or value is None:
                    continue
                self.session.cookies.set(
                    str(name),
                    str(value),
                    domain=str(rec.get("domain") or ""),
                    path=str(rec.get("path") or "/"),
                )
                loaded = True

        headers = data.get("headers")
        if isinstance(headers, dict):
            auth = headers.get("authorization") or headers.get("Authorization")
            if isinstance(auth, str) and auth.strip():
                self.session.headers["Authorization"] = auth.strip()
                loaded = True
            cookie_header = headers.get("cookie") or headers.get("Cookie")
            if isinstance(cookie_header, str) and cookie_header.strip():
                # Replay the exact browser Cookie header (most reliable transfer).
                self.session.headers["Cookie"] = cookie_header.strip()
                parsed = SimpleCookie()
                parsed.load(cookie_header)
                self.session.cookies.update({k: morsel.value for k, morsel in parsed.items()})
                loaded = True

        # Modern FPL uses PingOne OIDC; access_token lives in localStorage.
        token = _extract_oidc_access_token(data)
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
            loaded = True

        return loaded


def _extract_oidc_access_token(data: dict[str, Any]) -> str | None:
    """Pull Bearer access_token from captured browser localStorage / storage_state."""
    # auth-capture web_storage.localStorage dump (dict[str,str])
    web = data.get("web_storage")
    if isinstance(web, dict):
        local = web.get("localStorage")
        if isinstance(local, dict):
            for key, raw in local.items():
                if not str(key).startswith("oidc.user:"):
                    continue
                token = _token_from_oidc_raw(raw)
                if token:
                    return token

    # Playwright storage_state origins[].localStorage[{name,value}]
    storage = data.get("storage_state")
    if isinstance(storage, dict):
        for origin in storage.get("origins") or []:
            if not isinstance(origin, dict):
                continue
            for item in origin.get("localStorage") or []:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str) or not name.startswith("oidc.user:"):
                    continue
                token = _token_from_oidc_raw(item.get("value"))
                if token:
                    return token
    return None


def _token_from_oidc_raw(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    token = obj.get("access_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _resolve_current_gameweek(bootstrap: dict[str, Any]) -> int | None:
    events = bootstrap.get("events") or []
    for event in events:
        if event.get("is_next"):
            return int(event["id"])
    for event in events:
        if event.get("is_current"):
            return int(event["id"])
    finished = [int(e["id"]) for e in events if e.get("finished")]
    return max(finished) if finished else None
