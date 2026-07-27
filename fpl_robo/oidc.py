"""PingOne OIDC token refresh for modern FPL auth."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from fpl_robo.config import DEFAULT_SESSION_PATH
from fpl_robo.data.fpl_client import FPLClient, _extract_oidc_access_token

logger = logging.getLogger("fpl_robo.oidc")

TOKEN_URL = "https://account.premierleague.com/as/token"
DEFAULT_CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
# Refresh this many seconds before access-token expiry.
REFRESH_SKEW_SECONDS = 600  # 10 minutes
DAEMON_POLL_SECONDS = 60


@dataclass
class OidcUser:
    key: str
    data: dict[str, Any]

    @property
    def access_token(self) -> str | None:
        token = self.data.get("access_token")
        return token.strip() if isinstance(token, str) and token.strip() else None

    @property
    def refresh_token(self) -> str | None:
        token = self.data.get("refresh_token")
        return token.strip() if isinstance(token, str) and token.strip() else None

    @property
    def expires_at(self) -> int | None:
        raw = self.data.get("expires_at")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def client_id(self) -> str:
        # oidc.user:https://account.premierleague.com/as:<client_id>
        parts = self.key.rsplit(":", 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
        return DEFAULT_CLIENT_ID


def load_session_payload(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("session payload must be a JSON object")
    return data


def find_oidc_user(payload: dict[str, Any]) -> OidcUser | None:
    web = payload.get("web_storage")
    if isinstance(web, dict):
        local = web.get("localStorage")
        if isinstance(local, dict):
            for key, raw in local.items():
                if not str(key).startswith("oidc.user:"):
                    continue
                if not isinstance(raw, str):
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("refresh_token"):
                    return OidcUser(key=str(key), data=obj)

    storage = payload.get("storage_state")
    if isinstance(storage, dict):
        for origin in storage.get("origins") or []:
            if not isinstance(origin, dict):
                continue
            for item in origin.get("localStorage") or []:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                raw = item.get("value")
                if not isinstance(name, str) or not name.startswith("oidc.user:"):
                    continue
                if not isinstance(raw, str):
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("refresh_token"):
                    return OidcUser(key=name, data=obj)
    return None


def seconds_until_expiry(user: OidcUser) -> float | None:
    exp = user.expires_at
    if exp is None:
        return None
    return float(exp) - time.time()


def needs_refresh(user: OidcUser, *, skew_seconds: int = REFRESH_SKEW_SECONDS) -> bool:
    remaining = seconds_until_expiry(user)
    if remaining is None:
        return True
    return remaining <= skew_seconds


def refresh_access_token(
    user: OidcUser,
    *,
    token_url: str = TOKEN_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    if not user.refresh_token:
        raise RuntimeError("No refresh_token in OIDC session")
    response = requests.post(
        token_url,
        data={
            "grant_type": "refresh_token",
            "refresh_token": user.refresh_token,
            "client_id": user.client_id,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "FPL-Robo/0.1",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Token refresh failed ({response.status_code}): {response.text[:300]}")
    body = response.json()
    if not isinstance(body, dict) or not body.get("access_token"):
        raise RuntimeError("Token refresh returned no access_token")
    return body


def apply_token_response(user: OidcUser, body: dict[str, Any]) -> OidcUser:
    user.data["access_token"] = body["access_token"]
    if body.get("id_token"):
        user.data["id_token"] = body["id_token"]
    if body.get("refresh_token"):
        user.data["refresh_token"] = body["refresh_token"]
    if body.get("token_type"):
        user.data["token_type"] = body["token_type"]
    if body.get("scope"):
        user.data["scope"] = body["scope"]
    expires_in = int(body.get("expires_in") or 28800)
    user.data["expires_at"] = int(time.time()) + expires_in
    return user


def write_oidc_user(payload: dict[str, Any], user: OidcUser) -> None:
    raw = json.dumps(user.data)
    web = payload.setdefault("web_storage", {})
    if not isinstance(web, dict):
        web = {}
        payload["web_storage"] = web
    local = web.setdefault("localStorage", {})
    if not isinstance(local, dict):
        local = {}
        web["localStorage"] = local
    local[user.key] = raw

    storage = payload.get("storage_state")
    if isinstance(storage, dict):
        for origin in storage.get("origins") or []:
            if not isinstance(origin, dict):
                continue
            items = origin.get("localStorage")
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("name") == user.key:
                    item["value"] = raw


def refresh_session_file(
    path: str | Path = DEFAULT_SESSION_PATH,
    *,
    force: bool = False,
    skew_seconds: int = REFRESH_SKEW_SECONDS,
) -> dict[str, Any]:
    """Refresh OIDC tokens in session.json if near expiry (or force)."""
    path = Path(path)
    payload = load_session_payload(path)
    user = find_oidc_user(payload)
    if user is None:
        return {"ok": False, "reason": "No OIDC user/refresh_token in session file"}

    remaining = seconds_until_expiry(user)
    if not force and not needs_refresh(user, skew_seconds=skew_seconds):
        return {
            "ok": True,
            "refreshed": False,
            "expires_at": user.expires_at,
            "seconds_remaining": remaining,
            "session_path": str(path),
        }

    body = refresh_access_token(user)
    apply_token_response(user, body)
    write_oidc_user(payload, user)
    from fpl_robo.runtime.atomic import atomic_write_json

    atomic_write_json(path, payload)

    # Sanity-check the new access token against FPL.
    client = FPLClient()
    client.load_session(str(path))
    me = client.me()
    entry = ((me.get("player") or {}) if isinstance(me, dict) else {}).get("entry")
    if entry is None:
        return {
            "ok": False,
            "reason": "Refreshed token did not authenticate /api/me/",
            "session_path": str(path),
        }

    logger.info(
        "OIDC token refreshed; entry=%s expires_at=%s",
        entry,
        user.expires_at,
    )
    return {
        "ok": True,
        "refreshed": True,
        "entry_id": entry,
        "expires_at": user.expires_at,
        "seconds_remaining": seconds_until_expiry(user),
        "session_path": str(path),
    }


def ensure_fresh_session(
    path: str | Path = DEFAULT_SESSION_PATH,
    *,
    skew_seconds: int = REFRESH_SKEW_SECONDS,
) -> dict[str, Any]:
    """Best-effort refresh used by season runner / auth helpers."""
    try:
        return refresh_session_file(path, force=False, skew_seconds=skew_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OIDC refresh failed: %s", exc)
        return {"ok": False, "reason": str(exc)}


def run_refresh_daemon(
    *,
    session_path: str = DEFAULT_SESSION_PATH,
    skew_seconds: int = REFRESH_SKEW_SECONDS,
    poll_seconds: int = DAEMON_POLL_SECONDS,
) -> None:
    """Continuously refresh the OIDC access token before it expires."""
    from fpl_robo.runtime.interrupt import (
        install_stop_handlers,
        interruptible_sleep,
        stop_requested,
    )

    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("logs/fpl_robo_auth.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    install_stop_handlers()
    logger.info(
        "Auth refresh daemon started (skew=%ss poll=%ss path=%s)",
        skew_seconds,
        poll_seconds,
        session_path,
    )
    while not stop_requested():
        try:
            result = refresh_session_file(
                session_path,
                force=False,
                skew_seconds=skew_seconds,
            )
            if result.get("ok") and result.get("refreshed"):
                logger.info("Refreshed OK; expires_at=%s", result.get("expires_at"))
            elif result.get("ok"):
                logger.info(
                    "Token still valid; %.0fs remaining",
                    float(result.get("seconds_remaining") or 0),
                )
            else:
                logger.error("Refresh check failed: %s", result.get("reason"))
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected refresh daemon error")

        if stop_requested():
            break

        # Sleep until near expiry, but wake at least every poll_seconds to re-check.
        try:
            payload = load_session_payload(session_path)
            user = find_oidc_user(payload)
            remaining = seconds_until_expiry(user) if user else None
        except Exception:  # noqa: BLE001
            remaining = None

        if remaining is None:
            sleep_for = float(poll_seconds)
        else:
            sleep_for = max(float(poll_seconds), remaining - float(skew_seconds))
            # Cap long sleeps so process stays responsive / loggable.
            sleep_for = min(sleep_for, 3600.0)
        logger.info("Sleeping %.0fs until next refresh check", sleep_for)
        if not interruptible_sleep(sleep_for):
            break
    logger.info("Auth refresh daemon drained cleanly; exiting")


def session_has_oidc(path: str | Path = DEFAULT_SESSION_PATH) -> bool:
    try:
        return find_oidc_user(load_session_payload(path)) is not None
    except Exception:  # noqa: BLE001
        return False


# Re-export helper used elsewhere for typing clarity.
__all__ = [
    "REFRESH_SKEW_SECONDS",
    "ensure_fresh_session",
    "refresh_session_file",
    "run_refresh_daemon",
    "session_has_oidc",
    "_extract_oidc_access_token",
]
