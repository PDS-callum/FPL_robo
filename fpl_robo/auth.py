"""Credential loading and auth/session helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from fpl_robo.config import DEFAULT_MANAGER_ID, DEFAULT_SESSION_PATH
from fpl_robo.data.fpl_client import FPLClient


@dataclass
class AuthConfig:
    email: str
    password: str
    manager_id: int
    session_path: str


def load_auth_config() -> AuthConfig:
    load_dotenv()
    email = os.getenv("FPL_EMAIL", "").strip()
    password = os.getenv("FPL_PASSWORD", "").strip()
    manager_id = int(os.getenv("FPL_MANAGER_ID", str(DEFAULT_MANAGER_ID)))
    session_path = os.getenv("FPL_SESSION_PATH", DEFAULT_SESSION_PATH).strip()
    return AuthConfig(
        email=email,
        password=password,
        manager_id=manager_id,
        session_path=session_path,
    )


def ensure_authenticated(
    client: FPLClient,
    cfg: AuthConfig,
    *,
    load_session: bool,
    save_session: bool,
) -> dict[str, object]:
    if load_session and client.load_session(cfg.session_path):
        try:
            me = client.me()
            if (me.get("player") or {}).get("id") is not None:
                if save_session:
                    client.save_session(cfg.session_path)
                return {"ok": True, "method": "session", "entry_id": (me.get("player") or {}).get("entry")}
        except Exception:
            pass

    if cfg.email and cfg.password and client.authenticate(cfg.email, cfg.password):
        me = client.me()
        if save_session:
            client.save_session(cfg.session_path)
        return {
            "ok": True,
            "method": "password",
            "entry_id": (me.get("player") or {}).get("entry"),
        }

    return {
        "ok": False,
        "reason": "Authentication failed (session invalid and password login unavailable)",
    }


def auth_check(
    client: FPLClient | None = None,
    *,
    load_session: bool = True,
    save_session: bool = False,
) -> dict[str, object]:
    cfg = load_auth_config()
    client = client or FPLClient()

    if not load_session and (not cfg.email or not cfg.password):
        return {
            "ok": False,
            "reason": "Missing credentials and session loading disabled",
        }

    result = ensure_authenticated(
        client,
        cfg,
        load_session=load_session,
        save_session=save_session,
    )
    if not result.get("ok"):
        return result

    me = client.me()
    entry = ((me.get("player") or {}).get("entry")) if isinstance(me, dict) else None
    return {
        "ok": True,
        "method": result.get("method"),
        "entry_id": entry,
        "expected_manager_id": cfg.manager_id,
        "matches_expected_manager_id": entry == cfg.manager_id,
        "session_path": cfg.session_path,
    }
