"""Email alerts for FPL Robo (auth failures, etc.)."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fpl_robo.runtime.atomic import atomic_write_json

logger = logging.getLogger("fpl_robo.notify")

DEFAULT_COOLDOWN_SECONDS = 3600
DEFAULT_STAMP_PATH = "data/cache/last_auth_alert.json"


def _alert_config() -> dict[str, Any]:
    load_dotenv()
    to_addr = (os.getenv("ALERT_EMAIL_TO") or "").strip()
    from_addr = (os.getenv("ALERT_EMAIL_FROM") or os.getenv("SMTP_USER") or to_addr).strip()
    host = (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip()
    port = int(os.getenv("SMTP_PORT") or 587)
    user = (os.getenv("SMTP_USER") or "").strip()
    password = (os.getenv("SMTP_PASSWORD") or "").strip()
    cooldown = int(os.getenv("ALERT_COOLDOWN_SECONDS") or DEFAULT_COOLDOWN_SECONDS)
    enabled = (os.getenv("ALERT_EMAIL_ENABLED") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return {
        "enabled": enabled and bool(to_addr),
        "to": to_addr,
        "from": from_addr,
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "cooldown_seconds": max(60, cooldown),
        "stamp_path": os.getenv("ALERT_STAMP_PATH") or DEFAULT_STAMP_PATH,
    }


def alerts_configured() -> bool:
    cfg = _alert_config()
    return bool(cfg["enabled"] and cfg["to"] and cfg["user"] and cfg["password"])


def send_email(
    *,
    subject: str,
    body: str,
    to_addr: str | None = None,
) -> dict[str, Any]:
    """Send one email via SMTP. Returns {ok, reason?}."""
    cfg = _alert_config()
    to_addr = (to_addr or cfg["to"]).strip()
    if not cfg["enabled"]:
        return {"ok": False, "reason": "alerts_disabled"}
    if not to_addr:
        return {"ok": False, "reason": "ALERT_EMAIL_TO not set"}
    if not cfg["user"] or not cfg["password"]:
        return {"ok": False, "reason": "SMTP_USER/SMTP_PASSWORD not set"}

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from"] or cfg["user"]
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to send alert email")
        return {"ok": False, "reason": str(exc)}

    logger.info("Alert email sent to %s subject=%s", to_addr, subject)
    return {"ok": True, "to": to_addr}


def _load_stamp(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _within_cooldown(stamp: dict[str, Any], cooldown_seconds: int) -> bool:
    last = stamp.get("sent_at")
    if not last:
        return False
    try:
        sent = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        return False
    age = (datetime.now(timezone.utc) - sent.astimezone(timezone.utc)).total_seconds()
    return age < float(cooldown_seconds)


def notify_auth_failure(
    *,
    reason: str,
    source: str,
    force: bool = False,
) -> dict[str, Any]:
    """Email on auth failure, rate-limited so daemons do not spam."""
    cfg = _alert_config()
    if not cfg["enabled"]:
        return {"ok": False, "skipped": True, "reason": "alerts_disabled_or_no_to"}
    if not cfg["user"] or not cfg["password"]:
        logger.warning(
            "Auth failure alert suppressed (SMTP not configured): %s — %s",
            source,
            reason,
        )
        return {"ok": False, "skipped": True, "reason": "smtp_not_configured"}

    stamp_path = cfg["stamp_path"]
    stamp = _load_stamp(stamp_path)
    if not force and _within_cooldown(stamp, cfg["cooldown_seconds"]):
        logger.info(
            "Auth failure alert suppressed (cooldown): %s — %s",
            source,
            reason,
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "cooldown",
            "last_sent_at": stamp.get("sent_at"),
        }

    now = datetime.now(timezone.utc).isoformat()
    subject = f"[FPL Robo] Auth failed ({source})"
    body = (
        f"FPL Robo authentication failed.\n\n"
        f"Source: {source}\n"
        f"Time (UTC): {now}\n"
        f"Reason: {reason}\n\n"
        f"On the Pi, re-run:\n"
        f"  python -m fpl_robo.cli auth-capture\n"
        f"then copy data/session.json if needed, and:\n"
        f"  sudo systemctl restart fpl-robo-auth.service fpl-robo.service\n"
    )
    result = send_email(subject=subject, body=body)
    if result.get("ok"):
        atomic_write_json(
            stamp_path,
            {
                "sent_at": now,
                "source": source,
                "reason": reason,
                "to": cfg["to"],
            },
        )
    return {**result, "skipped": False}


def clear_auth_alert_stamp() -> None:
    """Clear cooldown stamp (e.g. after successful re-login)."""
    path = Path(_alert_config()["stamp_path"])
    if path.exists():
        path.unlink()
