"""Email alerts package."""

from fpl_robo.notify.email_alerts import (
    alerts_configured,
    clear_auth_alert_stamp,
    notify_auth_failure,
    send_email,
)

__all__ = [
    "alerts_configured",
    "clear_auth_alert_stamp",
    "notify_auth_failure",
    "send_email",
]
