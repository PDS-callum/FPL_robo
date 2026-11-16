"""LAN HTTP dashboard for FPL Robo."""

from __future__ import annotations

import json
import logging
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fpl_robo.config import DASHBOARD, DEFAULT_MANAGER_ID
from fpl_robo.dashboard.data import build_dashboard_payload
from fpl_robo.runtime.interrupt import install_stop_handlers, stop_requested

logger = logging.getLogger("fpl_robo.dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Inf so browser JSON.parse accepts the payload."""
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


class DashboardHandler(SimpleHTTPRequestHandler):
    manager_id: int = DEFAULT_MANAGER_ID

    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def end_headers(self) -> None:
        # Avoid sticky empty/partial caches for the dashboard HTML/JS bundle.
        path = urlparse(getattr(self, "path", "") or "").path
        if path.endswith((".html", ".js", ".css")) or path in ("/", "/index.html", ""):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/api/state", "/api/state.json"):
            self._send_json(build_dashboard_payload(manager_id=self.manager_id))
            return
        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
        elif parsed.path in ("/fpl_hub", "/fpl_hub/"):
            self.path = "/fpl_hub.html"
        return super().do_GET()

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(_json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_dashboard(
    *,
    host: str | None = None,
    port: int | None = None,
    manager_id: int = DEFAULT_MANAGER_ID,
) -> None:
    host = host or str(DASHBOARD["host"])
    port = int(port or DASHBOARD["port"])
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("logs/fpl_robo_dashboard.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    install_stop_handlers()
    DashboardHandler.manager_id = manager_id
    handler = partial(DashboardHandler, directory=str(STATIC_DIR))
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.timeout = 1.0
    logger.info("Dashboard listening on http://%s:%s", host, port)
    try:
        while not stop_requested():
            server.handle_request()
    finally:
        server.server_close()
        logger.info("Dashboard stopped")
