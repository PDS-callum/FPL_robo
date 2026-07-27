"""Interruptible daemon helpers (SIGTERM/SIGINT drain)."""

from __future__ import annotations

import logging
import signal
import time

logger = logging.getLogger("fpl_robo.runtime")

_stop_requested = False


def stop_requested() -> bool:
    return _stop_requested


def request_stop(signum: int | None = None, _frame=None) -> None:
    global _stop_requested
    _stop_requested = True
    if signum is not None:
        logger.info("Stop requested (signal=%s); draining current cycle", signum)
    else:
        logger.info("Stop requested; draining current cycle")


def install_stop_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that set a drain flag."""
    global _stop_requested
    _stop_requested = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def interruptible_sleep(seconds: float, *, chunk: float = 1.0) -> bool:
    """Sleep up to `seconds`, returning False if stop was requested."""
    deadline = time.time() + max(0.0, float(seconds))
    while time.time() < deadline:
        if _stop_requested:
            return False
        time.sleep(min(chunk, max(0.0, deadline - time.time())))
    return not _stop_requested
