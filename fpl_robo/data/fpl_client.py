"""Official FPL API client."""

from __future__ import annotations

from typing import Any

import requests

FPL_BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Thin client for public FPL endpoints used in step 1."""

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
        """Players, teams, events (gameweeks), element types."""
        return self.get_json("bootstrap-static/")

    def fixtures(self) -> list[dict[str, Any]]:
        """All fixtures for the current season."""
        data = self.get_json("fixtures/")
        if not isinstance(data, list):
            raise TypeError("Unexpected fixtures payload")
        return data

    def entry(self, manager_id: int) -> dict[str, Any]:
        """Manager summary (name, bank, overall rank, etc.)."""
        return self.get_json(f"entry/{manager_id}/")

    def entry_history(self, manager_id: int) -> dict[str, Any]:
        return self.get_json(f"entry/{manager_id}/history/")

    def entry_picks(self, manager_id: int, gameweek: int) -> dict[str, Any]:
        """Picks for a specific gameweek."""
        return self.get_json(f"entry/{manager_id}/event/{gameweek}/picks/")

    def entry_picks_latest(
        self, manager_id: int, gameweek: int | None = None, lookback: int = 3
    ) -> tuple[int, dict[str, Any]] | None:
        """Fetch picks for gameweek, falling back to previous GWs if needed."""
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
