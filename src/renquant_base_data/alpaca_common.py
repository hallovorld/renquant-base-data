"""Shared Alpaca data-refresh helpers."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path


class TokenBucket:
    """Sliding-window rate limiter."""

    def __init__(self, max_calls: int = 180, window_seconds: float = 60.0) -> None:
        self.max_calls = int(max_calls)
        self.window_seconds = float(window_seconds)
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        now = time.time()
        while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.window_seconds - (now - self._timestamps[0]) + 0.05
            time.sleep(max(0.05, sleep_for))
            now = time.time()
            while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
                self._timestamps.popleft()
        self._timestamps.append(now)


def load_strategy_watchlist(path: str | Path) -> list[str]:
    """Read symbols from strategy_config.json's top-level ``watchlist`` field.

    Other layouts such as ``symbols`` or ``data.watchlist`` are intentionally
    not accepted here; Alpaca refresh callers share one production config
    schema, and silent fallback between schemas can mask typos.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("watchlist")
    if not values:
        raise ValueError(f"strategy_config missing top-level 'watchlist': {path}")
    return [str(symbol).upper() for symbol in values if symbol and str(symbol) != "-"]
