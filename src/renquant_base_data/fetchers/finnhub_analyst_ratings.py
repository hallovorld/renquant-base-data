"""Finnhub analyst recommendation trends (free ``/stock/recommendation``).

Finnhub's free tier exposes monthly analyst recommendation distributions
(strongBuy/buy/hold/sell/strongSell), ~4 months per US stock, with **FULL stock
coverage** — unlike FMP free's ~30% plan-lock (HTTP 402). One call per ticker;
free tier is 60 calls/min. ETFs / indices have no analyst coverage (empty list →
``no_coverage``, not an error).

The 4-month window is short for a multi-year backtest, but it is enough for a
LIVE full-coverage consensus + 3-month REVISION feature, and a DAILY cron
accumulates the time-series over months (dedup by (ticker, period), keep latest).
Reuses the source-agnostic ``consensus_score`` / ``FetchResult`` / status
contract from the FMP fetcher so both feed the same downstream shape.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from renquant_base_data.fetchers.fmp_analyst_ratings import (
    FETCH_ERROR,
    NO_COVERAGE,
    QUOTA_ERROR,
    WITH_DATA,
    FetchResult,
    consensus_score,
)

log = logging.getLogger("kernel.finnhub_analyst_ratings")

RECO_URL = "https://finnhub.io/api/v1/stock/recommendation"
SOURCE = "finnhub_recommendation"
RATING_COLS = ("strongBuy", "buy", "hold", "sell", "strongSell")
_COLS = ["ticker", "period", *RATING_COLS, "consensus", "n_analysts", "source", "fetched_at"]


def parse_recommendations(ticker: str, payload: Any, asof=None) -> pd.DataFrame:
    """Finnhub recommendation JSON → tidy frame. ``period`` is the rating month;
    ``fetched_at`` (the staleness key) is WHEN we pulled it; ``source`` stamps
    provenance."""
    fetched = (pd.Timestamp(asof).normalize() if asof is not None
               else pd.Timestamp.today().normalize())
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=_COLS)
    rows = []
    for r in payload:
        if not isinstance(r, dict) or "period" not in r:
            continue
        sc, n = consensus_score(*(r.get(c) for c in RATING_COLS))
        rows.append({"ticker": ticker, "period": pd.to_datetime(r["period"]),
                     **{c: int(r.get(c, 0) or 0) for c in RATING_COLS},
                     "consensus": sc, "n_analysts": n,
                     "source": SOURCE, "fetched_at": fetched})
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def fetch_recommendations(ticker: str, api_key: str, *, timeout: float = 20.0,
                          asof=None, getter: Callable[..., Any] | None = None) -> FetchResult:
    """One ticker's recommendation trend as a :class:`FetchResult`. Never raises.
    Status: with_data / no_coverage (empty — e.g. an ETF) / quota_error (429) /
    fetch_error (bad key 401/403, schema, network). ``getter`` injectable for tests."""
    empty = parse_recommendations(ticker, None, asof=asof)
    try:
        if getter is None:
            import requests  # noqa: PLC0415
            resp = requests.get(RECO_URL, params={"symbol": ticker, "token": api_key},
                                timeout=timeout)
            if resp.status_code == 429:
                return FetchResult(empty, QUOTA_ERROR, "HTTP 429 rate limit")
            if resp.status_code in (401, 403):
                return FetchResult(empty, FETCH_ERROR, f"HTTP {resp.status_code} (key/plan)")
            payload = resp.json()
        else:
            payload = getter(ticker)
    except Exception as exc:  # noqa: BLE001
        log.warning("finnhub reco fetch failed for %s: %s", ticker, exc)
        return FetchResult(empty, FETCH_ERROR, str(exc)[:160])
    if isinstance(payload, dict) and any(k.lower() in ("error",) for k in payload):
        return FetchResult(empty, FETCH_ERROR, str(payload)[:160])
    if not isinstance(payload, list):
        return FetchResult(empty, FETCH_ERROR, f"unexpected payload {type(payload).__name__}")
    frame = parse_recommendations(ticker, payload, asof=asof)
    return FetchResult(frame, WITH_DATA if len(frame) else NO_COVERAGE)


@dataclass
class FinnhubRatingsStore:
    """Append-merge panel: one row per (ticker, period); keeps the latest write —
    so a daily cron accumulates the recommendation history over time."""

    path: Path

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.path) if Path(self.path).exists() else pd.DataFrame(columns=_COLS)

    def upsert(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return self.load()
        new = pd.concat(frames, ignore_index=True)
        existing = self.load()
        combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
        combined = (combined.drop_duplicates(subset=["ticker", "period"], keep="last")
                    .sort_values(["ticker", "period"]).reset_index(drop=True))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        return combined
