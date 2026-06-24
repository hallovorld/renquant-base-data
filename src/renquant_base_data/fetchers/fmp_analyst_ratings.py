"""FMP historical analyst rating distributions (free `grades-historical`).

FMP's free BASIC tier exposes ``/stable/grades-historical`` — ~7.5 years of
MONTHLY analyst rating distributions (strongBuy/buy/hold/sell/strongSell) per
US ticker. That is enough to build a consensus_score time series and, crucially,
the consensus REVISION (Δ over months) — the documented post-revision-drift
alpha — with real backtestable history (vs yfinance's 4-month recommendations).

Free-tier limits: 250 calls/day, and a per-minute cap (rapid pulls 429). One
ticker = one call returning its full history, so the ~142-name watchlist fits in
a day; the caller must THROTTLE (~1s) to avoid the per-minute cap. Designed for a
weekly cron (ratings update monthly).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.fmp_analyst_ratings")

GRADES_URL = "https://financialmodelingprep.com/stable/grades-historical"
RATING_COLS = ("analystRatingsStrongBuy", "analystRatingsBuy", "analystRatingsHold",
               "analystRatingsSell", "analystRatingsStrongSell")


def consensus_score(sb, b, h, s, ss) -> "tuple[float, int]":
    """(score in [-2,2], n_analysts). Empty coverage → (nan, 0)."""
    vals = [int(x or 0) for x in (sb, b, h, s, ss)]
    total = sum(vals)
    if total <= 0:
        return float("nan"), 0
    sb, b, h, s, ss = vals
    return (2 * sb + b - s - 2 * ss) / total, total


def parse_grades(ticker: str, payload: Any) -> pd.DataFrame:
    """FMP grades-historical JSON → tidy frame (date, counts, consensus, n)."""
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["ticker", "date", *RATING_COLS, "consensus", "n_analysts"])
    rows = []
    for r in payload:
        if not isinstance(r, dict) or "date" not in r:
            continue
        sc, n = consensus_score(*(r.get(c) for c in RATING_COLS))
        rows.append({"ticker": ticker, "date": pd.to_datetime(r["date"]),
                     **{c: int(r.get(c, 0) or 0) for c in RATING_COLS},
                     "consensus": sc, "n_analysts": n})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_grades_historical(ticker: str, api_key: str, *, timeout: float = 20.0,
                            getter: Callable[..., Any] | None = None) -> pd.DataFrame:
    """One ticker's full rating history. ``getter`` injectable for tests.
    Returns empty frame on any error (never raises)."""
    try:
        if getter is None:
            import requests  # noqa: PLC0415
            resp = requests.get(GRADES_URL, params={"symbol": ticker, "apikey": api_key},
                                timeout=timeout)
            payload = resp.json()
        else:
            payload = getter(ticker)
        # surface quota/restriction as a signal, not data
        if isinstance(payload, dict) and any(k.lower().startswith(("error", "message"))
                                             for k in payload):
            raise RuntimeError(str(payload)[:120])
        return parse_grades(ticker, payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("fmp grades fetch failed for %s: %s", ticker, exc)
        return parse_grades(ticker, None)


@dataclass
class FmpRatingsStore:
    """Append-merge panel: one row per (ticker, date); keeps the latest write."""

    path: Path

    def load(self) -> pd.DataFrame:
        if Path(self.path).exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=["ticker", "date", *RATING_COLS, "consensus", "n_analysts"])

    def upsert(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return self.load()
        new = pd.concat(frames, ignore_index=True)
        existing = self.load()
        combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
        combined = (combined.drop_duplicates(subset=["ticker", "date"], keep="last")
                    .sort_values(["ticker", "date"]).reset_index(drop=True))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        return combined
