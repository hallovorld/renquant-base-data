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
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.fmp_analyst_ratings")

GRADES_URL = "https://financialmodelingprep.com/stable/grades-historical"
SOURCE = "fmp_grades_historical"  # provenance stamp on every persisted row
RATING_COLS = ("analystRatingsStrongBuy", "analystRatingsBuy", "analystRatingsHold",
               "analystRatingsSell", "analystRatingsStrongSell")

# Per-ticker fetch outcomes — kept DISTINCT so a quota/API failure can never be
# silently collapsed into "this ticker has no ratings" (Codex #24 finding #1).
WITH_DATA = "with_data"      # got rating rows
NO_COVERAGE = "no_coverage"  # valid empty response: ticker genuinely has no grades
QUOTA_ERROR = "quota_error"  # free-tier 429 / "limit reached" (TRANSIENT — retry later)
FETCH_ERROR = "fetch_error"  # bad key, schema change, network/JSON failure
# PERMANENT plan ceiling, NOT a transient error: FMP free BASIC returns HTTP 402
# "Special Endpoint … not available under your current subscription" for symbols
# outside the free set (~70% of a large-cap watchlist, verified 2026-06-24). It
# can never be retried into success on this plan, so it is bucketed apart from
# the retryable errors and never counts against the error gate.
PREMIUM_RESTRICTED = "premium_restricted"
_QUOTA_HINTS = ("limit reach", "limit reached", "too many requests", "rate limit",
                "429", "quota", "exceeded")
_PREMIUM_HINTS = ("special endpoint", "not available under your current subscription",
                  "premium", "upgrade your plan")


class FetchResult(NamedTuple):
    """One ticker's pull: the parsed frame PLUS a non-collapsible status so the
    caller can gate on coverage vs. error (never treat a 429 as 'no data')."""

    frame: pd.DataFrame
    status: str
    detail: str = ""


def consensus_score(sb, b, h, s, ss) -> "tuple[float, int]":
    """(score in [-2,2], n_analysts). Empty coverage → (nan, 0)."""
    vals = [int(x or 0) for x in (sb, b, h, s, ss)]
    total = sum(vals)
    if total <= 0:
        return float("nan"), 0
    sb, b, h, s, ss = vals
    return (2 * sb + b - s - 2 * ss) / total, total


_COLS = ["ticker", "date", *RATING_COLS, "consensus", "n_analysts",
         "source", "fetched_at"]


def parse_grades(ticker: str, payload: Any, asof=None) -> pd.DataFrame:
    """FMP grades-historical JSON → tidy frame. ``source`` stamps the vendor
    endpoint and ``fetched_at`` stamps WHEN we pulled it (the staleness key for
    incremental refresh), distinct from the rating-month ``date`` — together they
    give every row auditable provenance (Codex #24 finding #3)."""
    fetched = (pd.Timestamp(asof).normalize() if asof is not None
               else pd.Timestamp.today().normalize())
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=_COLS)
    rows = []
    for r in payload:
        if not isinstance(r, dict) or "date" not in r:
            continue
        sc, n = consensus_score(*(r.get(c) for c in RATING_COLS))
        rows.append({"ticker": ticker, "date": pd.to_datetime(r["date"]),
                     **{c: int(r.get(c, 0) or 0) for c in RATING_COLS},
                     "consensus": sc, "n_analysts": n,
                     "source": SOURCE, "fetched_at": fetched})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _classify_text(detail: str, status_code: "int | None" = None) -> str:
    """Map an error body / HTTP status to a status. PERMANENT plan locks
    (402 / "Special Endpoint") rank ahead of transient quota/rate hints so an
    upgrade-required symbol is never retried forever as a quota_error."""
    low = (detail or "").lower()
    if status_code == 402 or any(h in low for h in _PREMIUM_HINTS):
        return PREMIUM_RESTRICTED
    if status_code == 429 or any(h in low for h in _QUOTA_HINTS):
        return QUOTA_ERROR
    return FETCH_ERROR


def fetch_grades_historical(ticker: str, api_key: str, *, timeout: float = 20.0,
                            asof=None, getter: Callable[..., Any] | None = None) -> FetchResult:
    """One ticker's full rating history as a :class:`FetchResult` (frame + status).

    Never raises. The status DISTINGUISHES an empty-but-valid response
    (``no_coverage``), a permanent plan lock (``premium_restricted`` — FMP free
    402 "Special Endpoint"), a transient quota hit (``quota_error``), and any
    other failure (``fetch_error``), so the refresh layer can fail-closed on real
    errors while treating the plan ceiling as expected. ``getter`` injectable for
    tests (return a list, an ``{"Error Message": ...}`` dict, or a raw error
    string to exercise each branch)."""
    empty = parse_grades(ticker, None, asof=asof)
    try:
        if getter is None:
            import requests  # noqa: PLC0415
            resp = requests.get(GRADES_URL, params={"symbol": ticker, "apikey": api_key},
                                timeout=timeout)
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001 — FMP 402/429 return a plain-text body
                st = _classify_text(resp.text, resp.status_code)
                log.warning("fmp grades %s for %s: HTTP %s %s",
                            st, ticker, resp.status_code, (resp.text or "")[:120])
                return FetchResult(empty, st, (resp.text or "")[:160])
        else:
            payload = getter(ticker)
    except Exception as exc:  # noqa: BLE001
        log.warning("fmp grades fetch failed for %s: %s", ticker, exc)
        return FetchResult(empty, FETCH_ERROR, str(exc)[:160])
    # FMP can also signal restriction/quota as a dict {"Error Message": ...} or a
    # raw string (injected getter) — classify, don't treat as data.
    if isinstance(payload, str):
        st = _classify_text(payload)
        log.warning("fmp grades %s for %s: %s", st, ticker, payload[:120])
        return FetchResult(empty, st, payload[:160])
    if isinstance(payload, dict) and any(k.lower().startswith(("error", "message"))
                                         for k in payload):
        detail = str(payload)[:160]
        status = _classify_text(detail)
        log.warning("fmp grades %s for %s: %s", status, ticker, detail)
        return FetchResult(empty, status, detail)
    if not isinstance(payload, list):
        detail = f"unexpected payload type {type(payload).__name__}"
        log.warning("fmp grades fetch_error for %s: %s", ticker, detail)
        return FetchResult(empty, FETCH_ERROR, detail)
    frame = parse_grades(ticker, payload, asof=asof)
    if len(frame):
        return FetchResult(frame, WITH_DATA)
    return FetchResult(frame, NO_COVERAGE)


@dataclass
class FmpRatingsStore:
    """Append-merge panel: one row per (ticker, date); keeps the latest write."""

    path: Path

    def load(self) -> pd.DataFrame:
        if Path(self.path).exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=_COLS)

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
