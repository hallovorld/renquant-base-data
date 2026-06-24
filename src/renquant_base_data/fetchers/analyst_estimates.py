"""Analyst estimate / revision snapshots (yfinance-backed, NO API key).

yfinance scrapes Yahoo Finance public endpoints — no key/token required. It
exposes richer analyst data than the Finnhub free tier (which is
recommendations-only): recommendation trends, timestamped upgrades/downgrades,
price targets, and EPS-estimate trend/revisions.

This module captures, per ticker, a point-in-time **snapshot row** stamped with
an ``asof`` date, plus derived revision features. Stamping each refresh asof and
appending means a clean PIT history accumulates forward (so a downstream
estimate-revision feature can be validated without look-ahead). The
``upgrades_downgrades`` events are already timestamped, so net-revision-over-
window is PIT from day one.

Schema of ``data/analyst_estimates.parquet`` (one row per ticker × asof):
    ticker, asof,
    consensus_score   float — (2*strongBuy + buy - sell - 2*strongSell)/total, [-2,2]
    n_analysts        int   — total recommending analysts (coverage / confidence)
    implied_upside    float — price_target_mean / current_price - 1
    pt_mean, pt_current float
    eps_fy_curr, eps_fy_30d_ago, eps_fy_90d_ago float — current-FY mean EPS estimate trajectory
    eps_rev_30d, eps_rev_90d float — fractional change in the FY estimate (the revision alpha)
    eps_up_30d, eps_down_30d int — # analysts revising up / down in last 30d
    net_upgrades_90d  int — (#upgrades - #downgrades) in trailing 90 days (timestamped events)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.analyst_estimates")

SNAPSHOT_COLS: list[str] = [
    "consensus_score", "n_analysts", "implied_upside", "pt_mean", "pt_current",
    "eps_fy_curr", "eps_fy_30d_ago", "eps_fy_90d_ago", "eps_rev_30d", "eps_rev_90d",
    "eps_up_30d", "eps_down_30d", "net_upgrades_90d",
]


def _safe(fn: Callable[[], Any], default=np.nan):
    try:
        v = fn()
        return v if v is not None else default
    except Exception:  # noqa: BLE001 — any yfinance shape error → NaN, never raise
        return default


def consensus_score(strong_buy, buy, hold, sell, strong_sell) -> tuple[float, int]:
    """(score in [-2,2], n_analysts). Neutral/no-coverage → (nan, 0)."""
    total = sum(int(x or 0) for x in (strong_buy, buy, hold, sell, strong_sell))
    if total <= 0:
        return float("nan"), 0
    score = (2 * (strong_buy or 0) + (buy or 0) - (sell or 0) - 2 * (strong_sell or 0)) / total
    return float(score), int(total)


def _frac_change(curr, past) -> float:
    try:
        curr = float(curr); past = float(past)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(curr) or not np.isfinite(past) or past == 0:
        return float("nan")
    return curr / past - 1.0


def fetch_analyst_snapshot(ticker: str, asof: pd.Timestamp,
                           ticker_factory: Callable[[str], Any] | None = None) -> dict:
    """One PIT snapshot row for ``ticker``. Defensive: any missing field → NaN."""
    import yfinance as yf  # noqa: PLC0415

    yt = (ticker_factory or yf.Ticker)(ticker)
    row: dict[str, Any] = {"ticker": ticker, "asof": pd.Timestamp(asof).normalize()}

    # 1) recommendation consensus (latest month)
    rec = _safe(lambda: yt.recommendations, default=None)
    sb = b = h = s = ss = 0
    if isinstance(rec, pd.DataFrame) and len(rec):
        last = rec.iloc[0]  # yfinance returns most-recent period first
        sb, b, h, s, ss = (int(last.get(k, 0) or 0) for k in
                           ("strongBuy", "buy", "hold", "sell", "strongSell"))
    score, n = consensus_score(sb, b, h, s, ss)
    row["consensus_score"], row["n_analysts"] = score, n

    # 2) price targets → implied upside
    pt = _safe(lambda: yt.analyst_price_targets, default=None)
    pt_mean = float(pt.get("mean")) if isinstance(pt, dict) and pt.get("mean") else float("nan")
    pt_cur = float(pt.get("current")) if isinstance(pt, dict) and pt.get("current") else float("nan")
    row["pt_mean"], row["pt_current"] = pt_mean, pt_cur
    row["implied_upside"] = _frac_change(pt_mean, pt_cur)

    # 3) EPS-estimate trend → revision drift (current fiscal-year row, label '+1y'/'0y')
    trend = _safe(lambda: yt.eps_trend, default=None)
    curr = d30 = d90 = float("nan")
    if isinstance(trend, pd.DataFrame) and len(trend):
        fy = trend.loc["+1y"] if "+1y" in trend.index else trend.iloc[-1]
        curr = float(_safe(lambda: fy.get("current")))
        d30 = float(_safe(lambda: fy.get("30daysAgo")))
        d90 = float(_safe(lambda: fy.get("90daysAgo")))
    row["eps_fy_curr"], row["eps_fy_30d_ago"], row["eps_fy_90d_ago"] = curr, d30, d90
    row["eps_rev_30d"] = _frac_change(curr, d30)
    row["eps_rev_90d"] = _frac_change(curr, d90)

    # 4) # analysts revising up/down (last 30d), current FY
    revs = _safe(lambda: yt.eps_revisions, default=None)
    up = down = float("nan")
    if isinstance(revs, pd.DataFrame) and len(revs):
        fy = revs.loc["+1y"] if "+1y" in revs.index else revs.iloc[-1]
        up = float(_safe(lambda: fy.get("upLast30days")))
        down = float(_safe(lambda: fy.get("downLast30days")))
    row["eps_up_30d"], row["eps_down_30d"] = up, down

    # 5) net upgrades over trailing 90d (timestamped events = PIT-clean)
    ud = _safe(lambda: yt.upgrades_downgrades, default=None)
    net_up = float("nan")
    if isinstance(ud, pd.DataFrame) and len(ud):
        try:
            idx = pd.to_datetime(ud.index)
            window = ud[idx >= (pd.Timestamp(asof) - pd.Timedelta(days=90))]
            acts = window.get("Action", pd.Series(dtype=str)).astype(str).str.lower()
            net_up = float((acts.str.contains("up")).sum() - (acts.str.contains("down")).sum())
        except Exception:  # noqa: BLE001
            net_up = float("nan")
    row["net_upgrades_90d"] = net_up
    return row


@dataclass
class AnalystEstimatesStore:
    """Append-merge panel store: one row per (ticker, asof), accumulating PIT history."""

    path: Path

    def load(self) -> pd.DataFrame:
        if Path(self.path).exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=["ticker", "asof", *SNAPSHOT_COLS])

    def upsert(self, rows: list[dict]) -> pd.DataFrame:
        new = pd.DataFrame(rows)
        if new.empty:
            return self.load()
        existing = self.load()
        # avoid the all-NA concat FutureWarning when the store is empty
        combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
        # keep the latest write for a given (ticker, asof)
        combined = (combined.drop_duplicates(subset=["ticker", "asof"], keep="last")
                    .sort_values(["ticker", "asof"]).reset_index(drop=True))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.path, index=False)
        return combined
