"""Regression: the incremental OHLCV fetch must refetch when the cache lags the
last COMPLETED trading session — not silently serve a cache that is merely
"within 2 calendar days".

The prior heuristic (``cache_last_date >= end - 2 calendar days``) called a
2-session-stale cache "fresh" and returned it without fetching the available
delta, deadlocking the retrain's 1-session freshness guard (32d-stale WF artifact).
"""
from __future__ import annotations

import pandas as pd
import pytest

import renquant_base_data.loaders.data as mod
from renquant_base_data.loaders.data import (
    LocalStore,
    _last_completed_nyse_session,
    fetch_ohlcv_incremental,
)


def _ohlcv(idx: pd.DatetimeIndex, v: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": v, "high": v, "low": v, "close": v, "volume": v}, index=idx
    )


def test_incremental_refetches_when_cache_lags_completed_session(tmp_path, monkeypatch):
    end = pd.Timestamp("2026-07-23")  # a Thursday
    last_complete = _last_completed_nyse_session(end)
    if last_complete is None:
        pytest.skip("NYSE calendar unavailable — fix reduces to the 2-day fallback")

    store = LocalStore(data_dir=tmp_path)
    # Seed a cache that lands in the OLD cache-hit window (>= end-2d) but is
    # still behind the last completed session — exactly the deadlock zone.
    cache_end = pd.Timestamp(last_complete) - pd.tseries.offsets.BDay(1)
    assert cache_end >= end - pd.Timedelta(days=2)  # old code WOULD cache-hit here
    store.save(_ohlcv(pd.bdate_range("2026-06-15", cache_end), 1.0), "XOM", "1d")

    calls = {"n": 0}

    def fake_call_with_timeout(fn, *a, **k):
        calls["n"] += 1
        new_idx = pd.bdate_range(cache_end + pd.tseries.offsets.BDay(1), end)
        return _ohlcv(new_idx, 2.0)

    monkeypatch.setattr(
        "renquant_common.net_safety.call_with_timeout", fake_call_with_timeout
    )

    out = fetch_ohlcv_incremental("XOM", end=str(end.date()), store=store, timeout_sec=30)

    assert calls["n"] >= 1, "must refetch the delta, not serve the stale cache"
    assert pd.to_datetime(out.index).max().date() >= last_complete, "cache must reach the completed session"


def test_incremental_serves_fresh_cache_without_network(tmp_path, monkeypatch):
    """Inverse: a cache that already reaches the completed session is served
    without any network call (no over-fetching regression)."""
    end = pd.Timestamp("2026-07-23")
    last_complete = _last_completed_nyse_session(end)
    if last_complete is None:
        pytest.skip("NYSE calendar unavailable")

    store = LocalStore(data_dir=tmp_path)
    store.save(_ohlcv(pd.bdate_range("2026-06-15", last_complete), 1.0), "XOM", "1d")

    def boom(*a, **k):  # network must NOT be called
        raise AssertionError("fresh cache should not hit the network")

    monkeypatch.setattr("renquant_common.net_safety.call_with_timeout", boom)
    out = fetch_ohlcv_incremental("XOM", end=str(end.date()), store=store)
    assert pd.to_datetime(out.index).max().date() >= last_complete
