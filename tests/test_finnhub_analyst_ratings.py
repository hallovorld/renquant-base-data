"""Tests for the Finnhub analyst-recommendation fetcher (no network)."""
from __future__ import annotations

import pandas as pd

from renquant_base_data.fetchers.finnhub_analyst_ratings import (
    SOURCE,
    FinnhubRatingsStore,
    fetch_recommendations,
    parse_recommendations,
)
from renquant_base_data.fetchers.fmp_analyst_ratings import (
    FETCH_ERROR,
    NO_COVERAGE,
    WITH_DATA,
)


def _payload():
    return [
        {"symbol": "AAPL", "period": "2026-06-01", "strongBuy": 12, "buy": 20,
         "hold": 8, "sell": 1, "strongSell": 0},
        {"symbol": "AAPL", "period": "2026-05-01", "strongBuy": 10, "buy": 21,
         "hold": 9, "sell": 1, "strongSell": 1},
    ]


def test_parse_computes_consensus_sorts_and_stamps_source():
    df = parse_recommendations("AAPL", _payload())
    assert list(df["period"]) == [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")]
    assert (df["source"] == SOURCE).all()
    last = df.iloc[-1]
    # consensus = (2*12 + 20 - 1 - 0) / 41
    assert abs(last["consensus"] - (2 * 12 + 20 - 1 - 0) / 41) < 1e-9
    assert last["n_analysts"] == 41


def test_fetch_with_data_and_etf_no_coverage():
    ok = fetch_recommendations("AAPL", "k", getter=lambda t: _payload())
    assert ok.status == WITH_DATA and len(ok.frame) == 2
    # an ETF returns an empty list → no_coverage (NOT an error)
    etf = fetch_recommendations("SPY", "k", getter=lambda t: [])
    assert etf.status == NO_COVERAGE and etf.frame.empty
    # a thrown error → fetch_error, never raises
    boom = fetch_recommendations("AAPL", "k",
                                 getter=lambda t: (_ for _ in ()).throw(ValueError("net")))
    assert boom.status == FETCH_ERROR


def test_store_append_merge_dedup(tmp_path):
    store = FinnhubRatingsStore(tmp_path / "r.parquet")
    store.upsert([parse_recommendations("AAPL", _payload())])
    p2 = _payload(); p2[0]["strongBuy"] = 15           # same period, changed count
    df = store.upsert([parse_recommendations("AAPL", p2)])
    aapl = df[df["ticker"] == "AAPL"]
    assert len(aapl) == 2                               # two distinct periods, no dup
    jun = aapl[aapl["period"] == pd.Timestamp("2026-06-01")].iloc[0]
    assert jun["strongBuy"] == 15                       # latest write kept


def test_refresh_buckets_etf_as_no_coverage_not_error(tmp_path):
    from renquant_base_data import finnhub_analyst_ratings_refresh as R
    out = tmp_path / "r.parquet"
    def getter(t):
        return _payload() if t in ("AAPL", "MSFT") else []   # SPY/GLD ETFs → empty
    s = R.refresh_finnhub_ratings(watchlist=["AAPL", "MSFT", "SPY", "GLD"], output=out,
                                  api_key="k", sleep_sec=0, max_pull=0,
                                  asof=pd.Timestamp("2026-06-25"), getter=getter)
    assert s["with_data"] == 2 and s["no_coverage"] == 2
    assert s["errors_total"] == 0                       # ETFs are NOT errors
    assert s["coverable"] == 2 and s["coverage_pct"] == 100.0   # over coverable (non-ETF)
