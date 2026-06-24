"""Tests for the yfinance-backed analyst-estimates fetcher (no API key)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.fetchers.analyst_estimates import (
    AnalystEstimatesStore,
    consensus_score,
    fetch_analyst_snapshot,
)


# ── consensus_score ───────────────────────────────────────────────────────
def test_consensus_score_all_strong_buy():
    s, n = consensus_score(10, 0, 0, 0, 0)
    assert s == 2.0 and n == 10


def test_consensus_score_balanced_is_zero():
    s, n = consensus_score(2, 0, 0, 0, 2)  # +2*2 - 2*2 = 0
    assert s == 0.0 and n == 4


def test_consensus_score_no_coverage_is_nan():
    s, n = consensus_score(0, 0, 0, 0, 0)
    assert np.isnan(s) and n == 0


# ── fetch_analyst_snapshot via a mock yfinance Ticker ─────────────────────
def _fake_ticker(_symbol):
    return SimpleNamespace(
        recommendations=pd.DataFrame([
            {"period": "2026-06", "strongBuy": 30, "buy": 18, "hold": 5, "sell": 1, "strongSell": 0},
            {"period": "2026-05", "strongBuy": 28, "buy": 18, "hold": 6, "sell": 2, "strongSell": 0},
        ]),
        analyst_price_targets={"current": 100.0, "high": 160.0, "low": 80.0,
                               "mean": 130.0, "median": 128.0},
        eps_trend=pd.DataFrame({
            "current": [5.0, 6.0], "30daysAgo": [4.9, 5.7], "90daysAgo": [4.5, 5.0],
        }, index=["0y", "+1y"]),
        eps_revisions=pd.DataFrame({
            "upLast30days": [12, 20], "downLast30days": [3, 4],
        }, index=["0y", "+1y"]),
        upgrades_downgrades=pd.DataFrame(
            {"Action": ["up", "up", "down", "main"]},
            index=pd.to_datetime(["2026-06-10", "2026-05-15", "2026-04-20", "2026-01-01"]),
        ),
    )


def test_fetch_snapshot_computes_features():
    r = fetch_analyst_snapshot("XYZ", pd.Timestamp("2026-06-24"), ticker_factory=_fake_ticker)
    assert r["ticker"] == "XYZ" and r["asof"] == pd.Timestamp("2026-06-24")
    # consensus from latest month (30/18/5/1/0, total 54)
    s, n = consensus_score(30, 18, 5, 1, 0)
    assert r["consensus_score"] == pytest.approx(s) and r["n_analysts"] == n
    # implied upside = 130/100 - 1 = 0.30
    assert r["implied_upside"] == pytest.approx(0.30)
    # +1y EPS revision 30d = 6.0/5.7 - 1
    assert r["eps_rev_30d"] == pytest.approx(6.0 / 5.7 - 1)
    assert r["eps_rev_90d"] == pytest.approx(6.0 / 5.0 - 1)
    assert r["eps_up_30d"] == 20 and r["eps_down_30d"] == 4
    # net upgrades in last 90d: up(06-10)+up(05-15)-down(04-20) = +1; 01-01 outside window
    assert r["net_upgrades_90d"] == 1.0


def test_fetch_snapshot_defensive_on_empty():
    r = fetch_analyst_snapshot("EMPTY", pd.Timestamp("2026-06-24"),
                               ticker_factory=lambda s: SimpleNamespace())
    assert r["ticker"] == "EMPTY"
    assert np.isnan(r["consensus_score"]) and r["n_analysts"] == 0
    assert np.isnan(r["implied_upside"])


# ── AnalystEstimatesStore append-merge ────────────────────────────────────
def test_store_upsert_accumulates_and_dedups(tmp_path):
    store = AnalystEstimatesStore(tmp_path / "analyst_estimates.parquet")
    store.upsert([{"ticker": "AAPL", "asof": pd.Timestamp("2026-06-23"), "consensus_score": 0.5}])
    store.upsert([{"ticker": "AAPL", "asof": pd.Timestamp("2026-06-24"), "consensus_score": 0.6}])
    # re-write same (ticker, asof) → keep latest, no dup
    df = store.upsert([{"ticker": "AAPL", "asof": pd.Timestamp("2026-06-24"), "consensus_score": 0.7}])
    aapl = df[df["ticker"] == "AAPL"]
    assert len(aapl) == 2  # two distinct asof dates, deduped
    assert aapl[aapl["asof"] == pd.Timestamp("2026-06-24")]["consensus_score"].iloc[0] == 0.7


# ── refresh CLI (mocked, no network) ──────────────────────────────────────
def test_refresh_appends_and_skips(tmp_path):
    from renquant_base_data.analyst_estimates_refresh import refresh_analyst_estimates
    out = tmp_path / "analyst_estimates.parquet"
    summary = refresh_analyst_estimates(
        watchlist=["XYZ", "EMPTY"], output=out,
        asof=pd.Timestamp("2026-06-24"), sleep_sec=0,
        ticker_factory=lambda s: _fake_ticker(s) if s == "XYZ" else SimpleNamespace(),
    )
    assert summary["ok"] == 1 and summary["skipped"] == 1
    df = pd.read_parquet(out)
    assert set(df["ticker"]) == {"XYZ"}  # EMPTY had no coverage → skipped


def test_load_watchlist_accepts_list_and_config(tmp_path):
    from renquant_base_data.analyst_estimates_refresh import load_watchlist
    p1 = tmp_path / "a.json"; p1.write_text('["AAPL","ZM","-"]')
    assert load_watchlist(p1) == ["AAPL", "ZM"]  # junk "-" filtered
    p2 = tmp_path / "b.json"; p2.write_text('{"watchlist":["NVDA"]}')
    assert load_watchlist(p2) == ["NVDA"]
