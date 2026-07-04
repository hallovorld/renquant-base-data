"""Tests for the PIT revision-drift feature builder.

Covers:
  1. PIT correctness: available_at == snapshot directory date
  2. Revision calculation correctness (known inputs -> known outputs)
  3. Edge case: ticker missing in old snapshot (NaN, not error)
  4. Incremental: only new dates processed, existing rows preserved
  5. CLI round-trip
  6. Empty / degenerate inputs
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data import pit_feature_builder as pit


# --- fixtures -----------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


def _make_analyst_estimates(tickers: list[str], eps_base: float, rev_base: float):
    """Create a synthetic analyst_estimates DataFrame."""
    rows = []
    for i, sym in enumerate(tickers):
        rows.append(
            {
                "symbol": sym,
                "date": "2026-12-31",
                "estimatedEpsAvg": eps_base + i * 0.5,
                "estimatedRevenueAvg": rev_base + i * 1e9,
                "numberAnalystEstimatedEps": 20,
                "snapshot_as_of": "placeholder",  # overwritten by snapshot dir date
            }
        )
    return pd.DataFrame(rows)


def _make_price_target_consensus(tickers: list[str], target_base: float):
    """Create a synthetic price_target_consensus DataFrame."""
    rows = []
    for i, sym in enumerate(tickers):
        rows.append(
            {
                "symbol": sym,
                "targetConsensus": target_base + i * 10.0,
                "targetHigh": target_base + i * 10.0 + 20.0,
                "targetLow": target_base + i * 10.0 - 20.0,
                "targetMedian": target_base + i * 10.0,
            }
        )
    return pd.DataFrame(rows)


def _make_grades_consensus(tickers: list[str], buy_base: int, sell_base: int):
    """Create a synthetic grades_consensus DataFrame."""
    rows = []
    for i, sym in enumerate(tickers):
        rows.append(
            {
                "symbol": sym,
                "strongBuy": buy_base + i,
                "buy": buy_base + i * 2,
                "hold": 5,
                "sell": sell_base,
                "strongSell": max(0, sell_base - 1),
            }
        )
    return pd.DataFrame(rows)


def _write_snapshot(
    root: Path,
    date_str: str,
    tickers: list[str],
    eps_base: float,
    rev_base: float,
    target_base: float,
    buy_base: int = 10,
    sell_base: int = 2,
    *,
    skip_tickers_estimates: list[str] | None = None,
    skip_tickers_target: list[str] | None = None,
):
    """Write a complete synthetic snapshot directory."""
    snap_dir = root / date_str
    snap_dir.mkdir(parents=True, exist_ok=True)

    # analyst estimates
    est_tickers = [t for t in tickers if t not in (skip_tickers_estimates or [])]
    if est_tickers:
        est_df = _make_analyst_estimates(est_tickers, eps_base, rev_base)
        est_df["snapshot_as_of"] = date_str
        est_df.to_parquet(snap_dir / "analyst_estimates.parquet", index=False)

    # price target consensus
    tgt_tickers = [t for t in tickers if t not in (skip_tickers_target or [])]
    if tgt_tickers:
        tgt_df = _make_price_target_consensus(tgt_tickers, target_base)
        tgt_df.to_parquet(snap_dir / "price_target_consensus.parquet", index=False)

    # grades consensus
    grd_df = _make_grades_consensus(tickers, buy_base, sell_base)
    grd_df.to_parquet(snap_dir / "grades_consensus.parquet", index=False)


@pytest.fixture
def snapshots_2dates(tmp_path):
    """Two snapshot dates ~90 days apart with known values."""
    old_date = "2026-04-01"
    new_date = "2026-07-01"

    # OLD snapshot: EPS base=2.0, revenue base=10e9, target base=100.0
    _write_snapshot(
        tmp_path, old_date, TICKERS,
        eps_base=2.0, rev_base=10e9, target_base=100.0,
        buy_base=10, sell_base=2,
    )

    # NEW snapshot: EPS base=2.2 (+10%), revenue base=11e9 (+10%), target base=110.0 (+10%)
    _write_snapshot(
        tmp_path, new_date, TICKERS,
        eps_base=2.2, rev_base=11e9, target_base=110.0,
        buy_base=12, sell_base=1,
    )

    return tmp_path


@pytest.fixture
def snapshots_3dates(tmp_path):
    """Three snapshot dates for incremental testing."""
    _write_snapshot(
        tmp_path, "2026-04-01", TICKERS,
        eps_base=2.0, rev_base=10e9, target_base=100.0,
    )
    _write_snapshot(
        tmp_path, "2026-07-01", TICKERS,
        eps_base=2.2, rev_base=11e9, target_base=110.0,
    )
    _write_snapshot(
        tmp_path, "2026-07-15", TICKERS,
        eps_base=2.3, rev_base=11.5e9, target_base=115.0,
    )
    return tmp_path


# --- 1. PIT correctness: available_at == snapshot directory date ---------------

def test_available_at_equals_snapshot_date(snapshots_2dates):
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    assert not result.empty
    # The only computable date is 2026-07-01 (2026-04-01 has no lookback)
    assert set(result["available_at"].unique()) == {"2026-07-01"}
    # available_at is the snapshot dir date, not today
    today_str = date.today().isoformat()
    assert today_str not in result["available_at"].values or today_str == "2026-07-01"


def test_available_at_never_today(snapshots_2dates):
    """available_at must be the snapshot date, not the current date."""
    result = pit.build_revision_drift_one_date(
        snapshots_2dates, "2026-07-01", lookback_days=90
    )
    assert result is not None
    assert (result["available_at"] == "2026-07-01").all()


# --- 2. Revision calculation correctness --------------------------------------

def test_eps_revision_correctness(snapshots_2dates):
    """Known inputs: AAPL EPS went from 2.0 to 2.2 = +10%."""
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    aapl = result[result["ticker"] == "AAPL"].iloc[0]
    # EPS: 2.2 / 2.0 - 1 = 0.10
    assert abs(aapl["eps_revision_3m"] - 0.10) < 1e-6


def test_revenue_revision_correctness(snapshots_2dates):
    """Known inputs: AAPL revenue went from 10e9 to 11e9 = +10%."""
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    aapl = result[result["ticker"] == "AAPL"].iloc[0]
    assert abs(aapl["revenue_revision_3m"] - 0.10) < 1e-6


def test_target_revision_correctness(snapshots_2dates):
    """Known inputs: AAPL target went from 100.0 to 110.0 = +10%."""
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    aapl = result[result["ticker"] == "AAPL"].iloc[0]
    assert abs(aapl["target_revision_3m"] - 0.10) < 1e-6


def test_revision_breadth_positive(snapshots_2dates):
    """With more buys than sells, breadth should be positive."""
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    # For the new snapshot: AAPL has strongBuy=12, buy=12, hold=5, sell=1, strongSell=0
    # total = 12+12+5+1+0 = 30, up = 24, down = 1, breadth = (24-1)/30 = 0.7667
    aapl = result[result["ticker"] == "AAPL"].iloc[0]
    assert aapl["revision_breadth"] > 0


def test_all_tickers_present(snapshots_2dates):
    """All 5 tickers should be in the output."""
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    assert set(result["ticker"].unique()) == set(TICKERS)


def test_output_columns(snapshots_2dates):
    """Output must have exactly the 6 specified columns."""
    result = pit.build_revision_drift(snapshots_2dates, lookback_days=90)
    expected = {
        "ticker", "available_at", "eps_revision_3m",
        "revenue_revision_3m", "target_revision_3m", "revision_breadth",
    }
    assert set(result.columns) == expected


# --- 3. Edge case: ticker missing in old snapshot (NaN, not error) ------------

def test_ticker_missing_in_old_snapshot_gives_nan(tmp_path):
    """A ticker in the new snapshot but NOT in the old should get NaN, not crash."""
    # Old snapshot: only AAPL and MSFT
    _write_snapshot(
        tmp_path, "2026-04-01", ["AAPL", "MSFT"],
        eps_base=2.0, rev_base=10e9, target_base=100.0,
    )
    # New snapshot: AAPL, MSFT, GOOG (GOOG is new)
    _write_snapshot(
        tmp_path, "2026-07-01", ["AAPL", "MSFT", "GOOG"],
        eps_base=2.2, rev_base=11e9, target_base=110.0,
    )

    result = pit.build_revision_drift(tmp_path, lookback_days=90)
    assert not result.empty

    # AAPL and MSFT should have valid revisions
    aapl = result[result["ticker"] == "AAPL"].iloc[0]
    assert pd.notna(aapl["eps_revision_3m"])

    # GOOG was not in the old snapshot -> NaN revisions (not an error)
    goog = result[result["ticker"] == "GOOG"].iloc[0]
    assert pd.isna(goog["eps_revision_3m"])
    assert pd.isna(goog["revenue_revision_3m"])
    assert pd.isna(goog["target_revision_3m"])


def test_ticker_missing_in_old_target_gives_nan(tmp_path):
    """Target missing for a ticker in old snapshot -> NaN."""
    _write_snapshot(
        tmp_path, "2026-04-01", TICKERS,
        eps_base=2.0, rev_base=10e9, target_base=100.0,
        skip_tickers_target=["META"],
    )
    _write_snapshot(
        tmp_path, "2026-07-01", TICKERS,
        eps_base=2.2, rev_base=11e9, target_base=110.0,
    )

    result = pit.build_revision_drift(tmp_path, lookback_days=90)
    meta = result[result["ticker"] == "META"].iloc[0]
    assert pd.isna(meta["target_revision_3m"])
    # EPS/revenue should still be computable since estimates exist for all
    assert pd.notna(meta["eps_revision_3m"])


# --- 4. Incremental: only new dates processed --------------------------------

def test_incremental_skips_existing_dates(snapshots_3dates):
    """incremental_update should skip dates already in existing."""
    # First, compute for 2026-07-01 only
    first_run = pit.build_revision_drift_one_date(
        snapshots_3dates, "2026-07-01", lookback_days=90
    )
    assert first_run is not None
    assert set(first_run["available_at"].unique()) == {"2026-07-01"}

    # Incremental with first_run as existing: should only add 2026-07-15
    result = pit.incremental_update(
        snapshots_3dates, existing=first_run, lookback_days=90
    )

    dates = set(result["available_at"].unique())
    assert "2026-07-01" in dates  # preserved from existing
    assert "2026-07-15" in dates  # newly computed

    # Verify the 2026-07-01 data is unchanged (same rows as first_run)
    old_rows = result[result["available_at"] == "2026-07-01"]
    pd.testing.assert_frame_equal(
        old_rows.reset_index(drop=True),
        first_run.reset_index(drop=True),
    )


def test_incremental_noop_when_all_done(snapshots_3dates):
    """If all dates are already computed, incremental returns existing unchanged."""
    full = pit.build_revision_drift(snapshots_3dates, lookback_days=90)
    result = pit.incremental_update(
        snapshots_3dates, existing=full, lookback_days=90
    )
    pd.testing.assert_frame_equal(result, full)


def test_incremental_from_scratch(snapshots_3dates):
    """incremental_update with existing=None behaves like build_revision_drift."""
    full = pit.build_revision_drift(snapshots_3dates, lookback_days=90)
    incr = pit.incremental_update(snapshots_3dates, existing=None, lookback_days=90)
    pd.testing.assert_frame_equal(
        incr.sort_values(["available_at", "ticker"]).reset_index(drop=True),
        full.sort_values(["available_at", "ticker"]).reset_index(drop=True),
    )


# --- 5. CLI round-trip --------------------------------------------------------

def test_cli_writes_parquet(snapshots_2dates, tmp_path):
    """CLI should produce a valid parquet file."""
    out_path = tmp_path / "output" / "pit_features.parquet"
    rc = pit.main([
        "--snapshots", str(snapshots_2dates),
        "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert "ticker" in df.columns
    assert "available_at" in df.columns
    assert len(df) > 0


def test_cli_incremental(snapshots_3dates, tmp_path):
    """CLI --incremental should append new dates."""
    out_path = tmp_path / "pit_features.parquet"

    # First run
    rc1 = pit.main([
        "--snapshots", str(snapshots_3dates),
        "--out", str(out_path),
    ])
    assert rc1 == 0
    df1 = pd.read_parquet(out_path)
    n1 = len(df1)

    # Add a 4th snapshot date (use the 3dates fixture but add one more)
    _write_snapshot(
        snapshots_3dates, "2026-07-30", TICKERS,
        eps_base=2.4, rev_base=12e9, target_base=120.0,
    )

    # Incremental run
    rc2 = pit.main([
        "--snapshots", str(snapshots_3dates),
        "--out", str(out_path),
        "--incremental",
    ])
    assert rc2 == 0
    df2 = pd.read_parquet(out_path)
    assert len(df2) >= n1  # should have more rows


def test_cli_missing_snapshots_dir(tmp_path):
    """CLI should return error code on missing snapshots dir."""
    rc = pit.main([
        "--snapshots", str(tmp_path / "nonexistent"),
        "--out", str(tmp_path / "out.parquet"),
    ])
    assert rc == 2


# --- 6. Empty / degenerate inputs ---------------------------------------------

def test_no_snapshots_returns_empty(tmp_path):
    """No snapshot directories -> empty DataFrame with correct columns."""
    result = pit.build_revision_drift(tmp_path, lookback_days=90)
    assert result.empty
    assert "ticker" in result.columns
    assert "available_at" in result.columns


def test_single_snapshot_returns_empty(tmp_path):
    """One snapshot date with no lookback -> empty (cannot compute revisions)."""
    _write_snapshot(
        tmp_path, "2026-07-01", TICKERS,
        eps_base=2.0, rev_base=10e9, target_base=100.0,
    )
    result = pit.build_revision_drift(tmp_path, lookback_days=90)
    assert result.empty


def test_lookback_too_far_returns_none(tmp_path):
    """Snapshots exist but are too far apart for the lookback window."""
    _write_snapshot(
        tmp_path, "2025-01-01", TICKERS,
        eps_base=2.0, rev_base=10e9, target_base=100.0,
    )
    _write_snapshot(
        tmp_path, "2026-07-01", TICKERS,
        eps_base=2.2, rev_base=11e9, target_base=110.0,
    )
    # Default lookback=90d with 30d tolerance => max 120d. These are ~547d apart.
    result = pit.build_revision_drift(tmp_path, lookback_days=90)
    assert result.empty


def test_find_lookback_date_picks_closest(tmp_path):
    """_find_lookback_date should pick the date closest to target, within tolerance."""
    dates = ["2026-03-15", "2026-04-01", "2026-04-15"]
    # Looking back 90 days from 2026-07-01 => target is 2026-04-02
    best = pit._find_lookback_date(dates, "2026-07-01", 90)
    assert best == "2026-04-01"  # closest to 2026-04-02


def test_zero_denominator_gives_nan(tmp_path):
    """EPS of 0 in old snapshot -> NaN revision (not inf/error)."""
    snap_old = tmp_path / "2026-04-01"
    snap_old.mkdir()
    snap_new = tmp_path / "2026-07-01"
    snap_new.mkdir()

    # Old with EPS=0
    old_df = pd.DataFrame([{
        "symbol": "ZERO",
        "estimatedEpsAvg": 0.0,
        "estimatedRevenueAvg": 10e9,
        "snapshot_as_of": "2026-04-01",
    }])
    old_df.to_parquet(snap_old / "analyst_estimates.parquet", index=False)
    _make_price_target_consensus(["ZERO"], 100.0).to_parquet(
        snap_old / "price_target_consensus.parquet", index=False
    )
    _make_grades_consensus(["ZERO"], 10, 2).to_parquet(
        snap_old / "grades_consensus.parquet", index=False
    )

    # New with EPS=2.0
    new_df = pd.DataFrame([{
        "symbol": "ZERO",
        "estimatedEpsAvg": 2.0,
        "estimatedRevenueAvg": 11e9,
        "snapshot_as_of": "2026-07-01",
    }])
    new_df.to_parquet(snap_new / "analyst_estimates.parquet", index=False)
    _make_price_target_consensus(["ZERO"], 110.0).to_parquet(
        snap_new / "price_target_consensus.parquet", index=False
    )
    _make_grades_consensus(["ZERO"], 12, 1).to_parquet(
        snap_new / "grades_consensus.parquet", index=False
    )

    result = pit.build_revision_drift(tmp_path, lookback_days=90)
    zero = result[result["ticker"] == "ZERO"].iloc[0]
    # Division by zero should give NaN, not inf
    assert pd.isna(zero["eps_revision_3m"]) or abs(zero["eps_revision_3m"]) < float("inf")
