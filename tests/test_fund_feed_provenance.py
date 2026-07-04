"""Per-entity fiscal provenance columns on the daily fundamentals feed.

Contract under test (the S12 promote refusal this fixes): the promote gate
(RenQuant scripts/promote_shadow_patchtst.py ``fundamentals_sla_verdict``) and
the pipeline P-FUND-FRESHNESS gate judge QUARTERLY fundamentals freshness PER
ENTITY. They require a real entity id (``ticker``) plus a fiscal-period /
available-at column on ``sec_fundamentals_daily.parquet``; with neither of
``fiscal_period_end`` / ``available_at`` present the verdict is "QUARTERLY
UNVERIFIABLE ... fail-closed until it exists". These tests pin:

1. the ADDITIVE schema (old columns and values unchanged; consumers that select
   columns explicitly are unaffected),
2. the LATE-FILER case the gate guards (one entity quarters-stale while the
   global max fiscal date stays fresh — a global max must not certify it),
3. PIT never-precedes (``available_at <= date``; ``fiscal_period_end <=
   available_at``; values never appear before their filing's availability),
4. the availability tiers (SEC ``filed`` > FMP ``acceptedDate`` join [the C2
   same-filing assumption] > period end + 45d expected-filing-lag fallback).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data.sec_fundamentals import (
    BASE_FEATURE_COLS,
    FILING_LAG_FALLBACK_DAYS,
    PROVENANCE_COLS,
    build_daily_fundamentals,
    build_quarterly_panel,
    forward_fill_to_daily,
    load_fmp_accepted_dates,
    validate_pit_provenance,
)

pytest.importorskip("pyarrow")


# --- fixtures ----------------------------------------------------------------

def _sec_raw(ticker: str, quarters: list[tuple[str, str | None]]) -> pd.DataFrame:
    """SEC frames rows for one ticker; ``quarters`` = [(end, filed|None), ...]."""
    rows = []
    for end, filed in quarters:
        for concept, val in (
            ("NetIncomeLoss", 10.0),
            ("GrossProfit", 25.0),
            ("Assets", 200.0),
            ("StockholdersEquity", 80.0),
            ("CommonStockSharesOutstanding", 10.0),
        ):
            rows.append({"ticker": ticker, "cik": 1, "end": end,
                         "filed": filed, "concept": concept, "val": val})
    return pd.DataFrame(rows)


def _write_ohlcv(data_dir: Path, tickers: list[str], start: str, end: str) -> None:
    dates = pd.date_range(start, end, freq="B")
    for ticker in tickers:
        d = data_dir / "ohlcv" / ticker
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"close": 10.0}, index=dates).to_parquet(d / "1d.parquet")


def _write_fmp_harvest(data_dir: Path, rows: list[dict]) -> Path:
    harvest = data_dir / "fmp_harvest_5y"
    harvest.mkdir(parents=True, exist_ok=True)
    path = harvest / "income_statement_annual.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# Mirror of the promote gate's per-entity staleness formula
# (RenQuant scripts/promote_shadow_patchtst.py ``entity_quarters_behind``):
# quarters one entity lags from its OWN latest fiscal-period end, with the
# 45d filing-lag grace and a 92d nominal quarter, no calendar snapping.
def _gate_quarters_behind(fpe: pd.Timestamp, today: pd.Timestamp,
                          filing_lag_days: int = 45, quarter_days: int = 92) -> int:
    staleness = (today - fpe).days
    return max(0, (staleness - filing_lag_days) // quarter_days)


# --- 1. additive schema regression -------------------------------------------

def test_daily_feed_schema_is_additive_and_values_unchanged(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, ["AAA"], "2020-05-10", "2020-09-01")
    out = build_daily_fundamentals(
        raw=_sec_raw("AAA", [("2020-03-31", "2020-05-10")]),
        universe=["AAA"],
        cik_to_ticker={1: "AAA"},
        data_dir=tmp_path,
        output_path=tmp_path / "daily.parquet",
    )
    daily = pd.read_parquet(out)

    # Old contract intact: same entity/date/feature columns, same values.
    assert {"ticker", "date", *BASE_FEATURE_COLS}.issubset(daily.columns)
    first = daily.sort_values("date").iloc[0]
    assert first["earnings_yield"] == pytest.approx(10.0 / (10.0 * 10.0))
    assert first["book_to_price"] == pytest.approx(80.0 / (10.0 * 10.0))

    # New provenance columns present (ADDITIVE) and populated once available.
    assert set(PROVENANCE_COLS).issubset(daily.columns)
    last = daily.sort_values("date").iloc[-1]
    assert pd.Timestamp(last["fiscal_period_end"]) == pd.Timestamp("2020-03-31")
    assert pd.Timestamp(last["available_at"]) == pd.Timestamp("2020-05-10")
    assert last["available_source"] == "sec_filed"


def test_downstream_fund_panel_consumer_ignores_provenance(tmp_path: Path) -> None:
    """build_alpha158_fund_panel selects columns explicitly -> unaffected."""
    from renquant_base_data.alpha158_fund_panel import build_alpha158_fund_panel

    dates = pd.date_range("2020-05-11", "2020-06-30", freq="B")
    pd.DataFrame({"ticker": "AAA", "date": dates, "alpha_f1": 1.0}).to_parquet(
        tmp_path / "alpha158_qlib_dataset.parquet", index=False)
    fund = pd.DataFrame({
        "ticker": "AAA", "date": dates,
        **{c: 0.5 for c in BASE_FEATURE_COLS},
        "fiscal_period_end": pd.Timestamp("2020-03-31"),
        "available_at": pd.Timestamp("2020-05-10"),
        "available_source": "sec_filed",
    })
    fund.to_parquet(tmp_path / "sec_fundamentals_daily.parquet", index=False)

    out = build_alpha158_fund_panel(tmp_path, output_path=tmp_path / "panel.parquet")
    panel = pd.read_parquet(out)
    assert len(panel) == len(dates)
    assert not set(PROVENANCE_COLS) & set(panel.columns)
    assert "available_source" not in panel.columns
    assert panel["earnings_yield"].eq(0.5).all()


# --- 2. the late-filer case (the exact guarded failure) ----------------------

def test_late_filer_is_visible_per_entity_while_global_max_is_fresh(tmp_path: Path) -> None:
    """One current issuer must not certify the panel: AAA is current, BBB's
    latest filing is ~3 quarters old. The global max(fiscal_period_end) equals
    AAA's fresh quarter, but the per-entity view the new columns enable shows
    BBB >= 1 quarter behind — exactly what the promote gate's coverage
    distribution guards (and what the pre-provenance schema made UNVERIFIABLE).
    """
    today = pd.Timestamp("2020-11-20")
    _write_ohlcv(tmp_path, ["AAA", "BBB"], "2020-01-02", "2020-11-20")
    raw = pd.concat([
        _sec_raw("AAA", [("2020-03-31", "2020-05-10"),
                          ("2020-06-30", "2020-08-10"),
                          ("2020-09-30", "2020-11-09")]),
        _sec_raw("BBB", [("2019-12-31", "2020-02-10")]),  # late filer: stopped
    ], ignore_index=True)
    out = build_daily_fundamentals(
        raw=raw, universe=["AAA", "BBB"], cik_to_ticker={1: "AAA"},
        data_dir=tmp_path, output_path=tmp_path / "daily.parquet",
    )
    daily = pd.read_parquet(out)

    fiscal_by_entity = (
        daily.dropna(subset=["fiscal_period_end"])
        .groupby("ticker")["fiscal_period_end"].max()
    )
    # Axis the OLD schema could offer: one global max -> looks current.
    assert _gate_quarters_behind(fiscal_by_entity.max(), today) == 0
    # Per-entity axis the new columns enable: the late filer is caught.
    assert _gate_quarters_behind(fiscal_by_entity["AAA"], today) == 0
    assert _gate_quarters_behind(fiscal_by_entity["BBB"], today) >= 1


# --- 3. PIT: available_at never precedes real availability -------------------

def test_values_never_appear_before_their_availability(tmp_path: Path) -> None:
    _write_ohlcv(tmp_path, ["AAA"], "2020-05-10", "2020-09-01")
    out = build_daily_fundamentals(
        raw=_sec_raw("AAA", [("2020-03-31", "2020-05-10"),
                              ("2020-06-30", "2020-08-10")]),
        universe=["AAA"], cik_to_ticker={1: "AAA"},
        data_dir=tmp_path, output_path=tmp_path / "daily.parquet",
    )
    daily = pd.read_parquet(out)
    stamped = daily.dropna(subset=["available_at"])

    # Row-level PIT invariants.
    assert (pd.to_datetime(stamped["available_at"])
            <= pd.to_datetime(stamped["date"])).all()
    assert (pd.to_datetime(stamped["fiscal_period_end"])
            <= pd.to_datetime(stamped["available_at"])).all()

    # The Q2 filing (available 2020-08-10) must NOT be reflected on 2020-08-07.
    before = daily[daily["date"] == pd.Timestamp("2020-08-07")].iloc[0]
    after = daily[daily["date"] == pd.Timestamp("2020-08-10")].iloc[0]
    assert pd.Timestamp(before["fiscal_period_end"]) == pd.Timestamp("2020-03-31")
    assert pd.Timestamp(after["fiscal_period_end"]) == pd.Timestamp("2020-06-30")


def test_validate_pit_provenance_fails_closed_on_lookahead() -> None:
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2020-05-01")],
        "available_at": [pd.Timestamp("2020-05-02")],   # after the serving date
        "fiscal_period_end": [pd.Timestamp("2020-03-31")],
    })
    with pytest.raises(RuntimeError, match="look-ahead"):
        validate_pit_provenance(frame)

    frame = pd.DataFrame({
        "date": [pd.Timestamp("2020-05-01")],
        "available_at": [pd.Timestamp("2020-03-01")],   # before the period end
        "fiscal_period_end": [pd.Timestamp("2020-03-31")],
    })
    with pytest.raises(RuntimeError, match="BEFORE their fiscal-period end"):
        validate_pit_provenance(frame)


def test_rows_before_first_filing_have_no_provenance(tmp_path: Path) -> None:
    """Pre-first-filing rows keep NaT provenance (nothing was available), so a
    gate counts them from later rows' stamps, never as spuriously fresh."""
    quarterly = build_quarterly_panel(
        _sec_raw("AAA", [("2020-03-31", "2020-05-10")]), {1: "AAA"})
    daily = forward_fill_to_daily(
        quarterly, pd.date_range("2020-05-08", "2020-05-12", freq="D"), ["AAA"],
        value_cols=["NetIncomeLoss"])
    before = daily[daily["date"] < pd.Timestamp("2020-05-10")]
    assert before["fiscal_period_end"].isna().all()
    assert before["available_at"].isna().all()


# --- 4. availability tiers ----------------------------------------------------

def test_tier1_sec_filed_wins_over_fmp_and_fallback() -> None:
    accepted = {("AAA", pd.Timestamp("2020-03-31")): pd.Timestamp("2020-05-20")}
    panel = build_quarterly_panel(
        _sec_raw("AAA", [("2020-03-31", "2020-05-10")]), {1: "AAA"},
        accepted_dates=accepted)
    assert panel.loc[0, "available_date"] == pd.Timestamp("2020-05-10")
    assert panel.loc[0, "available_source"] == "sec_filed"


def test_tier2_fmp_accepted_backfills_missing_filed_dates() -> None:
    accepted = {("AAA", pd.Timestamp("2020-03-31")): pd.Timestamp("2020-05-20")}
    panel = build_quarterly_panel(
        _sec_raw("AAA", [("2020-03-31", None)]), {1: "AAA"},
        accepted_dates=accepted)
    assert panel.loc[0, "available_date"] == pd.Timestamp("2020-05-20")
    assert panel.loc[0, "available_source"] == "fmp_accepted"


def test_tier3_expected_filing_lag_is_the_last_resort() -> None:
    panel = build_quarterly_panel(
        _sec_raw("AAA", [("2020-03-31", None)]), {1: "AAA"})
    assert panel.loc[0, "available_date"] == \
        pd.Timestamp("2020-03-31") + pd.Timedelta(days=FILING_LAG_FALLBACK_DAYS)
    assert panel.loc[0, "available_source"] == "expected_filing_lag"


def test_load_fmp_accepted_dates_c2_join(tmp_path: Path) -> None:
    _write_fmp_harvest(tmp_path, [
        # post-close acceptance: filingDate (next day) must win (never-precedes).
        {"symbol": "AAA", "date": "2020-03-31",
         "acceptedDate": "2020-05-20 18:08:27", "filingDate": "2020-05-21"},
        # corrupt: availability before the period end -> dropped, not clamped.
        {"symbol": "BBB", "date": "2020-03-31",
         "acceptedDate": "2020-02-01 06:00:00", "filingDate": "2020-02-01"},
    ])
    lookup = load_fmp_accepted_dates(tmp_path / "fmp_harvest_5y")
    assert lookup[("AAA", pd.Timestamp("2020-03-31"))] == pd.Timestamp("2020-05-21")
    assert ("BBB", pd.Timestamp("2020-03-31")) not in lookup


def test_load_fmp_accepted_dates_missing_dir_is_empty(tmp_path: Path) -> None:
    assert load_fmp_accepted_dates(tmp_path / "does_not_exist") == {}


def test_end_to_end_fmp_tier_reaches_the_daily_feed(tmp_path: Path) -> None:
    """No SEC filed dates anywhere (the production frames-API case): the FMP
    acceptedDate join must stamp available_at, not the 45d assumption."""
    _write_ohlcv(tmp_path, ["AAA"], "2020-05-10", "2020-09-01")
    _write_fmp_harvest(tmp_path, [
        {"symbol": "AAA", "date": "2020-03-31",
         "acceptedDate": "2020-05-20 06:01:26", "filingDate": "2020-05-20"},
    ])
    out = build_daily_fundamentals(
        raw=_sec_raw("AAA", [("2020-03-31", None)]),
        universe=["AAA"], cik_to_ticker={1: "AAA"},
        data_dir=tmp_path, output_path=tmp_path / "daily.parquet",
    )
    daily = pd.read_parquet(out)
    last = daily.sort_values("date").iloc[-1]
    assert pd.Timestamp(last["available_at"]) == pd.Timestamp("2020-05-20")
    assert last["available_source"] == "fmp_accepted"
    # And not a single row shows the value before 2020-05-20.
    stamped = daily.dropna(subset=["available_at"])
    assert (pd.to_datetime(stamped["date"]) >= pd.Timestamp("2020-05-20")).all()
