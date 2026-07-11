"""Ratio-coverage fix + IMPUTED-SHARE guard (orchestrator PR #475 finding).

Pre-fix, earnings_yield / book_to_price / gross_profitability were finite for
only 67/70/317 of 826 served universe members (META never finite) because:

1. the as-of daily join took the WHOLE latest filing row, wiping concepts the
   newest filing did not tag (``CommonStockSharesOutstanding`` is usually
   10-K-only) — fixed by ``carry_forward_within_ticker``;
2. single-tag concept mappings missed issuers on variant XBRL tags
   (multi-class filers tag shares dimensionally; many issuers never present a
   ``GrossProfit`` subtotal) — fixed by the fallback tag chains;
3. dual-class listings sharing one CIK dropped one class entirely (last-wins
   scalar map) — fixed by the multi-ticker CIK fan-out.

The fix is behavior-ADDITIVE: every cell that was finite pre-fix keeps its
exact value (pinned here on a primary-tag fixture and enforced structurally:
primary tags lead every chain; carry-forward only fills NaN cells).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.crypto_bars import manifest_fingerprint
from renquant_base_data.sec_fundamentals import (
    BASE_FEATURE_COLS,
    DAILY_MANIFEST_FILENAME,
    DEFAULT_FEATURE_COVERAGE_FLOORS,
    build_daily_fundamentals,
    build_extended_fundamentals,
    build_quarterly_panel,
    compute_feature_coverage,
    forward_fill_to_daily,
    main,
    verify_daily_feed,
)

pytest.importorskip("pyarrow")

PRICE = 10.0


def _frames_rows(ticker: str, cik: int, end: str, filed: str, values: dict[str, float]) -> list[dict]:
    return [
        {"ticker": ticker, "cik": cik, "end": end, "filed": filed, "concept": concept, "val": val}
        for concept, val in values.items()
    ]


def _write_runtime_inputs(data_dir: Path, tickers: list[str]) -> None:
    dates = pd.date_range("2020-05-10", "2021-06-01", freq="D")
    for ticker in tickers:
        ohlcv_dir = data_dir / "ohlcv" / ticker
        ohlcv_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"close": PRICE}, index=dates).to_parquet(ohlcv_dir / "1d.parquet")


def _build_daily(tmp_path: Path, raw: pd.DataFrame, tickers: list[str],
                 cik_map: dict | None = None) -> pd.DataFrame:
    _write_runtime_inputs(tmp_path, tickers)
    out = build_daily_fundamentals(
        raw=raw,
        universe=tickers,
        cik_to_ticker=cik_map,
        data_dir=tmp_path,
        output_path=tmp_path / "daily.parquet",
    )
    return pd.read_parquet(out)


# ---------------------------------------------------------------------------
# 1. Behavior-additive regression pin: primary-tag values are unchanged.
# ---------------------------------------------------------------------------

def test_primary_tag_values_pinned_to_prefix_formulas(tmp_path: Path) -> None:
    """A ticker fully served by the PRIMARY tags must produce values from the
    exact pre-fix formulas (ni/(shares*price+1e-9) etc.) — pins that the
    fallback chains and carry-forward change NOTHING for already-finite
    cells."""
    raw = pd.DataFrame(
        _frames_rows("AAA", 1, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 10.0,
            "GrossProfit": 25.0,
            "Revenues": 100.0,
            "Assets": 200.0,
            "StockholdersEquity": 80.0,
            "CommonStockSharesOutstanding": 10.0,
            # decoy fallback values that must all LOSE to the primary tags
            "WeightedAverageNumberOfDilutedSharesOutstanding": 999.0,
            "CommonStockSharesIssued": 888.0,
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": 777.0,
            "RevenueFromContractWithCustomerExcludingAssessedTax": 666.0,
            "CostOfRevenue": 555.0,
        })
    )
    daily = _build_daily(tmp_path, raw, ["AAA"])
    row = daily[daily["date"] == pd.Timestamp("2020-05-10")].iloc[0]

    market_cap = 10.0 * PRICE
    assert row["earnings_yield"] == 10.0 / (market_cap + 1e-9)
    assert row["book_to_price"] == 80.0 / (market_cap + 1e-9)
    assert row["gross_profitability"] == 25.0 / (200.0 + 1e-9)
    assert row["roe"] == 10.0 / (80.0 + 1e-9)


# ---------------------------------------------------------------------------
# 2. Fallback tag chains recover never-finite tickers (the META class).
# ---------------------------------------------------------------------------

def test_shares_and_grossprofit_fallbacks_recover_meta_class_ticker(tmp_path: Path) -> None:
    """META shape: no non-dimensional CommonStockSharesOutstanding ever, no
    GrossProfit subtotal — pre-fix ey/b2p/gp were NaN forever; the weighted-
    average shares and revenue−cost fallbacks recover all three."""
    raw = pd.DataFrame(
        _frames_rows("MMM", 7, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 20.0,
            "Assets": 400.0,
            "StockholdersEquity": 160.0,
            "WeightedAverageNumberOfDilutedSharesOutstanding": 8.0,
            "RevenueFromContractWithCustomerExcludingAssessedTax": 120.0,
            "CostOfRevenue": 30.0,
        })
    )
    daily = _build_daily(tmp_path, raw, ["MMM"])
    row = daily[daily["date"] == pd.Timestamp("2020-05-10")].iloc[0]

    market_cap = 8.0 * PRICE
    assert row["earnings_yield"] == pytest.approx(20.0 / market_cap, rel=1e-9)
    assert row["book_to_price"] == pytest.approx(160.0 / market_cap, rel=1e-9)
    assert row["gross_profitability"] == pytest.approx((120.0 - 30.0) / 400.0, rel=1e-9)


def test_gross_profit_fallback_requires_both_revenue_and_cost(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        _frames_rows("RRR", 8, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 5.0,
            "Assets": 100.0,
            "Revenues": 50.0,  # revenue without any cost-of-revenue tag
        })
    )
    daily = _build_daily(tmp_path, raw, ["RRR"])
    row = daily[daily["date"] == pd.Timestamp("2020-05-10")].iloc[0]

    assert pd.isna(row["gross_profitability"])  # no partial math


# ---------------------------------------------------------------------------
# 3. Carry-forward fixes the whole-row-wipe (the JPM/NVDA class).
# ---------------------------------------------------------------------------

def test_carry_forward_survives_filing_without_shares(tmp_path: Path) -> None:
    """10-K tags shares; the next 10-Q does not (the dominant pre-fix hole:
    the newer filing row wiped the known share count). The daily rows after
    the 10-Q must keep the 10-K share count and stay finite."""
    raw = pd.DataFrame(
        _frames_rows("JJJ", 9, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 10.0,
            "Assets": 200.0,
            "StockholdersEquity": 80.0,
            "CommonStockSharesOutstanding": 10.0,
        })
        + _frames_rows("JJJ", 9, "2020-06-30", "2020-08-01", {
            "NetIncomeLoss": 12.0,
            "Assets": 210.0,
            "StockholdersEquity": 82.0,
            # no share tag in this filing
        })
    )
    daily = _build_daily(tmp_path, raw, ["JJJ"])
    daily = daily.set_index("date")

    q1_row = daily.loc[pd.Timestamp("2020-05-10")]
    q2_row = daily.loc[pd.Timestamp("2020-08-10")]
    market_cap = 10.0 * PRICE
    # Q1 row untouched; Q2 row uses fresh NI with the CARRIED share count.
    assert q1_row["earnings_yield"] == 10.0 / (market_cap + 1e-9)
    assert q2_row["earnings_yield"] == 12.0 / (market_cap + 1e-9)
    # provenance still describes the LATEST filing (never carried)
    assert q2_row["fiscal_period_end"] == pd.Timestamp("2020-06-30")


def test_forward_fill_carry_is_opt_in_default_off() -> None:
    quarterly = pd.DataFrame(
        [
            {"ticker": "AAA", "end": pd.Timestamp("2020-03-31"),
             "available_date": pd.Timestamp("2020-05-01"), "available_source": "sec_filed",
             "NetIncomeLoss": 10.0, "CommonStockSharesOutstanding": 10.0},
            {"ticker": "AAA", "end": pd.Timestamp("2020-06-30"),
             "available_date": pd.Timestamp("2020-08-01"), "available_source": "sec_filed",
             "NetIncomeLoss": 12.0, "CommonStockSharesOutstanding": np.nan},
        ]
    )
    index = pd.DatetimeIndex([pd.Timestamp("2020-08-10")])
    cols = ["NetIncomeLoss", "CommonStockSharesOutstanding"]

    legacy = forward_fill_to_daily(quarterly, index, ["AAA"], value_cols=cols)
    carried = forward_fill_to_daily(
        quarterly, index, ["AAA"], value_cols=cols, carry_forward_within_ticker=True
    )

    # default (extended feed path) keeps the wipe: byte-identical legacy behavior
    assert pd.isna(legacy.loc[0, "CommonStockSharesOutstanding"])
    assert carried.loc[0, "CommonStockSharesOutstanding"] == 10.0
    # already-present cells are untouched by the carry
    assert legacy.loc[0, "NetIncomeLoss"] == carried.loc[0, "NetIncomeLoss"] == 12.0


# ---------------------------------------------------------------------------
# 4. Dual-class CIK fan-out.
# ---------------------------------------------------------------------------

def test_dual_class_cik_maps_to_both_tickers(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {"cik": 1, "end": "2020-03-31", "filed": "2020-05-01", "concept": c, "val": v}
            for c, v in {
                "NetIncomeLoss": 10.0, "Assets": 200.0, "StockholdersEquity": 80.0,
                "CommonStockSharesOutstanding": 10.0,
            }.items()
        ]
    )
    daily = _build_daily(tmp_path, raw, ["GOG", "GOGL"], cik_map={1: ("GOG", "GOGL")})
    day = daily[daily["date"] == pd.Timestamp("2020-05-10")].set_index("ticker")

    assert set(day.index) == {"GOG", "GOGL"}
    assert day.loc["GOG", "earnings_yield"] == day.loc["GOGL", "earnings_yield"]

    # legacy scalar map still accepted (extended feed path)
    panel = build_quarterly_panel(raw, {1: "GOG"})
    assert list(panel["ticker"].unique()) == ["GOG"]


# ---------------------------------------------------------------------------
# 5. Extended z-scored feed stays byte-identical.
# ---------------------------------------------------------------------------

def test_extended_feed_unchanged_by_fallback_concept_rows(tmp_path: Path) -> None:
    """mode=both shares one fetch, so the extended build now receives
    FALLBACK_CONCEPTS rows — it must produce the exact same parquet as from a
    primary-only fetch (its train-window z-parameters must not move)."""
    base_values = {
        "NetIncomeLoss": 10.0, "GrossProfit": 25.0, "Revenues": 100.0,
        "Assets": 200.0, "StockholdersEquity": 80.0, "Liabilities": 120.0,
        "CommonStockSharesOutstanding": 10.0,
    }
    rows_primary = []
    rows_with_fallback = []
    for i, (end, filed) in enumerate([
        ("2020-03-31", "2020-05-01"), ("2020-06-30", "2020-08-01"),
        ("2020-09-30", "2020-11-01"), ("2020-12-31", "2021-02-01"),
        ("2021-03-31", "2021-05-01"),
    ]):
        values = {k: v + i for k, v in base_values.items()}
        rows_primary += _frames_rows("AAA", 1, end, filed, values)
        rows_with_fallback += _frames_rows("AAA", 1, end, filed, {
            **values,
            "WeightedAverageNumberOfDilutedSharesOutstanding": 500.0 + i,
            "RevenueFromContractWithCustomerExcludingAssessedTax": 400.0 + i,
            "CostOfRevenue": 300.0 + i,
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": 200.0 + i,
        })
    _write_runtime_inputs(tmp_path, ["AAA"])

    outputs = []
    for name, rows in (("a.parquet", rows_primary), ("b.parquet", rows_with_fallback)):
        build_extended_fundamentals(
            raw=pd.DataFrame(rows),
            universe=["AAA"],
            cik_to_ticker={1: "AAA"},
            data_dir=tmp_path,
            output_path=tmp_path / name,
            train_end="2021-01-01",
        )
        outputs.append(pd.read_parquet(tmp_path / name))

    pd.testing.assert_frame_equal(outputs[0], outputs[1])


# ---------------------------------------------------------------------------
# 6. IMPUTED-SHARE guard: manifest stamping + verify CLI.
# ---------------------------------------------------------------------------

def test_ingestion_manifest_stamps_coverage_and_fingerprint(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        _frames_rows("AAA", 1, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 10.0, "GrossProfit": 25.0, "Assets": 200.0,
            "StockholdersEquity": 80.0, "CommonStockSharesOutstanding": 10.0,
        })
    )
    _build_daily(tmp_path, raw, ["AAA"])
    manifest = json.loads((tmp_path / DAILY_MANIFEST_FILENAME).read_text())

    assert manifest["dataset_id"] == "sec-fundamentals-daily"
    assert manifest["fingerprint"] == manifest_fingerprint(manifest)
    entry = manifest["feature_coverage"]["earnings_yield"]
    # data_contracts.v1-aligned vocabulary
    assert entry == {
        "coverage": 1.0, "n_have": 1, "n_expected": 1,
        "denominator": "priced_tickers",
        "min_coverage": DEFAULT_FEATURE_COVERAGE_FLOORS["earnings_yield"], "ok": True,
    }
    assert manifest["feature_coverage"]["gross_profitability"]["denominator"] == "served_tickers"
    assert manifest["coverage_ok"] is True
    assert manifest["n_served"] == 1
    assert manifest["expected_universe"] == ["AAA"]


def test_verify_daily_feed_passes_fails_floor_and_detects_tamper(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        _frames_rows("AAA", 1, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 10.0, "Assets": 200.0,
            "StockholdersEquity": 80.0, "CommonStockSharesOutstanding": 10.0,
            # no GrossProfit and no revenue/cost pair -> gp coverage 0.0
        })
    )
    _build_daily(tmp_path, raw, ["AAA"])

    lenient = dict.fromkeys(DEFAULT_FEATURE_COVERAGE_FLOORS, 0.0)
    report = verify_daily_feed(
        data_dir=tmp_path, daily_output=tmp_path / "daily.parquet", floors=lenient
    )
    assert report["ok"] is True

    # gp finite coverage is 0.0 -> the default floor must FAIL the feed
    report = verify_daily_feed(data_dir=tmp_path, daily_output=tmp_path / "daily.parquet")
    assert report["ok"] is False
    assert report["checks"]["coverage_ok"] is False
    assert report["feature_coverage"]["gross_profitability"]["ok"] is False

    # tampered manifest -> fingerprint check fails closed
    manifest_path = tmp_path / DAILY_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text())
    payload["coverage_ok"] = "tampered"
    manifest_path.write_text(json.dumps(payload))
    report = verify_daily_feed(
        data_dir=tmp_path, daily_output=tmp_path / "daily.parquet", floors=lenient
    )
    assert report["ok"] is False
    assert report["checks"]["fingerprint_ok"] is False


def test_cli_verify_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = pd.DataFrame(
        _frames_rows("AAA", 1, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 10.0, "GrossProfit": 25.0, "Assets": 200.0,
            "StockholdersEquity": 80.0, "CommonStockSharesOutstanding": 10.0,
        })
    )
    _build_daily(tmp_path, raw, ["AAA"])

    rc = main(["--verify", "--data-dir", str(tmp_path),
               "--daily-output", str(tmp_path / "daily.parquet")])
    assert rc == 0
    assert '"ok": true' in capsys.readouterr().out

    rc = main(["--verify", "--data-dir", str(tmp_path),
               "--daily-output", str(tmp_path / "daily.parquet"),
               "--coverage-floor", "earnings_yield=1.01"])
    assert rc == 1


def test_compute_feature_coverage_price_denominator() -> None:
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-05-10")] * 3,
            "ticker": ["A", "B", "C"],
            "price": [10.0, np.nan, 10.0],   # B has no same-day close
            "earnings_yield": [0.1, np.nan, np.nan],
            "book_to_price": [0.5, np.nan, 0.4],
            "gross_profitability": [0.2, 0.1, np.nan],
            "roe": [0.1, 0.2, 0.3],
            "asset_growth": [0.0, 0.0, 0.0],
        }
    )
    coverage = compute_feature_coverage(frame, feature_cols=BASE_FEATURE_COLS)

    assert coverage["n_served"] == 3
    assert coverage["n_priced"] == 2
    assert coverage["features"]["earnings_yield"]["n_expected"] == 2   # priced only
    assert coverage["features"]["earnings_yield"]["coverage"] == 0.5
    assert coverage["features"]["book_to_price"]["coverage"] == 1.0
    assert coverage["features"]["gross_profitability"]["n_expected"] == 3
    assert coverage["features"]["roe"]["coverage"] == 1.0
