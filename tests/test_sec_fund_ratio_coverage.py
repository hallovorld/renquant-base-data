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
    DEFAULT_FEATURE_MAX_AGE_DAYS,
    DEFAULT_PREREQUISITE_PRICE_COVERAGE_FLOOR,
    DEFAULT_UNIVERSE_COVERAGE_FLOORS,
    RAW_VALUE_COLS,
    build_daily_fundamentals,
    build_extended_fundamentals,
    build_quarterly_panel,
    compute_derived_features,
    compute_feature_coverage,
    compute_feature_freshness,
    forward_fill_to_daily,
    main,
    verify_daily_feed,
)

# Lenient overrides so tests that exercise COVERAGE/tamper-detection (not
# freshness) aren't tripped by staleness — these fixtures' single filing is
# always older than the OHLCV date range's tail by construction.
_LENIENT_MAX_AGE = dict.fromkeys(DEFAULT_FEATURE_MAX_AGE_DAYS, 10_000)

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
                 cik_map: dict | None = None,
                 max_age_days: dict | None = _LENIENT_MAX_AGE) -> pd.DataFrame:
    """Most fixtures here span a full year of OHLCV dates from a SINGLE
    filing near the start, so the (real, working) freshness check would
    correctly flag every feature stale by the last serving date — these
    tests are about COVERAGE, not freshness, so default to a lenient
    max_age_days unless a test explicitly wants to exercise freshness."""
    _write_runtime_inputs(tmp_path, tickers)
    out = build_daily_fundamentals(
        raw=raw,
        universe=tickers,
        cik_to_ticker=cik_map,
        data_dir=tmp_path,
        output_path=tmp_path / "daily.parquet",
        max_age_days=max_age_days,
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
    # data_contracts.v1-aligned vocabulary — LEGACY (priced-or-served) fields...
    assert entry["coverage"] == 1.0
    assert entry["n_have"] == 1
    assert entry["n_expected"] == 1
    assert entry["denominator"] == "priced_tickers"
    assert entry["min_coverage"] == DEFAULT_FEATURE_COVERAGE_FLOORS["earnings_yield"]
    assert entry["ok"] is True
    # ...and the NEW dual-denominator fields (P1 coverage-denominator fix,
    # 2026-07-11 Codex CHANGES_REQUESTED, PR #43): same finite count, honest
    # denominator against the full declared/scored universe (here == priced,
    # both size 1, but the FIELD is what matters for the 826-name production
    # case where n_universe_expected >> n_priced).
    assert entry["universe_coverage"] == 1.0
    assert entry["n_universe_expected"] == 1
    assert entry["universe_denominator"] == "declared_scored_universe"
    assert entry["universe_min_coverage"] == DEFAULT_UNIVERSE_COVERAGE_FLOORS["earnings_yield"]
    assert entry["universe_ok"] is True
    assert manifest["feature_coverage"]["gross_profitability"]["denominator"] == "served_tickers"
    # axis-level prerequisite price coverage (also against the declared universe)
    price = manifest["prerequisite_price_coverage"]
    assert price["denominator"] == "declared_scored_universe"
    assert price["min_coverage"] == DEFAULT_PREREQUISITE_PRICE_COVERAGE_FLOOR
    assert price["ok"] is True
    assert manifest["prerequisite_price_coverage_ok"] is True
    assert manifest["universe_coverage_ok"] is True
    assert manifest["legacy_coverage_ok"] is True
    assert manifest["coverage_ok"] is True
    assert manifest["n_served"] == 1
    assert manifest["expected_universe"] == ["AAA"]
    # freshness block present (lenient max_age here; the dedicated
    # regression test below exercises a tight, failing bound).
    assert manifest["freshness_ok"] is True
    assert "earnings_yield" in manifest["feature_freshness"]


def test_verify_daily_feed_passes_fails_floor_and_detects_tamper(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        _frames_rows("AAA", 1, "2020-03-31", "2020-05-01", {
            "NetIncomeLoss": 10.0, "Assets": 200.0,
            "StockholdersEquity": 80.0, "CommonStockSharesOutstanding": 10.0,
            # no GrossProfit and no revenue/cost pair -> gp coverage 0.0
        })
    )
    _build_daily(tmp_path, raw, ["AAA"])

    # Isolate the coverage-floor/tamper checks under test from freshness (a
    # separate, dedicated regression covers that): lenient everywhere here.
    lenient = dict.fromkeys(DEFAULT_FEATURE_COVERAGE_FLOORS, 0.0)
    lenient_universe = dict.fromkeys(DEFAULT_UNIVERSE_COVERAGE_FLOORS, 0.0)
    report = verify_daily_feed(
        data_dir=tmp_path, daily_output=tmp_path / "daily.parquet",
        floors=lenient, universe_floors=lenient_universe,
        prerequisite_price_floor=0.0, max_age_days=_LENIENT_MAX_AGE,
    )
    assert report["ok"] is True

    # gp finite coverage is 0.0 -> the default floor must FAIL the feed
    report = verify_daily_feed(
        data_dir=tmp_path, daily_output=tmp_path / "daily.parquet",
        max_age_days=_LENIENT_MAX_AGE,
    )
    assert report["ok"] is False
    assert report["checks"]["coverage_ok"] is False
    assert report["feature_coverage"]["gross_profitability"]["ok"] is False

    # tampered manifest -> fingerprint check fails closed
    manifest_path = tmp_path / DAILY_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text())
    payload["coverage_ok"] = "tampered"
    manifest_path.write_text(json.dumps(payload))
    report = verify_daily_feed(
        data_dir=tmp_path, daily_output=tmp_path / "daily.parquet",
        floors=lenient, universe_floors=lenient_universe,
        prerequisite_price_floor=0.0, max_age_days=_LENIENT_MAX_AGE,
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
    # Lenient max-age via CLI so freshness (a separate, dedicated regression
    # covers it) doesn't interfere with these coverage-floor exit-code checks.
    lenient_age_args = [f"{feature}={days}" for feature, days in _LENIENT_MAX_AGE.items()]

    rc = main(["--verify", "--data-dir", str(tmp_path),
               "--daily-output", str(tmp_path / "daily.parquet"),
               "--max-age-days", *lenient_age_args])
    assert rc == 0
    assert '"ok": true' in capsys.readouterr().out

    rc = main(["--verify", "--data-dir", str(tmp_path),
               "--daily-output", str(tmp_path / "daily.parquet"),
               "--max-age-days", *lenient_age_args,
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


# ---------------------------------------------------------------------------
# 7. P1 coverage-denominator fix (2026-07-11 Codex CHANGES_REQUESTED, PR #43):
#    a good priced-relative number must not hide a bad declared-universe one.
# ---------------------------------------------------------------------------

def test_universe_denominator_exposes_degraded_price_coverage_hidden_by_priced_denominator() -> None:
    """Reproduces the exact review finding: 1 of 10 declared/scored names is
    priced, and earnings_yield is finite for that 1 name — the LEGACY
    priced-relative number reports a perfect 1.0 (1/1 priced), which would
    look "healthy" in isolation. The NEW universe-denominator number and the
    prerequisite price-coverage number both correctly show only 10% coverage,
    and the combined axis verdict must be UNHEALTHY."""
    universe = [f"T{i}" for i in range(10)]
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-05-10")] * 10,
            "ticker": universe,
            "price": [10.0] + [np.nan] * 9,          # only 1/10 priced
            "earnings_yield": [0.1] + [np.nan] * 9,  # finite for the 1 priced name
            "book_to_price": [0.2] + [np.nan] * 9,
            "gross_profitability": [0.3] * 10,        # fully served (not price-dependent)
            "roe": [0.1] * 10,
            "asset_growth": [0.0] * 10,
        }
    )
    coverage = compute_feature_coverage(frame, universe=universe)

    ey = coverage["features"]["earnings_yield"]
    # LEGACY number: perfect and misleading in isolation.
    assert ey["coverage"] == 1.0
    assert ey["n_expected"] == 1
    assert ey["denominator"] == "priced_tickers"
    # NEW: the SAME 1 finite cell against the FULL declared universe.
    assert ey["universe_coverage"] == pytest.approx(0.1)
    assert ey["n_universe_expected"] == 10
    assert ey["universe_denominator"] == "declared_scored_universe"
    assert ey["universe_ok"] is False  # 0.1 < DEFAULT_UNIVERSE_COVERAGE_FLOORS["earnings_yield"] (0.3)

    price = coverage["prerequisite_price_coverage"]
    assert price["coverage"] == pytest.approx(0.1)
    assert price["n_expected"] == 10
    assert price["denominator"] == "declared_scored_universe"
    assert price["ok"] is False  # 0.1 < DEFAULT_PREREQUISITE_PRICE_COVERAGE_FLOOR (0.5)

    # gross_profitability (price-independent, fully served) stays healthy on
    # BOTH denominators — only the price-dependent features are degraded.
    gp = coverage["features"]["gross_profitability"]
    assert gp["coverage"] == 1.0
    assert gp["universe_coverage"] == 1.0
    assert gp["universe_ok"] is True

    # The combined axis verdict must be UNHEALTHY: a good ratio-among-priced
    # number can no longer certify the axis while the universe/price
    # prerequisite contracts fail.
    assert coverage["legacy_coverage_ok"] is True   # the (misleading) old number alone
    assert coverage["universe_coverage_ok"] is False
    assert coverage["prerequisite_price_coverage_ok"] is False
    assert coverage["coverage_ok"] is False


# ---------------------------------------------------------------------------
# 8. P1 provenance-freshness fix (2026-07-11 Codex CHANGES_REQUESTED, PR #43):
#    a ratio finite via carry-forward must not look as fresh as the newest
#    filing when a DIFFERENT, older filing actually supplied its operand.
# ---------------------------------------------------------------------------

def test_carried_concept_retains_older_source_age_and_fails_tight_freshness_bound(
    tmp_path: Path,
) -> None:
    """The exact scenario from the review: a 10-Q (2020-06-30, available
    2020-08-01) refreshes NetIncomeLoss/Assets/StockholdersEquity but does
    NOT tag CommonStockSharesOutstanding, which only the prior 10-K
    (2020-03-31, available 2020-05-01) tagged.

    (a) coverage still increases: earnings_yield/book_to_price are FINITE
        post-carry-forward (pre-fix they were NaN forever after the 10-Q).
    (b) the shares-backed ratios retain the OLDER 10-K source age in their
        OWN provenance — not the newer 10-Q's row-level timestamp, which
        still (correctly, unchanged) describes only the latest filing.
    (c) a tight "current-quarter" (60d) freshness assertion FAILS for
        earnings_yield/book_to_price, while roe (NI + equity, both freshly
        refreshed by the 10-Q) correctly PASSES — freshness is genuinely
        PER-FEATURE, driven by each ratio's own operands.
    """
    quarterly = pd.DataFrame(
        [
            {"ticker": "SSS", "end": pd.Timestamp("2020-03-31"),
             "available_date": pd.Timestamp("2020-05-01"), "available_source": "sec_filed",
             "NetIncomeLoss": 10.0, "Assets": 200.0, "StockholdersEquity": 80.0,
             "CommonStockSharesOutstanding": 10.0},
            {"ticker": "SSS", "end": pd.Timestamp("2020-06-30"),
             "available_date": pd.Timestamp("2020-08-01"), "available_source": "sec_filed",
             "NetIncomeLoss": 12.0, "Assets": 210.0, "StockholdersEquity": 82.0,
             "CommonStockSharesOutstanding": np.nan},  # 10-Q omits shares
        ]
    )
    _write_runtime_inputs(tmp_path, ["SSS"])
    index = pd.date_range("2020-08-05", "2020-08-10", freq="D")

    # PRE-FIX behavior (no carry-forward): the whole-row wipe leaves
    # earnings_yield/book_to_price permanently NaN after the 10-Q, even
    # though NetIncomeLoss/Assets/StockholdersEquity all refreshed.
    pre = forward_fill_to_daily(quarterly, index, ["SSS"], value_cols=RAW_VALUE_COLS)
    pre_features = compute_derived_features(pre, tmp_path / "ohlcv")
    pre_row = pre_features[pre_features["date"] == pd.Timestamp("2020-08-10")].iloc[0]
    assert pd.isna(pre_row["earnings_yield"])
    assert pd.isna(pre_row["book_to_price"])

    # POST-FIX: carry-forward (+ per-concept provenance tracking) recovers it.
    post = forward_fill_to_daily(
        quarterly, index, ["SSS"], value_cols=RAW_VALUE_COLS,
        carry_forward_within_ticker=True, track_concept_provenance=True,
    )
    post_features = compute_derived_features(post, tmp_path / "ohlcv")
    row = post_features[post_features["date"] == pd.Timestamp("2020-08-10")].iloc[0]

    # (a) coverage increases: the ratio is now finite.
    assert np.isfinite(row["earnings_yield"])
    assert np.isfinite(row["book_to_price"])

    # Row-level (latest-filing) provenance is UNCHANGED — still describes
    # only the newest filing. This is the insufficient signal a naive
    # freshness check keyed on these columns alone would wrongly trust.
    assert row["fiscal_period_end"] == pd.Timestamp("2020-06-30")
    assert row["available_at"] == pd.Timestamp("2020-08-01")

    # (b) the RATIO's own provenance instead attributes to the OLDER 10-K
    # filing that actually supplied the carried shares operand.
    assert row["earnings_yield_source_available_at"] == pd.Timestamp("2020-05-01")
    assert row["earnings_yield_source_fiscal_period_end"] == pd.Timestamp("2020-03-31")
    assert row["book_to_price_source_available_at"] == pd.Timestamp("2020-05-01")
    assert row["book_to_price_source_fiscal_period_end"] == pd.Timestamp("2020-03-31")
    expected_age = (pd.Timestamp("2020-08-10") - pd.Timestamp("2020-05-01")).days
    assert row["earnings_yield_source_age_days"] == expected_age
    assert row["book_to_price_source_age_days"] == expected_age

    # roe's operands (NI + equity) both came from the fresh 10-Q.
    assert row["roe_source_available_at"] == pd.Timestamp("2020-08-01")

    # (c) a tight "current-quarter" freshness bound correctly FAILS ey/b2p
    # but PASSES roe.
    tight = {**DEFAULT_FEATURE_MAX_AGE_DAYS, "earnings_yield": 60, "book_to_price": 60, "roe": 60}
    freshness = compute_feature_freshness(post_features, max_age_days=tight)
    assert freshness["features"]["earnings_yield"]["fresh_ok"] is False
    assert freshness["features"]["earnings_yield"]["n_stale"] == 1
    assert freshness["features"]["book_to_price"]["fresh_ok"] is False
    assert freshness["features"]["roe"]["fresh_ok"] is True
    assert freshness["freshness_ok"] is False

    # A LENIENT (e.g. default 150d) bound still passes at 101 days old —
    # the mechanism trips on genuinely stale filings, not on jitter.
    lenient_freshness = compute_feature_freshness(post_features)
    assert lenient_freshness["features"]["earnings_yield"]["fresh_ok"] is True


def test_forward_fill_provenance_tracking_is_independent_of_carry_forward() -> None:
    """``track_concept_provenance`` can be requested WITHOUT carry-forward:
    the companion source columns then simply mirror the row-level
    provenance wherever a concept has its own value, and are NaT wherever
    the (uncarried) value itself is NaT — never silently invented."""
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

    tracked_no_carry = forward_fill_to_daily(
        quarterly, index, ["AAA"], value_cols=cols, track_concept_provenance=True,
    )
    row = tracked_no_carry.iloc[0]
    # value not carried -> still NaN, and its provenance is NaT (no invented
    # source for a value that isn't there).
    assert pd.isna(row["CommonStockSharesOutstanding"])
    assert pd.isna(row["CommonStockSharesOutstanding__source_available_at"])
    # the concept WITH a value on the selected (latest) row gets that row's
    # own provenance.
    assert row["NetIncomeLoss__source_available_at"] == pd.Timestamp("2020-08-01")
