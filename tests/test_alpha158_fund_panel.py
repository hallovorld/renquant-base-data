"""Tests for the subrepo-owned alpha158+fund panel builder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data.alpha158_fund_panel import (
    FUND_COLS,
    PEAD_COLS,
    SENT_COLS,
    SUE_COLS,
    build_alpha158_fund_panel,
)


pytest.importorskip("pyarrow")


def _write_inputs(data_dir: Path, *, extra_alpha_day: bool = False) -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    if extra_alpha_day:
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp("2026-01-07")]))

    alpha_rows = []
    for date in dates:
        for ticker, base in (("AAA", 1.0), ("BBB", 2.0)):
            alpha_rows.append({
                "ticker": ticker,
                "date": date,
                "KMID": base,
                "ROC5": base + 0.5,
                "fwd_5d_excess": base / 100.0,
                "fwd_20d_excess": base / 50.0,
                "fwd_60d_excess": base / 25.0,
            })
    pd.DataFrame(alpha_rows).to_parquet(data_dir / "alpha158_qlib_dataset.parquet", index=False)

    fund_dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    fund_rows = []
    for date in fund_dates:
        fund_rows.append({
            "ticker": "AAA",
            "date": date,
            "earnings_yield": 0.05,
            "book_to_price": 0.4,
            "gross_profitability": 0.3,
            "roe": 0.2,
            "asset_growth": 0.1,
        })
        # BBB is intentionally missing some dates so median imputation is exercised.
        if date == pd.Timestamp("2026-01-02"):
            fund_rows.append({
                "ticker": "BBB",
                "date": date,
                "earnings_yield": 0.07,
                "book_to_price": 0.5,
                "gross_profitability": 0.4,
                "roe": 0.3,
                "asset_growth": 0.2,
            })
    pd.DataFrame(fund_rows).to_parquet(data_dir / "sec_fundamentals_daily.parquet", index=False)

    earn_dir = data_dir / "earnings_surprise"
    earn_dir.mkdir()
    earnings = pd.DataFrame(
        {
            "eps_actual": [1.0, 1.2, 1.4],
            "eps_estimate": [0.9, 1.0, 1.1],
            "surprise_abs": [0.1, 0.2, 0.3],
            "surprise_pct": [0.10, 0.20, 0.30],
        },
        index=pd.DatetimeIndex(["2025-12-31", "2025-10-01", "2025-07-01"], name="date"),
    )
    earnings.to_parquet(earn_dir / "AAA.parquet")

    sent_dir = data_dir / "news_sentiment_alpaca"
    sent_dir.mkdir()
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "date": pd.to_datetime(["2019-12-31", "2026-01-05"]),
            "mean_sentiment": [0.99, 0.2],
            "n_articles": [1, 3],
            "sentiment_pos_share": [1.0, 0.75],
        }
    ).to_parquet(sent_dir / "AAA.parquet", index=False)


def test_build_alpha158_fund_panel_merges_and_fills(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    out = build_alpha158_fund_panel(tmp_path)

    panel = pd.read_parquet(out)

    assert len(panel) == 6
    for col in FUND_COLS + PEAD_COLS + SUE_COLS + SENT_COLS:
        assert col in panel.columns
        assert panel[col].isna().sum() == 0
    bbb_later = panel[(panel["ticker"] == "BBB") & (panel["date"] == pd.Timestamp("2026-01-05"))].iloc[0]
    assert bbb_later["earnings_yield"] == pytest.approx(0.05)
    aaa_later = panel[(panel["ticker"] == "AAA") & (panel["date"] == pd.Timestamp("2026-01-05"))].iloc[0]
    assert aaa_later["pead_signal"] > 0
    assert aaa_later["n_articles_log"] > 0


def test_sec_coverage_guard_can_truncate(tmp_path: Path) -> None:
    _write_inputs(tmp_path, extra_alpha_day=True)

    with pytest.raises(RuntimeError, match="SEC coverage guard"):
        build_alpha158_fund_panel(tmp_path)

    out = build_alpha158_fund_panel(tmp_path, truncate_to_sec_max=True)
    panel = pd.read_parquet(out)

    assert panel["date"].max() == pd.Timestamp("2026-01-06")
    assert len(panel) == 6
