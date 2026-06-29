"""Regression: the SERVING fundamentals axis must reach the latest price date,
decoupled from the alpha158 ``fwd_60d_excess`` TRAINING-label clip.

Bug (2026-06): ``build_daily_fundamentals`` bound the live feed's daily date
axis to the alpha158 training dataset. That dataset drops its last ~60 trading
days because ``fwd_60d_excess`` is unlabeled there, so the live feed inherited
the clip and sat ~88 calendar days behind the latest price — permanently
failing the P-FUND-FRESHNESS gate and blocking all new buys.

Fix: derive the serving axis from the OHLCV price calendar (fresh to today)
while leaving the alpha158 training panel — which LEFT-joins the feed on its
own clipped dates — unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data.sec_fundamentals import (
    BASE_FEATURE_COLS,
    SecFundamentalsConfig,
    load_daily_index,
    load_price_calendar_index,
    refresh_sec_fundamentals,
    resolve_serving_daily_index,
)


pytest.importorskip("pyarrow")


# Price calendar reaches "today"; the alpha (training) dataset is CLIPPED ~60
# trading days behind it, exactly reproducing the production fwd_60d clip.
PRICE_START = "2020-05-10"
PRICE_END = "2021-06-01"
ALPHA_CLIP_END = "2021-03-08"  # ~60 trading days behind PRICE_END


def _raw_fixture() -> pd.DataFrame:
    rows = []
    quarters = [
        ("2020-03-31", "2020-05-10", 10.0, 25.0, 100.0, 200.0, 80.0, 120.0, 10.0),
        ("2020-06-30", "2020-08-10", 12.0, 28.0, 110.0, 220.0, 82.0, 138.0, 10.0),
        ("2020-09-30", "2020-11-10", 14.0, 31.0, 120.0, 240.0, 84.0, 156.0, 10.0),
        ("2020-12-31", "2021-02-10", 16.0, 34.0, 130.0, 260.0, 86.0, 174.0, 10.0),
        ("2021-03-31", "2021-05-10", 18.0, 37.0, 150.0, 300.0, 90.0, 210.0, 10.0),
    ]
    for end, filed, ni, gp, revenue, assets, equity, liabilities, shares in quarters:
        values = {
            "NetIncomeLoss": ni,
            "GrossProfit": gp,
            "Revenues": revenue,
            "Assets": assets,
            "StockholdersEquity": equity,
            "Liabilities": liabilities,
            "CommonStockSharesOutstanding": shares,
        }
        for concept, value in values.items():
            rows.append(
                {
                    "ticker": "AAA",
                    "cik": 1,
                    "end": end,
                    "filed": filed,
                    "concept": concept,
                    "val": value,
                }
            )
    return pd.DataFrame(rows)


def _write_clipped_inputs(data_dir: Path) -> Path:
    """OHLCV fresh to PRICE_END; alpha dataset clipped to ALPHA_CLIP_END."""
    price_dates = pd.date_range(PRICE_START, PRICE_END, freq="D")
    alpha_dates = pd.date_range(PRICE_START, ALPHA_CLIP_END, freq="D")

    alpha_path = data_dir / "alpha158_qlib_dataset.parquet"
    pd.DataFrame({"ticker": "AAA", "date": alpha_dates}).to_parquet(alpha_path, index=False)

    ohlcv_dir = data_dir / "ohlcv" / "AAA"
    ohlcv_dir.mkdir(parents=True)
    # Index-dated, no ``date`` column — matches the production OHLCV cache.
    pd.DataFrame({"close": 10.0}, index=price_dates).to_parquet(ohlcv_dir / "1d.parquet")
    return alpha_path


def test_price_calendar_index_reaches_today_not_alpha_clip(tmp_path: Path) -> None:
    alpha_path = _write_clipped_inputs(tmp_path)

    alpha_axis = load_daily_index(alpha_path)
    calendar_axis = load_price_calendar_index(tmp_path / "ohlcv", ["AAA"])

    assert alpha_axis.max() == pd.Timestamp(ALPHA_CLIP_END)
    # The serving calendar reaches the latest PRICE date, not the training clip.
    assert calendar_axis.max() == pd.Timestamp(PRICE_END)
    assert calendar_axis.max() > alpha_axis.max()


def test_resolve_serving_index_prefers_price_calendar(tmp_path: Path) -> None:
    _write_clipped_inputs(tmp_path)
    serving = resolve_serving_daily_index(data_dir=tmp_path, universe=["AAA"])
    assert serving.max() == pd.Timestamp(PRICE_END)


def test_resolve_serving_index_falls_back_to_alpha_without_ohlcv(tmp_path: Path) -> None:
    alpha_path = tmp_path / "alpha158_qlib_dataset.parquet"
    alpha_dates = pd.date_range(PRICE_START, ALPHA_CLIP_END, freq="D")
    pd.DataFrame({"ticker": "AAA", "date": alpha_dates}).to_parquet(alpha_path, index=False)
    # No OHLCV directory -> must fall back to the alpha-derived axis.
    serving = resolve_serving_daily_index(data_dir=tmp_path, universe=["AAA"])
    assert serving.max() == pd.Timestamp(ALPHA_CLIP_END)


def test_daily_feed_reaches_today_with_sane_recent_features(tmp_path: Path) -> None:
    alpha_path = _write_clipped_inputs(tmp_path)
    config = SecFundamentalsConfig(
        data_dir=tmp_path,
        mode="daily",
        symbols=("AAA",),
        alpha_path=alpha_path,
        daily_output=tmp_path / "daily.parquet",
        train_end="2021-01-01",
    )

    summary = refresh_sec_fundamentals(
        config,
        raw_daily=_raw_fixture(),
        ticker_cik={"AAA": 1},
    )
    assert summary["ok"] is True

    daily = pd.read_parquet(tmp_path / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    # 1) Serving feed reaches the latest PRICE date, NOT the alpha clip.
    assert daily["date"].max() == pd.Timestamp(PRICE_END)
    assert daily["date"].max() > load_daily_index(alpha_path).max()

    # 2) The recent (previously-clipped) rows have SANE feature values: the
    #    price-dependent features are computed, not NaN-exploded.
    recent = daily[daily["date"] > pd.Timestamp(ALPHA_CLIP_END)]
    assert not recent.empty
    assert set(BASE_FEATURE_COLS).issubset(daily.columns)
    # book_to_price = equity / (shares * price). After the 2021-05-10 filing it
    # is equity=90 / (10 * 10) = 0.9 on the latest date.
    latest = daily[daily["date"] == pd.Timestamp(PRICE_END)].iloc[0]
    assert latest["book_to_price"] == pytest.approx(90.0 / (10.0 * 10.0))
    assert pd.notna(latest["earnings_yield"])
    assert pd.notna(latest["roe"])


def test_training_alpha_axis_unchanged_by_serving_fix(tmp_path: Path) -> None:
    """The alpha158 (training) date axis is untouched: serving extends past it,
    but the training panel left-joins the feed on these clipped dates."""
    alpha_path = _write_clipped_inputs(tmp_path)
    alpha_axis = load_daily_index(alpha_path)
    # Training axis remains label-clipped (decoupling proven).
    assert alpha_axis.max() == pd.Timestamp(ALPHA_CLIP_END)
    assert alpha_axis.min() == pd.Timestamp(PRICE_START)
