"""Tests for crypto SMA feature computation (data-layer primitive)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.crypto_trend_signal import (
    SMAConfig,
    PairSMA,
    compute_sma_for_pair,
    compute_sma_features,
    DEFAULT_SMA_PERIOD,
)


def _make_close(n: int = 200, drift: float = 0.001, vol: float = 0.02, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(close, index=dates, name="close")


def _populate_store(tmp_path: Path, pairs_data: dict[str, pd.Series]) -> Path:
    store_dir = tmp_path / "crypto_ohlcv"
    for slug, close in pairs_data.items():
        pair_dir = store_dir / slug
        pair_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(len(close)) * 1e6,
        }, index=close.index)
        df.to_parquet(pair_dir / "1d.parquet")
    return store_dir


class TestComputeSmaForPair:
    def test_returns_close_and_sma(self) -> None:
        close = _make_close(100, drift=0.008, vol=0.01)
        result = compute_sma_for_pair(close, sma_period=50)
        assert result is not None
        last_close, sma = result
        assert isinstance(last_close, float)
        assert isinstance(sma, float)
        assert last_close > 0
        assert sma > 0

    def test_insufficient_data(self) -> None:
        close = _make_close(30)
        assert compute_sma_for_pair(close, sma_period=50) is None

    def test_exact_minimum_bars(self) -> None:
        close = _make_close(50)
        result = compute_sma_for_pair(close, sma_period=50)
        assert result is not None

    def test_nan_close_returns_none(self) -> None:
        close = pd.Series([np.nan] * 60, index=pd.date_range("2025-01-01", periods=60))
        assert compute_sma_for_pair(close, sma_period=50) is None


class TestComputeSmaFeatures:
    def test_basic_features(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, drift=0.003, seed=1),
            "ETH-USD": _make_close(200, drift=-0.003, seed=2),
        })
        cfg = SMAConfig(crypto_ohlcv_dir=store_dir)
        features = compute_sma_features(["BTC-USD", "ETH-USD"], cfg, as_of=date(2025, 7, 1))

        assert len(features) == 2
        for f in features:
            assert isinstance(f, PairSMA)
            assert f.close > 0
            assert f.sma > 0
            assert isinstance(f.bar_date, date)

    def test_missing_pair_skipped(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, seed=1),
        })
        cfg = SMAConfig(crypto_ohlcv_dir=store_dir)
        features = compute_sma_features(["BTC-USD", "MISSING-USD"], cfg)

        assert len(features) == 1
        assert features[0].pair == "BTC-USD"

    def test_insufficient_bars_skipped(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(30, seed=1),
        })
        cfg = SMAConfig(crypto_ohlcv_dir=store_dir, min_bars=60)
        features = compute_sma_features(["BTC-USD"], cfg)

        assert len(features) == 0

    def test_as_of_date_filters_bars(self, tmp_path: Path) -> None:
        close = _make_close(200, drift=0.003, seed=1)
        store_dir = _populate_store(tmp_path, {"BTC-USD": close})
        cfg = SMAConfig(crypto_ohlcv_dir=store_dir, min_bars=60)

        features = compute_sma_features(["BTC-USD"], cfg, as_of=date(2025, 3, 1))
        assert len(features) == 1

    def test_multi_pair(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, drift=0.005, seed=1),
            "ETH-USD": _make_close(200, drift=-0.005, seed=2),
            "SOL-USD": _make_close(200, drift=0.003, seed=3),
        })
        cfg = SMAConfig(crypto_ohlcv_dir=store_dir)
        features = compute_sma_features(["BTC-USD", "ETH-USD", "SOL-USD"], cfg)

        assert len(features) == 3
        pairs = {f.pair for f in features}
        assert pairs == {"BTC-USD", "ETH-USD", "SOL-USD"}

    def test_empty_universe(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "crypto_ohlcv"
        store_dir.mkdir()
        cfg = SMAConfig(crypto_ohlcv_dir=store_dir)
        features = compute_sma_features([], cfg)
        assert features == []
