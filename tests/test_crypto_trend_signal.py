"""Tests for crypto SMA50 trend-following signal (G2 v3)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.crypto_trend_signal import (
    TrendSignalConfig,
    PairSignal,
    SignalSnapshot,
    compute_signal_for_pair,
    compute_signals,
    _universe_hash,
    _snapshot_digest,
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


class TestComputeSignalForPair:
    def test_long_when_above_sma(self) -> None:
        close = _make_close(100, drift=0.008, vol=0.01)
        result = compute_signal_for_pair(close, sma_period=50)
        assert result is not None
        signal, last_close, sma = result
        assert signal == 1
        assert last_close > sma

    def test_cash_when_below_sma(self) -> None:
        close = _make_close(100, drift=-0.005, seed=99)
        result = compute_signal_for_pair(close, sma_period=50)
        assert result is not None
        signal, last_close, sma = result
        assert signal == 0
        assert last_close <= sma

    def test_insufficient_data(self) -> None:
        close = _make_close(30)
        assert compute_signal_for_pair(close, sma_period=50) is None

    def test_exact_minimum_bars(self) -> None:
        close = _make_close(50)
        result = compute_signal_for_pair(close, sma_period=50)
        assert result is not None

    def test_nan_close_returns_none(self) -> None:
        close = pd.Series([np.nan] * 60, index=pd.date_range("2025-01-01", periods=60))
        assert compute_signal_for_pair(close, sma_period=50) is None


class TestComputeSignals:
    def test_basic_signal_snapshot(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, drift=0.003, seed=1),
            "ETH-USD": _make_close(200, drift=-0.003, seed=2),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals(["BTC-USD", "ETH-USD"], cfg, as_of=date(2025, 7, 1))

        assert isinstance(snap, SignalSnapshot)
        assert len(snap.signals) == 2
        assert snap.n_long + snap.n_cash == 2
        assert snap.digest.startswith("sha256:")
        assert snap.universe_hash.startswith("sha256:")

    def test_missing_pair_skipped(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, seed=1),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals(["BTC-USD", "MISSING-USD"], cfg)

        assert len(snap.signals) == 1
        assert snap.signals[0].pair == "BTC-USD"

    def test_insufficient_bars_skipped(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(30, seed=1),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir, min_bars=60)
        snap = compute_signals(["BTC-USD"], cfg)

        assert len(snap.signals) == 0
        assert snap.n_long == 0
        assert snap.n_cash == 0

    def test_as_of_date_filters_bars(self, tmp_path: Path) -> None:
        close = _make_close(200, drift=0.003, seed=1)
        store_dir = _populate_store(tmp_path, {"BTC-USD": close})
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir, min_bars=60)

        snap = compute_signals(["BTC-USD"], cfg, as_of=date(2025, 3, 1))
        assert len(snap.signals) == 1
        assert snap.as_of_date == date(2025, 3, 1)

    def test_snapshot_to_dict(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, seed=1),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals(["BTC-USD"], cfg)
        d = snap.to_dict()

        assert "as_of_date" in d
        assert "signals" in d
        assert len(d["signals"]) == 1
        assert d["signals"][0]["pair"] == "BTC-USD"
        assert d["signals"][0]["signal"] in (0, 1)
        assert isinstance(d["signals"][0]["close"], float)
        assert isinstance(d["signals"][0]["sma"], float)
        assert d["digest"].startswith("sha256:")

    def test_digest_deterministic(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, seed=1),
            "ETH-USD": _make_close(200, seed=2),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap1 = compute_signals(["BTC-USD", "ETH-USD"], cfg, as_of=date(2025, 7, 1))
        snap2 = compute_signals(["BTC-USD", "ETH-USD"], cfg, as_of=date(2025, 7, 1))
        assert snap1.digest == snap2.digest

    def test_digest_changes_with_data(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, seed=1),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap1 = compute_signals(["BTC-USD"], cfg, as_of=date(2025, 6, 1))
        snap2 = compute_signals(["BTC-USD"], cfg, as_of=date(2025, 7, 1))
        assert snap1.digest != snap2.digest

    def test_pair_order_does_not_affect_digest(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, seed=1),
            "ETH-USD": _make_close(200, seed=2),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap1 = compute_signals(["BTC-USD", "ETH-USD"], cfg, as_of=date(2025, 7, 1))
        snap2 = compute_signals(["ETH-USD", "BTC-USD"], cfg, as_of=date(2025, 7, 1))
        assert snap1.digest == snap2.digest

    def test_empty_universe(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "crypto_ohlcv"
        store_dir.mkdir()
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals([], cfg)
        assert len(snap.signals) == 0
        assert snap.n_long == 0
        assert snap.n_cash == 0


class TestUniverseHash:
    def test_deterministic(self) -> None:
        h1 = _universe_hash(["BTC-USD", "ETH-USD"])
        h2 = _universe_hash(["BTC-USD", "ETH-USD"])
        assert h1 == h2

    def test_order_independent(self) -> None:
        h1 = _universe_hash(["ETH-USD", "BTC-USD"])
        h2 = _universe_hash(["BTC-USD", "ETH-USD"])
        assert h1 == h2

    def test_different_sets(self) -> None:
        h1 = _universe_hash(["BTC-USD"])
        h2 = _universe_hash(["ETH-USD"])
        assert h1 != h2


class TestSignalIntegration:
    def test_uptrend_detected(self, tmp_path: Path) -> None:
        """Strong uptrend should produce LONG signal."""
        close = _make_close(200, drift=0.005, vol=0.01, seed=42)
        store_dir = _populate_store(tmp_path, {"BTC-USD": close})
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals(["BTC-USD"], cfg)

        assert snap.n_long == 1
        sig = snap.signals[0]
        assert sig.signal == 1
        assert sig.close > sig.sma

    def test_downtrend_detected(self, tmp_path: Path) -> None:
        """Strong downtrend should produce CASH signal."""
        close = _make_close(200, drift=-0.005, vol=0.01, seed=42)
        store_dir = _populate_store(tmp_path, {"BTC-USD": close})
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals(["BTC-USD"], cfg)

        assert snap.n_cash == 1
        sig = snap.signals[0]
        assert sig.signal == 0
        assert sig.close <= sig.sma

    def test_multi_pair_mixed_signals(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, {
            "BTC-USD": _make_close(200, drift=0.005, seed=1),
            "ETH-USD": _make_close(200, drift=-0.005, seed=2),
            "SOL-USD": _make_close(200, drift=0.003, seed=3),
        })
        cfg = TrendSignalConfig(crypto_ohlcv_dir=store_dir)
        snap = compute_signals(["BTC-USD", "ETH-USD", "SOL-USD"], cfg)

        assert len(snap.signals) == 3
        assert snap.n_long + snap.n_cash == 3
        assert snap.n_long >= 1
        assert snap.n_cash >= 1
