"""Tests for crypto vol-rank signal module."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.crypto_vol_rank import (
    VolRankConfig,
    backtest_vol_rank,
    compute_trailing_vol,
    rank_by_vol,
    ANNUALIZATION_FACTOR,
)


def _make_ohlcv(n_days: int = 200, vol: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with controllable volatility."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    log_returns = rng.normal(0, vol, n_days)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0.005, 0.03, n_days))
    low = close * (1 - rng.uniform(0.005, 0.03, n_days))
    opening = close * (1 + rng.normal(0, vol * 0.5, n_days))
    volume = rng.uniform(1e6, 1e8, n_days)
    return pd.DataFrame(
        {"open": opening, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _populate_store(tmp_path: Path, pairs_vols: list[tuple[str, float]]) -> Path:
    store_dir = tmp_path / "crypto_ohlcv"
    for i, (slug, vol) in enumerate(pairs_vols):
        pair_dir = store_dir / slug
        pair_dir.mkdir(parents=True)
        ohlcv = _make_ohlcv(n_days=200, vol=vol, seed=42 + i)
        ohlcv.to_parquet(pair_dir / "1d.parquet")
    return store_dir


class TestComputeTrailingVol:
    def test_returns_annualized_vol(self) -> None:
        ohlcv = _make_ohlcv(n_days=100, vol=0.03)
        vol = compute_trailing_vol(ohlcv["close"], window=20)
        assert vol is not None
        assert 0.3 < vol < 0.8  # ~3% daily * sqrt(365) ~ 0.57

    def test_insufficient_bars_returns_none(self) -> None:
        ohlcv = _make_ohlcv(n_days=10, vol=0.02)
        assert compute_trailing_vol(ohlcv["close"], window=20) is None

    def test_higher_daily_vol_gives_higher_annual(self) -> None:
        low_vol = compute_trailing_vol(_make_ohlcv(vol=0.01, seed=1)["close"])
        high_vol = compute_trailing_vol(_make_ohlcv(vol=0.05, seed=2)["close"])
        assert low_vol is not None and high_vol is not None
        assert high_vol > low_vol


class TestRankByVol:
    def test_ranks_ascending_by_vol(self, tmp_path: Path) -> None:
        pairs = [
            ("DOGE-USD", 0.06),
            ("ETH-USD", 0.03),
            ("BTC-USD", 0.01),
        ]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2)
        result = rank_by_vol(cfg)

        assert result.n_pairs_scored == 3
        assert result.rankings[0]["pair"] == "BTC-USD"
        assert result.rankings[-1]["pair"] == "DOGE-USD"
        assert result.rankings[0]["annualized_vol"] < result.rankings[-1]["annualized_vol"]

    def test_selects_top_n_lowest(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.01), ("ETH-USD", 0.02), ("SOL-USD", 0.04), ("DOGE-USD", 0.06)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2)
        result = rank_by_vol(cfg)

        assert len(result.selected) == 2
        assert set(result.selected) == {"BTC-USD", "ETH-USD"}
        assert abs(result.weights["BTC-USD"] - 0.5) < 1e-6
        assert abs(result.weights["ETH-USD"] - 0.5) < 1e-6

    def test_equal_weights(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.01), ("ETH-USD", 0.02), ("SOL-USD", 0.03)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=3)
        result = rank_by_vol(cfg)

        assert len(result.weights) == 3
        for w in result.weights.values():
            assert abs(w - 1/3) < 1e-4

    def test_exclude_pairs(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.03), ("ETH-USD", 0.04), ("SOL-USD", 0.05)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2, exclude_pairs=["BTC-USD"])
        result = rank_by_vol(cfg)

        assert result.n_pairs_scored == 2
        assert result.n_pairs_excluded == 1
        assert "BTC-USD" not in [r["pair"] for r in result.rankings]

    def test_top_n_capped_at_available(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.01), ("ETH-USD", 0.02)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=10)
        result = rank_by_vol(cfg)

        assert result.top_n == 2
        assert len(result.selected) == 2

    def test_empty_store_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        cfg = VolRankConfig(crypto_ohlcv_dir=empty)
        with pytest.raises((RuntimeError, FileNotFoundError)):
            rank_by_vol(cfg)

    def test_to_dict_roundtrip(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.01), ("ETH-USD", 0.02)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2)
        result = rank_by_vol(cfg)
        d = result.to_dict()

        assert d["n_pairs_scored"] == 2
        assert len(d["selected"]) == 2
        assert len(d["rankings"]) == 2

    def test_as_of_date(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.01), ("ETH-USD", 0.02)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2)
        result = rank_by_vol(cfg, as_of=date(2025, 5, 1))

        assert result.as_of_date == date(2025, 5, 1)


class TestBacktest:
    def test_returns_dataframe_with_port_return(self, tmp_path: Path) -> None:
        pairs = [("BTC-USD", 0.01), ("ETH-USD", 0.03), ("SOL-USD", 0.05)]
        store_dir = _populate_store(tmp_path, pairs)
        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2, min_bars=30)
        bt = backtest_vol_rank(cfg, rebalance_days=20)

        assert "port_return" in bt.columns
        assert "rebalance" in bt.columns
        assert len(bt) > 100
        assert bt["rebalance"].sum() >= 1

    def test_low_vol_outperforms_high_vol_on_average(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(123)
        store_dir = tmp_path / "crypto_ohlcv"

        # Create pairs where low-vol has positive drift, high-vol has negative
        for slug, daily_vol, drift in [
            ("BTC-USD", 0.01, 0.001), ("ETH-USD", 0.03, 0.0),
            ("SOL-USD", 0.06, -0.002), ("DOGE-USD", 0.08, -0.003),
        ]:
            pair_dir = store_dir / slug
            pair_dir.mkdir(parents=True)
            n = 300
            dates = pd.date_range("2025-01-01", periods=n, freq="D")
            log_ret = rng.normal(drift, daily_vol, n)
            close = 100.0 * np.exp(np.cumsum(log_ret))
            df = pd.DataFrame({
                "open": close * (1 + rng.normal(0, 0.005, n)),
                "high": close * 1.02, "low": close * 0.98,
                "close": close, "volume": np.ones(n) * 1e6,
            }, index=dates)
            df.to_parquet(pair_dir / "1d.parquet")

        cfg = VolRankConfig(crypto_ohlcv_dir=store_dir, top_n=2, min_bars=30)
        bt = backtest_vol_rank(cfg, rebalance_days=20)

        total_return = (1 + bt["port_return"]).prod() - 1
        # Low-vol pairs (STABLE, MEDIUM) have positive drift
        assert total_return > 0
