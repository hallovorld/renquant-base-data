"""Tests for the crypto alpha158 panel builder (D-C3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.crypto_alpha158_panel import (
    BTC_SLUG,
    CryptoPanelConfig,
    build_crypto_panel,
    build_features_for_pair,
    compute_forward_returns,
    discover_pairs,
    EXPECTED_FEATURE_COUNT,
)
from renquant_base_data.crypto_bars import CryptoLocalStore


def _make_ohlcv(n_days: int = 200, start: str = "2025-01-01", seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, freq="D")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
    high = close * (1 + rng.uniform(0.005, 0.05, n_days))
    low = close * (1 - rng.uniform(0.005, 0.05, n_days))
    opening = close * (1 + rng.normal(0, 0.01, n_days))
    volume = rng.uniform(1e6, 1e8, n_days)
    df = pd.DataFrame(
        {"open": opening, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def _populate_store(tmp_path: Path, pairs: list[str], n_days: int = 200) -> Path:
    """Create a fake crypto OHLCV store with synthetic data."""
    store_dir = tmp_path / "crypto_ohlcv"
    for i, slug in enumerate(pairs):
        pair_dir = store_dir / slug
        pair_dir.mkdir(parents=True)
        ohlcv = _make_ohlcv(n_days=n_days, seed=42 + i)
        ohlcv.to_parquet(pair_dir / "1d.parquet")
    return store_dir


class TestDiscoverPairs:
    def test_finds_pairs(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, ["BTC-USD", "ETH-USD", "SOL-USD"])
        slugs = discover_pairs(store_dir)
        assert slugs == ["BTC-USD", "ETH-USD", "SOL-USD"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_pairs(tmp_path / "nonexistent") == []

    def test_skips_dirs_without_parquet(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "crypto_ohlcv"
        (store_dir / "BTC-USD").mkdir(parents=True)
        (store_dir / "BTC-USD" / "1d.parquet").touch()
        (store_dir / "EMPTY-PAIR").mkdir(parents=True)
        assert discover_pairs(store_dir) == ["BTC-USD"]


class TestBuildFeaturesForPair:
    def test_produces_158_features(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, ["BTC-USD"])
        store = CryptoLocalStore(store_dir)
        feats = build_features_for_pair("BTC-USD", store)
        assert feats is not None
        feature_cols = [c for c in feats.columns if c not in ("pair", "date")]
        assert len(feature_cols) == EXPECTED_FEATURE_COUNT
        assert "pair" in feats.columns
        assert (feats["pair"] == "BTC-USD").all()

    def test_insufficient_bars_returns_none(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, ["SHORT-PAIR"], n_days=30)
        store = CryptoLocalStore(store_dir)
        assert build_features_for_pair("SHORT-PAIR", store, min_bars=70) is None

    def test_missing_pair_returns_none(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "empty_store"
        store_dir.mkdir()
        store = CryptoLocalStore(store_dir)
        assert build_features_for_pair("MISSING-PAIR", store) is None


class TestComputeForwardReturns:
    def test_raw_returns(self) -> None:
        ohlcv = _make_ohlcv(n_days=100)
        labels = compute_forward_returns(ohlcv, "BTC-USD", horizons=(5, 20))
        assert "fwd_5d" in labels.columns
        assert "fwd_20d" in labels.columns
        assert "pair" in labels.columns
        assert labels["fwd_5d"].notna().sum() > 50

    def test_btc_excess(self) -> None:
        ohlcv = _make_ohlcv(n_days=100, seed=1)
        btc = _make_ohlcv(n_days=100, seed=2)
        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc["close"],
        )
        assert "fwd_5d_btc_excess" in labels.columns
        assert "fwd_5d" in labels.columns

    def test_no_btc_when_none(self) -> None:
        ohlcv = _make_ohlcv(n_days=100)
        labels = compute_forward_returns(ohlcv, "BTC-USD", horizons=(5,), btc_close=None)
        assert "fwd_5d" in labels.columns
        assert "fwd_5d_btc_excess" not in labels.columns


class TestBuildCryptoPanel:
    def test_full_build(self, tmp_path: Path) -> None:
        pairs = ["BTC-USD", "ETH-USD", "SOL-USD"]
        store_dir = _populate_store(tmp_path, pairs)
        out = tmp_path / "panel.parquet"
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=out,
            min_panel_dates=50,
            min_pairs=2,
        )
        result = build_crypto_panel(cfg)
        assert result == out
        assert out.exists()

        panel = pd.read_parquet(out)
        assert panel["pair"].nunique() == 3
        feature_cols = [c for c in panel.columns if c not in ("pair", "date") and not c.startswith("fwd_")]
        assert len(feature_cols) == EXPECTED_FEATURE_COUNT
        assert "fwd_20d" in panel.columns
        assert "fwd_20d_btc_excess" in panel.columns

        manifest_path = out.with_suffix(".manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["n_pairs"] == 3
        assert manifest["n_features"] == EXPECTED_FEATURE_COUNT
        assert manifest["btc_excess"] is True

    def test_too_few_pairs_raises(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, ["BTC-USD"])
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=tmp_path / "panel.parquet",
            min_pairs=3,
            min_panel_dates=10,
        )
        with pytest.raises(RuntimeError, match="only 1 pairs"):
            build_crypto_panel(cfg)

    def test_too_few_dates_raises(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, ["BTC-USD", "ETH-USD"], n_days=100)
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=tmp_path / "panel.parquet",
            min_panel_dates=200,
            min_pairs=2,
        )
        with pytest.raises(RuntimeError, match="only .* dates"):
            build_crypto_panel(cfg)

    def test_no_btc_excess(self, tmp_path: Path) -> None:
        pairs = ["BTC-USD", "ETH-USD"]
        store_dir = _populate_store(tmp_path, pairs)
        out = tmp_path / "panel.parquet"
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=out,
            btc_excess=False,
            min_panel_dates=50,
            min_pairs=2,
        )
        build_crypto_panel(cfg)
        panel = pd.read_parquet(out)
        btc_excess_cols = [c for c in panel.columns if "btc_excess" in c]
        assert len(btc_excess_cols) == 0

    def test_specific_pairs(self, tmp_path: Path) -> None:
        store_dir = _populate_store(tmp_path, ["BTC-USD", "ETH-USD", "SOL-USD"])
        out = tmp_path / "panel.parquet"
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=out,
            pairs=["BTC/USD", "ETH/USD"],
            min_panel_dates=50,
            min_pairs=2,
        )
        build_crypto_panel(cfg)
        panel = pd.read_parquet(out)
        assert set(panel["pair"].unique()) == {"BTC-USD", "ETH-USD"}

    def test_empty_store_raises(self, tmp_path: Path) -> None:
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=tmp_path / "empty",
            output_path=tmp_path / "panel.parquet",
        )
        with pytest.raises(RuntimeError, match="No crypto pairs"):
            build_crypto_panel(cfg)
