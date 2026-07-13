"""Tests for the crypto alpha158 panel builder (D-C3)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.crypto_alpha158_panel import (
    BTC_SLUG,
    CryptoPanelConfig,
    FFILL_LIMIT_DAYS,
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

    def test_gap_calendar_shift_is_calendar_days(self) -> None:
        """shift(-n) must mean n calendar days, not n observed rows.

        Create OHLCV with a 7-day gap (Jan 5 -> Jan 13).  With row-based
        shift, fwd_5d at Jan 4 would reach row index+5 = Jan 17 (skipping
        the gap entirely).  With calendar-day shift, it reaches Jan 9
        which is in the middle of the gap (4 days after last observation
        on Jan 5) — beyond FFILL_LIMIT_DAYS (3), so NaN.
        """
        dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05",
            # 7-day gap: Jan 6-12 missing
            "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17",
        ])
        close = [100, 101, 102, 103, 104, 200, 201, 202, 203, 204]
        ohlcv = pd.DataFrame({
            "open": close, "high": close, "low": close,
            "close": close, "volume": [1e6] * len(close),
        }, index=dates)

        labels = compute_forward_returns(ohlcv, "TEST", horizons=(5,))

        # Jan 4 + 5 calendar days = Jan 9.  Jan 9 is 4 days after
        # the last pre-gap observation (Jan 5), beyond FFILL_LIMIT_DAYS=3
        # -> NaN.  A row-based shift would have returned 204/103-1.
        row_jan4 = labels.loc[labels["date"] == pd.Timestamp("2025-01-04")]
        assert row_jan4["fwd_5d"].isna().all(), (
            "fwd_5d at Jan 4 should be NaN: the 5-calendar-day target "
            "falls beyond the ffill limit in the gap"
        )

        # Jan 1 + 5 calendar days = Jan 6.  Jan 6 is 1 day after Jan 5
        # (within ffill limit) -> ffilled close = 104.
        row_jan1 = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        expected = 104 / 100 - 1
        actual = row_jan1["fwd_5d"].iloc[0]
        assert abs(actual - expected) < 1e-10, (
            f"fwd_5d at Jan 1 should be {expected:.6f} (ffill from Jan 5), "
            f"got {actual:.6f}"
        )

        # Jan 13 + 5 = Jan 18.  Jan 17 is 1 day before, within limit.
        # ffilled close = 204.
        row_jan13 = labels.loc[labels["date"] == pd.Timestamp("2025-01-13")]
        expected_13 = 204 / 200 - 1
        actual_13 = row_jan13["fwd_5d"].iloc[0]
        assert abs(actual_13 - expected_13) < 1e-10

    def test_btc_excess_uses_same_calendar(self) -> None:
        """BTC-excess labels must reindex BTC to the same daily calendar
        so both legs represent the same n-calendar-day horizon."""
        dates = pd.to_datetime([
            "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
            "2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17",
        ])
        pair_close = [100, 102, 104, 103, 105, 108, 110, 112, 111, 113]
        btc_close_vals = [50000, 50500, 51000, 50800, 51200, 52000, 52500, 53000, 52800, 53200]
        ohlcv = pd.DataFrame({
            "open": pair_close, "high": pair_close, "low": pair_close,
            "close": pair_close, "volume": [1e6] * len(dates),
        }, index=dates)
        btc_series = pd.Series(btc_close_vals, index=dates, name="btc_close")
        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc_series,
        )
        # Jan 6 + 5 calendar days = Jan 11 (Sat).
        # With ffill limit 3: Jan 10 (Fri) is 1 day before, within limit.
        # So Jan 11 close = ffill from Jan 10 = 105 (pair), 51200 (BTC).
        row = labels.loc[labels["date"] == pd.Timestamp("2025-01-06")]
        fwd_raw = row["fwd_5d"].iloc[0]
        fwd_excess = row["fwd_5d_btc_excess"].iloc[0]
        expected_raw = 105 / 100 - 1
        expected_btc = 51200 / 50000 - 1
        assert abs(fwd_raw - expected_raw) < 1e-10
        assert abs(fwd_excess - (expected_raw - expected_btc)) < 1e-10


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

    def test_manifest_integrity(self, tmp_path: Path) -> None:
        """Manifest must carry full parquet SHA-256, input bar digests,
        label contract, and feature config digest — not a truncated
        CSV-derived presentation hash."""
        pairs = ["BTC-USD", "ETH-USD"]
        store_dir = _populate_store(tmp_path, pairs)
        out = tmp_path / "panel.parquet"
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=out,
            min_panel_dates=50,
            min_pairs=2,
        )
        build_crypto_panel(cfg)

        manifest = json.loads(out.with_suffix(".manifest.json").read_text())

        # Full parquet SHA-256 matches actual file.
        actual_sha = hashlib.sha256(out.read_bytes()).hexdigest()
        assert manifest["parquet_sha256"] == actual_sha
        assert len(manifest["parquet_sha256"]) == 64

        # Truncated prefix is gone.
        assert "content_sha256_prefix" not in manifest

        # Input bar digests present and correct.
        assert "input_bar_digests" in manifest
        for slug in pairs:
            bar_path = store_dir / slug / "1d.parquet"
            expected_digest = hashlib.sha256(bar_path.read_bytes()).hexdigest()
            assert manifest["input_bar_digests"][slug] == expected_digest

        # Label contract present with correct fields.
        lc = manifest["label_contract"]
        assert lc["type"] == "calendar_day_forward_return"
        assert lc["horizons_calendar_days"] == [5, 20, 60]
        assert lc["ffill_limit_days"] == FFILL_LIMIT_DAYS
        assert isinstance(lc["btc_excess"], bool)

        # Feature config digest present.
        assert "feature_config_digest" in manifest
        assert len(manifest["feature_config_digest"]) == 64
