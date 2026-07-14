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
        {"open": opening, "high": high, "low": low, "close": close, "volume": volume,
         "bar_close_utc": dates + pd.Timedelta(days=1)},
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
        which is in the middle of the gap — no real observation, so NaN.
        Labels are only valid when the terminal date has a real observation.
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
            "bar_close_utc": dates + pd.Timedelta(days=1),
        }, index=dates)

        labels = compute_forward_returns(ohlcv, "TEST", horizons=(5,))

        # Jan 4 + 5 = Jan 9 — no real observation at Jan 9 (in 7-day gap)
        # -> NaN, both from ffill gap AND terminal-obs requirement.
        row_jan4 = labels.loc[labels["date"] == pd.Timestamp("2025-01-04")]
        assert row_jan4["fwd_5d"].isna().all(), (
            "fwd_5d at Jan 4 should be NaN: the 5-calendar-day target "
            "falls in the gap with no real observation"
        )

        # Jan 1 + 5 = Jan 6 — no real observation at Jan 6 (gap starts)
        # -> NaN due to terminal-obs requirement.
        row_jan1 = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        assert row_jan1["fwd_5d"].isna().all(), (
            "fwd_5d at Jan 1 should be NaN: no real observation at Jan 6"
        )

        # Jan 13 + 5 = Jan 18 — no real observation at Jan 18 (past end)
        # -> NaN due to terminal-obs requirement.
        row_jan13 = labels.loc[labels["date"] == pd.Timestamp("2025-01-13")]
        assert row_jan13["fwd_5d"].isna().all(), (
            "fwd_5d at Jan 13 should be NaN: no real observation at Jan 18"
        )

        # But: Jan 13 + (17-13) = verify a date where terminal IS real.
        labels_2d = compute_forward_returns(ohlcv, "TEST", horizons=(1,))
        row_jan13_2 = labels_2d.loc[labels_2d["date"] == pd.Timestamp("2025-01-13")]
        expected = 201 / 200 - 1
        actual = row_jan13_2["fwd_1d"].iloc[0]
        assert abs(actual - expected) < 1e-10, (
            f"fwd_1d at Jan 13 should be {expected:.6f} (real obs at Jan 14), "
            f"got {actual:.6f}"
        )

    def test_btc_excess_uses_same_calendar(self) -> None:
        """BTC-excess labels must reindex BTC to the same daily calendar
        so both legs represent the same n-calendar-day horizon. Both
        legs require a real observation at the terminal date."""
        # Use consecutive dates (no gaps) to avoid terminal-obs issues.
        dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05",
            "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
        ])
        pair_close = [100, 102, 104, 103, 105, 108, 110, 112, 111, 113]
        btc_close_vals = [50000, 50500, 51000, 50800, 51200, 52000, 52500, 53000, 52800, 53200]
        ohlcv = pd.DataFrame({
            "open": pair_close, "high": pair_close, "low": pair_close,
            "close": pair_close, "volume": [1e6] * len(dates),
            "bar_close_utc": dates + pd.Timedelta(days=1),
        }, index=dates)
        btc_series = pd.Series(btc_close_vals, index=dates, name="btc_close")
        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc_series,
        )
        # Jan 1 + 5 = Jan 6 (real observation exists for both pair and BTC).
        row = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        fwd_raw = row["fwd_5d"].iloc[0]
        fwd_excess = row["fwd_5d_btc_excess"].iloc[0]
        expected_raw = 108 / 100 - 1
        expected_btc = 52000 / 50000 - 1
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
        feature_cols = [c for c in panel.columns if c not in ("pair", "date", "feature_available_after") and not c.startswith("fwd_")]
        assert len(feature_cols) == EXPECTED_FEATURE_COUNT
        assert "feature_available_after" in panel.columns
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

        # PIT provenance: observation_end, label_available_at, watermarks, calendar digest.
        assert "observation_end" in manifest
        assert "label_available_at" in manifest
        assert "input_bar_watermarks" in manifest
        assert "calendar_identity_digest" in manifest
        assert len(manifest["calendar_identity_digest"]) == 64
        for slug in pairs:
            assert slug in manifest["observation_end"]
            assert slug in manifest["label_available_at"]
            assert slug in manifest["input_bar_watermarks"]

        # terminal_obs_required flag in label contract.
        assert lc["terminal_obs_required"] is True
        assert lc["row_level_pit_fields"] is True
        assert lc["btc_start_obs_required"] is True
        assert lc["bar_timestamp_convention"] == "UTC_daily_open"
        assert lc["bar_close_offset_days"] == 1
        assert "availability_rule" in lc

    def test_btc_benchmark_in_manifest_when_excluded_from_pairs(self, tmp_path: Path) -> None:
        """When pairs exclude BTC but btc_excess=True, BTC must appear in
        input_bar_digests, input_bar_watermarks, and benchmark_inputs."""
        pairs = ["BTC-USD", "ETH-USD", "SOL-USD"]
        store_dir = _populate_store(tmp_path, pairs)
        out = tmp_path / "panel.parquet"
        cfg = CryptoPanelConfig(
            crypto_ohlcv_dir=store_dir,
            output_path=out,
            pairs=["ETH/USD", "SOL/USD"],
            btc_excess=True,
            min_panel_dates=50,
            min_pairs=2,
        )
        build_crypto_panel(cfg)
        manifest = json.loads(out.with_suffix(".manifest.json").read_text())

        assert BTC_SLUG in manifest["input_bar_digests"], (
            "BTC must appear in input_bar_digests when btc_excess=True"
        )
        assert BTC_SLUG in manifest["input_bar_watermarks"], (
            "BTC must appear in input_bar_watermarks when btc_excess=True"
        )
        assert "benchmark_inputs" in manifest
        assert BTC_SLUG in manifest["benchmark_inputs"]
        assert manifest["benchmark_inputs"][BTC_SLUG]["role"] == "excess_return_denominator"

        # BTC should also appear in observation_end.
        assert BTC_SLUG in manifest["observation_end"]

        # Target pairs should NOT include BTC.
        assert BTC_SLUG not in manifest["pairs"]


class TestRowLevelPIT:
    def test_available_after_columns_exist(self) -> None:
        """Each label horizon must have a companion _available_after column
        with the terminal observation date (NaT where label is NaN)."""
        ohlcv = _make_ohlcv(n_days=100)
        labels = compute_forward_returns(ohlcv, "TEST", horizons=(5, 20))

        assert "fwd_5d_available_after" in labels.columns
        assert "fwd_20d_available_after" in labels.columns

        # Where label is NaN, available_after must be NaT.
        for col in ["fwd_5d", "fwd_20d"]:
            avail_col = f"{col}_available_after"
            nan_mask = labels[col].isna()
            assert labels.loc[nan_mask, avail_col].isna().all(), (
                f"{avail_col} must be NaT where {col} is NaN"
            )
            # Where label is valid, available_after is date + horizon + 1
            # (bar close offset: bar at date D closes at D+1).
            valid_mask = labels[col].notna()
            if valid_mask.any():
                horizon_days = int(col.split("_")[1].rstrip("d"))
                expected_dates = labels.loc[valid_mask, "date"] + pd.Timedelta(days=horizon_days + 1)
                actual_dates = labels.loc[valid_mask, avail_col]
                pd.testing.assert_series_equal(
                    actual_dates.reset_index(drop=True),
                    expected_dates.reset_index(drop=True),
                    check_names=False,
                )

    def test_btc_excess_available_after(self) -> None:
        """BTC-excess _available_after must be NaT where excess is NaN."""
        ohlcv = _make_ohlcv(n_days=100, seed=1)
        btc = _make_ohlcv(n_days=100, seed=2)
        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc["close"],
        )
        assert "fwd_5d_btc_excess_available_after" in labels.columns

        nan_mask = labels["fwd_5d_btc_excess"].isna()
        assert labels.loc[nan_mask, "fwd_5d_btc_excess_available_after"].isna().all()


class TestTerminalGapProtection:
    def test_no_labels_after_last_real_observation(self) -> None:
        """Labels must be NaN when the terminal observation date has no
        real close.  At the end of the series, the last real observation
        is Jan 10.  For horizon=5, labels at dates Jan 6+ need a real
        observation at date+5 (Jan 11+), which doesn't exist, so must
        be NaN."""
        dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09", "2025-01-10",
        ])
        close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ohlcv = pd.DataFrame({
            "open": close, "high": close, "low": close,
            "close": close, "volume": [1e6] * len(close),
            "bar_close_utc": dates + pd.Timedelta(days=1),
        }, index=dates)

        labels = compute_forward_returns(ohlcv, "TEST", horizons=(5,))

        # Jan 6 + 5 = Jan 11 — no real observation at Jan 11 -> NaN.
        for check_date in ["2025-01-06", "2025-01-07", "2025-01-08",
                           "2025-01-09", "2025-01-10"]:
            row = labels.loc[labels["date"] == pd.Timestamp(check_date)]
            assert row["fwd_5d"].isna().all(), (
                f"fwd_5d at {check_date} should be NaN: terminal date "
                f"has no real observation"
            )

        # Jan 5 + 5 = Jan 10 — real observation exists -> valid label.
        row_jan5 = labels.loc[labels["date"] == pd.Timestamp("2025-01-05")]
        expected = 109 / 104 - 1
        actual = row_jan5["fwd_5d"].iloc[0]
        assert abs(actual - expected) < 1e-10, (
            f"fwd_5d at Jan 5 should be {expected:.6f}, got {actual:.6f}"
        )

    def test_btc_excess_terminal_gap(self) -> None:
        """BTC-excess labels must also be NaN when either the pair or BTC
        terminal observation is missing."""
        pair_dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05", "2025-01-06", "2025-01-07",
        ])
        btc_dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05",
        ])
        pair_close = [100, 101, 102, 103, 104, 105, 106]
        btc_close = [50000, 50100, 50200, 50300, 50400]
        ohlcv = pd.DataFrame({
            "open": pair_close, "high": pair_close, "low": pair_close,
            "close": pair_close, "volume": [1e6] * len(pair_close),
            "bar_close_utc": pair_dates + pd.Timedelta(days=1),
        }, index=pair_dates)
        btc_series = pd.Series(btc_close, index=btc_dates, name="btc_close")

        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc_series,
        )

        # Jan 3 + 5 = Jan 8 — pair has no real obs at Jan 8 -> NaN.
        row = labels.loc[labels["date"] == pd.Timestamp("2025-01-03")]
        assert row["fwd_5d"].isna().all()
        assert row["fwd_5d_btc_excess"].isna().all()

        # Jan 1 + 5 = Jan 6 — BTC has no real obs at Jan 6 -> excess NaN.
        row_jan1 = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        assert row_jan1["fwd_5d_btc_excess"].isna().all()

    def test_btc_start_missing_nullifies_excess(self) -> None:
        """BTC-excess labels must be NaN when BTC has no real observation
        at the pair's start date, even if both endpoints are real.

        Pair has daily data Jan 1-10. BTC has data Jan 1, Jan 4-10
        (missing Jan 2-3, within 3-day ffill tolerance).  At pair date
        Jan 2, BTC(t=Jan 2) is forward-filled from Jan 1, but pair(t=Jan 2)
        is real.  The BTC leg would measure [btc(Jan 1), btc(Jan 7)]
        instead of [btc(Jan 2), btc(Jan 7)] — different effective horizon.
        The excess label must be NaN."""
        pair_dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09", "2025-01-10",
        ])
        # BTC missing Jan 2-3 (within 3-day ffill limit).
        btc_dates = pd.to_datetime([
            "2025-01-01", "2025-01-04", "2025-01-05", "2025-01-06",
            "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10",
        ])
        pair_close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        btc_close = [50000, 50300, 50400, 50500, 50600, 50700, 50800, 50900]
        ohlcv = pd.DataFrame({
            "open": pair_close, "high": pair_close, "low": pair_close,
            "close": pair_close, "volume": [1e6] * len(pair_close),
            "bar_close_utc": pair_dates + pd.Timedelta(days=1),
        }, index=pair_dates)
        btc_series = pd.Series(btc_close, index=btc_dates, name="btc_close")

        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc_series,
        )

        # Jan 2: pair(t) real, BTC(t=Jan 2) is ffilled from Jan 1 -> excess NaN.
        row_jan2 = labels.loc[labels["date"] == pd.Timestamp("2025-01-02")]
        assert row_jan2["fwd_5d"].notna().all(), "raw fwd should be valid"
        assert row_jan2["fwd_5d_btc_excess"].isna().all(), (
            "BTC-excess at Jan 2 must be NaN: BTC start is forward-filled, "
            "not a real observation"
        )

        # Jan 1: BTC(t=Jan 1) is real, BTC terminal Jan 6 is real -> valid.
        row_jan1 = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        assert row_jan1["fwd_5d_btc_excess"].notna().all(), (
            "BTC-excess at Jan 1 should be valid: both BTC start and terminal are real"
        )

        # Jan 4: BTC(t=Jan 4) is real, BTC terminal Jan 9 is real -> valid.
        row_jan4 = labels.loc[labels["date"] == pd.Timestamp("2025-01-04")]
        assert row_jan4["fwd_5d_btc_excess"].notna().all(), (
            "BTC-excess at Jan 4 should be valid: both BTC start and terminal are real"
        )


class TestBarClosePIT:
    def test_label_available_after_uses_bar_close_offset(self) -> None:
        """Bar index is bar OPEN timestamp. Close[D] is known at D+1 00:00 UTC.
        Label fwd_5d at date D uses close[D+5], available at D+6 (not D+5).
        Feature at date D uses close[D], available at D+1."""
        dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09", "2025-01-10",
        ])
        close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        ohlcv = pd.DataFrame({
            "open": close, "high": close, "low": close,
            "close": close, "volume": [1e6] * len(close),
            "bar_close_utc": dates + pd.Timedelta(days=1),
        }, index=dates)

        labels = compute_forward_returns(ohlcv, "TEST", horizons=(5,))

        # Jan 1 + 5 = Jan 6 (terminal real). Available after Jan 7 (D+5+1).
        row_jan1 = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        assert row_jan1["fwd_5d"].notna().all()
        avail = row_jan1["fwd_5d_available_after"].iloc[0]
        assert avail == pd.Timestamp("2025-01-07"), (
            f"fwd_5d at Jan 1 available_after should be Jan 7 (D+5+1), got {avail}"
        )

    def test_btc_excess_available_after_bar_close_offset(self) -> None:
        """BTC-excess label at D with horizon N is available at D+N+1."""
        dates = pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
            "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
            "2025-01-09", "2025-01-10",
        ])
        pair_close = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        btc_close = [50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900]
        ohlcv = pd.DataFrame({
            "open": pair_close, "high": pair_close, "low": pair_close,
            "close": pair_close, "volume": [1e6] * len(pair_close),
            "bar_close_utc": dates + pd.Timedelta(days=1),
        }, index=dates)
        btc_series = pd.Series(btc_close, index=dates, name="btc_close")

        labels = compute_forward_returns(
            ohlcv, "ETH-USD", horizons=(5,), btc_close=btc_series,
        )

        row_jan1 = labels.loc[labels["date"] == pd.Timestamp("2025-01-01")]
        assert row_jan1["fwd_5d_btc_excess"].notna().all()
        avail = row_jan1["fwd_5d_btc_excess_available_after"].iloc[0]
        assert avail == pd.Timestamp("2025-01-07"), (
            f"excess available_after should be Jan 7 (D+5+1), got {avail}"
        )

    def test_feature_available_after_in_panel(self, tmp_path: Path) -> None:
        """Panel must include feature_available_after = date + 1 day
        for every row (features use close[D], known at D+1)."""
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
        panel = pd.read_parquet(out)

        assert "feature_available_after" in panel.columns
        expected = pd.to_datetime(panel["date"]) + pd.Timedelta(days=1)
        pd.testing.assert_series_equal(
            panel["feature_available_after"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


class TestBarCloseValidation:
    def test_missing_bar_close_utc_raises(self, tmp_path: Path) -> None:
        """_load_ohlcv must raise ValueError if bar_close_utc is absent."""
        from renquant_base_data.crypto_alpha158_panel import _load_ohlcv

        store_dir = tmp_path / "crypto_ohlcv"
        pair_dir = store_dir / "TEST-USD"
        pair_dir.mkdir(parents=True)
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "open": range(10), "high": range(10), "low": range(10),
            "close": range(10), "volume": [1e6] * 10,
        }, index=dates)
        df.to_parquet(pair_dir / "1d.parquet")

        store = CryptoLocalStore(store_dir)
        with pytest.raises(ValueError, match="bar_close_utc.*required"):
            _load_ohlcv(store, "TEST-USD")

    def test_misaligned_bar_close_utc_raises(self, tmp_path: Path) -> None:
        """_load_ohlcv must raise ValueError if bar_close_utc != index + 1 day."""
        from renquant_base_data.crypto_alpha158_panel import _load_ohlcv

        store_dir = tmp_path / "crypto_ohlcv"
        pair_dir = store_dir / "TEST-USD"
        pair_dir.mkdir(parents=True)
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "open": range(10), "high": range(10), "low": range(10),
            "close": range(10), "volume": [1e6] * 10,
            "bar_close_utc": dates + pd.Timedelta(days=2),
        }, index=dates)
        df.to_parquet(pair_dir / "1d.parquet")

        store = CryptoLocalStore(store_dir)
        with pytest.raises(ValueError, match="does not match UTC daily convention"):
            _load_ohlcv(store, "TEST-USD")

    def test_valid_bar_close_utc_accepted(self, tmp_path: Path) -> None:
        """_load_ohlcv must accept bar_close_utc == index + 1 day."""
        from renquant_base_data.crypto_alpha158_panel import _load_ohlcv

        store_dir = tmp_path / "crypto_ohlcv"
        pair_dir = store_dir / "TEST-USD"
        pair_dir.mkdir(parents=True)
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "open": range(10), "high": range(10), "low": range(10),
            "close": range(10), "volume": [1e6] * 10,
            "bar_close_utc": dates + pd.Timedelta(days=1),
        }, index=dates)
        df.to_parquet(pair_dir / "1d.parquet")

        store = CryptoLocalStore(store_dir)
        result = _load_ohlcv(store, "TEST-USD")
        assert result is not None
        assert "bar_close_utc" in result.columns

    def test_manifest_bar_close_validated(self, tmp_path: Path) -> None:
        """Manifest must carry bar_close_convention_validated and
        availability_derived_from fields."""
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
        lc = manifest["label_contract"]
        assert lc["bar_close_convention_validated"] is True
        assert lc["availability_derived_from"] == "bar_close_utc"
