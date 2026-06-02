"""Tests for the subrepo-owned alpha158 Qlib panel builder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.alpha158_qlib_panel import (
    EXPECTED_ALPHA158_FEATURES,
    InsufficientTrainHistoryError,
    MIN_TRACK_B_TRAIN_OBS,
    build_alpha158_qlib_panel,
)


pytest.importorskip("pyarrow")

# Default fixture row count must exceed one 252-day Track B warmup window
# plus enough train rows to satisfy MIN_TRACK_B_TRAIN_OBS finite-observation
# gate. 600 rows: ~503 train + ~50 val + ~47 test leaves > MIN_TRACK_B_TRAIN_OBS
# finite samples for each 252-day feature in train.
_DEFAULT_ROWS = 600


def _write_ohlcv(
    data_dir: Path,
    ticker: str,
    *,
    base: float,
    slope: float,
    phase: float,
    rows: int = _DEFAULT_ROWS,
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    x = np.arange(len(dates), dtype=float)
    close = base + slope * x + 0.35 * np.sin(x / 8.0 + phase)
    open_ = close * (1.0 + 0.001 * np.cos(x / 7.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_000_000 + (x * 137).astype(int) + int(base) * 10
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    out_dir = data_dir / "ohlcv" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "1d.parquet")


def _write_inputs(
    data_dir: Path, *, include_spy: bool = True, rows: int = _DEFAULT_ROWS
) -> None:
    (data_dir / "transformer_universe_inventory.json").write_text(
        json.dumps({"tier_A_tickers": ["AAA"], "tier_B_tickers": ["BBB"]})
    )
    (data_dir / "transformer_data_integrity_report.json").write_text(
        json.dumps(
            {
                "per_ticker": {
                    "A": [{"ticker": "AAA", "ok": True}],
                    "B": [{"ticker": "BBB", "ok": True}],
                }
            }
        )
    )

    dates = pd.bdate_range("2024-01-02", periods=rows)
    split = pd.Series("test", index=dates)
    # ~84% train / ~8% val / ~8% test so train comfortably covers >252 finite
    # observations per Track B feature even after the 252-day warmup is NaN.
    train_end = int(rows * 0.84)
    val_end = int(rows * 0.92)
    split.iloc[:train_end] = "train"
    split.iloc[train_end:val_end] = "val"
    pd.DataFrame({"date": dates, "split_label": split.values}).to_parquet(
        data_dir / "transformer_dataset_engineered.parquet",
        index=False,
    )

    _write_ohlcv(data_dir, "AAA", base=50.0, slope=0.08, phase=0.0, rows=rows)
    _write_ohlcv(data_dir, "BBB", base=80.0, slope=0.03, phase=0.9, rows=rows)
    if include_spy:
        _write_ohlcv(data_dir, "SPY", base=300.0, slope=0.05, phase=0.4, rows=rows)


def test_build_alpha158_qlib_panel_writes_dataset_and_stats(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    out = build_alpha158_qlib_panel(tmp_path, max_workers=1)

    panel = pd.read_parquet(out)
    stats = json.loads(out.with_suffix(".stats.json").read_text())

    assert set(panel["ticker"]) == {"AAA", "BBB"}
    assert set(["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "split_label"]).issubset(panel.columns)
    assert len(stats["feature_cols"]) == EXPECTED_ALPHA158_FEATURES
    assert len(stats["feature_means"]) == EXPECTED_ALPHA158_FEATURES
    assert stats["feature_preprocess_version"] == 2
    assert stats["n_train_rows"] > 0
    assert not panel[stats["feature_cols"]].isna().any().any()
    assert panel["split_label"].isin({"train", "val", "test"}).all()


def test_build_alpha158_qlib_panel_requires_spy_for_excess_labels(tmp_path: Path) -> None:
    _write_inputs(tmp_path, include_spy=False)

    with pytest.raises(FileNotFoundError, match="SPY OHLCV"):
        build_alpha158_qlib_panel(tmp_path, max_workers=1)


def test_build_alpha158_qlib_panel_with_track_b_appends_four_features(tmp_path: Path) -> None:
    from renquant_base_data.track_b_features import TRACK_B_FEATURES

    _write_inputs(tmp_path)

    out = build_alpha158_qlib_panel(tmp_path, max_workers=1, include_track_b=True)
    panel = pd.read_parquet(out)
    stats = json.loads(out.with_suffix(".stats.json").read_text())

    # All 4 Track B features are present, ordered after baseline alpha158.
    for col in TRACK_B_FEATURES:
        assert col in panel.columns, f"missing Track B column {col}"
        assert col in stats["feature_cols"], f"Track B column {col} not in stats"
    # The baseline 158 alpha features survive too.
    assert len(stats["feature_cols"]) == EXPECTED_ALPHA158_FEATURES + len(TRACK_B_FEATURES)
    # Train-only stats fit applied (no infinities, no NaNs in feature cols).
    assert not panel[stats["feature_cols"]].isna().any().any()

    # HIGH-finding regression guard (codex PR #16 review, 2026-06-02): the
    # Track B feature means/stds in the stats artifact MUST be finite, and
    # the materialized columns MUST NOT be identically zero. A bare-NaN
    # mean here would mean NormalizeAndAnnotateJob silently turned the
    # advertised feature into all-zero downstream, the exact silent-NaN
    # propagation the reviewer caught.
    for col in TRACK_B_FEATURES:
        col_idx = stats["feature_cols"].index(col)
        mean = stats["feature_means"][col_idx]
        std = stats["feature_stds"][col_idx]
        assert np.isfinite(mean), f"Track B {col} train mean not finite: {mean}"
        assert np.isfinite(std), f"Track B {col} train std not finite: {std}"
        # The materialized column has SOME non-zero values across the panel
        # (otherwise the feature is effectively dead even if stats look ok).
        col_values = panel[col].to_numpy(dtype=float)
        assert np.any(np.abs(col_values) > 1e-12), (
            f"Track B {col} column is identically zero post-normalize "
            "(silent NaN-fill collapse)"
        )


def test_build_alpha158_qlib_panel_track_b_raises_on_insufficient_train_history(
    tmp_path: Path,
) -> None:
    """Negative test for codex PR #16 HIGH finding: when the train split has
    fewer than ``MIN_TRACK_B_TRAIN_OBS`` finite observations for any 252-day
    Track B feature, the build must raise ``InsufficientTrainHistoryError``
    at the source — NOT silently emit zero columns advertised as live
    features.
    """
    _write_inputs(tmp_path, rows=150)  # < 252, both 252-day features all-NaN
    with pytest.raises(InsufficientTrainHistoryError) as excinfo:
        build_alpha158_qlib_panel(tmp_path, max_workers=1, include_track_b=True)
    msg = str(excinfo.value)
    assert "mom_carry_12_1" in msg
    assert "beta_dm" in msg
    assert str(MIN_TRACK_B_TRAIN_OBS) in msg
