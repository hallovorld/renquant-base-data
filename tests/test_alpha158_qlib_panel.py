"""Tests for the subrepo-owned alpha158 Qlib panel builder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.alpha158_qlib_panel import (
    EXPECTED_ALPHA158_FEATURES,
    build_alpha158_qlib_panel,
)


pytest.importorskip("pyarrow")


def _write_ohlcv(data_dir: Path, ticker: str, *, base: float, slope: float, phase: float) -> None:
    dates = pd.bdate_range("2025-01-02", periods=150)
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


def _write_inputs(data_dir: Path, *, include_spy: bool = True) -> None:
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

    dates = pd.bdate_range("2025-01-02", periods=150)
    split = pd.Series("test", index=dates)
    split.iloc[:90] = "train"
    split.iloc[90:120] = "val"
    pd.DataFrame({"date": dates, "split_label": split.values}).to_parquet(
        data_dir / "transformer_dataset_engineered.parquet",
        index=False,
    )

    _write_ohlcv(data_dir, "AAA", base=50.0, slope=0.08, phase=0.0)
    _write_ohlcv(data_dir, "BBB", base=80.0, slope=0.03, phase=0.9)
    if include_spy:
        _write_ohlcv(data_dir, "SPY", base=300.0, slope=0.05, phase=0.4)


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
