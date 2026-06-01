from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.watchlist_screen import perf_stats, screen_watchlist


pytest.importorskip("pyarrow")


def _write_ohlcv(data_dir: Path, ticker: str, closes: np.ndarray) -> None:
    path = data_dir / "ohlcv" / ticker
    path.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    pd.DataFrame({"close": closes}, index=dates).to_parquet(path / "1d.parquet")


def test_perf_stats_rejects_short_series() -> None:
    assert perf_stats(pd.Series([1.0, 2.0, 3.0])) == {}


def test_screen_watchlist_writes_report_and_recommendations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RENQUANT_NO_NOTIFY", "1")
    cfg = {
        "watchlist": ["AAA", "BBB"],
        "defensive_tickers": ["BBB"],
    }
    cfg_path = tmp_path / "strategy_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    n = 60
    wiggle = np.sin(np.linspace(0, 8, n))
    _write_ohlcv(tmp_path, "SPY", np.linspace(100, 112, n) + wiggle)
    _write_ohlcv(tmp_path, "AAA", np.linspace(100, 75, n) + wiggle)
    _write_ohlcv(tmp_path, "BBB", np.linspace(100, 70, n) + wiggle)
    _write_ohlcv(tmp_path, "CCC", np.linspace(100, 150, n) + wiggle)

    result = screen_watchlist(
        strategy_config=cfg_path,
        data_dir=tmp_path,
        output_dir=tmp_path / "reports",
        lookback_days=90,
        top_add_candidates=3,
        send_notify=False,
    )

    assert result.report_path.exists()
    assert [item["ticker"] for item in result.drops] == ["AAA"]
    assert "CCC" in [item["ticker"] for item in result.adds]
    text = result.report_path.read_text(encoding="utf-8")
    assert "Drop candidates" in text
    assert "Add candidates" in text
