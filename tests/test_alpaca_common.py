from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_base_data.alpaca_common import load_strategy_watchlist


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_strategy_watchlist_reads_top_level_watchlist(tmp_path: Path) -> None:
    path = _write(tmp_path / "strategy_config.json", {"watchlist": ["aapl", "-", "msft"]})

    assert load_strategy_watchlist(path) == ["AAPL", "MSFT"]


def test_load_strategy_watchlist_rejects_symbols_schema(tmp_path: Path) -> None:
    path = _write(tmp_path / "strategy_config.json", {"symbols": ["AAPL"]})

    with pytest.raises(ValueError, match="top-level 'watchlist'"):
        load_strategy_watchlist(path)


def test_load_strategy_watchlist_rejects_nested_data_watchlist_schema(tmp_path: Path) -> None:
    path = _write(tmp_path / "strategy_config.json", {"data": {"watchlist": ["AAPL"]}})

    with pytest.raises(ValueError, match="top-level 'watchlist'"):
        load_strategy_watchlist(path)
