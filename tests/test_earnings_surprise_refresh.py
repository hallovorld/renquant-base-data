from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from renquant_base_data.earnings_surprise_refresh import (
    load_watchlist,
    refresh_earnings_surprise,
)


def _strategy_config(path: Path) -> None:
    path.write_text(json.dumps({"watchlist": ["aapl", "msft"]}), encoding="utf-8")


def test_load_watchlist_from_strategy_config(tmp_path: Path) -> None:
    cfg = tmp_path / "strategy_config.json"
    _strategy_config(cfg)

    assert load_watchlist(cfg) == ["AAPL", "MSFT"]


def test_refresh_earnings_surprise_writes_under_data_dir(tmp_path: Path) -> None:
    def provider(symbol: str) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "eps_actual": [2.0],
                "eps_estimate": [1.5],
                "surprise_abs": [0.5],
                "surprise_pct": [0.3333],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2026-01-15")], name="date"),
        )
        return frame

    summary = refresh_earnings_surprise(
        watchlist=["AAPL", "MSFT"],
        data_dir=tmp_path / "data",
        provider_fn=provider,
        total_budget_sec=30,
        per_ticker_sec=5,
    )

    assert summary["ok"] is True
    assert summary["non_empty"] == 2
    assert (tmp_path / "data" / "earnings_surprise" / "AAPL.parquet").exists()
    assert (tmp_path / "data" / "earnings_surprise" / "MSFT.parquet").exists()


def test_cli_json_dry_no_network_with_empty_symbols(tmp_path: Path) -> None:
    cfg = tmp_path / "strategy_config.json"
    _strategy_config(cfg)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(Path(__file__).resolve().parents[1] / "src"),
        str(Path(__file__).resolve().parents[2] / "renquant-common" / "src"),
        env.get("PYTHONPATH", ""),
    ])
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "renquant_base_data.earnings_surprise_refresh",
            "--strategy-config",
            str(cfg),
            "--data-dir",
            str(tmp_path / "data"),
            "--symbols",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert '"ok": true' in proc.stdout
    assert '"n_symbols": 0' in proc.stdout
