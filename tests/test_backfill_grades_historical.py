"""Tests for grades-historical PIT backfill."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from renquant_base_data.backfill_grades_historical import (
    backfill,
    group_by_month,
    load_universe,
    main,
    transform_grades_row,
    write_snapshot,
)


SAMPLE_GRADES = [
    {
        "symbol": "AAPL",
        "date": "2026-06-01",
        "analystRatingsStrongBuy": 7,
        "analystRatingsBuy": 23,
        "analystRatingsHold": 16,
        "analystRatingsSell": 2,
        "analystRatingsStrongSell": 2,
    },
    {
        "symbol": "AAPL",
        "date": "2026-05-01",
        "analystRatingsStrongBuy": 7,
        "analystRatingsBuy": 25,
        "analystRatingsHold": 16,
        "analystRatingsSell": 1,
        "analystRatingsStrongSell": 2,
    },
    {
        "symbol": "AAPL",
        "date": "2026-04-01",
        "analystRatingsStrongBuy": 8,
        "analystRatingsBuy": 24,
        "analystRatingsHold": 15,
        "analystRatingsSell": 1,
        "analystRatingsStrongSell": 1,
    },
]

SAMPLE_GOOG = [
    {
        "symbol": "GOOG",
        "date": "2026-06-01",
        "analystRatingsStrongBuy": 10,
        "analystRatingsBuy": 30,
        "analystRatingsHold": 5,
        "analystRatingsSell": 0,
        "analystRatingsStrongSell": 0,
    },
    {
        "symbol": "GOOG",
        "date": "2026-05-01",
        "analystRatingsStrongBuy": 9,
        "analystRatingsBuy": 28,
        "analystRatingsHold": 6,
        "analystRatingsSell": 1,
        "analystRatingsStrongSell": 0,
    },
]


def test_transform_grades_row():
    row = SAMPLE_GRADES[0]
    result = transform_grades_row(row)
    assert result["symbol"] == "AAPL"
    assert result["strongBuy"] == 7
    assert result["buy"] == 23
    assert result["hold"] == 16
    assert result["sell"] == 2
    assert result["strongSell"] == 2
    assert "analystRatingsStrongBuy" not in result


def test_transform_grades_row_missing_fields():
    row = {"symbol": "TEST", "date": "2026-01-01"}
    result = transform_grades_row(row)
    assert result["symbol"] == "TEST"
    assert result["strongBuy"] == 0
    assert result["buy"] == 0


def test_group_by_month():
    all_data = {"AAPL": SAMPLE_GRADES, "GOOG": SAMPLE_GOOG}
    by_month = group_by_month(all_data)
    assert len(by_month) == 3
    assert "2026-06-01" in by_month
    assert "2026-05-01" in by_month
    assert "2026-04-01" in by_month
    june = by_month["2026-06-01"]
    assert len(june) == 2
    assert set(june["symbol"]) == {"AAPL", "GOOG"}
    april = by_month["2026-04-01"]
    assert len(april) == 1
    assert april["symbol"].iloc[0] == "AAPL"


def test_write_snapshot(tmp_path):
    df = pd.DataFrame([
        {"symbol": "AAPL", "strongBuy": 7, "buy": 23, "hold": 16, "sell": 2, "strongSell": 2},
        {"symbol": "GOOG", "strongBuy": 10, "buy": 30, "hold": 5, "sell": 0, "strongSell": 0},
    ])
    result = write_snapshot(tmp_path, "2026-06-01", df)
    assert result is not None
    assert result["pit_source"] == "grades_historical_backfill"
    assert result["n_symbols"] == 2

    written = pd.read_parquet(tmp_path / "2026-06-01" / "grades_consensus.parquet")
    assert len(written) == 2
    assert "strongBuy" in written.columns
    assert "buy" in written.columns

    manifest = json.loads((tmp_path / "2026-06-01" / "grades_consensus.manifest.json").read_text())
    assert manifest["snapshot_as_of"] == "2026-06-01"
    assert manifest["sha256"] is not None


def test_write_snapshot_skip_existing(tmp_path):
    df = pd.DataFrame([{"symbol": "AAPL", "strongBuy": 1, "buy": 2, "hold": 3, "sell": 0, "strongSell": 0}])
    write_snapshot(tmp_path, "2026-06-01", df)
    result = write_snapshot(tmp_path, "2026-06-01", df, overwrite=False)
    assert result is None


def test_write_snapshot_overwrite(tmp_path):
    df = pd.DataFrame([{"symbol": "AAPL", "strongBuy": 1, "buy": 2, "hold": 3, "sell": 0, "strongSell": 0}])
    write_snapshot(tmp_path, "2026-06-01", df)
    df2 = pd.DataFrame([
        {"symbol": "AAPL", "strongBuy": 10, "buy": 20, "hold": 3, "sell": 0, "strongSell": 0},
        {"symbol": "GOOG", "strongBuy": 5, "buy": 15, "hold": 2, "sell": 0, "strongSell": 0},
    ])
    result = write_snapshot(tmp_path, "2026-06-01", df2, overwrite=True)
    assert result is not None
    assert result["n_symbols"] == 2


def _mock_fetch(ticker, api_key, **kwargs):
    data = {"AAPL": SAMPLE_GRADES, "GOOG": SAMPLE_GOOG}
    return data.get(ticker, [])


def test_backfill_dry_run(tmp_path):
    with patch(
        "renquant_base_data.backfill_grades_historical.fetch_grades_historical",
        side_effect=_mock_fetch,
    ):
        result = backfill(
            ["AAPL", "GOOG"],
            "fake_key",
            tmp_path,
            execute=False,
        )
    assert result["status"] == "dry_run"
    assert result["tickers_ok"] == 2
    assert result["months_total"] == 3
    assert result["months_written"] == 3
    assert not (tmp_path / "2026-06-01").exists()


def test_backfill_execute(tmp_path):
    with patch(
        "renquant_base_data.backfill_grades_historical.fetch_grades_historical",
        side_effect=_mock_fetch,
    ):
        result = backfill(
            ["AAPL", "GOOG"],
            "fake_key",
            tmp_path,
            execute=True,
        )
    assert result["status"] == "ok"
    assert result["months_written"] == 3
    assert (tmp_path / "2026-06-01" / "grades_consensus.parquet").exists()
    assert (tmp_path / "2026-05-01" / "grades_consensus.parquet").exists()
    assert (tmp_path / "2026-04-01" / "grades_consensus.parquet").exists()


def test_backfill_skip_existing(tmp_path):
    (tmp_path / "2026-06-01").mkdir()
    pd.DataFrame([{"symbol": "X", "strongBuy": 1, "buy": 1, "hold": 1, "sell": 0, "strongSell": 0}]).to_parquet(
        tmp_path / "2026-06-01" / "grades_consensus.parquet"
    )
    with patch(
        "renquant_base_data.backfill_grades_historical.fetch_grades_historical",
        side_effect=_mock_fetch,
    ):
        result = backfill(
            ["AAPL", "GOOG"],
            "fake_key",
            tmp_path,
            execute=True,
        )
    assert result["months_written"] == 2
    assert result["months_skipped"] == 1


def test_backfill_below_coverage(tmp_path):
    def fail_fetch(ticker, api_key, **kw):
        return []

    with patch(
        "renquant_base_data.backfill_grades_historical.fetch_grades_historical",
        side_effect=fail_fetch,
    ):
        result = backfill(
            ["AAPL", "GOOG"],
            "fake_key",
            tmp_path,
            execute=True,
            min_coverage=0.5,
        )
    assert result["status"] == "error"
    assert result["reason"] == "below_coverage_floor"


def test_load_universe_json(tmp_path):
    cfg = {"watchlist": ["GOOG", "AAPL", "MSFT"]}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    tickers = load_universe(str(cfg_path))
    assert tickers == ["AAPL", "GOOG", "MSFT"]


def test_load_universe_txt(tmp_path):
    txt = "AAPL\nGOOG\n# comment\nMSFT\n"
    txt_path = tmp_path / "tickers.txt"
    txt_path.write_text(txt)
    tickers = load_universe(str(txt_path))
    assert tickers == ["AAPL", "GOOG", "MSFT"]


def test_cli_dry_run(tmp_path, capsys):
    with patch(
        "renquant_base_data.backfill_grades_historical.fetch_grades_historical",
        side_effect=_mock_fetch,
    ), patch(
        "renquant_base_data.backfill_grades_historical.load_api_key",
        return_value="fake",
    ), patch(
        "renquant_base_data.backfill_grades_historical.load_universe",
        return_value=["AAPL", "GOOG"],
    ):
        rc = main(["--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "3 total" in out


def test_cli_json_output(tmp_path, capsys):
    with patch(
        "renquant_base_data.backfill_grades_historical.fetch_grades_historical",
        side_effect=_mock_fetch,
    ), patch(
        "renquant_base_data.backfill_grades_historical.load_api_key",
        return_value="fake",
    ), patch(
        "renquant_base_data.backfill_grades_historical.load_universe",
        return_value=["AAPL", "GOOG"],
    ):
        rc = main(["--out", str(tmp_path), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "dry_run"
    assert data["months_total"] == 3


def test_cli_no_api_key(capsys):
    with patch(
        "renquant_base_data.backfill_grades_historical.load_api_key",
        return_value=None,
    ):
        rc = main(["--out", "/tmp/x"])
    assert rc == 1


def test_pit_feature_builder_reads_backfilled_snapshots(tmp_path):
    """Integration: backfilled grades-only snapshots produce grade_score features."""
    from datetime import date as _date

    from renquant_base_data.pit_revision_features import build_features

    months = ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
    for month in months:
        snap_dir = tmp_path / month
        snap_dir.mkdir()
        df = pd.DataFrame([
            {"symbol": "AAPL", "strongBuy": 7, "buy": 23, "hold": 16, "sell": 2, "strongSell": 2},
            {"symbol": "GOOG", "strongBuy": 10, "buy": 30, "hold": 5, "sell": 0, "strongSell": 0},
        ])
        df.to_parquet(snap_dir / "grades_consensus.parquet")

    days = [_date.fromisoformat(m) for m in months]
    result = build_features(tmp_path, days=days)
    assert not result.empty
    assert "grade_migration_1m" in result.columns
    assert set(result["symbol"]) == {"AAPL", "GOOG"}
    later = result[result["as_of"] >= _date(2026, 2, 1)]
    assert not later["grade_migration_1m"].isna().all()
