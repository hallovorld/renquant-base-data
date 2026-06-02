from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data.sec_fundamentals import (
    EXTENDED_FEATURE_COLS,
    BASE_FEATURE_COLS,
    SecFundamentalsConfig,
    build_quarterly_panel,
    main,
    refresh_sec_fundamentals,
    sec_headers,
)


pytest.importorskip("pyarrow")


def test_sec_headers_use_env_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "RenQuant ops@example.com")

    assert sec_headers()["User-Agent"] == "RenQuant ops@example.com"


def test_sec_headers_fallback_has_no_personal_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    user_agent = sec_headers()["User-Agent"]
    assert user_agent == "renquant-base-data sec-edgar-contact@invalid.example"
    assert "renhao.overflow@gmail.com" not in user_agent


def _raw_fixture() -> pd.DataFrame:
    rows = []
    quarters = [
        ("2020-03-31", "2020-05-10", 10.0, 25.0, 100.0, 200.0, 80.0, 120.0, 10.0),
        ("2020-06-30", "2020-08-10", 12.0, 28.0, 110.0, 220.0, 82.0, 138.0, 10.0),
        ("2020-09-30", "2020-11-10", 14.0, 31.0, 120.0, 240.0, 84.0, 156.0, 10.0),
        ("2020-12-31", "2021-02-10", 16.0, 34.0, 130.0, 260.0, 86.0, 174.0, 10.0),
        ("2021-03-31", "2021-05-10", 18.0, 37.0, 150.0, 300.0, 90.0, 210.0, 10.0),
    ]
    for end, filed, ni, gp, revenue, assets, equity, liabilities, shares in quarters:
        values = {
            "NetIncomeLoss": ni,
            "GrossProfit": gp,
            "Revenues": revenue,
            "Assets": assets,
            "StockholdersEquity": equity,
            "Liabilities": liabilities,
            "CommonStockSharesOutstanding": shares,
        }
        for concept, value in values.items():
            rows.append(
                {
                    "ticker": "AAA",
                    "cik": 1,
                    "end": end,
                    "filed": filed,
                    "concept": concept,
                    "val": value,
                }
            )
    return pd.DataFrame(rows)


def _write_runtime_inputs(data_dir: Path) -> Path:
    dates = pd.date_range("2020-05-10", "2021-06-01", freq="D")
    alpha_path = data_dir / "alpha158_816_dataset.parquet"
    pd.DataFrame({"ticker": "AAA", "date": dates}).to_parquet(alpha_path, index=False)

    ohlcv_dir = data_dir / "ohlcv" / "AAA"
    ohlcv_dir.mkdir(parents=True)
    pd.DataFrame({"close": 10.0}, index=dates).to_parquet(ohlcv_dir / "1d.parquet")
    return alpha_path


def test_build_quarterly_panel_uses_latest_contributing_filing_date() -> None:
    raw = pd.DataFrame(
        [
            {"ticker": "AAA", "end": "2020-03-31", "filed": "2020-05-01", "concept": "NetIncomeLoss", "val": 10},
            {"ticker": "AAA", "end": "2020-03-31", "filed": "2020-05-05", "concept": "NetIncomeLoss", "val": 11},
            {"ticker": "AAA", "end": "2020-03-31", "filed": "2020-05-03", "concept": "Assets", "val": 100},
        ]
    )

    panel = build_quarterly_panel(raw)

    assert len(panel) == 1
    assert panel.loc[0, "NetIncomeLoss"] == 11
    assert panel.loc[0, "available_date"] == pd.Timestamp("2020-05-05")


def test_refresh_sec_fundamentals_with_injected_raw_builds_outputs(tmp_path: Path) -> None:
    alpha_path = _write_runtime_inputs(tmp_path)
    config = SecFundamentalsConfig(
        data_dir=tmp_path,
        mode="both",
        symbols=("AAA",),
        alpha_path=alpha_path,
        daily_output=tmp_path / "daily.parquet",
        extended_output=tmp_path / "extended.parquet",
        train_end="2021-01-01",
    )

    summary = refresh_sec_fundamentals(
        config,
        raw_daily=_raw_fixture(),
        raw_extended=_raw_fixture(),
        ticker_cik={"AAA": 1},
    )

    assert summary["ok"] is True
    daily = pd.read_parquet(tmp_path / "daily.parquet")
    extended = pd.read_parquet(tmp_path / "extended.parquet")
    assert set(BASE_FEATURE_COLS).issubset(daily.columns)
    assert set(EXTENDED_FEATURE_COLS).issubset(extended.columns)
    first_available = daily[daily["date"] == pd.Timestamp("2020-05-10")].iloc[0]
    assert first_available["earnings_yield"] == pytest.approx(10.0 / (10.0 * 10.0))
    assert extended[list(EXTENDED_FEATURE_COLS)].isna().sum().sum() == 0


def test_cli_dry_run_plans_without_alpha_or_network(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "--mode",
            "daily",
            "--symbols",
            "AAA",
            "--data-dir",
            str(tmp_path),
            "--start-year",
            "2020",
            "--end-year",
            "2020",
            "--dry-run",
            "--json",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert '"dry_run": true' in out
    assert '"daily_planned_requests": 24' in out
