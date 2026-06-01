from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data.alpaca_news_refresh import iter_chunks, merge_news, refresh_alpaca_news
from renquant_base_data.options_iv_refresh import (
    merge_iv_snapshot,
    nearest_atm_iv,
    parse_occ,
    refresh_options_iv,
)


pytest.importorskip("pyarrow")


def test_iter_chunks_covers_window() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 6, tzinfo=timezone.utc)
    chunks = list(iter_chunks(start, end, days=2))

    assert chunks == [
        (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 3, tzinfo=timezone.utc)),
        (datetime(2026, 1, 3, tzinfo=timezone.utc), datetime(2026, 1, 5, tzinfo=timezone.utc)),
        (datetime(2026, 1, 5, tzinfo=timezone.utc), datetime(2026, 1, 6, tzinfo=timezone.utc)),
    ]


def test_merge_news_deduplicates_symbol_timestamp_headline() -> None:
    prior = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "created_at": ["2026-01-01T12:00:00Z"],
            "updated_at": ["2026-01-01T12:01:00Z"],
            "headline": ["same"],
            "summary": ["old"],
            "author": ["a"],
            "url": ["u"],
            "all_symbols": ["AAA"],
        }
    )
    new = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "created_at": ["2026-01-01T12:00:00Z", "2026-01-02T12:00:00Z"],
            "updated_at": ["2026-01-01T12:02:00Z", "2026-01-02T12:01:00Z"],
            "headline": ["same", "new"],
            "summary": ["fresh", "fresh"],
            "author": ["b", "b"],
            "url": ["u", "u2"],
            "all_symbols": ["AAA", "AAA"],
        }
    )

    merged = merge_news(prior, new)

    assert len(merged) == 2
    assert list(merged["headline"]) == ["same", "new"]


def test_refresh_alpaca_news_writes_per_symbol_cache(tmp_path: Path) -> None:
    def fake_fetch(_client, _bucket, symbol, _start, _end, **_kwargs):
        return pd.DataFrame(
            {
                "symbol": [symbol],
                "created_at": ["2026-01-01T12:00:00Z"],
                "updated_at": ["2026-01-01T12:01:00Z"],
                "headline": ["headline"],
                "summary": [""],
                "author": ["a"],
                "url": ["u"],
                "all_symbols": [symbol],
            }
        )

    summary = refresh_alpaca_news(
        symbols=["aaa"],
        data_dir=tmp_path,
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        fetch_symbol_fn=fake_fetch,
    )

    assert summary["ok"] is True
    out = pd.read_parquet(tmp_path / "news_alpaca" / "AAA.parquet")
    assert len(out) == 1
    assert out.loc[0, "symbol"] == "AAA"


def test_parse_occ_accepts_alpaca_and_standard_symbols() -> None:
    assert parse_occ("AAPL260529C00170000") == {
        "underlying": "AAPL",
        "expiry": date(2026, 5, 29),
        "option_type": "C",
        "strike": 170.0,
    }
    assert parse_occ("AAPL_260529P00165000")["option_type"] == "P"
    assert parse_occ("bad") is None


def test_nearest_atm_iv_prefers_nearest_dte_then_strike() -> None:
    today = date(2026, 1, 1)
    contracts = [
        {"expiry": date(2026, 1, 31), "option_type": "C", "strike": 90.0, "iv": 0.30},
        {"expiry": date(2026, 1, 31), "option_type": "C", "strike": 101.0, "iv": 0.20},
        {"expiry": date(2026, 2, 5), "option_type": "C", "strike": 100.0, "iv": 0.10},
    ]

    assert nearest_atm_iv(contracts, 30, "C", 100.0, today=today) == (0.20, 30, 101.0)


def test_refresh_options_iv_writes_and_dedupes(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fake_spot(_symbol: str) -> float:
        return 100.0

    def fake_features(_client, symbol, spot, _bucket):
        calls["n"] += 1
        return {
            "symbol": symbol,
            "as_of": "2026-01-01",
            "spot": spot,
            "iv_30d_call_atm": 0.2 + calls["n"] / 100,
            "iv_30d_put_atm": 0.25,
            "iv_60d_call_atm": 0.30,
            "iv_60d_put_atm": 0.35,
            "iv_skew_30d": 0.05,
            "iv_term_struct": 0.10,
            "dte_30": 30,
            "dte_60": 60,
            "n_valid_iv_contracts": 4,
        }

    refresh_options_iv(symbols=["aaa"], data_dir=tmp_path, spot_fn=fake_spot, feature_fn=fake_features)
    refresh_options_iv(symbols=["aaa"], data_dir=tmp_path, spot_fn=fake_spot, feature_fn=fake_features)

    out = pd.read_parquet(tmp_path / "options_iv_alpaca" / "AAA.parquet")
    assert len(out) == 1
    assert out.loc[0, "iv_30d_call_atm"] == pytest.approx(0.22)


def test_merge_iv_snapshot_replaces_same_as_of() -> None:
    prior = pd.DataFrame([{"symbol": "AAA", "as_of": "2026-01-01", "iv_30d_call_atm": 0.1}])
    merged = merge_iv_snapshot(prior, {"symbol": "AAA", "as_of": "2026-01-01", "iv_30d_call_atm": 0.2})

    assert len(merged) == 1
    assert merged.loc[0, "iv_30d_call_atm"] == pytest.approx(0.2)
