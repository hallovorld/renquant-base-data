"""Network-free unit tests for edgar_filing_dates (the EDGAR PIT backfill)."""
import pandas as pd
import pytest

from renquant_base_data import edgar_filing_dates
from renquant_base_data.edgar_filing_dates import (
    build_filing_dates, cik_variants, fetch_filing_dates_for_cik,
    stamp_available_v2, restamp_fundamentals)


def test_cik_variants_share_classes():
    assert cik_variants("BRK.B") == ["BRK.B", "BRK-B"]
    assert cik_variants("BF-B") == ["BF-B", "BF.B"]
    assert cik_variants("AAPL") == ["AAPL"]


def test_stamp_available_v2_is_next_business_day():
    df = pd.DataFrame({
        "ticker": ["X", "X"],
        "period_end": ["2024-03-31", "2024-06-30"],
        "form": ["10-Q", "10-Q"],
        # Friday evening acceptance -> available Monday; Tue -> Wed
        "accepted_at": ["2024-05-03T21:10:00.000Z", "2024-08-06T12:00:00.000Z"],
    })
    out = stamp_available_v2(df)
    assert str(out.loc[0, "available_at_v2"].date()) == "2024-05-06"
    assert str(out.loc[1, "available_at_v2"].date()) == "2024-08-07"


def test_restamp_keeps_v1_when_unmatched_and_flags_source():
    fundamentals = pd.DataFrame({
        "ticker": ["X", "Y"],
        "fiscal_period_end": ["2024-03-31", "2024-03-31"],
        "available_at": ["2024-05-15", "2024-05-15"],
        "roe": [0.1, 0.2],
    })
    stamps = pd.DataFrame({
        "ticker": ["X"],
        "period_end": [pd.Timestamp("2024-03-31")],
        "form": ["10-Q"],
        "available_at_v2": [pd.Timestamp("2024-05-06")],
    })
    out = restamp_fundamentals(fundamentals, stamps)
    x = out[out.ticker == "X"].iloc[0]
    y = out[out.ticker == "Y"].iloc[0]
    assert str(x["available_at_v2"].date()) == "2024-05-06"
    assert x["available_source_v2"] == "edgar_accepted"
    assert str(y["available_at_v2"].date()) == "2024-05-15"
    assert y["available_source_v2"] == "carried_v1"


def _root_with_one_paginated_file():
    empty = {"form": [], "acceptanceDateTime": [], "reportDate": [], "filingDate": []}
    return {"filings": {"recent": empty, "files": [{"name": "CIK0000000001-001.json"}]}}


def test_paginated_file_failure_fails_closed(monkeypatch):
    """A failed paginated history-file fetch must propagate, not be silently
    dropped (codex P1, PR #52 round 2) — a partial history must not be
    returned as if it were complete coverage for the CIK."""
    calls = {"n": 0}

    def fake_get(url):
        calls["n"] += 1
        if calls["n"] == 1:
            return _root_with_one_paginated_file()
        raise RuntimeError("SEC rate limit")

    monkeypatch.setattr(edgar_filing_dates, "_get", fake_get)
    with pytest.raises(RuntimeError):
        fetch_filing_dates_for_cik(1)


def test_build_filing_dates_marks_ticker_missing_on_paginated_failure(monkeypatch):
    """The fail-closed propagation must reach build_filing_dates's per-ticker
    handling: the ticker lands in `missing`, not partially in the output df."""
    calls = {"n": 0}

    def fake_get(url):
        calls["n"] += 1
        if calls["n"] == 1:
            return _root_with_one_paginated_file()
        raise RuntimeError("SEC rate limit")

    monkeypatch.setattr(edgar_filing_dates, "load_cik_map", lambda: {"ZZZZ": 1})
    monkeypatch.setattr(edgar_filing_dates, "_get", fake_get)

    df, missing = build_filing_dates(["ZZZZ"])
    assert df.empty
    assert missing == ["ZZZZ"]
