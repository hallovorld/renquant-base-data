"""Network-free unit tests for edgar_filing_dates (the EDGAR PIT backfill)."""
import pandas as pd
from renquant_base_data.edgar_filing_dates import (
    cik_variants, stamp_available_v2, restamp_fundamentals)


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
