"""The fallback availability stamp must not sit below the measured 10-K p95.

When SEC frames return no `filed` date — the production case, 90.37% of rows —
availability is stamped as fiscal-period-end + `FILING_LAG_FALLBACK_DAYS`. At 45
days that was look-ahead on **77.6% of 10-K filings**, median **+10 days**,
measured against 36,564 real filing dates. 10-Ks are 24.6% of filings, so ~19% of
filing events claimed a value was knowable before it was filed.

2026-07-30 review correction: raising the fallback to the measured 10-K p95
(60d) is a RISK REDUCTION, not a look-ahead-free guarantee — by definition of
"p95", ~5% of the measured 10-K filings still file later than the new stamp.
These tests pin the DIRECTION of the trade (a fallback below the measured 10-K
p95 reintroduces the 45d-era MAJORITY look-ahead) and a sanity ceiling on
staleness; they do NOT certify the fallback is point-in-time.
"""
from __future__ import annotations

from renquant_base_data.sec_fundamentals import (
    FILING_LAG_FALLBACK_DAYS,
    MEASURED_10K_P95_FILING_LAG_DAYS,
)


def test_fallback_is_not_below_the_measured_10k_p95():
    """The regression pin. 45 < 53 (10-K median) is look-ahead by construction."""
    assert FILING_LAG_FALLBACK_DAYS >= MEASURED_10K_P95_FILING_LAG_DAYS, (
        f"fallback {FILING_LAG_FALLBACK_DAYS}d is below the measured 10-K p95 "
        f"{MEASURED_10K_P95_FILING_LAG_DAYS}d, which reintroduces annual-report "
        f"look-ahead")


def test_the_old_value_would_fail_this_pin():
    """Guards against the pin being vacuous: it must actually reject 45."""
    assert 45 < MEASURED_10K_P95_FILING_LAG_DAYS


def test_fallback_stays_within_one_reporting_cycle_plus_a_quarter():
    """A conservative bound must not become unbounded: past ~1 quarter of extra
    lag the feature is stale enough that admitting it is its own defect."""
    assert FILING_LAG_FALLBACK_DAYS <= 120
