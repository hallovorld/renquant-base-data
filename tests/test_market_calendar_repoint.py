"""Campaign B5 lockstep tests: ``_last_completed_nyse_session`` is now a
thin explicitly-lenient wrap of the canonical
``renquant_common.market_calendar.last_completed_session`` (audit #296 §4.1
row 3 / XC-2 — this was the original the orchestrator freshness guard
hand-mirrored and diverged from)."""
from __future__ import annotations

import builtins
import datetime as dt

import pandas as pd
import pytest

from renquant_base_data.loaders.data import _last_completed_nyse_session
from renquant_common.market_calendar import last_completed_session


def test_lockstep_with_canonical_on_golden_vectors() -> None:
    vectors = [
        ("2025-11-25 14:00", dt.date(2025, 11, 24)),  # intra-session -> prior
        ("2025-11-25 16:30", dt.date(2025, 11, 25)),  # after close -> today
        ("2025-11-28 14:00", dt.date(2025, 11, 28)),  # half-day close passed
        ("2025-11-28 12:00", dt.date(2025, 11, 26)),  # half-day open (Thu hol)
        ("2026-06-28 09:00", dt.date(2026, 6, 26)),   # Sunday -> Friday
        ("2026-07-03 12:00", dt.date(2026, 7, 2)),    # Jul-4-observed holiday
    ]
    for now, expected in vectors:
        ts = pd.Timestamp(now, tz="America/New_York")
        got = _last_completed_nyse_session(ts)
        assert got == expected
        assert got == last_completed_session(ts)


def test_lenient_wrap_swallows_calendar_unavailability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical is fail-closed; THIS call-site keeps base-data's lenient
    contract — calendar backend unavailable => None (the caller then applies
    its conservative 2-calendar-day staleness cap)."""
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name.startswith("renquant_common"):
            raise ImportError("simulated stale renquant_common install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert _last_completed_nyse_session(
        pd.Timestamp("2026-06-30 12:00", tz="America/New_York")
    ) is None
