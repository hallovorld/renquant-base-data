"""resolve_alpha_path must pick the FRESHEST candidate, not the first-listed.

2026-06-12: an abandoned alpha158_816_dataset.parquet shadowed the
daily-rebuilt alpha158_qlib_dataset.parquet, clipping the
sec_fundamentals_daily date axis 121 days into the past.
"""
from __future__ import annotations

import os
import time

from renquant_base_data.sec_fundamentals import (
    DEFAULT_ALPHA_CANDIDATES,
    resolve_alpha_path,
)


def test_prefers_freshest_candidate(tmp_path):
    old = tmp_path / DEFAULT_ALPHA_CANDIDATES[0]
    new = tmp_path / DEFAULT_ALPHA_CANDIDATES[1]
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    past = time.time() - 86400 * 30
    os.utime(old, (past, past))
    assert resolve_alpha_path(tmp_path) == new.resolve()


def test_explicit_path_still_wins(tmp_path):
    p = tmp_path / "explicit.parquet"
    p.write_bytes(b"x")
    assert resolve_alpha_path(tmp_path, p).name == "explicit.parquet"


def test_fallback_when_none_exist(tmp_path):
    assert resolve_alpha_path(tmp_path).name == DEFAULT_ALPHA_CANDIDATES[0]
