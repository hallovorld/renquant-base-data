"""Focused fixture tests for tools/build_split_factor.py's measured-factor route and
the cover-date anchoring in tools/build_pit_panel_v2.py (renquant-base-data#58 review
round 2). Complements tests/test_split_factor_fail_closed.py, which covers the
reconstructed-route fail-closed contract; this file covers the MEASURED route and
factor_at, which that file does not touch.

No live FMP calls, no writes outside pytest's tmp_path. Covers:
  * measured_factor: a real, persisting split step is detected and the tail is
    correctly anchored to 1.0
  * measured_factor: a short-lived, uncorroborated edge blip is merged away (does
    NOT poison the tail anchor) -- the exact bug the module's docstring documents
    for HUM/CF/ELV/... at their file's last date
  * measured_factor: the same short-lived step IS kept when a real ex-date
    corroborates it
  * measured_factor: NaN fail-closed when the two price series don't agree cleanly
    (residual exceeds MAX_RESID) -- never publish an untrustworthy factor
  * build_pit_panel_v2.factor_at: cover-date anchoring, both in-axis and the
    pre-axis calendar extension
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import build_split_factor as bsf  # noqa: E402
import build_pit_panel_v2 as bpp  # noqa: E402


def _close(n=60, start=100.0, step=0.1, origin="2020-01-01"):
    idx = pd.bdate_range(origin, periods=n)
    return pd.Series(start + np.arange(n) * step, index=idx)


def _write_raw_cache(cache_dir: Path, ticker: str, close: pd.Series, ratio: np.ndarray):
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": close.index, "raw_close": close.values * ratio})
    df.to_parquet(cache_dir / f"{ticker}.parquet", index=False)


# --------------------------------------------------------------------- measured_factor

def test_measured_factor_detects_persisting_split_step(tmp_path, monkeypatch):
    close = _close(60)
    ratio = np.where(np.arange(60) < 30, 2.0, 1.0)  # 2:1 split at index 30, F ends at 1.0
    monkeypatch.setattr(bsf, "CACHE", tmp_path)
    _write_raw_cache(tmp_path, "SPLT", close, ratio)

    full, info = bsf.measured_factor("SPLT", close)

    assert full is not None
    assert info["why"] == "measured"
    assert full.iloc[-1] == pytest.approx(1.0)
    assert full.iloc[0] == pytest.approx(2.0, rel=1e-3)
    assert len(info["steps"]) == 1


def test_measured_factor_drops_uncorroborated_edge_blip(tmp_path, monkeypatch):
    """A step that lives on only the file's LAST bar and is not corroborated by any
    real ex-date must be merged away, not anchor the tail (HUM/CF/ELV.. bug class)."""
    close = _close(60)
    ratio = np.ones(60)
    ratio[-1] = 1.05  # single-day vendor disagreement on the final bar
    monkeypatch.setattr(bsf, "CACHE", tmp_path)
    _write_raw_cache(tmp_path, "EDGE", close, ratio)

    full, info = bsf.measured_factor("EDGE", close)

    assert full is not None
    # tail must anchor to 1.0 -- if the 1-row blip were kept as its own segment, the
    # whole history would be rescaled by 1/1.05 instead
    assert full.iloc[-1] == pytest.approx(1.0)
    assert full.iloc[0] == pytest.approx(1.0)
    assert info["steps"] == []


def test_measured_factor_keeps_edge_blip_when_corroborated(tmp_path, monkeypatch):
    """The same short-lived step IS kept when a real calendar ex-date corroborates it."""
    close = _close(60)
    ratio = np.ones(60)
    ratio[-1] = 1.05
    monkeypatch.setattr(bsf, "CACHE", tmp_path)
    _write_raw_cache(tmp_path, "EDGE2", close, ratio)
    cal_dates = np.array([close.index[-1].to_datetime64()])

    full, info = bsf.measured_factor("EDGE2", close, cal_dates)

    assert full is not None
    assert len(info["steps"]) == 1
    assert full.iloc[-1] == pytest.approx(1.0)
    assert full.iloc[0] == pytest.approx(1.0 / 1.05, rel=1e-3)


def test_measured_factor_fails_closed_on_unreliable_series(tmp_path, monkeypatch):
    """The two price series disagreeing (not a clean step) must publish NO factor."""
    close = _close(60)
    ratio = np.where(np.arange(60) % 2 == 0, 1.08, 0.92)  # alternating vendor noise
    monkeypatch.setattr(bsf, "CACHE", tmp_path)
    _write_raw_cache(tmp_path, "NOISY", close, ratio)

    full, info = bsf.measured_factor("NOISY", close)

    assert full is None
    assert info["why"].startswith("unreliable-resid")


def test_measured_factor_no_raw_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bsf, "CACHE", tmp_path)
    full, info = bsf.measured_factor("MISSING", _close(60))
    assert full is None
    assert info["why"] == "no-raw-cache"


# --------------------------------------------------------- build_pit_panel_v2.factor_at

def test_factor_at_in_axis_lookup():
    dates = pd.bdate_range("2020-01-01", periods=10)
    factors = np.where(np.arange(10) < 5, 2.0, 1.0)
    per = {"T": (dates.values.astype("datetime64[ns]"), factors)}
    calper = {}

    out = bpp.factor_at("T", [dates[3], dates[7]], per, calper)

    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(1.0)


def test_factor_at_pre_axis_extends_with_calendar():
    """A share fact's cover date before the priced axis starts must still pick up
    any calendar event between it and the axis start -- an early share count left
    unextended would be on the wrong basis."""
    dates = pd.bdate_range("2020-03-01", periods=10)
    factors = np.full(10, 1.0)
    per = {"T": (dates.values.astype("datetime64[ns]"), factors)}
    pre_date = pd.Timestamp("2020-01-15")  # before the axis start
    cal = pd.DataFrame({"date": [pd.Timestamp("2020-02-01")], "ratio": [3.0]})
    calper = {"T": cal}

    out = bpp.factor_at("T", [pre_date], per, calper)

    assert out[0] == pytest.approx(1.0 * 3.0)


def test_factor_at_unknown_ticker_is_nan():
    out = bpp.factor_at("NOPE", [pd.Timestamp("2020-01-01")], {}, {})
    assert np.isnan(out[0])
