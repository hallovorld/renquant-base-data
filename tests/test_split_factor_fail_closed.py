"""An empty split calendar is not an authoritative "never split" answer.

`build_split_factor.reconstructed_factor` emitted `cum_factor = 1.0` whenever the
union calendar had no row for a ticker. That is correct when FMP answered and has
no split history, and WRONG when the request failed — in which case nothing is
known and 1.0 publishes an unadjusted market-cap basis as though it had been
verified. `harvest_splits_830.py` already recorded the difference and its own
manifest note says "errors are NOT authoritative and must be treated as
unknown"; the builder simply never read it.

These pin the fail-closed contract, including the exact error-plus-missing-raw
case from the review.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "build_split_factor.py"


def _load(monkeypatch, work_dir: Path):
    """Import the tool with its work dir pointed at a fixture directory."""
    monkeypatch.setenv("RQ_SPLIT_FIX_DIR", str(work_dir))
    spec = importlib.util.spec_from_file_location("_bsf_fixture", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bsf_fixture"] = mod
    spec.loader.exec_module(mod)
    return mod


def _manifest(work_dir: Path, *, no_data: list[str], errors: list[str]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "fmp_splits_830.manifest.json").write_text(json.dumps({
        "no_data_tickers": no_data,
        "error_tickers": errors,
        "error_samples": [{"ticker": t, "err": "HTTPError: 500"} for t in errors],
    }))


def _close(n: int = 30) -> pd.Series:
    idx = pd.bdate_range("2026-01-02", periods=n)
    return pd.Series(100.0, index=idx)


EMPTY_CAL = pd.DataFrame(columns=["ticker", "date", "ratio", "src"])


def test_errored_ticker_with_empty_calendar_FAILS_not_1_0(tmp_path, monkeypatch):
    """The exact review case: request errored AND no raw-price fallback."""
    _manifest(tmp_path, no_data=["CLEAN"], errors=["BROKEN"])
    m = _load(monkeypatch, tmp_path)
    no_ev, unknown = m.load_split_retrieval_status()
    assert "BROKEN" in unknown and "BROKEN" not in no_ev

    f, info = m.reconstructed_factor("BROKEN", _close(), EMPTY_CAL, no_ev)
    assert f is None, "an errored ticker must not receive factor 1.0"
    assert info["why"] == "no-authoritative-no-event-answer"


def test_authoritative_no_event_ticker_DOES_get_1_0(tmp_path, monkeypatch):
    """The guard must not become blanket-deny: a real no-split answer works."""
    _manifest(tmp_path, no_data=["CLEAN"], errors=["BROKEN"])
    m = _load(monkeypatch, tmp_path)
    no_ev, _ = m.load_split_retrieval_status()

    f, info = m.reconstructed_factor("CLEAN", _close(), EMPTY_CAL, no_ev)
    assert f is not None
    assert (f == 1.0).all()
    assert info["why"] == "reconstructed-no-events"


def test_a_ticker_absent_from_the_manifest_entirely_also_FAILS(tmp_path, monkeypatch):
    """Never requested is also not an authoritative no-event answer."""
    _manifest(tmp_path, no_data=["CLEAN"], errors=["BROKEN"])
    m = _load(monkeypatch, tmp_path)
    no_ev, _ = m.load_split_retrieval_status()
    f, info = m.reconstructed_factor("NEVER_ASKED", _close(), EMPTY_CAL, no_ev)
    assert f is None
    assert info["why"] == "no-authoritative-no-event-answer"


def test_a_MISSING_manifest_fails_every_empty_calendar_ticker(tmp_path, monkeypatch):
    """No manifest means no authority to claim "never split" for anyone."""
    (tmp_path / "raw_price_cache").mkdir(parents=True, exist_ok=True)
    m = _load(monkeypatch, tmp_path)
    no_ev, unknown = m.load_split_retrieval_status()
    assert no_ev == frozenset() and unknown == frozenset()
    f, info = m.reconstructed_factor("ANY", _close(), EMPTY_CAL, no_ev)
    assert f is None
    assert info["why"] == "no-authoritative-no-event-answer"


def test_omitting_the_status_argument_is_fail_closed_too(tmp_path, monkeypatch):
    """A caller that forgets to pass the set must not silently get 1.0 —
    the default is None, which denies, not permits."""
    _manifest(tmp_path, no_data=["CLEAN"], errors=[])
    m = _load(monkeypatch, tmp_path)
    f, info = m.reconstructed_factor("CLEAN", _close(), EMPTY_CAL)
    assert f is None, "default must deny, so a forgetful caller cannot fail open"
    assert info["why"] == "no-authoritative-no-event-answer"


def test_a_real_split_event_still_reconstructs(tmp_path, monkeypatch):
    """Non-empty calendar path is unchanged by the guard."""
    _manifest(tmp_path, no_data=[], errors=[])
    m = _load(monkeypatch, tmp_path)
    close = _close(30)
    ex = close.index[10]
    cal = pd.DataFrame([{"ticker": "SPLIT", "date": ex, "ratio": 2.0, "src": "fmp"}])
    f, info = m.reconstructed_factor("SPLIT", close, cal, frozenset())
    assert f is not None
    assert f.loc[f.index < ex].eq(2.0).all(), "pre-split dates carry the ratio"
    assert f.loc[f.index >= ex].eq(1.0).all(), "post-split dates are unadjusted"
    assert info["n_segments"] == 2


def test_legacy_manifest_without_error_tickers_still_derives_what_it_can(tmp_path, monkeypatch):
    """Older manifests carried only a truncated error_samples list."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fmp_splits_830.manifest.json").write_text(json.dumps({
        "no_data_tickers": ["CLEAN"],
        "error_samples": [{"ticker": "OLDBROKEN", "err": "HTTPError: 500"}],
    }))
    m = _load(monkeypatch, tmp_path)
    no_ev, unknown = m.load_split_retrieval_status()
    assert "OLDBROKEN" in unknown
    assert m.reconstructed_factor("OLDBROKEN", _close(), EMPTY_CAL, no_ev)[0] is None
