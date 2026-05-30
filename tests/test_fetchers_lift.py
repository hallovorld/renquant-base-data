"""Smoke test for the external-data fetchers lift (Track C2.8).

Six top-level kernel/*.py → renquant_base_data.fetchers (per inventory D):
  * fundamentals.py      — SEC fundamentals fetch
  * macro.py             — macro features
  * macro_per_ticker.py  — per-ticker macro overlay
  * fred_macro.py        — FRED macro feed
  * insider_trades.py    — Form-4 insider trades
  * earnings_surprise.py — earnings surprise feeds

All six are clean of kernel.* deps. Phase 1 invariant: byte-equivalent +
clean import.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[1] / "src" / "renquant_base_data" / "fetchers"
_UMBRELLA = Path(__file__).resolve().parents[2] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel"

_LIFTED = (
    "fundamentals.py",
    "macro.py",
    "macro_per_ticker.py",
    "fred_macro.py",
    "insider_trades.py",
    "earnings_surprise.py",
)


def test_byte_equivalent_to_umbrella() -> None:
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    for name in _LIFTED:
        bt = _BT_PKG / name
        um = _UMBRELLA / name
        assert bt.exists(), f"missing in subrepo: {name}"
        assert um.exists(), f"missing in umbrella: {name}"
        assert hashlib.md5(bt.read_bytes()).hexdigest() == hashlib.md5(um.read_bytes()).hexdigest(), \
            f"byte-mismatch: {name}"


def test_expected_files_present() -> None:
    present = {f.name for f in _BT_PKG.glob("*.py")}
    missing = set(_LIFTED) - present
    assert not missing, f"missing: {missing}"


@pytest.mark.parametrize("name", [
    "renquant_base_data.fetchers.fundamentals",
    "renquant_base_data.fetchers.macro",
    "renquant_base_data.fetchers.macro_per_ticker",
    "renquant_base_data.fetchers.fred_macro",
    "renquant_base_data.fetchers.insider_trades",
    "renquant_base_data.fetchers.earnings_surprise",
])
def test_imports_cleanly_or_optional_dep_skip(name: str) -> None:
    """Either imports cleanly OR raises ImportError on an OPTIONAL data-vendor dep."""
    try:
        __import__(name)
    except ImportError as exc:
        # acceptable: optional data-vendor SDKs (fredapi, edgar, etc.)
        keep = ("fredapi", "edgar", "sec_edgar", "polygon", "yfinance")
        if not any(k in str(exc).lower() for k in keep):
            raise
