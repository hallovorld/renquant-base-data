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
import os
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[1] / "src" / "renquant_base_data" / "fetchers"

# Marker file unique to a GENUINE RenQuant umbrella checkout (RENQUANT_REPOS.md
# is auto-generated FROM it in every subrepo). Distinguishes the real umbrella
# from an incidentally same-named directory.
_UMBRELLA_MARKER = "subrepos.lock.json"
_UMBRELLA_SUFFIX = Path("backtesting") / "renquant_104" / "kernel"
_UMBRELLA_ENV = "RENQUANT_UMBRELLA_PATH"


def _resolve_umbrella_kernel_dir() -> Path | None:
    """Find the ``RenQuant`` umbrella's ``kernel`` dir, or ``None``.

    2026-07-11 Codex CHANGES_REQUESTED (PR #43): the prior fixed-depth
    ``parents[2] / "RenQuant"`` heuristic can find a WRONG directory that
    happens to be named "RenQuant" in an ad hoc scratch/worktree layout
    (observed: a stale, non-git directory copy with no ``subrepos.lock.json``
    and a divergent kernel/ — that produced a real byte-MISMATCH failure,
    not a clean skip, purely from siting, not from an actual drift). This
    resolver only trusts a candidate that also carries the umbrella's own
    marker file, so an unrelated/stale same-named directory is treated the
    same as "umbrella absent" (clean skip) instead of a false failure.

    Resolution order:
      1. ``RENQUANT_UMBRELLA_PATH`` env var, if set — explicit, for CI/worktree
         configs that don't use the canonical sibling-checkout layout.
      2. The canonical sibling-of-repo-root layout
         (``<repo-root>/../RenQuant``), the common local dev layout.
      3. A couple of extra ancestor depths, for nested worktree layouts,
         still gated on the marker file so a false positive can't slip in.
    """
    env_override = os.environ.get(_UMBRELLA_ENV)
    if env_override:
        root = Path(env_override).expanduser()
        return root / _UMBRELLA_SUFFIX if (root / _UMBRELLA_MARKER).is_file() else None
    repo_root = Path(__file__).resolve().parents[1]
    # ``.parents`` slicing needs a list() on Python 3.9 (no slice support).
    ancestors = [repo_root.parent] + list(repo_root.parents)[1:3]
    for ancestor in ancestors:
        candidate = ancestor / "RenQuant"
        if (candidate / _UMBRELLA_MARKER).is_file():
            return candidate / _UMBRELLA_SUFFIX
    return None


_UMBRELLA = _resolve_umbrella_kernel_dir()

_LIFTED = (
    "fundamentals.py",
    "macro.py",
    "macro_per_ticker.py",
    "fred_macro.py",
    "insider_trades.py",
    "earnings_surprise.py",
)


def test_byte_equivalent_to_umbrella() -> None:
    if _UMBRELLA is None or not _UMBRELLA.exists():
        pytest.skip(
            "no genuine RenQuant umbrella checkout found (checked "
            f"${_UMBRELLA_ENV} and sibling-of-repo-root layouts; a directory "
            "merely NAMED 'RenQuant' without subrepos.lock.json does not "
            "count — see _resolve_umbrella_kernel_dir)"
        )
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
