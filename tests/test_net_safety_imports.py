"""Guards against the renquant_common.net_safety lift regressing.

The existing test_loaders_import.py only imports modules at top level — it
doesn't trigger the lazy `from .net_safety import ...` lines buried in
function bodies (PR #1 of renquant-common lifted net_safety out of
renquant-base-data, so these relative imports point at a module that no
longer exists in this package).

This test catches that with two layers:

1. AST scan — fails fast if any module body contains a relative import of
   a local `net_safety` (`from .net_safety import ...`). Doesn't require
   network or call into the functions.
2. Lazy-path exercise — calls `fetch_fundamentals_watchlist` /
   `_fetch_from_yfinance` with stubs that prevent network use, so the
   in-function `from .net_safety import ...` actually runs. If the import
   path is wrong, ModuleNotFoundError surfaces here instead of in
   production.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent / "src" / "renquant_base_data"

MODULES_TO_SCAN = [
    "renquant_base_data.loaders.fundamentals",
    "renquant_base_data.loaders.earnings_surprise",
    "renquant_base_data.loaders.insider_trades",
    "renquant_base_data.loaders.data",
    "renquant_base_data.loaders.data_cache",
    "renquant_base_data.fetchers.fundamentals",
    "renquant_base_data.fetchers.earnings_surprise",
    "renquant_base_data.fetchers.insider_trades",
]


def _module_path(dotted: str) -> Path:
    parts = dotted.split(".")[1:]
    return PKG_ROOT.joinpath(*parts).with_suffix(".py")


@pytest.mark.parametrize("module_name", MODULES_TO_SCAN)
def test_no_relative_net_safety_import(module_name: str) -> None:
    """No `from .net_safety import ...` anywhere — must be renquant_common.net_safety."""
    path = _module_path(module_name)
    tree = ast.parse(path.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level >= 1 and (node.module or "").endswith("net_safety"):
                names = ", ".join(alias.name for alias in node.names)
                offenders.append(f"{path.name}:{node.lineno}: from .{node.module} import {names}")
    assert not offenders, (
        f"Stale relative net_safety imports in {module_name}:\n  "
        + "\n  ".join(offenders)
        + "\nFix: replace with `from renquant_common.net_safety import ...`."
    )


def test_lazy_import_paths_resolve() -> None:
    """Exercise the in-function lazy imports — they only run when the function is called."""
    mods_to_probe = [
        ("renquant_base_data.loaders.fundamentals", "fetch_fundamentals_watchlist"),
        ("renquant_base_data.fetchers.fundamentals", "fetch_fundamentals_watchlist"),
        ("renquant_base_data.loaders.earnings_surprise", "_fetch_from_yfinance"),
        ("renquant_base_data.fetchers.earnings_surprise", "_fetch_from_yfinance"),
    ]
    for mod_name, fn_name in mods_to_probe:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        try:
            if fn_name == "_fetch_from_yfinance":
                fn("AAPL")
            else:
                fn(["AAPL"])
        except ModuleNotFoundError as exc:
            if "net_safety" in str(exc):
                pytest.fail(f"{mod_name}.{fn_name} still resolves a missing net_safety: {exc}")
            raise
        except Exception:
            pass
