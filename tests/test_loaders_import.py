"""Smoke-import tests for lifted data-layer modules (copy-not-move slice)."""
from __future__ import annotations

import importlib

import pytest

LIFTED_MODULES = [
    "renquant_base_data.alpha158_qlib_panel",
    "renquant_base_data.alpha158_fund_panel",
    "renquant_base_data.loaders.data_cache",
    "renquant_base_data.loaders.data_coverage",
    "renquant_base_data.loaders.fundamentals",
    "renquant_base_data.loaders.macro_per_ticker",
    "renquant_base_data.loaders.row_coverage",
    "renquant_base_data.loaders.data",
    "renquant_base_data.loaders.indicators",
    "renquant_base_data.loaders.macro",
    "renquant_base_data.loaders.fred_macro",
    "renquant_base_data.loaders.earnings_surprise",
    "renquant_base_data.loaders.insider_trades",
]


@pytest.mark.parametrize("module_name", LIFTED_MODULES)
def test_lifted_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
