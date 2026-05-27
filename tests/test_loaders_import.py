"""Smoke-import tests for lifted data-layer modules (copy-not-move slice)."""
from __future__ import annotations

import importlib

import pytest

LIFTED_MODULES = [
    "renquant_base_data.loaders.data_cache",
    "renquant_base_data.loaders.data_coverage",
    "renquant_base_data.loaders.fundamentals",
    "renquant_base_data.loaders.macro_per_ticker",
    "renquant_base_data.loaders.row_coverage",
]


@pytest.mark.parametrize("module_name", LIFTED_MODULES)
def test_lifted_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
