"""Data-layer modules lifted from the umbrella (copy-not-move).

Per RFC §"Backfill Plan" functional-lift, these data-access / coverage /
fundamentals modules are copied verbatim from
`backtesting/renquant_104/kernel/` into renquant-base-data and verified
import-clean. The umbrella keeps its working copy until cutover.

* ``data_cache``      — OHLCV cache access
* ``data_coverage``   — dataset coverage / gap reporting
* ``fundamentals``    — fundamental feature loaders
* ``macro_per_ticker``— per-ticker macro overlays
* ``row_coverage``    — panel row-coverage checks
"""
from __future__ import annotations

__all__: list[str] = []
