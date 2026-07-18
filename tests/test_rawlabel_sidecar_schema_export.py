"""Drift guard: the committed 176-column schema export matches the builder.

``tests/rawlabel_sidecar_columns_176.json`` is the canonical, reviewable
export of ``RAWLABEL_SIDECAR_COLUMNS`` (the AC-1 sidecar contract). The
consumer-repo fixture tests (backtesting / model / orchestrator companion
PRs of the AC-1 evidence appendix) embed copies of this list to build their
176-column fixtures without a cross-repo import. If the builder's contract
ever changes, this test fails first and the embedded copies must be
re-exported — the appendix names every embed site.
"""
import json
from pathlib import Path

from renquant_base_data.rawlabel_sidecar import (
    RAW_LABEL_COL,
    RAWLABEL_SIDECAR_COLUMNS,
    SENTIMENT_COLS,
)

EXPORT = Path(__file__).parent / "rawlabel_sidecar_columns_176.json"


def test_export_matches_builder_contract_exactly():
    exported = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert exported == list(RAWLABEL_SIDECAR_COLUMNS)


def test_contract_shape():
    cols = list(RAWLABEL_SIDECAR_COLUMNS)
    assert len(cols) == 176
    assert not set(SENTIMENT_COLS) & set(cols)
    assert cols[-1] == RAW_LABEL_COL
    assert cols[:2] == ["ticker", "date"]
    # The z-scored labels the WF-gate sanity path scores against stay carried.
    for label in ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "split_label"):
        assert label in cols
