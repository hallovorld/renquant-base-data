"""Drift guard: the committed 179-column schema export matches the builder.

``tests/rawlabel_sidecar_columns_179.json`` is the canonical, reviewable
export of ``RAWLABEL_SIDECAR_COLUMNS`` — the single-writer canonical sidecar
contract (base-data#48 §2.2: the full 178-column fund-panel schema, sentiment
INCLUDED, + ``fwd_60d_excess_raw``). If the builder's contract ever changes,
this test fails first; any consumer-repo fixture embedding a copy of this list
must be re-exported in lockstep.
"""
import json
from pathlib import Path

from renquant_base_data.rawlabel_sidecar import (
    RAW_LABEL_COL,
    RAWLABEL_SIDECAR_COLUMNS,
    SENTIMENT_COLS,
)

EXPORT = Path(__file__).parent / "rawlabel_sidecar_columns_179.json"


def test_export_matches_builder_contract_exactly():
    exported = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert exported == list(RAWLABEL_SIDECAR_COLUMNS)


def test_contract_shape():
    cols = list(RAWLABEL_SIDECAR_COLUMNS)
    assert len(cols) == 179
    # base-data#48 §2.2: sentiment is now CARRIED (un-frozen), not dropped.
    assert set(SENTIMENT_COLS) <= set(cols)
    assert cols[-1] == RAW_LABEL_COL
    assert cols[:2] == ["ticker", "date"]
    # The three sentiment columns sit at the panel tail, just before the
    # appended raw label (the served σ-head recipe's order).
    assert cols[-4:] == [
        "sentiment_pos_share",
        "mean_sentiment",
        "n_articles_log",
        RAW_LABEL_COL,
    ]
    # The z-scored labels the WF-gate sanity path scores against stay carried.
    for label in ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "split_label"):
        assert label in cols
