"""Tests for the committed PatchTST shadow-corpus recipe (S12 B1).

The served ``transformer_v4_wl200_clean.parquet`` was an ad-hoc snapshot with no
committed builder; the true recipe is the production fund panel subset to the
live watchlist with forward labels dropna'd, preserving the exact served
178-column schema. These tests assert, on a small full-schema fixture, that the
built corpus matches the served schema contract EXACTLY (column set + order +
dtypes), applies the watchlist subset and label-dropna semantics, orders rows
deterministically, and fails CLOSED on schema drift / bad watchlist / duplicate
rows. Fixtures live in tmp_path only — no production file is read or written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.transformer_corpus import (
    LABEL_COLS,
    SPLIT_COL,
    TRANSFORMER_CORPUS_COLUMNS,
    TransformerCorpusError,
    build_transformer_corpus,
    load_watchlist,
    main,
)


pytest.importorskip("pyarrow")

FEATURE_COLS = [
    c
    for c in TRANSFORMER_CORPUS_COLUMNS
    if c not in ("ticker", "date", SPLIT_COL) + tuple(LABEL_COLS)
]


def _fund_panel_frame(
    tickers=("AAA", "BBB", "CCC"),
    n_dates: int = 6,
    n_unlabeled_tail: int = 2,
) -> pd.DataFrame:
    """A production-fund-panel-shaped frame carrying the FULL 178-column served
    schema: real dtypes, per-ticker trailing rows with NaN forward labels (the
    bar rows past the label frontier), and a ``split_label`` annotation."""
    dates = pd.bdate_range("2026-01-05", periods=n_dates)
    rows = []
    for t_i, ticker in enumerate(tickers):
        for d_i, date in enumerate(dates):
            row = {"ticker": ticker, "date": date}
            for f_i, col in enumerate(FEATURE_COLS):
                row[col] = float(t_i + 1) + 0.01 * d_i + 0.0001 * f_i
            unlabeled = d_i >= n_dates - n_unlabeled_tail
            for label in LABEL_COLS:
                row[label] = np.nan if unlabeled else 0.001 * (t_i + d_i + 1)
            row[SPLIT_COL] = "train" if d_i < n_dates - 3 else "test"
            rows.append(row)
    frame = pd.DataFrame(rows, columns=list(TRANSFORMER_CORPUS_COLUMNS))
    frame["ticker"] = frame["ticker"].astype("string")
    frame[SPLIT_COL] = frame[SPLIT_COL].astype("string")
    return frame


def _write_panel(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    path = tmp_path / "alpha158_291_fundamental_dataset.parquet"
    (frame if frame is not None else _fund_panel_frame()).to_parquet(path, index=False)
    return path


# ── schema contract ──────────────────────────────────────────────────────────


def test_contract_is_the_served_178_column_schema() -> None:
    # 2 keys + 158 alpha158 + 3 labels + split_label + 14 fund-family = 178.
    assert len(TRANSFORMER_CORPUS_COLUMNS) == 178
    assert TRANSFORMER_CORPUS_COLUMNS[:2] == ("ticker", "date")
    assert TRANSFORMER_CORPUS_COLUMNS[2] == "KMID"
    assert TRANSFORMER_CORPUS_COLUMNS[-1] == "n_articles_log"
    assert "fwd_60d_excess" in TRANSFORMER_CORPUS_COLUMNS
    assert len(set(TRANSFORMER_CORPUS_COLUMNS)) == 178


def test_output_schema_matches_served_contract_exactly(tmp_path: Path) -> None:
    panel_path = _write_panel(tmp_path)
    out = tmp_path / "corpus.parquet"
    build_transformer_corpus(panel_path, ["AAA", "BBB"], out)

    built = pd.read_parquet(out)
    # Column SET and ORDER match the served contract exactly.
    assert list(built.columns) == list(TRANSFORMER_CORPUS_COLUMNS)
    # Dtype contract: string keys, datetime date, float64 everything else.
    assert built["ticker"].dtype == "string"
    assert built[SPLIT_COL].dtype == "string"
    assert str(built["date"].dtype) == "datetime64[ns]"
    for col in FEATURE_COLS + list(LABEL_COLS):
        assert built[col].dtype == np.float64, col

    import pyarrow.parquet as pq

    arrow_types = {n: str(t) for n, t in zip(pq.read_schema(out).names, pq.read_schema(out).types)}
    assert arrow_types["ticker"] == "string"
    assert arrow_types[SPLIT_COL] == "string"
    assert arrow_types["date"] == "timestamp[ns]"
    assert arrow_types["KMID"] == "double"
    assert arrow_types["fwd_60d_excess"] == "double"


# ── recipe semantics ─────────────────────────────────────────────────────────


def test_watchlist_subset_and_missing_names_reported(tmp_path: Path) -> None:
    panel_path = _write_panel(tmp_path)
    out = tmp_path / "corpus.parquet"
    report = build_transformer_corpus(panel_path, ["AAA", "BBB", "ZZZ"], out)

    built = pd.read_parquet(out)
    assert sorted(built["ticker"].unique()) == ["AAA", "BBB"]  # CCC excluded, ZZZ absent
    assert report["n_tickers"] == 2
    assert report["missing_watchlist_tickers"] == ["ZZZ"]


def test_label_dropna_clips_to_the_labeled_frontier(tmp_path: Path) -> None:
    frame = _fund_panel_frame(n_dates=6, n_unlabeled_tail=2)
    panel_path = _write_panel(tmp_path, frame)
    out = tmp_path / "corpus.parquet"
    report = build_transformer_corpus(panel_path, ["AAA", "BBB", "CCC"], out)

    built = pd.read_parquet(out)
    labeled_frontier = frame.dropna(subset=list(LABEL_COLS))["date"].max()
    assert built["date"].max() == labeled_frontier
    assert not built[list(LABEL_COLS)].isna().any().any()
    assert report["n_label_rows_dropped"] == 2 * 3  # 2 unlabeled tail dates x 3 tickers
    assert report["max_date"] == labeled_frontier.date().isoformat()


def test_deterministic_ticker_date_ordering(tmp_path: Path) -> None:
    frame = _fund_panel_frame().sample(frac=1.0, random_state=7)  # shuffled input
    panel_path = _write_panel(tmp_path, frame)
    out = tmp_path / "corpus.parquet"
    build_transformer_corpus(panel_path, ["BBB", "AAA"], out)

    built = pd.read_parquet(out)
    keys = built[["ticker", "date"]]
    expected = keys.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(keys.reset_index(drop=True), expected)

    # Two builds from differently-ordered inputs produce identical frames.
    out2 = tmp_path / "corpus2.parquet"
    build_transformer_corpus(_write_panel(tmp_path, _fund_panel_frame()), ["BBB", "AAA"], out2)
    pd.testing.assert_frame_equal(built, pd.read_parquet(out2))


def test_split_label_carried_through(tmp_path: Path) -> None:
    panel_path = _write_panel(tmp_path)
    out = tmp_path / "corpus.parquet"
    build_transformer_corpus(panel_path, ["AAA"], out)
    built = pd.read_parquet(out)
    assert set(built[SPLIT_COL].unique()) <= {"train", "test"}
    assert (built[SPLIT_COL] == "train").any()


# ── fail-closed contract ─────────────────────────────────────────────────────


def test_missing_contract_column_fails_closed(tmp_path: Path) -> None:
    frame = _fund_panel_frame().drop(columns=["ROC60"])
    panel_path = _write_panel(tmp_path, frame)
    with pytest.raises(TransformerCorpusError, match="missing.*contract column"):
        build_transformer_corpus(panel_path, ["AAA"], tmp_path / "corpus.parquet")


def test_extra_column_fails_closed_by_default_but_droppable(tmp_path: Path) -> None:
    frame = _fund_panel_frame()
    frame["surprise_extra"] = 1.0
    panel_path = _write_panel(tmp_path, frame)
    out = tmp_path / "corpus.parquet"
    with pytest.raises(TransformerCorpusError, match="unexpected column"):
        build_transformer_corpus(panel_path, ["AAA"], out)

    build_transformer_corpus(panel_path, ["AAA"], out, require_exact_schema=False)
    assert list(pd.read_parquet(out).columns) == list(TRANSFORMER_CORPUS_COLUMNS)


def test_empty_watchlist_and_no_overlap_fail_closed(tmp_path: Path) -> None:
    panel_path = _write_panel(tmp_path)
    with pytest.raises(TransformerCorpusError, match="watchlist is empty"):
        build_transformer_corpus(panel_path, [], tmp_path / "corpus.parquet")
    with pytest.raises(TransformerCorpusError, match="no watchlist ticker present"):
        build_transformer_corpus(panel_path, ["ZZZ"], tmp_path / "corpus.parquet")


def test_all_labels_nan_fails_closed(tmp_path: Path) -> None:
    frame = _fund_panel_frame()
    frame[list(LABEL_COLS)] = np.nan
    panel_path = _write_panel(tmp_path, frame)
    with pytest.raises(TransformerCorpusError, match="empty after the label dropna"):
        build_transformer_corpus(panel_path, ["AAA"], tmp_path / "corpus.parquet")


def test_duplicate_ticker_date_rows_fail_closed(tmp_path: Path) -> None:
    frame = _fund_panel_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    panel_path = _write_panel(tmp_path, frame)
    with pytest.raises(TransformerCorpusError, match="duplicate"):
        build_transformer_corpus(panel_path, ["AAA"], tmp_path / "corpus.parquet")


def test_unreadable_panel_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TransformerCorpusError, match="unreadable"):
        build_transformer_corpus(tmp_path / "missing.parquet", ["AAA"], tmp_path / "c.parquet")


# ── watchlist loader ─────────────────────────────────────────────────────────


def test_load_watchlist_from_strategy_config(tmp_path: Path) -> None:
    cfg = tmp_path / "strategy_config.json"
    cfg.write_text(json.dumps({"watchlist": ["aapl", "MSFT", "AAPL"]}))
    assert load_watchlist(cfg) == ["AAPL", "MSFT"]


def test_load_watchlist_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TransformerCorpusError, match="unreadable"):
        load_watchlist(tmp_path / "missing.json")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"watchlist": []}))
    with pytest.raises(TransformerCorpusError, match="no non-empty 'watchlist'"):
        load_watchlist(empty)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    with pytest.raises(TransformerCorpusError, match="unreadable|corrupt"):
        load_watchlist(corrupt)


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    panel_path = _write_panel(tmp_path)
    cfg = tmp_path / "strategy_config.json"
    cfg.write_text(json.dumps({"watchlist": ["AAA", "BBB"]}))
    out = tmp_path / "corpus.parquet"

    rc = main(
        [
            "--fund-panel", str(panel_path),
            "--strategy-config", str(cfg),
            "--output", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["n_tickers"] == 2
    assert list(pd.read_parquet(out).columns) == list(TRANSFORMER_CORPUS_COLUMNS)
