"""Tests for the committed RAWLABEL calibration-sidecar recipe (S12, B1 pattern).

Under the single-writer amendment (base-data#48) this builder is the SOLE
writer of the served ``alpha158_291_fundamental_dataset_rawlabel.parquet``.
The canonical recipe is the production fund panel (full universe, NO label
dropna, sentiment columns CARRIED — §2.2) with a raw (un-z-scored)
``fwd_60d_excess_raw`` recomputed point-in-time from OHLCV closes vs the
benchmark, and — for THIS served artifact — ZERO bar-frontier axis extension
by default (§2.3); the opt-in extension path stays exercised for a separate
artifact. These tests assert, on a small full-schema fixture, that the built
sidecar matches the 179-column contract EXACTLY (sentiment included), computes
the raw label correctly and point-in-time, keeps unlabeled rows, never
forward-fills missing sentiment (§2.4), extends the axis honestly only when
opted in (NaN features — never fabricated values), emits exactly the canonical
179-column contract with zero bar-frontier extension rows (§2.5), orders rows
deterministically, and fails CLOSED on schema drift / missing benchmark /
corrupt OHLCV / future-dated bars / duplicate rows. Fixtures live in tmp_path
only — no production file is read or written.

The cross-repo umbrella refresh guard is NOT reimplemented here: base-data tests
only its own schema + row-domain contract; exact guard execution against this
built output is AC-C umbrella integration/runbook work against the pinned
base-data revision.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.rawlabel_sidecar import (
    RAW_LABEL_COL,
    RAWLABEL_SIDECAR_COLUMNS,
    SENTIMENT_COLS,
    RawLabelSidecarError,
    build_rawlabel_sidecar,
    main,
)
from renquant_base_data.transformer_corpus import (
    LABEL_COLS,
    SPLIT_COL,
    TRANSFORMER_CORPUS_COLUMNS,
)


pytest.importorskip("pyarrow")

# Small horizon keeps fixtures tiny; the recipe is horizon-parametric.
HORIZON = 3
TODAY = dt.date(2026, 7, 3)

PANEL_FEATURE_COLS = [
    c
    for c in TRANSFORMER_CORPUS_COLUMNS
    if c not in ("ticker", "date", SPLIT_COL) + tuple(LABEL_COLS)
]
SIDECAR_FLOAT_COLS = [
    c for c in RAWLABEL_SIDECAR_COLUMNS if c not in ("ticker", "date", SPLIT_COL)
]

# 10 bar dates; the panel covers the first 5 (label-clipped frontier), bars
# extend 5 more to the bar frontier.
BAR_DATES = pd.bdate_range("2026-06-01", periods=10)
PANEL_DATES = BAR_DATES[:5]


def _fund_panel_frame(tickers=("AAA", "BBB")) -> pd.DataFrame:
    """A production-fund-panel-shaped frame carrying the FULL 178-column served
    schema (real dtypes, sentiment columns present, split_label annotated)."""
    rows = []
    for t_i, ticker in enumerate(tickers):
        for d_i, date in enumerate(PANEL_DATES):
            row = {"ticker": ticker, "date": date}
            for f_i, col in enumerate(PANEL_FEATURE_COLS):
                row[col] = float(t_i + 1) + 0.01 * d_i + 0.0001 * f_i
            for label in LABEL_COLS:
                # A NaN z-scored label row must be KEPT (raw semantics).
                row[label] = np.nan if d_i == 0 else 0.001 * (t_i + d_i + 1)
            row[SPLIT_COL] = "train" if d_i < 3 else "test"
            rows.append(row)
    frame = pd.DataFrame(rows, columns=list(TRANSFORMER_CORPUS_COLUMNS))
    frame["ticker"] = frame["ticker"].astype("string")
    frame[SPLIT_COL] = frame[SPLIT_COL].astype("string")
    return frame


def _write_panel(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    path = tmp_path / "alpha158_291_fundamental_dataset.parquet"
    (frame if frame is not None else _fund_panel_frame()).to_parquet(path, index=False)
    return path


def _closes(ticker: str, dates=BAR_DATES) -> pd.Series:
    base = {"AAA": 100.0, "BBB": 50.0, "SPY": 400.0}.get(ticker, 10.0)
    return pd.Series(
        [base * (1.0 + 0.01 * i) for i in range(len(dates))], index=dates, name="close"
    )


def _write_ohlcv(tmp_path: Path, ticker: str, close: pd.Series | None = None) -> Path:
    close = close if close is not None else _closes(ticker)
    path = tmp_path / "ohlcv" / ticker / "1d.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"close": close}).to_parquet(path)
    return path


def _ohlcv_dir(tmp_path: Path, tickers=("AAA", "BBB", "SPY")) -> Path:
    for t in tickers:
        _write_ohlcv(tmp_path, t)
    return tmp_path / "ohlcv"


def _build(tmp_path: Path, **kw) -> tuple:
    panel_path = kw.pop("panel_path", None) or _write_panel(tmp_path)
    ohlcv = kw.pop("ohlcv", None) or _ohlcv_dir(tmp_path)
    out = kw.pop("out", None) or tmp_path / "rawlabel.parquet"
    kw.setdefault("horizon_trading_days", HORIZON)
    kw.setdefault("today", TODAY)
    report = build_rawlabel_sidecar(panel_path, ohlcv, out, **kw)
    return pd.read_parquet(out), report


def _expected_excess(ticker: str, date: pd.Timestamp) -> float:
    close, spy = _closes(ticker), _closes("SPY")
    i = list(BAR_DATES).index(date)
    j = i + HORIZON
    if j >= len(BAR_DATES):
        return np.nan
    return (close.iloc[j] / close.iloc[i] - 1.0) - (spy.iloc[j] / spy.iloc[i] - 1.0)


# ── schema contract ──────────────────────────────────────────────────────────


def test_contract_is_the_canonical_179_column_schema() -> None:
    # base-data#48 §2.2: full 178-column fund-panel contract (sentiment
    # INCLUDED) + fwd_60d_excess_raw = 179.
    assert len(RAWLABEL_SIDECAR_COLUMNS) == 179
    assert RAWLABEL_SIDECAR_COLUMNS[:2] == ("ticker", "date")
    assert RAWLABEL_SIDECAR_COLUMNS[-1] == RAW_LABEL_COL
    # Sentiment is un-frozen: all three columns are carried, at the panel tail
    # just before the appended raw label (the served σ-head recipe's order).
    assert set(SENTIMENT_COLS) <= set(RAWLABEL_SIDECAR_COLUMNS)
    assert RAWLABEL_SIDECAR_COLUMNS[-4:-1] == SENTIMENT_COLS
    assert set(LABEL_COLS) < set(RAWLABEL_SIDECAR_COLUMNS)  # z-labels carried
    assert len(set(RAWLABEL_SIDECAR_COLUMNS)) == 179


def test_output_schema_matches_served_contract_exactly(tmp_path: Path) -> None:
    built, _ = _build(tmp_path)
    # Column SET and ORDER match the served contract exactly.
    assert list(built.columns) == list(RAWLABEL_SIDECAR_COLUMNS)
    assert built["ticker"].dtype == "string"
    assert built[SPLIT_COL].dtype == "string"
    assert str(built["date"].dtype) == "datetime64[ns]"
    for col in SIDECAR_FLOAT_COLS:
        assert built[col].dtype == np.float64, col

    import pyarrow.parquet as pq

    out = tmp_path / "rawlabel.parquet"
    arrow_types = {n: str(t) for n, t in zip(pq.read_schema(out).names, pq.read_schema(out).types)}
    assert arrow_types["ticker"] == "string"
    assert arrow_types[SPLIT_COL] == "string"
    assert arrow_types["date"] == "timestamp[ns]"
    assert arrow_types["KMID"] == "double"
    assert arrow_types[RAW_LABEL_COL] == "double"


# ── recipe semantics ─────────────────────────────────────────────────────────


def test_raw_label_is_pit_forward_excess_vs_benchmark(tmp_path: Path) -> None:
    # Opt in to the extension so the full BAR_DATES axis (incl trailing
    # incomplete-window rows) exists to exercise both the labeled and PIT-NaN
    # branches; the label math is identical whether or not the axis is extended.
    built, _ = _build(tmp_path, extend_to_bar_frontier=True)
    for ticker in ("AAA", "BBB"):
        rows = built[built["ticker"] == ticker].set_index("date")
        for date in BAR_DATES:
            expected = _expected_excess(ticker, date)
            got = rows.loc[date, RAW_LABEL_COL]
            if np.isnan(expected):
                # PIT: the forward window is incomplete — unknowable, stays NaN.
                assert np.isnan(got), (ticker, date)
            else:
                assert got == pytest.approx(expected), (ticker, date)


def test_no_label_dropna_and_axis_extends_to_bar_frontier(tmp_path: Path) -> None:
    # Opt-in extension path (base-data#48 §2.3: NOT the served-file default).
    built, report = _build(tmp_path, extend_to_bar_frontier=True)
    # Raw semantics: the NaN-z-label panel row (d_i == 0) is KEPT.
    first = built[(built["ticker"] == "AAA") & (built["date"] == PANEL_DATES[0])]
    assert len(first) == 1
    assert first[list(LABEL_COLS)].isna().all().all()
    assert not np.isnan(first[RAW_LABEL_COL].iloc[0])  # raw label still computed
    # The axis reaches the BAR frontier, not the panel's label-clipped frontier.
    assert built["date"].max() == BAR_DATES[-1]
    assert report["max_date"] == BAR_DATES[-1].date().isoformat()
    assert report["n_extension_rows"] == 2 * (len(BAR_DATES) - len(PANEL_DATES))
    assert report["n_rows"] == report["n_panel_rows"] + report["n_extension_rows"]
    # Labeled frontier trails the bar frontier by the horizon.
    assert report["max_labeled_date"] == BAR_DATES[-1 - HORIZON].date().isoformat()


def test_extension_rows_are_honest_nan_not_fabricated(tmp_path: Path) -> None:
    built, _ = _build(tmp_path, extend_to_bar_frontier=True)
    ext = built[built["date"] > PANEL_DATES[-1]]
    assert len(ext) > 0
    # Features, z-labels, and split_label are NaN/NA on extension rows — the
    # builder never fabricates values it did not compute.
    feature_cols = [c for c in SIDECAR_FLOAT_COLS if c != RAW_LABEL_COL]
    assert ext[feature_cols].isna().all().all()
    assert ext[SPLIT_COL].isna().all()
    # Extension rows inside the completed-window range still get a REAL raw
    # label (knowable today); the trailing HORIZON rows stay NaN.
    labeled_ext = ext[ext["date"] <= BAR_DATES[-1 - HORIZON]]
    assert not labeled_ext.empty
    assert labeled_ext[RAW_LABEL_COL].notna().all()
    tail_ext = ext[ext["date"] > BAR_DATES[-1 - HORIZON]]
    assert tail_ext[RAW_LABEL_COL].isna().all()


def test_no_extend_flag_keeps_the_panel_axis(tmp_path: Path) -> None:
    built, report = _build(tmp_path, extend_to_bar_frontier=False)
    assert built["date"].max() == PANEL_DATES[-1]
    assert report["n_extension_rows"] == 0


def test_default_carries_zero_extension_rows_ac_b_prime(tmp_path: Path) -> None:
    # AC-B' (base-data#48 §2.3): the CANONICAL served-file recipe — the plain
    # default, no kwarg — carries ZERO bar-frontier extension rows. This is the
    # last recipe divergence from the σ-head writer, now closed.
    built, report = _build(tmp_path)  # default extend_to_bar_frontier is OFF
    assert report["n_extension_rows"] == 0
    assert report["n_rows"] == report["n_panel_rows"]
    # Every emitted row is a real panel (ticker, date) row — none is a key-only
    # extension row (which would carry an all-NaN feature vector).
    assert built["date"].max() == PANEL_DATES[-1]
    feature_cols = [c for c in SIDECAR_FLOAT_COLS if c != RAW_LABEL_COL]
    assert built[feature_cols].notna().any(axis=1).all()


def test_missing_ohlcv_ticker_gets_nan_label_and_no_extension(tmp_path: Path) -> None:
    ohlcv = _ohlcv_dir(tmp_path, tickers=("AAA", "SPY"))  # no BBB cache
    # Opt in to extension: a ticker with no cache stays at the panel frontier
    # even when extension is requested (no fabricated axis).
    built, report = _build(tmp_path, ohlcv=ohlcv, extend_to_bar_frontier=True)
    bbb = built[built["ticker"] == "BBB"]
    assert bbb[RAW_LABEL_COL].isna().all()
    assert bbb["date"].max() == PANEL_DATES[-1]  # not extended
    assert report["tickers_without_ohlcv"] == ["BBB"]
    # AAA is unaffected.
    aaa = built[built["ticker"] == "AAA"]
    assert aaa[RAW_LABEL_COL].notna().any()
    assert aaa["date"].max() == BAR_DATES[-1]


def test_deterministic_ordering_and_rebuild_equality(tmp_path: Path) -> None:
    shuffled = _fund_panel_frame().sample(frac=1.0, random_state=7)
    built, _ = _build(tmp_path, panel_path=_write_panel(tmp_path, shuffled))
    keys = built[["ticker", "date"]]
    expected = keys.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(keys.reset_index(drop=True), expected)

    # A rebuild from the in-order panel produces the identical frame.
    out2 = tmp_path / "rawlabel2.parquet"
    build_rawlabel_sidecar(
        _write_panel(tmp_path, _fund_panel_frame()),
        tmp_path / "ohlcv",
        out2,
        horizon_trading_days=HORIZON,
        today=TODAY,
    )
    pd.testing.assert_frame_equal(built, pd.read_parquet(out2))


# ── fail-closed contract ─────────────────────────────────────────────────────


def test_missing_contract_column_fails_closed(tmp_path: Path) -> None:
    frame = _fund_panel_frame().drop(columns=["ROC60"])
    with pytest.raises(RawLabelSidecarError, match="missing.*contract column"):
        _build(tmp_path, panel_path=_write_panel(tmp_path, frame))


def test_sentiment_columns_are_carried_from_the_panel(tmp_path: Path) -> None:
    # base-data#48 §2.2: sentiment is un-frozen — the built sidecar carries the
    # three columns with the panel's real values (no drop, no zeroing).
    frame = _fund_panel_frame()
    frame.loc[frame.index[0], "mean_sentiment"] = 0.73
    built, _ = _build(tmp_path, panel_path=_write_panel(tmp_path, frame))
    for col in SENTIMENT_COLS:
        assert col in built.columns
    first_key = (frame.loc[0, "ticker"], frame.loc[0, "date"])
    got = built[(built["ticker"] == first_key[0]) & (built["date"] == first_key[1])]
    assert got["mean_sentiment"].iloc[0] == pytest.approx(0.73)


def test_missing_sentiment_columns_fail_closed(tmp_path: Path) -> None:
    # Sentiment is now a REQUIRED contract column (base-data#48 §2.2); a panel
    # dropping it is schema drift, not a tolerated legacy shape — fail closed.
    frame = _fund_panel_frame().drop(columns=list(SENTIMENT_COLS))
    with pytest.raises(RawLabelSidecarError, match="missing.*contract column"):
        _build(tmp_path, panel_path=_write_panel(tmp_path, frame))


def test_missing_sentiment_value_stays_nan_never_ffilled(tmp_path: Path) -> None:
    # base-data#48 §2.4: sentiment is event-driven; a MISSING value must stay
    # NaN — a forward-fill would fabricate a stale prior reading as fresh
    # signal. AAA carries a real reading on date[1] and a missing one on the
    # later date[2] (an "unlabeled-tail-like" row a naive ffill would clobber).
    frame = _fund_panel_frame()
    aaa = frame["ticker"] == "AAA"
    d1 = frame.loc[aaa & (frame["date"] == PANEL_DATES[1])].index[0]
    d2 = frame.loc[aaa & (frame["date"] == PANEL_DATES[2])].index[0]
    frame.loc[d1, "mean_sentiment"] = 0.42
    frame.loc[d2, "mean_sentiment"] = np.nan
    built, _ = _build(tmp_path, panel_path=_write_panel(tmp_path, frame))
    rows = built[built["ticker"] == "AAA"].set_index("date")
    assert rows.loc[PANEL_DATES[1], "mean_sentiment"] == pytest.approx(0.42)
    # Preserved as NaN — NOT forward-filled from date[1].
    assert np.isnan(rows.loc[PANEL_DATES[2], "mean_sentiment"])


def test_extra_column_fails_closed_by_default_but_droppable(tmp_path: Path) -> None:
    frame = _fund_panel_frame()
    frame["surprise_extra"] = 1.0
    panel_path = _write_panel(tmp_path, frame)
    with pytest.raises(RawLabelSidecarError, match="unexpected column"):
        _build(tmp_path, panel_path=panel_path)
    built, _ = _build(tmp_path, panel_path=panel_path, require_exact_schema=False)
    assert list(built.columns) == list(RAWLABEL_SIDECAR_COLUMNS)


def test_panel_already_carrying_raw_label_fails_closed(tmp_path: Path) -> None:
    frame = _fund_panel_frame()
    frame[RAW_LABEL_COL] = 0.0
    with pytest.raises(RawLabelSidecarError, match="already carries"):
        _build(tmp_path, panel_path=_write_panel(tmp_path, frame))


def test_missing_benchmark_fails_closed(tmp_path: Path) -> None:
    ohlcv = _ohlcv_dir(tmp_path, tickers=("AAA", "BBB"))  # no SPY
    with pytest.raises(RawLabelSidecarError, match="benchmark OHLCV"):
        _build(tmp_path, ohlcv=ohlcv)


def test_corrupt_ohlcv_cache_fails_closed(tmp_path: Path) -> None:
    ohlcv = _ohlcv_dir(tmp_path)
    # Present-but-broken cache (no close column) is corruption, not absence.
    pd.DataFrame({"open": [1.0]}, index=BAR_DATES[:1]).to_parquet(
        tmp_path / "ohlcv" / "AAA" / "1d.parquet"
    )
    with pytest.raises(RawLabelSidecarError, match="no 'close' column"):
        _build(tmp_path, ohlcv=ohlcv)


def test_duplicate_ohlcv_bar_dates_fail_closed(tmp_path: Path) -> None:
    _ohlcv_dir(tmp_path)
    dup = pd.Series([1.0, 2.0], index=[BAR_DATES[0], BAR_DATES[0]], name="close")
    pd.DataFrame({"close": dup}).to_parquet(tmp_path / "ohlcv" / "AAA" / "1d.parquet")
    with pytest.raises(RawLabelSidecarError, match="duplicate bar dates"):
        _build(tmp_path, ohlcv=tmp_path / "ohlcv")


def test_future_dated_bars_or_panel_fail_closed(tmp_path: Path) -> None:
    _ohlcv_dir(tmp_path)
    # "today" == the panel frontier: the panel is fine but the benchmark bars
    # postdate it — the bar-side future-dated guard fires.
    with pytest.raises(RawLabelSidecarError, match="future-dated"):
        _build(tmp_path, today=PANEL_DATES[-1].date())
    frame = _fund_panel_frame()
    frame.loc[frame.index[-1], "date"] = pd.Timestamp(TODAY) + pd.Timedelta(days=30)
    with pytest.raises(RawLabelSidecarError, match="future-dated"):
        _build(tmp_path, panel_path=_write_panel(tmp_path, frame))


def test_duplicate_ticker_date_rows_fail_closed(tmp_path: Path) -> None:
    frame = _fund_panel_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(RawLabelSidecarError, match="duplicate"):
        _build(tmp_path, panel_path=_write_panel(tmp_path, frame))


def test_no_usable_ohlcv_at_all_fails_closed(tmp_path: Path) -> None:
    # Benchmark present, but NO panel ticker has a cache: an all-NaN raw label
    # is a vacuous sidecar — refuse it.
    ohlcv = _ohlcv_dir(tmp_path, tickers=("SPY",))
    with pytest.raises(RawLabelSidecarError, match="no row received a raw label"):
        _build(tmp_path, ohlcv=ohlcv)


def test_unreadable_or_empty_panel_fails_closed(tmp_path: Path) -> None:
    _ohlcv_dir(tmp_path)
    with pytest.raises(RawLabelSidecarError, match="unreadable"):
        _build(tmp_path, panel_path=tmp_path / "missing.parquet")
    empty = _fund_panel_frame().iloc[0:0]
    with pytest.raises(RawLabelSidecarError, match="empty"):
        _build(tmp_path, panel_path=_write_panel(tmp_path, empty))


def test_bad_horizon_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RawLabelSidecarError, match="horizon"):
        _build(tmp_path, horizon_trading_days=0)


# ── canonical-contract parity (base-data#48 §2.5: base-data's OWN contract) ────
#
# base-data owns and tests only its 179-column schema + row-domain contract.
# Exact execution of the RenQuant umbrella refresh guard (its dropped-columns /
# date-advance / row-ratio / ticker-coverage decision surface) against this
# built output — and the proof that the pre-amendment 176-col recipe reproduces
# the guard's Saturday rejection — is AC-C umbrella integration/runbook work
# against the PINNED base-data revision. It is NOT ported across the repo
# boundary here (a partial port would only test one branch of a foreign guard
# and silently diverge as the umbrella evolves).


def test_builder_output_is_exactly_the_canonical_179_contract(tmp_path: Path) -> None:
    # base-data#48 §2.5: the served-file builder's OWN contract — the default
    # (canonical) build emits EXACTLY the 179-column schema, in order, sentiment
    # INCLUDED, with ZERO bar-frontier extension rows. This is the schema +
    # row-domain surface base-data owns and can audit here; whether the umbrella
    # refresh guard admits it is asserted in the umbrella runbook stage (AC-C),
    # not reimplemented across the boundary.
    built, report = _build(tmp_path)  # default: the canonical served recipe
    # Column SET and ORDER are the canonical 179-col contract, exactly.
    assert list(built.columns) == list(RAWLABEL_SIDECAR_COLUMNS)
    assert set(built.columns) == set(RAWLABEL_SIDECAR_COLUMNS)
    assert len(built.columns) == 179
    # Sentiment is carried (the un-frozen columns the pre-amendment recipe dropped).
    assert set(SENTIMENT_COLS) <= set(built.columns)
    # ZERO bar-frontier extension rows — every emitted row is a real panel
    # (ticker, date), never a key-only frontier extension.
    assert report["n_extension_rows"] == 0
    assert report["n_rows"] == report["n_panel_rows"]
    assert built["date"].max() == PANEL_DATES[-1]


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    import json

    panel_path = _write_panel(tmp_path)
    ohlcv = _ohlcv_dir(tmp_path)
    out = tmp_path / "rawlabel.parquet"
    rc = main(
        [
            "--fund-panel", str(panel_path),
            "--ohlcv-dir", str(ohlcv),
            "--output", str(out),
            "--horizon-trading-days", str(HORIZON),
        ]
    )
    assert rc == 0
    assert out.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["n_tickers"] == 2
    assert list(pd.read_parquet(out).columns) == list(RAWLABEL_SIDECAR_COLUMNS)
