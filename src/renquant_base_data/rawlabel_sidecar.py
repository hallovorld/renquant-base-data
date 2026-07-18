"""Build the RAWLABEL calibration sidecar — the TRUE committed recipe (S12, B1 pattern).

WHY (S12 promote refusal, 2026-07-02: ``source[fast] rawlabel: cutoff=2026-02-11
age=142d sla=28d OFF-SLA``): the served
``alpha158_291_fundamental_dataset_rawlabel.parquet`` — the UN-standardized
forward-label sidecar the PatchTST/GBDT calibrator fits read — was a one-off
research build (RenQuant ``scripts/build_raw_fwd60d_label.py``, Track A) with NO
refresh mechanism, so its ``max(date)`` froze at the 2026-02-11 panel vintage
while the transformer panel gained a committed recipe + weekly refresh (S12 B1:
``transformer_corpus`` here + RenQuant ``scripts/refresh_transformer_corpus.py``).
The promote gate holds the rawlabel source to the raw 28-day fast-axis SLA
because a healthy rawlabel KEEPS unlabeled rows — its ``max(date)`` tracks the
panel's weekly-refreshed feature frontier, not a frozen vintage (nor the
label-clipped training frontier; the raw sidecar never label-dropnas). The
default no longer extends past the panel frontier — see the amendment below.

SINGLE-WRITER AMENDMENT (base-data#48, 2026-07-18): this builder is now the
SOLE writer of the served sidecar. The canonical contract CARRIES the three
sentiment columns (179 cols) and, for THIS artifact, drops the bar-frontier
axis extension — the two former recipe divergences between this builder and
the orchestrator σ-head writer that produced the weekly deadlock. See
``doc/design/2026-07-18-sidecar-single-writer-amendment.md`` §2.

THE RECIPE (committed + deterministic):

  1. take the daily-refreshed production fund panel
     (``alpha158_291_fundamental_dataset.parquet``, the committed
     ``alpha158_fund_panel`` output) over its FULL ticker universe — no
     watchlist subset, NO forward-label dropna (raw-sidecar semantics);
  2. carry the panel's full column schema THROUGH — sentiment columns
     INCLUDED (amendment §2.2: the sentiment columns are ACTIVE features of
     the prod XGB recipe and both GBDT WF corpora, not vestigial). Missing
     sentiment values stay NaN, never forward-filled (amendment §2.4:
     event-driven features; a ffill would fabricate staleness as signal);
  3. by DEFAULT do NOT extend each ticker's (ticker, date) axis past the panel
     frontier (amendment §2.3: no consumer of the served file needs
     bar-frontier rows and the σ-head validator rejected them; the served
     file carries none). ``extend_to_bar_frontier=True`` remains available for
     a SEPARATE artifact; when set, extension rows carry NaN features / NaN
     split_label (honestly not computed — no fabricated values);
  4. compute ``fwd_60d_excess_raw`` point-in-time from OHLCV closes exactly as
     the original build: per ticker, ``close[d+60td]/close[d] - 1`` minus the
     same-window benchmark (SPY) return. A label for date ``d`` is knowable
     only once ``d+60td`` has traded; rows whose forward window is incomplete
     stay NaN (the calibrator fits dropna on the label);
  5. keep the EXACT served 179-column schema — column set, order, dtypes —
     and emit deterministic (ticker, date) row ordering.

FAIL-CLOSED CONTRACT: a fund panel missing contract columns (sentiment now
included), carrying unexpected extra columns (strict default), duplicate
(ticker, date) rows, a missing/corrupt benchmark OHLCV cache, future-dated
bars or panel rows, or an all-NaN raw label (broken OHLCV dir) all raise
``RawLabelSidecarError`` — the builder never silently emits a divergent or
vacuous sidecar. The builder writes ONLY to the caller-specified output path;
it never defaults to (or touches) a served production file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from renquant_base_data.transformer_corpus import (
    SPLIT_COL,
    TRANSFORMER_CORPUS_COLUMNS,
)


log = logging.getLogger("renquant_base_data.rawlabel_sidecar")

#: The raw (un-z-scored) forward excess-return label the sidecar exists for.
RAW_LABEL_COL = "fwd_60d_excess_raw"

#: Trading-day horizon of the raw label (matches the production fwd_60d label).
DEFAULT_HORIZON_TRADING_DAYS = 60

#: Benchmark whose same-window forward return is subtracted (raw EXCESS return).
DEFAULT_BENCHMARK_TICKER = "SPY"

#: The panel's event-driven sentiment columns. The canonical sidecar contract
#: now CARRIES them (base-data#48 §2.2 — un-frozen: they are ACTIVE features
#: of the prod XGB recipe + both GBDT WF corpora; AC-1 falsified the "the
#: served sidecar predates them" freeze). Missing sentiment values are kept as
#: NaN, never forward-filled (§2.4).
SENTIMENT_COLS = ("sentiment_pos_share", "mean_sentiment", "n_articles_log")

#: The exact served-sidecar column contract, IN ORDER: the full 178-column
#: fund-panel schema (sentiment INCLUDED) with ``fwd_60d_excess_raw`` appended
#: = 179 columns (base-data#48 §2.2, matching the served file the 99
#: active/candidate sanity contracts require).
RAWLABEL_SIDECAR_COLUMNS = tuple(TRANSFORMER_CORPUS_COLUMNS) + (RAW_LABEL_COL,)

#: Panel-side columns the sidecar carries through (everything but the raw label).
_PANEL_SIDE_COLUMNS = RAWLABEL_SIDECAR_COLUMNS[:-1]

_STRING_COLS = ("ticker", SPLIT_COL)
_DATETIME_COLS = ("date",)


class RawLabelSidecarError(RuntimeError):
    """Fail-closed sidecar-build failure (schema drift / bad OHLCV / bad panel).
    Subclasses RuntimeError so callers can catch either."""


def _read_close_series(ohlcv_dir: Path, ticker: str) -> "pd.Series | None":
    """A ticker's daily close series from ``ohlcv_dir/<ticker>/1d.parquet``
    (DatetimeIndex, sorted). None when the cache file is absent (the original
    build's skip semantics: NaN labels, no axis extension). A PRESENT but
    unreadable/duplicated/close-less cache fails CLOSED — that is corruption,
    not absence."""
    path = ohlcv_dir / ticker / "1d.parquet"
    if not path.exists():
        return None
    try:
        ohlcv = pd.read_parquet(path)
    except Exception as exc:
        raise RawLabelSidecarError(f"OHLCV cache unreadable: {path}: {exc}") from exc
    if "close" not in ohlcv.columns:
        raise RawLabelSidecarError(f"OHLCV cache has no 'close' column: {path}")
    close = ohlcv["close"]
    close.index = pd.to_datetime(ohlcv.index)
    close = close.sort_index()
    if close.index.has_duplicates:
        raise RawLabelSidecarError(f"OHLCV cache has duplicate bar dates: {path}")
    if close.empty:
        return None
    return close


def _forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Forward ``horizon``-trading-day simple return per bar date; NaN where the
    forward window extends past the series end (PIT: unknowable yet)."""
    return close.shift(-horizon) / close - 1.0


def build_rawlabel_sidecar(
    fund_panel_path: str | Path,
    ohlcv_dir: str | Path,
    output_path: str | Path,
    *,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    require_exact_schema: bool = True,
    extend_to_bar_frontier: bool = False,
    today: "dt.date | None" = None,
) -> dict:
    """Build the rawlabel calibration sidecar from the production fund panel.

    Parameters
    ----------
    fund_panel_path:
        The daily-refreshed production fund panel parquet (the committed
        ``alpha158_fund_panel`` output). Read-only input.
    ohlcv_dir:
        Directory of per-ticker daily bar caches (``<ticker>/1d.parquet``);
        supplies both the raw-label closes and the bar-frontier axis extension.
        Read-only input.
    output_path:
        REQUIRED caller-specified output parquet path. This builder never
        writes anywhere else and has no production-path default.
    horizon_trading_days:
        Forward-label horizon in trading days (default 60, the served label).
    benchmark_ticker:
        Benchmark whose same-window forward return is subtracted. Its OHLCV
        cache is REQUIRED — no benchmark, no honest excess label, fail closed.
    require_exact_schema:
        When True (default) the panel must carry ONLY the fund-panel contract
        columns (sentiment INCLUDED); any extra column fails closed. When
        False, extras are dropped (the output still matches the 179-column
        contract exactly).
    extend_to_bar_frontier:
        When False (default, base-data#48 §2.3) the sidecar carries ZERO
        bar-frontier extension rows — the canonical served-file recipe (no
        consumer of the served file needs them; the σ-head validator rejected
        them, and the served file carries none). When True each ticker's axis
        is extended past the panel frontier with its own OHLCV bar dates
        (extension rows carry NaN features/labels/split) — reserved for a
        SEPARATE artifact, never a recipe fork of the served file.
    today:
        Injectable "no bar may postdate this" guard date (default: the wall
        clock). A future-dated panel row or bar fails closed — mirrors the
        promote gate's future-dated guard.

    Returns
    -------
    dict
        Build report: rows / tickers / date frontier / labeled frontier /
        extension + unlabeled row counts / tickers without OHLCV.
    """
    fund_panel_path = Path(fund_panel_path).expanduser()
    ohlcv_dir = Path(ohlcv_dir).expanduser()
    output_path = Path(output_path).expanduser()
    today = today or dt.date.today()
    if horizon_trading_days <= 0:
        raise RawLabelSidecarError(
            f"horizon_trading_days must be positive: {horizon_trading_days}"
        )

    try:
        panel = pd.read_parquet(fund_panel_path)
    except Exception as exc:
        raise RawLabelSidecarError(
            f"fund panel unreadable: {fund_panel_path}: {exc}"
        ) from exc
    if panel.empty:
        raise RawLabelSidecarError(f"fund panel is empty: {fund_panel_path}")

    # ── schema contract (fail closed on drift) ──────────────────────────────
    panel_cols = [str(c) for c in panel.columns]
    missing = [c for c in _PANEL_SIDE_COLUMNS if c not in panel_cols]
    if missing:
        raise RawLabelSidecarError(
            f"fund panel is missing {len(missing)} sidecar contract column(s) "
            f"(schema drift): {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    if RAW_LABEL_COL in panel_cols:
        raise RawLabelSidecarError(
            f"fund panel already carries {RAW_LABEL_COL!r} — refusing to "
            "re-derive a raw label on top of one (wrong input file?)"
        )
    extras = [c for c in panel_cols if c not in _PANEL_SIDE_COLUMNS]
    if extras and require_exact_schema:
        raise RawLabelSidecarError(
            f"fund panel carries {len(extras)} unexpected column(s) "
            f"(schema drift; pass require_exact_schema=False to drop them): "
            f"{extras[:8]}{'...' if len(extras) > 8 else ''}"
        )
    if extras:
        log.warning("dropping %d extra fund-panel column(s): %s", len(extras), extras[:8])
    sidecar = panel.loc[:, list(_PANEL_SIDE_COLUMNS)]

    # ── keys + integrity (raw semantics: NO label dropna, full universe) ─────
    sidecar = sidecar.assign(
        ticker=sidecar["ticker"].astype("string").str.upper(),
        date=pd.to_datetime(sidecar["date"]),
    )
    dupes = sidecar.duplicated(subset=["ticker", "date"])
    if bool(dupes.any()):
        sample = sidecar.loc[dupes, ["ticker", "date"]].head(3).to_dict("records")
        raise RawLabelSidecarError(
            f"fund panel has duplicate (ticker, date) rows (broken panel): {sample}"
        )
    panel_max = sidecar["date"].max().date()
    if panel_max > today:
        raise RawLabelSidecarError(
            f"fund panel is future-dated: max(date)={panel_max.isoformat()} > "
            f"{today.isoformat()} (clock bug / corrupted date column)"
        )
    n_panel_rows = len(sidecar)

    # ── benchmark forward return (REQUIRED — no benchmark, no excess label) ──
    bench_close = _read_close_series(ohlcv_dir, benchmark_ticker)
    if bench_close is None:
        raise RawLabelSidecarError(
            f"benchmark OHLCV cache missing/empty: "
            f"{ohlcv_dir / benchmark_ticker / '1d.parquet'} — cannot compute "
            f"{RAW_LABEL_COL} without the benchmark leg"
        )
    if bench_close.index.max().date() > today:
        raise RawLabelSidecarError(
            f"benchmark OHLCV is future-dated: {bench_close.index.max().date()} > "
            f"{today.isoformat()}"
        )
    bench_fwd = _forward_return(bench_close, horizon_trading_days)

    # ── per-ticker: extend axis to the bar frontier + raw excess label ───────
    blocks = []
    tickers_without_ohlcv: list = []
    n_extension_rows = 0
    for ticker, group in sidecar.groupby("ticker", sort=True):
        group = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        close = _read_close_series(ohlcv_dir, str(ticker))
        if close is None:
            # Original build's skip semantics: NaN label, no axis extension.
            tickers_without_ohlcv.append(str(ticker))
            group[RAW_LABEL_COL] = np.nan
            blocks.append(group)
            continue
        bar_max = close.index.max().date()
        if bar_max > today:
            raise RawLabelSidecarError(
                f"OHLCV is future-dated for {ticker}: {bar_max.isoformat()} > "
                f"{today.isoformat()}"
            )
        if extend_to_bar_frontier:
            ext_dates = close.index[close.index > group["date"].max()]
            if len(ext_dates):
                # Honest axis extension: keys only; features/labels/split stay
                # NaN (not computed — never fabricated).
                ext = pd.DataFrame(
                    {"ticker": str(ticker), "date": pd.DatetimeIndex(ext_dates)}
                )
                group = pd.concat([group, ext], ignore_index=True)
                n_extension_rows += len(ext)
        fwd = _forward_return(close, horizon_trading_days)
        group[RAW_LABEL_COL] = (
            fwd.reindex(group["date"]).to_numpy()
            - bench_fwd.reindex(group["date"]).to_numpy()
        )
        blocks.append(group)

    out = pd.concat(blocks, ignore_index=True)
    n_labeled = int(out[RAW_LABEL_COL].notna().sum())
    if n_labeled == 0:
        raise RawLabelSidecarError(
            "no row received a raw label — OHLCV dir has no usable closes for "
            f"any of {sidecar['ticker'].nunique()} panel tickers (wrong dir?)"
        )

    # ── dtype contract ───────────────────────────────────────────────────────
    try:
        out = out.assign(
            date=pd.to_datetime(out["date"]),
            **{c: out[c].astype("string") for c in _STRING_COLS},
        )
        float_cols = [
            c
            for c in RAWLABEL_SIDECAR_COLUMNS
            if c not in _STRING_COLS + _DATETIME_COLS
        ]
        out[float_cols] = out[float_cols].astype("float64")
    except (TypeError, ValueError) as exc:
        raise RawLabelSidecarError(
            f"fund panel dtype(s) not castable to the sidecar contract: {exc}"
        ) from exc

    # ── deterministic ordering + exact served column order ──────────────────
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    out = out.loc[:, list(RAWLABEL_SIDECAR_COLUMNS)]
    if list(out.columns) != list(RAWLABEL_SIDECAR_COLUMNS):  # pragma: no cover
        raise RawLabelSidecarError("internal error: output column order drifted")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    labeled_max = out.loc[out[RAW_LABEL_COL].notna(), "date"].max()
    report = {
        "fund_panel_path": str(fund_panel_path),
        "ohlcv_dir": str(ohlcv_dir),
        "output_path": str(output_path),
        "horizon_trading_days": int(horizon_trading_days),
        "benchmark_ticker": str(benchmark_ticker),
        "n_rows": int(len(out)),
        "n_panel_rows": int(n_panel_rows),
        "n_extension_rows": int(n_extension_rows),
        "n_tickers": int(out["ticker"].nunique()),
        "n_columns": int(len(out.columns)),
        "min_date": out["date"].min().date().isoformat(),
        "max_date": out["date"].max().date().isoformat(),
        "max_labeled_date": labeled_max.date().isoformat() if pd.notna(labeled_max) else None,
        "n_labeled_rows": n_labeled,
        "n_unlabeled_rows": int(len(out) - n_labeled),
        "tickers_without_ohlcv": tickers_without_ohlcv,
    }
    log.info(
        "rawlabel sidecar built: rows=%d (panel=%d ext=%d) tickers=%d cols=%d "
        "dates=%s..%s labeled<=%s (labeled=%d unlabeled=%d; no-OHLCV tickers=%d) -> %s",
        report["n_rows"],
        report["n_panel_rows"],
        report["n_extension_rows"],
        report["n_tickers"],
        report["n_columns"],
        report["min_date"],
        report["max_date"],
        report["max_labeled_date"],
        report["n_labeled_rows"],
        report["n_unlabeled_rows"],
        len(tickers_without_ohlcv),
        output_path,
    )
    return report


def parse_args(argv: "list | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fund-panel",
        type=Path,
        required=True,
        help="Production fund panel parquet (the alpha158_fund_panel build).",
    )
    parser.add_argument(
        "--ohlcv-dir",
        type=Path,
        required=True,
        help="Per-ticker daily-bar cache dir (<ticker>/1d.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output sidecar parquet path (never defaults to a served file).",
    )
    parser.add_argument(
        "--horizon-trading-days", type=int, default=DEFAULT_HORIZON_TRADING_DAYS
    )
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help="Drop unexpected fund-panel columns instead of failing closed.",
    )
    parser.add_argument(
        "--extend-to-bar-frontier",
        action="store_true",
        help="Opt IN to the bar-frontier axis extension (default OFF for the "
        "canonical served sidecar; base-data#48 §2.3 — reserved for a "
        "separate artifact, never the served file).",
    )
    return parser.parse_args(argv)


def main(argv: "list | None" = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args(argv)
    report = build_rawlabel_sidecar(
        args.fund_panel,
        args.ohlcv_dir,
        args.output,
        horizon_trading_days=args.horizon_trading_days,
        benchmark_ticker=args.benchmark,
        require_exact_schema=not args.allow_extra_columns,
        extend_to_bar_frontier=args.extend_to_bar_frontier,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
