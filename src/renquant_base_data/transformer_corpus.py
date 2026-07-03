"""Build the PatchTST shadow training corpus — the TRUE committed recipe (S12 B1).

WHY (S12 panel-refresh diagnosis, RenQuant
``doc/research/2026-07-02-s12-panel-refresh-diagnosis.md`` sections 1 and 5.1):
the served shadow corpus ``transformer_v4_wl200_clean.parquet`` was an ad-hoc
2026-05-18 research snapshot with NO committed builder — measured ground truth
is an alpha158+fund-family panel (178 columns) restricted to the live strategy
watchlist with forward labels dropna'd, NOT the raw-OHLCV 292-ticker output of
``transformer_dataset_builder.py`` that the RenQuant #424 refresh chain's
default ``builder_fn`` invokes (so its swap gate fail-closes forever).

THE RECIPE (cheapest correct, per the diagnosis's remediation step 1): derive
the corpus from the already-daily-refreshed production fund panel
(``alpha158_291_fundamental_dataset.parquet``, itself the committed
``alpha158_fund_panel`` output):

  1. subset the panel rows to the live strategy watchlist (the pinned
     ``renquant-strategy-104`` ``strategy_config.json`` ``watchlist`` key);
  2. drop rows with any NaN forward label (``fwd_5d/20d/60d_excess`` — the
     training-axis label clip; unlabeled rows are untrainable and the trainer
     re-drops them anyway);
  3. keep the EXACT served 178-column schema — column set, order, and dtypes —
     and emit deterministic ``(ticker, date)`` row ordering.

The panel already carries the label dropna and ``split_label``, so the subset
inherits them; the dropna here is a defensive re-assertion of the served
semantics. Because the production panel's schema is identical to the served
corpus's (verified column-for-column, dtype-for-dtype), a rebuild passes the
refresh chain's swap gate (schema / label-horizon / ticker-coverage parity) by
construction, and its max(date) sits at the achievable frontier
(~bar frontier - 60 trading days; the correct structural fwd_60d lag).

FAIL-CLOSED CONTRACT: a fund panel missing contract columns, carrying
unexpected extra columns (strict default), an empty / non-overlapping
watchlist, non-castable feature dtypes, or duplicate (ticker, date) rows all
raise ``TransformerCorpusError`` — the builder never silently emits a
divergent corpus. The builder writes ONLY to the caller-specified output path;
it never defaults to (or touches) a served production file.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd


log = logging.getLogger("renquant_base_data.transformer_corpus")

#: Forward-label columns whose NaN rows are dropped (the training-axis clip).
LABEL_COLS = ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess")

#: Walk-forward split annotation carried through from the fund panel.
SPLIT_COL = "split_label"

#: Fund-family feature columns appended after ``split_label`` (fund panel order).
FUND_FAMILY_COLS = (
    "earnings_yield",
    "book_to_price",
    "gross_profitability",
    "roe",
    "asset_growth",
    "days_since_earnings",
    "pead_signal",
    "pead_quintile_rank",
    "sue_signal",
    "surprise_momentum",
    "surprise_streak",
    "sentiment_pos_share",
    "mean_sentiment",
    "n_articles_log",
)

#: Default production fund panel filename (the ``alpha158_fund_panel`` output).
DEFAULT_FUND_PANEL_FILENAME = "alpha158_291_fundamental_dataset.parquet"

_ALPHA158_BASE = (
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
)
_ALPHA158_ROLLING = (
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
)
_ALPHA158_WINDOWS = (5, 10, 20, 30, 60)


def _alpha158_columns() -> tuple:
    cols = list(_ALPHA158_BASE)
    for window in _ALPHA158_WINDOWS:
        cols.extend(f"{name}{window}" for name in _ALPHA158_ROLLING)
    return tuple(cols)


#: The exact served-corpus column contract, IN ORDER (verified 2026-07-02
#: against the served ``transformer_v4_wl200_clean.parquet`` AND the production
#: ``alpha158_291_fundamental_dataset.parquet`` — both are exactly this,
#: column-for-column): ticker + date + 158 alpha158 features + 3 forward labels
#: + split_label + 14 FUND/PEAD/SUE/SENT columns = 178 columns.
TRANSFORMER_CORPUS_COLUMNS = (
    ("ticker", "date")
    + _alpha158_columns()
    + LABEL_COLS
    + (SPLIT_COL,)
    + FUND_FAMILY_COLS
)

#: Served dtype contract: pandas "string" for ticker/split_label,
#: datetime64[ns] for date, float64 for every feature / label column.
_STRING_COLS = ("ticker", SPLIT_COL)
_DATETIME_COLS = ("date",)


class TransformerCorpusError(RuntimeError):
    """Fail-closed corpus-build failure (schema drift / bad watchlist / bad
    panel). Subclasses RuntimeError so callers can catch either."""


def load_watchlist(strategy_config_path: str | Path) -> list:
    """Load the live ticker watchlist from a strategy config JSON.

    Reads the ``watchlist`` key of the pinned ``renquant-strategy-104``
    ``strategy_config.json``. Fails CLOSED (raises) on a missing / unreadable /
    corrupt config or an empty watchlist — the corpus universe must never
    silently degrade to nothing.
    """
    path = Path(strategy_config_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TransformerCorpusError(
            f"strategy config unreadable/corrupt: {path}: {exc}"
        ) from exc
    watchlist = payload.get("watchlist") if isinstance(payload, dict) else None
    if not isinstance(watchlist, list) or not watchlist:
        raise TransformerCorpusError(
            f"strategy config has no non-empty 'watchlist' list: {path}"
        )
    # Uppercase + de-dupe, preserving config order (row ordering is enforced
    # deterministically downstream regardless).
    return list(dict.fromkeys(str(t).upper() for t in watchlist))


def build_transformer_corpus(
    fund_panel_path: str | Path,
    watchlist: Sequence[str],
    output_path: str | Path,
    *,
    label_columns: Sequence[str] = LABEL_COLS,
    require_exact_schema: bool = True,
) -> dict:
    """Build the transformer shadow corpus from the production fund panel.

    Parameters
    ----------
    fund_panel_path:
        The daily-refreshed production fund panel parquet (the committed
        ``alpha158_fund_panel`` output, e.g.
        ``data/alpha158_291_fundamental_dataset.parquet``). Read-only input.
    watchlist:
        Live strategy watchlist tickers (see :func:`load_watchlist`). The
        corpus rows are ``watchlist ∩ panel tickers``; watchlist names absent
        from the panel are reported, not fatal (new adds lag the panel build).
    output_path:
        REQUIRED caller-specified output parquet path. This builder never
        writes anywhere else and has no production-path default.
    label_columns:
        Forward-label columns whose NaN rows are dropped (served semantics).
    require_exact_schema:
        When True (default) the panel must carry EXACTLY the served 178-column
        contract; unexpected extra columns fail closed. When False, extras are
        dropped (the output still matches the contract exactly).

    Returns
    -------
    dict
        Build report: rows / tickers / date frontier / dropped-label rows /
        watchlist names missing from the panel.
    """
    fund_panel_path = Path(fund_panel_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not watchlist:
        raise TransformerCorpusError("watchlist is empty; refusing to build an empty corpus")
    wanted = list(dict.fromkeys(str(t).upper() for t in watchlist))

    try:
        panel = pd.read_parquet(fund_panel_path)
    except Exception as exc:
        raise TransformerCorpusError(
            f"fund panel unreadable: {fund_panel_path}: {exc}"
        ) from exc

    # ── schema contract (fail closed on drift) ──────────────────────────────
    panel_cols = [str(c) for c in panel.columns]
    missing = [c for c in TRANSFORMER_CORPUS_COLUMNS if c not in panel_cols]
    if missing:
        raise TransformerCorpusError(
            f"fund panel is missing {len(missing)} contract column(s) "
            f"(schema drift): {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    extras = [c for c in panel_cols if c not in TRANSFORMER_CORPUS_COLUMNS]
    if extras and require_exact_schema:
        raise TransformerCorpusError(
            f"fund panel carries {len(extras)} unexpected column(s) "
            f"(schema drift; pass require_exact_schema=False to drop them): "
            f"{extras[:8]}{'...' if len(extras) > 8 else ''}"
        )
    if extras:
        log.warning("dropping %d extra fund-panel column(s): %s", len(extras), extras[:8])
    bad_labels = [c for c in label_columns if c not in TRANSFORMER_CORPUS_COLUMNS]
    if bad_labels:
        raise TransformerCorpusError(f"label column(s) not in the corpus contract: {bad_labels}")
    corpus = panel.loc[:, list(TRANSFORMER_CORPUS_COLUMNS)]

    # ── watchlist subset ─────────────────────────────────────────────────────
    corpus = corpus.assign(ticker=corpus["ticker"].astype("string").str.upper())
    panel_tickers = set(corpus["ticker"].dropna())
    present = [t for t in wanted if t in panel_tickers]
    missing_tickers = [t for t in wanted if t not in panel_tickers]
    if not present:
        raise TransformerCorpusError(
            f"no watchlist ticker present in the fund panel "
            f"({len(wanted)} wanted, {len(panel_tickers)} in panel) — wrong panel?"
        )
    corpus = corpus[corpus["ticker"].isin(present)]

    # ── label dropna (training-axis clip; served semantics) ─────────────────
    n_before = len(corpus)
    corpus = corpus.dropna(subset=list(label_columns))
    n_label_rows_dropped = n_before - len(corpus)
    if corpus.empty:
        raise TransformerCorpusError(
            "corpus empty after the label dropna — fund panel has no labeled "
            "rows for the watchlist"
        )

    # ── dtype contract ───────────────────────────────────────────────────────
    try:
        corpus = corpus.assign(
            date=pd.to_datetime(corpus["date"]),
            **{c: corpus[c].astype("string") for c in _STRING_COLS},
        )
        float_cols = [
            c
            for c in TRANSFORMER_CORPUS_COLUMNS
            if c not in _STRING_COLS + _DATETIME_COLS
        ]
        corpus[float_cols] = corpus[float_cols].astype("float64")
    except (TypeError, ValueError) as exc:
        raise TransformerCorpusError(
            f"fund panel dtype(s) not castable to the corpus contract: {exc}"
        ) from exc

    # ── deterministic ordering + integrity ──────────────────────────────────
    corpus = corpus.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    dupes = corpus.duplicated(subset=["ticker", "date"])
    if bool(dupes.any()):
        sample = corpus.loc[dupes, ["ticker", "date"]].head(3).to_dict("records")
        raise TransformerCorpusError(
            f"fund panel has duplicate (ticker, date) rows (broken panel): {sample}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(output_path, index=False)

    report = {
        "fund_panel_path": str(fund_panel_path),
        "output_path": str(output_path),
        "n_rows": int(len(corpus)),
        "n_tickers": int(corpus["ticker"].nunique()),
        "n_columns": int(len(corpus.columns)),
        "min_date": corpus["date"].min().date().isoformat(),
        "max_date": corpus["date"].max().date().isoformat(),
        "n_label_rows_dropped": int(n_label_rows_dropped),
        "missing_watchlist_tickers": missing_tickers,
    }
    log.info(
        "transformer corpus built: rows=%d tickers=%d cols=%d dates=%s..%s "
        "(label rows dropped=%d; watchlist names absent from panel=%d) -> %s",
        report["n_rows"],
        report["n_tickers"],
        report["n_columns"],
        report["min_date"],
        report["max_date"],
        report["n_label_rows_dropped"],
        len(missing_tickers),
        output_path,
    )
    return report


def parse_args(argv: "list | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fund-panel",
        type=Path,
        required=True,
        help=f"Production fund panel parquet (the {DEFAULT_FUND_PANEL_FILENAME} build).",
    )
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument(
        "--strategy-config",
        type=Path,
        help="Pinned strategy_config.json whose 'watchlist' key is the corpus universe.",
    )
    universe.add_argument(
        "--tickers",
        help="Comma-separated explicit watchlist (testing/override).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output corpus parquet path (never defaults to a served file).",
    )
    parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help="Drop unexpected fund-panel columns instead of failing closed.",
    )
    return parser.parse_args(argv)


def main(argv: "list | None" = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    if args.strategy_config is not None:
        watchlist = load_watchlist(args.strategy_config)
    else:
        watchlist = [t.strip() for t in str(args.tickers).split(",") if t.strip()]
    report = build_transformer_corpus(
        args.fund_panel,
        watchlist,
        args.output,
        require_exact_schema=not args.allow_extra_columns,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
