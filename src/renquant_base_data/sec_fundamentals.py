"""SEC EDGAR fundamentals refresh pipeline.

This module owns the data-side materialization formerly kept in the
RenQuant umbrella scripts:

* ``scripts/fetch_sec_fundamentals.py``
* ``scripts/build_extended_fundamentals.py``

The public CLI is intentionally explicit about ``data_dir`` and input files so
scheduled wrappers can call the base-data package without relying on an
umbrella repo root.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import requests

from renquant_common import Job, Pipeline, Task
from renquant_common.net_safety import FetchBudget, call_with_timeout


log = logging.getLogger("renquant_base_data.sec_fundamentals")

SEC_HEADERS = {"User-Agent": "RenQuant renhao.overflow@gmail.com"}
FRAMES_BASE = "https://data.sec.gov/api/xbrl/frames"
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_START_YEAR = 2010
DEFAULT_DAILY_OUTPUT = "sec_fundamentals_daily.parquet"
DEFAULT_EXTENDED_OUTPUT = "sec_fundamentals_extended.parquet"
DEFAULT_ALPHA_CANDIDATES = ("alpha158_816_dataset.parquet", "alpha158_qlib_dataset.parquet")

ConceptSpec = tuple[str, str, str, str]

BASE_CONCEPTS: tuple[ConceptSpec, ...] = (
    ("NetIncomeLoss", "us-gaap", "USD", "duration"),
    ("GrossProfit", "us-gaap", "USD", "duration"),
    ("Revenues", "us-gaap", "USD", "duration"),
    ("Assets", "us-gaap", "USD", "instant"),
    ("StockholdersEquity", "us-gaap", "USD", "instant"),
    ("CommonStockSharesOutstanding", "us-gaap", "shares", "instant"),
)

EXTENDED_CONCEPTS: tuple[ConceptSpec, ...] = BASE_CONCEPTS + (
    ("Liabilities", "us-gaap", "USD", "instant"),
)

RAW_VALUE_COLS = tuple(concept for concept, *_ in BASE_CONCEPTS)
BASE_FEATURE_COLS = (
    "earnings_yield",
    "book_to_price",
    "gross_profitability",
    "roe",
    "asset_growth",
)
EXTENDED_FEATURE_COLS = (
    "asset_turnover",
    "profit_margin",
    "return_on_assets",
    "debt_to_assets",
    "rev_growth_yoy",
    "ni_growth_yoy",
    "equity_growth",
)


class _MissingFrame:
    pass


MISSING_FRAME = _MissingFrame()


@dataclass(frozen=True)
class SecFundamentalsConfig:
    data_dir: Path
    mode: str = "both"
    start_year: int = DEFAULT_START_YEAR
    end_year: int = 2026
    universe_path: Path | None = None
    symbols: tuple[str, ...] | None = None
    alpha_path: Path | None = None
    daily_output: Path | None = None
    extended_output: Path | None = None
    dry_run: bool = False
    sleep_sec: float = 0.12
    per_request_sec: float = 30.0
    total_budget_sec: float = 900.0
    train_end: str = "2022-11-01"


@dataclass
class SecFundamentalsContext:
    config: SecFundamentalsConfig
    universe: list[str] = field(default_factory=list)
    ticker_cik: dict[str, int] = field(default_factory=dict)
    raw_daily: pd.DataFrame | None = None
    daily_features: pd.DataFrame | None = None
    raw_extended: pd.DataFrame | None = None
    extended_features: pd.DataFrame | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def cik_ticker(self) -> dict[int, str]:
        return {cik: ticker for ticker, cik in self.ticker_cik.items()}


def load_universe(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("watchlist") or payload.get("symbols") or payload.get("data", {}).get("watchlist")
    else:
        values = payload
    if not values:
        raise ValueError(f"universe is empty or missing in {path}")
    return [str(symbol).upper() for symbol in values if symbol and str(symbol) != "-"]


def resolve_alpha_path(data_dir: str | Path, alpha_path: str | Path | None = None) -> Path:
    if alpha_path is not None:
        return Path(alpha_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    for name in DEFAULT_ALPHA_CANDIDATES:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    return data_dir / DEFAULT_ALPHA_CANDIDATES[0]


def load_daily_index(alpha_path: str | Path) -> pd.DatetimeIndex:
    alpha = pd.read_parquet(alpha_path, columns=["date"]).drop_duplicates()
    dates = pd.to_datetime(alpha["date"])
    return pd.DatetimeIndex(sorted(dates.unique()))


def period_for(year: int, quarter: int, period_type: str) -> str:
    suffix = "I" if period_type == "instant" else ""
    return f"CY{year}Q{quarter}{suffix}"


def planned_request_count(concepts: Sequence[ConceptSpec], start_year: int, end_year: int) -> int:
    return len(concepts) * max(0, end_year - start_year + 1) * 4


def _download_frame_json(
    concept: str,
    taxonomy: str,
    unit: str,
    period: str,
    *,
    session: requests.Session | None,
) -> object:
    client = session or requests
    response = client.get(
        f"{FRAMES_BASE}/{taxonomy}/{concept}/{unit}/{period}.json",
        headers=SEC_HEADERS,
        timeout=30,
    )
    if response.status_code == 404:
        return MISSING_FRAME
    response.raise_for_status()
    return response.json()


def fetch_frame(
    concept: str,
    taxonomy: str,
    unit: str,
    period: str,
    *,
    session: requests.Session | None = None,
    max_retries: int = 3,
    backoff_sec: float = 5.0,
    timeout_sec: float = 30.0,
    budget: FetchBudget | None = None,
) -> pd.DataFrame | None:
    """Fetch one SEC frames endpoint, returning ``None`` for missing data."""
    for attempt in range(1, max_retries + 1):
        payload = call_with_timeout(
            _download_frame_json,
            concept,
            taxonomy,
            unit,
            period,
            session=session,
            timeout_sec=timeout_sec,
            label=f"sec.frames({concept}/{period})",
            budget=budget,
        )
        if payload is MISSING_FRAME:
            return None
        if isinstance(payload, dict):
            data = payload.get("data", [])
            if not data:
                return None
            frame = pd.DataFrame(data)
            frame["concept"] = concept
            frame["period"] = period
            return frame
        if attempt < max_retries:
            time.sleep(backoff_sec * attempt)
    return None


def fetch_all_concepts(
    *,
    start_year: int,
    end_year: int,
    concepts: Sequence[ConceptSpec],
    sleep_sec: float = 0.12,
    fetcher: Callable[..., pd.DataFrame | None] = fetch_frame,
    budget: FetchBudget | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    total = planned_request_count(concepts, start_year, end_year)
    done = 0
    for concept, taxonomy, unit, period_type in concepts:
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                period = period_for(year, quarter, period_type)
                frame = fetcher(
                    concept,
                    taxonomy,
                    unit,
                    period,
                    budget=budget,
                )
                if frame is not None and not frame.empty:
                    rows.append(frame)
                done += 1
                if done % 50 == 0 or done == total:
                    log.info("SEC frames fetched: %d/%d", done, total)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _download_ticker_map(*, session: requests.Session | None) -> dict[str, Any]:
    client = session or requests
    response = client.get(TICKER_CIK_URL, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def build_ticker_cik_map(
    universe: Sequence[str],
    *,
    session: requests.Session | None = None,
    timeout_sec: float = 30.0,
    budget: FetchBudget | None = None,
) -> dict[str, int]:
    payload = call_with_timeout(
        _download_ticker_map,
        session=session,
        timeout_sec=timeout_sec,
        label="sec.ticker_cik_map",
        budget=budget,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("failed to fetch SEC ticker CIK map")
    full_map = {str(item["ticker"]).upper(): int(item["cik_str"]) for item in payload.values()}
    return {str(symbol).upper(): full_map[str(symbol).upper()] for symbol in universe if str(symbol).upper() in full_map}


def build_quarterly_panel(raw: pd.DataFrame, cik_to_ticker: dict[int, str] | None = None) -> pd.DataFrame:
    """Pivot SEC frames data to one PIT row per ticker and period end."""
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "ticker" not in frame.columns:
        if not cik_to_ticker:
            raise ValueError("raw SEC frame requires ticker column or cik_to_ticker map")
        frame["ticker"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64").map(cik_to_ticker)
    elif cik_to_ticker and "cik" in frame.columns:
        mapped = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64").map(cik_to_ticker)
        frame["ticker"] = frame["ticker"].fillna(mapped)
    frame = frame.dropna(subset=["ticker"]).copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["end"] = pd.to_datetime(frame["end"])
    frame["filed"] = pd.to_datetime(frame.get("filed"), errors="coerce")
    frame["val"] = pd.to_numeric(frame["val"], errors="coerce")
    frame = frame.sort_values(["ticker", "end", "concept", "filed"])

    rows: list[dict[str, Any]] = []
    for (ticker, end_date), group in frame.groupby(["ticker", "end"], sort=True):
        row: dict[str, Any] = {"ticker": ticker, "end": end_date}
        selected_filed: list[pd.Timestamp] = []
        for concept, concept_group in group.groupby("concept", sort=False):
            latest = concept_group.dropna(subset=["val"]).tail(1)
            if latest.empty:
                continue
            item = latest.iloc[0]
            row[str(concept)] = item["val"]
            if pd.notna(item.get("filed")):
                selected_filed.append(pd.Timestamp(item["filed"]))
        row["available_date"] = max(selected_filed) if selected_filed else end_date + pd.Timedelta(days=45)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ticker", "end"]).reset_index(drop=True)


def forward_fill_to_daily(
    quarterly: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    tickers: Sequence[str],
    *,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    if quarterly.empty:
        return pd.DataFrame()
    out: list[pd.DataFrame] = []
    dates = pd.DataFrame({"date": pd.DatetimeIndex(daily_index).sort_values()})
    cols = [col for col in value_cols if col in quarterly.columns]
    for ticker in tickers:
        ticker_q = quarterly[quarterly["ticker"] == str(ticker).upper()].sort_values("available_date")
        if ticker_q.empty:
            continue
        updates = ticker_q[["available_date", *cols]].rename(columns={"available_date": "date"})
        updates["date"] = pd.to_datetime(updates["date"])
        updates = updates.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        daily = pd.merge_asof(dates, updates, on="date", direction="backward")
        daily["ticker"] = str(ticker).upper()
        out.append(daily)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def _read_price_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    price = pd.read_parquet(path)
    if "close" not in price.columns:
        return None
    if "date" in price.columns:
        index = pd.to_datetime(price["date"])
    else:
        index = pd.to_datetime(price.index)
    return pd.Series(pd.to_numeric(price["close"], errors="coerce").to_numpy(), index=index, name="price")


def _numeric_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def compute_derived_features(daily_raw: pd.DataFrame, ohlcv_dir: str | Path) -> pd.DataFrame:
    """Compute market-cap-normalized daily fundamental features."""
    if daily_raw.empty:
        return pd.DataFrame()
    ohlcv_dir = Path(ohlcv_dir).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    for ticker, group in daily_raw.groupby("ticker"):
        price = _read_price_series(ohlcv_dir / str(ticker) / "1d.parquet")
        if price is None:
            continue
        group = group.copy()
        group["date"] = pd.to_datetime(group["date"])
        merged = group.set_index("date").join(price, how="left")

        ni = _numeric_series(merged, "NetIncomeLoss")
        gp = _numeric_series(merged, "GrossProfit")
        assets = _numeric_series(merged, "Assets")
        equity = _numeric_series(merged, "StockholdersEquity")
        shares = _numeric_series(merged, "CommonStockSharesOutstanding")
        market_cap = shares * _numeric_series(merged, "price")

        result = pd.DataFrame(index=merged.index)
        result["ticker"] = str(ticker)
        with np.errstate(invalid="ignore", divide="ignore"):
            result["earnings_yield"] = ni / (market_cap + 1e-9)
            result["book_to_price"] = equity / (market_cap + 1e-9)
            result["gross_profitability"] = gp / (assets + 1e-9)
            result["roe"] = ni / (equity + 1e-9)
            result["asset_growth"] = assets.pct_change(periods=252).clip(-0.99, 5.0)
        result = result.replace([np.inf, -np.inf], np.nan)
        rows.append(result.reset_index().rename(columns={"index": "date"}))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_extended_quarterly_features(quarterly: pd.DataFrame) -> pd.DataFrame:
    if quarterly.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for ticker, group in quarterly.groupby("ticker"):
        group = group.sort_values("end").copy()
        ni = _numeric_series(group, "NetIncomeLoss")
        revenue = _numeric_series(group, "Revenues")
        assets = _numeric_series(group, "Assets")
        equity = _numeric_series(group, "StockholdersEquity")
        liabilities = _numeric_series(group, "Liabilities") if "Liabilities" in group else None

        with np.errstate(invalid="ignore", divide="ignore"):
            group["asset_turnover"] = revenue / (assets + 1e-9)
            group["profit_margin"] = ni / (revenue + 1e-9)
            group["return_on_assets"] = ni / (assets + 1e-9)
            if liabilities is not None:
                group["debt_to_assets"] = liabilities / (assets + 1e-9)
            else:
                group["debt_to_assets"] = (assets - equity) / (assets + 1e-9)
            group["rev_growth_yoy"] = revenue.pct_change(periods=4)
            group["ni_growth_yoy"] = ni.pct_change(periods=4)
            group["equity_growth"] = equity.pct_change(periods=4)
        group[list(EXTENDED_FEATURE_COLS)] = group[list(EXTENDED_FEATURE_COLS)].replace([np.inf, -np.inf], np.nan)
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def robust_zscore_train_window(
    frame: pd.DataFrame,
    *,
    cols: Sequence[str],
    train_end: str | pd.Timestamp = "2022-11-01",
) -> pd.DataFrame:
    out = frame.copy()
    train_end_ts = pd.Timestamp(train_end)
    for col in cols:
        train = pd.to_numeric(out.loc[out["date"] < train_end_ts, col], errors="coerce").dropna()
        median = float(train.median()) if len(train) else 0.0
        mad = float((train - median).abs().median()) if len(train) else 1.0
        denom = max(mad * 1.4826, 1e-9)
        out[col] = ((pd.to_numeric(out[col], errors="coerce") - median) / denom).clip(-3.0, 3.0)
    out[list(cols)] = out[list(cols)].fillna(0.0)
    return out


def build_daily_fundamentals(
    *,
    raw: pd.DataFrame,
    universe: Sequence[str],
    cik_to_ticker: dict[int, str] | None,
    data_dir: str | Path,
    alpha_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_DAILY_OUTPUT
    quarterly = build_quarterly_panel(raw, cik_to_ticker)
    daily_index = load_daily_index(resolve_alpha_path(data_dir, alpha_path))
    daily_raw = forward_fill_to_daily(quarterly, daily_index, universe, value_cols=RAW_VALUE_COLS)
    features = compute_derived_features(daily_raw, data_dir / "ohlcv")
    if features.empty:
        raise RuntimeError("SEC daily fundamentals produced no feature rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    return out


def build_extended_fundamentals(
    *,
    raw: pd.DataFrame,
    universe: Sequence[str],
    cik_to_ticker: dict[int, str] | None,
    data_dir: str | Path,
    alpha_path: str | Path | None = None,
    output_path: str | Path | None = None,
    train_end: str | pd.Timestamp = "2022-11-01",
) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_EXTENDED_OUTPUT
    quarterly = build_quarterly_panel(raw, cik_to_ticker)
    quarterly_ext = compute_extended_quarterly_features(quarterly)
    daily_index = load_daily_index(resolve_alpha_path(data_dir, alpha_path))
    daily_ext = forward_fill_to_daily(quarterly_ext, daily_index, universe, value_cols=EXTENDED_FEATURE_COLS)
    if daily_ext.empty:
        raise RuntimeError("SEC extended fundamentals produced no feature rows")
    daily_ext = robust_zscore_train_window(daily_ext, cols=EXTENDED_FEATURE_COLS, train_end=train_end)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily_ext.to_parquet(out, index=False)
    return out


class LoadUniverseTask(Task):
    def run(self, ctx: SecFundamentalsContext) -> bool | None:
        if ctx.config.symbols is not None:
            ctx.universe = [symbol.upper() for symbol in ctx.config.symbols]
        elif ctx.config.universe_path is not None:
            ctx.universe = load_universe(ctx.config.universe_path)
        else:
            raise ValueError("provide symbols or universe_path")
        ctx.summary["n_symbols"] = len(ctx.universe)
        return True


class LoadTickerMapTask(Task):
    def run(self, ctx: SecFundamentalsContext) -> bool | None:
        if ctx.ticker_cik:
            ctx.summary["cik_found"] = len(ctx.ticker_cik)
            ctx.summary["cik_missing"] = []
            return True
        if ctx.config.dry_run:
            ctx.summary["ticker_cik"] = "skipped_dry_run"
            return True
        budget = FetchBudget(total_sec=min(60.0, ctx.config.total_budget_sec), label="sec_ticker_map")
        ctx.ticker_cik = build_ticker_cik_map(
            ctx.universe,
            timeout_sec=ctx.config.per_request_sec,
            budget=budget,
        )
        missing = sorted(set(ctx.universe) - set(ctx.ticker_cik))
        ctx.summary["cik_found"] = len(ctx.ticker_cik)
        ctx.summary["cik_missing"] = missing[:20]
        return True


class PrepareSecFundamentalsJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [LoadUniverseTask(), LoadTickerMapTask()]


class FetchDailyFramesTask(Task):
    def run(self, ctx: SecFundamentalsContext) -> bool | None:
        concepts = EXTENDED_CONCEPTS if ctx.config.mode == "both" else BASE_CONCEPTS
        planned = planned_request_count(concepts, ctx.config.start_year, ctx.config.end_year)
        ctx.summary["daily_planned_requests"] = planned
        if ctx.config.mode == "both":
            ctx.summary["shared_sec_fetch"] = "extended_concepts_reused_for_daily_and_extended"
        if ctx.raw_daily is not None:
            if ctx.config.mode == "both" and ctx.raw_extended is None:
                ctx.raw_extended = ctx.raw_daily
            ctx.summary["daily_raw_rows"] = int(len(ctx.raw_daily))
            return True
        if ctx.config.dry_run:
            return False
        budget = FetchBudget(total_sec=ctx.config.total_budget_sec, label="sec_daily_frames")
        ctx.raw_daily = fetch_all_concepts(
            start_year=ctx.config.start_year,
            end_year=ctx.config.end_year,
            concepts=concepts,
            sleep_sec=ctx.config.sleep_sec,
            budget=budget,
        )
        if ctx.config.mode == "both":
            ctx.raw_extended = ctx.raw_daily
        ctx.summary["daily_raw_rows"] = int(0 if ctx.raw_daily is None else len(ctx.raw_daily))
        return True


class BuildDailyFundamentalsTask(Task):
    def run(self, ctx: SecFundamentalsContext) -> bool | None:
        if ctx.raw_daily is None or ctx.raw_daily.empty:
            raise RuntimeError("daily SEC raw frame is empty")
        output = build_daily_fundamentals(
            raw=ctx.raw_daily,
            universe=ctx.universe,
            cik_to_ticker=ctx.cik_ticker,
            data_dir=ctx.config.data_dir,
            alpha_path=ctx.config.alpha_path,
            output_path=ctx.config.daily_output,
        )
        ctx.summary["daily_output"] = str(output)
        return True


class DailySecFundamentalsJob(Job):
    def should_skip(self, ctx: SecFundamentalsContext) -> bool:
        return ctx.config.mode not in {"daily", "both"}

    @property
    def tasks(self) -> list[Task]:
        return [FetchDailyFramesTask(), BuildDailyFundamentalsTask()]


class FetchExtendedFramesTask(Task):
    def run(self, ctx: SecFundamentalsContext) -> bool | None:
        planned = planned_request_count(EXTENDED_CONCEPTS, ctx.config.start_year, ctx.config.end_year)
        ctx.summary["extended_planned_requests"] = planned
        if ctx.raw_extended is not None:
            ctx.summary["extended_raw_rows"] = int(len(ctx.raw_extended))
            return True
        if ctx.config.dry_run:
            return False
        budget = FetchBudget(total_sec=ctx.config.total_budget_sec, label="sec_extended_frames")
        ctx.raw_extended = fetch_all_concepts(
            start_year=ctx.config.start_year,
            end_year=ctx.config.end_year,
            concepts=EXTENDED_CONCEPTS,
            sleep_sec=ctx.config.sleep_sec,
            budget=budget,
        )
        ctx.summary["extended_raw_rows"] = int(0 if ctx.raw_extended is None else len(ctx.raw_extended))
        return True


class BuildExtendedFundamentalsTask(Task):
    def run(self, ctx: SecFundamentalsContext) -> bool | None:
        if ctx.raw_extended is None or ctx.raw_extended.empty:
            raise RuntimeError("extended SEC raw frame is empty")
        output = build_extended_fundamentals(
            raw=ctx.raw_extended,
            universe=ctx.universe,
            cik_to_ticker=ctx.cik_ticker,
            data_dir=ctx.config.data_dir,
            alpha_path=ctx.config.alpha_path,
            output_path=ctx.config.extended_output,
            train_end=ctx.config.train_end,
        )
        ctx.summary["extended_output"] = str(output)
        return True


class ExtendedSecFundamentalsJob(Job):
    def should_skip(self, ctx: SecFundamentalsContext) -> bool:
        return ctx.config.mode not in {"extended", "both"}

    @property
    def tasks(self) -> list[Task]:
        return [FetchExtendedFramesTask(), BuildExtendedFundamentalsTask()]


class SecFundamentalsRefreshPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__(
            [PrepareSecFundamentalsJob(), DailySecFundamentalsJob(), ExtendedSecFundamentalsJob()],
            name="sec-fundamentals-refresh",
        )


def refresh_sec_fundamentals(
    config: SecFundamentalsConfig,
    *,
    raw_daily: pd.DataFrame | None = None,
    raw_extended: pd.DataFrame | None = None,
    ticker_cik: dict[str, int] | None = None,
) -> dict[str, Any]:
    ctx = SecFundamentalsContext(
        config=config,
        raw_daily=raw_daily,
        raw_extended=raw_extended,
        ticker_cik=ticker_cik or {},
    )
    pipeline = SecFundamentalsRefreshPipeline()
    result = pipeline.run(ctx)
    ctx.summary.update({
        "ok": bool(result.ok),
        "mode": config.mode,
        "dry_run": bool(config.dry_run),
        "elapsed_sec": result.elapsed_sec,
        "steps": [record.job_name for record in result.steps if not record.skipped],
    })
    return ctx.summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["daily", "extended", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--alpha-path", type=Path, default=None)
    parser.add_argument("--daily-output", type=Path, default=None)
    parser.add_argument("--extended-output", type=Path, default=None)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--sleep-sec", type=float, default=0.12)
    parser.add_argument("--per-request-sec", type=float, default=30.0)
    parser.add_argument("--total-budget-sec", type=float, default=900.0)
    parser.add_argument("--train-end", default="2022-11-01")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    if args.symbols is None and args.universe is None:
        raise SystemExit("--symbols or --universe is required")
    config = SecFundamentalsConfig(
        data_dir=args.data_dir,
        mode=args.mode,
        start_year=args.start_year,
        end_year=args.end_year,
        universe_path=args.universe,
        symbols=tuple(args.symbols) if args.symbols is not None else None,
        alpha_path=args.alpha_path,
        daily_output=args.daily_output,
        extended_output=args.extended_output,
        dry_run=args.dry_run,
        sleep_sec=args.sleep_sec,
        per_request_sec=args.per_request_sec,
        total_budget_sec=args.total_budget_sec,
        train_end=args.train_end,
    )
    summary = refresh_sec_fundamentals(config)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
