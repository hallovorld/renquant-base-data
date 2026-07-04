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
import os
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

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"
DEFAULT_SEC_USER_AGENT = "renquant-base-data sec-edgar-contact@invalid.example"
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
# Per-entity fiscal provenance columns stamped on every (ticker, date) row of the
# daily feeds (ADDITIVE schema; every audited consumer selects columns explicitly,
# so extra columns are ignored). They exist so freshness gates (renquant-pipeline
# P-FUND-FRESHNESS and RenQuant scripts/promote_shadow_patchtst.py
# ``fundamentals_sla_verdict``) can verify QUARTERLY coverage PER ENTITY instead
# of failing closed as UNVERIFIABLE:
#   fiscal_period_end  fiscal-period end date of the LATEST filing whose values
#                      that daily row carries (the ``end`` of the forward-filled
#                      quarterly snapshot).
#   available_at       point-in-time date the filing's values became available
#                      (see AVAILABILITY TIERS in ``build_quarterly_panel``);
#                      never precedes real availability where a genuine filing
#                      timestamp exists, and never exceeds the row's ``date``
#                      (enforced fail-closed by ``validate_pit_provenance``).
PROVENANCE_COLS = ("fiscal_period_end", "available_at")
# Days after a fiscal-period end at which a 10-Q/10-K is conservatively assumed
# filed+available when NO genuine filing timestamp exists (SEC 10-Q deadlines:
# 40d large-accelerated / 45d accelerated + non-accelerated). Matches the
# ``filing_lag_days`` convention of the consuming freshness gates. This is an
# ASSUMPTION tier, never a substitute for a genuine timestamp when one exists.
FILING_LAG_FALLBACK_DAYS = 45
# Default location of the FMP fundamentals harvest whose income-statement
# ``acceptedDate`` backfills availability when SEC frames carry no ``filed``
# date (the production case: the XBRL frames API returns no filing timestamp).
DEFAULT_FMP_HARVEST_DIRNAME = "fmp_harvest_5y"
FMP_INCOME_STATEMENT_GLOB = "income_statement*.parquet"


class _MissingFrame:
    pass


MISSING_FRAME = _MissingFrame()


def sec_headers() -> dict[str, str]:
    return {"User-Agent": os.environ.get(SEC_USER_AGENT_ENV, DEFAULT_SEC_USER_AGENT)}


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
    fmp_harvest_dir: Path | None = None
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
    # 2026-06-12 staleness fix: pick the FRESHEST existing candidate (by
    # mtime), not the first-listed. Pre-fix, an abandoned
    # alpha158_816_dataset.parquet (last built 2026-05-07, dates ending
    # 2026-02-10) shadowed the daily-rebuilt alpha158_qlib_dataset.parquet,
    # silently clipping the sec_fundamentals_daily date axis 121 days into
    # the past — the live pipeline then warned "fundamentals feed STALE"
    # every run even though the SEC refresh itself was working.
    existing = [data_dir / name for name in DEFAULT_ALPHA_CANDIDATES
                if (data_dir / name).exists()]
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)
    return data_dir / DEFAULT_ALPHA_CANDIDATES[0]


def load_daily_index(alpha_path: str | Path) -> pd.DatetimeIndex:
    alpha = pd.read_parquet(alpha_path, columns=["date"]).drop_duplicates()
    dates = pd.to_datetime(alpha["date"])
    return pd.DatetimeIndex(sorted(dates.unique()))


def _read_price_calendar(path: Path) -> pd.DatetimeIndex | None:
    """Read just the trading-date axis of one OHLCV ``1d.parquet`` file.

    Mirrors :func:`_read_price_series` date handling: the date is either a
    ``date`` column or the (datetime) index, depending on how the OHLCV cache
    was written.
    """
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path, columns=["date"])
        index = pd.to_datetime(frame["date"])
    except (KeyError, ValueError):
        # No ``date`` column -> dates live in the parquet index. Read a single
        # cheap column and use its index for the calendar.
        frame = pd.read_parquet(path)
        index = pd.to_datetime(frame.index)
    return pd.DatetimeIndex(index).dropna()


def load_price_calendar_index(
    ohlcv_dir: str | Path,
    tickers: Sequence[str],
) -> pd.DatetimeIndex:
    """Build the SERVING daily date axis from the OHLCV price calendar.

    The fundamentals feed is a SERVING artifact whose features
    (``book_to_price`` etc.) are price-dependent and computable to the latest
    price date. Historically the daily axis was bound to the alpha158 training
    dataset, which drops its last ~60 trading days because ``fwd_60d_excess``
    is unlabeled there. That training-label clip leaked into the live feed,
    pinning it ~88 calendar days behind the latest price and making the
    P-FUND-FRESHNESS gate structurally unsatisfiable.

    Deriving the axis from the OHLCV trading calendar (the union of trading
    dates across the universe's price files) decouples serving from the
    training label clip: the feed reaches the latest price date while the
    alpha158 training panel — which LEFT-joins the feed on its own clipped
    dates — is unaffected.

    Returns an empty index when no OHLCV calendar is available so callers can
    fall back to the alpha-derived axis.
    """
    ohlcv_dir = Path(ohlcv_dir).expanduser().resolve()
    collected: list[pd.DatetimeIndex] = []
    for ticker in tickers:
        idx = _read_price_calendar(ohlcv_dir / str(ticker).upper() / "1d.parquet")
        if idx is not None and len(idx):
            collected.append(idx)
    if not collected:
        return pd.DatetimeIndex([])
    union = collected[0]
    for idx in collected[1:]:
        union = union.union(idx)
    return pd.DatetimeIndex(sorted(union.unique()))


def resolve_serving_daily_index(
    *,
    data_dir: str | Path,
    universe: Sequence[str],
    alpha_path: str | Path | None = None,
) -> pd.DatetimeIndex:
    """Resolve the SERVING daily date axis for the fundamentals feed.

    Prefers the OHLCV price calendar (fresh to the latest price date) so the
    live feed is NOT clipped to the alpha158 training dataset's
    ``fwd_60d_excess`` label horizon. Falls back to the alpha-derived axis only
    when no OHLCV calendar is available (e.g. an explicit ``alpha_path`` is
    supplied without an OHLCV cache), preserving prior behaviour for those
    callers.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    calendar = load_price_calendar_index(data_dir / "ohlcv", universe)
    if len(calendar):
        return calendar
    log.warning(
        "no OHLCV price calendar under %s for the serving fundamentals axis; "
        "falling back to the alpha158-derived (training-clipped) date axis",
        data_dir / "ohlcv",
    )
    return load_daily_index(resolve_alpha_path(data_dir, alpha_path))


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
        headers=sec_headers(),
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
    response = client.get(TICKER_CIK_URL, headers=sec_headers(), timeout=30)
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


def load_fmp_accepted_dates(harvest_dir: str | Path) -> dict[tuple[str, pd.Timestamp], pd.Timestamp]:
    """PIT availability lookup ``{(ticker, fiscal_period_end) -> available date}``
    from the FMP fundamentals harvest's income statements.

    C2 same-filing assumption (renquant-orchestrator
    doc/design/2026-07-02-m-sig-signal-stack-spec.md): the income statement's
    ``acceptedDate`` is the EDGAR acceptance timestamp of the WHOLE filing
    (10-K/10-Q), so it stamps availability for every fundamental concept of the
    same (ticker, fiscal-period end) — balance-sheet items arrive in the same
    filing.

    Day-granularity PIT rule — never precede real availability: the stamp is
    ``max(date(acceptedDate), filingDate)``. A post-close acceptance (e.g.
    18:08 ET) is disseminated the NEXT business day, which FMP already encodes
    as ``filingDate`` > ``date(acceptedDate)``; taking the max never stamps a
    filing earlier than either field. Rows whose stamp precedes the fiscal
    period end (impossible → corrupt vendor row) are DROPPED, not clamped, so
    they fall through to the conservative ``FILING_LAG_FALLBACK_DAYS`` tier.
    Duplicate (ticker, period) rows keep the LATEST stamp (never-precedes).

    Missing directory / files / columns -> empty lookup (the caller's fallback
    tier still applies; this loader must never fail the refresh)."""
    harvest_dir = Path(harvest_dir).expanduser()
    out: dict[tuple[str, pd.Timestamp], pd.Timestamp] = {}
    if not harvest_dir.is_dir():
        return out
    for path in sorted(harvest_dir.glob(FMP_INCOME_STATEMENT_GLOB)):
        try:
            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - corrupt harvest must not kill the refresh
            log.warning("FMP accepted-dates: unreadable %s; skipping", path)
            continue
        if "symbol" not in frame.columns or "date" not in frame.columns \
                or "acceptedDate" not in frame.columns:
            continue
        symbols = frame["symbol"].astype(str).str.upper()
        period_end = pd.to_datetime(frame["date"], errors="coerce")
        accepted = pd.to_datetime(frame["acceptedDate"], errors="coerce").dt.normalize()
        if "filingDate" in frame.columns:
            filing = pd.to_datetime(frame["filingDate"], errors="coerce")
            available = pd.concat([accepted, filing], axis=1).max(axis=1)
        else:
            available = accepted
        ok = period_end.notna() & available.notna() & (available >= period_end)
        n_dropped = int((period_end.notna() & available.notna() & (available < period_end)).sum())
        if n_dropped:
            log.warning(
                "FMP accepted-dates: dropped %d row(s) in %s whose availability "
                "precedes the fiscal-period end (corrupt; fallback tier applies)",
                n_dropped, path.name,
            )
        for symbol, end, avail in zip(symbols[ok], period_end[ok], available[ok]):
            key = (symbol, end)
            prev = out.get(key)
            if prev is None or avail > prev:
                out[key] = avail
    return out


def build_quarterly_panel(
    raw: pd.DataFrame,
    cik_to_ticker: dict[int, str] | None = None,
    *,
    accepted_dates: dict[tuple[str, pd.Timestamp], pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Pivot SEC frames data to one PIT row per ticker and period end.

    AVAILABILITY TIERS for ``available_date`` (each row records its tier in
    ``available_source``; a tier is only used when every earlier tier has no
    genuine timestamp):
      1. ``sec_filed``      max ``filed`` date over the concepts whose values the
                            row carries — the direct SEC statement of when the
                            filing landed (frames-API responses usually lack it).
      2. ``fmp_accepted``   the FMP income-statement ``acceptedDate`` join for the
                            same (ticker, fiscal-period end) — the C2 same-filing
                            assumption (see :func:`load_fmp_accepted_dates`).
      3. ``expected_filing_lag``  period end + ``FILING_LAG_FALLBACK_DAYS`` — the
                            conservative EXPECTED-availability assumption (never
                            zero-lag) used only when no genuine timestamp exists.
    """
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
        accepted = (accepted_dates or {}).get((str(ticker), pd.Timestamp(end_date)))
        if selected_filed:
            row["available_date"] = max(selected_filed)
            row["available_source"] = "sec_filed"
        elif accepted is not None:
            row["available_date"] = pd.Timestamp(accepted)
            row["available_source"] = "fmp_accepted"
        else:
            row["available_date"] = end_date + pd.Timedelta(days=FILING_LAG_FALLBACK_DAYS)
            row["available_source"] = "expected_filing_lag"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ticker", "end"]).reset_index(drop=True)


def forward_fill_to_daily(
    quarterly: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    tickers: Sequence[str],
    *,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    """As-of forward-fill the quarterly panel onto the daily serving axis.

    Each daily row carries the values of the LATEST filing whose
    ``available_date`` is on/before that row's ``date`` (``merge_asof``
    backward — the PIT join). When the quarterly panel has ``end`` /
    ``available_date`` / ``available_source``, they are carried ADDITIVELY as
    the per-row provenance columns ``fiscal_period_end`` / ``available_at`` /
    ``available_source``, so every daily row states WHICH fiscal period it
    reflects and WHEN that filing became available. Rows before a ticker's
    first filing keep NaT provenance (nothing is available yet)."""
    if quarterly.empty:
        return pd.DataFrame()
    out: list[pd.DataFrame] = []
    dates = pd.DataFrame({"date": pd.DatetimeIndex(daily_index).sort_values()})
    cols = [col for col in value_cols if col in quarterly.columns]
    has_period_end = "end" in quarterly.columns
    for ticker in tickers:
        sort_cols = ["available_date", "end"] if has_period_end else ["available_date"]
        ticker_q = quarterly[quarterly["ticker"] == str(ticker).upper()].sort_values(sort_cols)
        if ticker_q.empty:
            continue
        updates = ticker_q[["available_date", *cols]].copy()
        if has_period_end:
            updates["fiscal_period_end"] = pd.to_datetime(ticker_q["end"].to_numpy())
        updates["available_at"] = pd.to_datetime(ticker_q["available_date"].to_numpy())
        if "available_source" in ticker_q.columns:
            updates["available_source"] = ticker_q["available_source"].to_numpy()
        updates = updates.rename(columns={"available_date": "date"})
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
        # Carry the per-row fiscal provenance (ADDITIVE) so the serving feed
        # states which fiscal period each row reflects and when it became
        # available — the columns the P-FUND-FRESHNESS / promote-gate quarterly
        # verification requires to exist per entity.
        for col in (*PROVENANCE_COLS, "available_source"):
            if col in merged.columns:
                result[col] = merged[col]
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


def validate_pit_provenance(frame: pd.DataFrame) -> None:
    """Fail CLOSED on any point-in-time violation in the provenance columns.

    Invariants (rows with NaT provenance — before a ticker's first filing —
    are exempt; the gates count those entities as MISSING, never as fresh):
      1. ``available_at <= date``: a row must never carry values from a filing
         that was not yet available on that serving date.
      2. ``fiscal_period_end <= available_at``: a filing cannot be available
         before the fiscal period it reports on has ended.
    """
    if "available_at" not in frame.columns:
        return
    date = pd.to_datetime(frame["date"], errors="coerce")
    available = pd.to_datetime(frame["available_at"], errors="coerce")
    lookahead = available.notna() & date.notna() & (available > date)
    if bool(lookahead.any()):
        raise RuntimeError(
            f"PIT violation: {int(lookahead.sum())} row(s) carry an available_at "
            "AFTER the serving date (look-ahead); refusing to write the feed"
        )
    if "fiscal_period_end" in frame.columns:
        period_end = pd.to_datetime(frame["fiscal_period_end"], errors="coerce")
        impossible = available.notna() & period_end.notna() & (available < period_end)
        if bool(impossible.any()):
            raise RuntimeError(
                f"PIT violation: {int(impossible.sum())} row(s) claim availability "
                "BEFORE their fiscal-period end (impossible); refusing to write the feed"
            )


def resolve_fmp_harvest_dir(
    data_dir: str | Path, fmp_harvest_dir: str | Path | None = None
) -> Path:
    if fmp_harvest_dir is not None:
        return Path(fmp_harvest_dir).expanduser()
    return Path(data_dir).expanduser() / DEFAULT_FMP_HARVEST_DIRNAME


def build_daily_fundamentals(
    *,
    raw: pd.DataFrame,
    universe: Sequence[str],
    cik_to_ticker: dict[int, str] | None,
    data_dir: str | Path,
    alpha_path: str | Path | None = None,
    output_path: str | Path | None = None,
    fmp_harvest_dir: str | Path | None = None,
) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_DAILY_OUTPUT
    accepted_dates = load_fmp_accepted_dates(resolve_fmp_harvest_dir(data_dir, fmp_harvest_dir))
    quarterly = build_quarterly_panel(raw, cik_to_ticker, accepted_dates=accepted_dates)
    daily_index = resolve_serving_daily_index(
        data_dir=data_dir, universe=universe, alpha_path=alpha_path
    )
    daily_raw = forward_fill_to_daily(quarterly, daily_index, universe, value_cols=RAW_VALUE_COLS)
    features = compute_derived_features(daily_raw, data_dir / "ohlcv")
    if features.empty:
        raise RuntimeError("SEC daily fundamentals produced no feature rows")
    validate_pit_provenance(features)
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
    fmp_harvest_dir: str | Path | None = None,
) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_EXTENDED_OUTPUT
    accepted_dates = load_fmp_accepted_dates(resolve_fmp_harvest_dir(data_dir, fmp_harvest_dir))
    quarterly = build_quarterly_panel(raw, cik_to_ticker, accepted_dates=accepted_dates)
    quarterly_ext = compute_extended_quarterly_features(quarterly)
    daily_index = resolve_serving_daily_index(
        data_dir=data_dir, universe=universe, alpha_path=alpha_path
    )
    daily_ext = forward_fill_to_daily(quarterly_ext, daily_index, universe, value_cols=EXTENDED_FEATURE_COLS)
    if daily_ext.empty:
        raise RuntimeError("SEC extended fundamentals produced no feature rows")
    daily_ext = robust_zscore_train_window(daily_ext, cols=EXTENDED_FEATURE_COLS, train_end=train_end)
    validate_pit_provenance(daily_ext)
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
            fmp_harvest_dir=ctx.config.fmp_harvest_dir,
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
            fmp_harvest_dir=ctx.config.fmp_harvest_dir,
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
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "SEC EDGAR requests identify with SEC_USER_AGENT. "
            f"Set {SEC_USER_AGENT_ENV}='RenQuant ops@example.com' in production."
        ),
    )
    parser.add_argument("--mode", choices=["daily", "extended", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--alpha-path", type=Path, default=None)
    parser.add_argument("--daily-output", type=Path, default=None)
    parser.add_argument("--extended-output", type=Path, default=None)
    parser.add_argument(
        "--fmp-harvest-dir", type=Path, default=None,
        help="FMP fundamentals harvest dir whose income-statement acceptedDate "
             f"backfills available_at (default: <data-dir>/{DEFAULT_FMP_HARVEST_DIRNAME})",
    )
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
        fmp_harvest_dir=args.fmp_harvest_dir,
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
