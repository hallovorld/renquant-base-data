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

# 2026-07-11 ratio-coverage fix (orchestrator PR #475 META attribution finding:
# earnings_yield/book_to_price/gross_profitability finite for only 67/70/317 of
# 826 universe members; the panel model was valuation-blind on most of the
# universe and silently median-imputed). Multiple XBRL tags carry the SAME
# economic concept, varying by issuer and accounting era (the
# ``sec_edgar_companyfacts_harvester.CANONICAL_CONCEPTS`` precedent):
#
#   * shares — most filers do NOT tag ``us-gaap:CommonStockSharesOutstanding``
#     outside the 10-K balance sheet, and multi-class filers (META, GOOGL, …)
#     tag it per-class WITH dimensions, which the frames API excludes entirely.
#     Weighted-average share counts from the EPS block are tagged consolidated
#     and quarter-aligned by virtually everyone (945/992 of the 2026-07
#     universe vs 654 for the primary tag).
#   * gross profit — issuers that do not present a gross-profit subtotal
#     (banks, META, AMZN, NFLX, …) never tag ``GrossProfit`` (400/992);
#     revenue − cost-of-revenue reconstructs it where both are tagged.
#   * revenue — ASC 606 adopters moved off plain ``Revenues`` onto
#     ``RevenueFromContractWithCustomer*`` (harvester precedent).
#   * equity — some filers only tag the including-noncontrolling-interest
#     total.
#
# Fallbacks are fetched AFTER the primary concepts so a fetch-budget
# exhaustion degrades to pre-fix coverage instead of hurting primaries.
FALLBACK_CONCEPTS: tuple[ConceptSpec, ...] = (
    ("CommonStockSharesIssued", "us-gaap", "shares", "instant"),
    ("WeightedAverageNumberOfDilutedSharesOutstanding", "us-gaap", "shares", "duration"),
    ("WeightedAverageNumberOfSharesOutstandingBasic", "us-gaap", "shares", "duration"),
    ("RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap", "USD", "duration"),
    ("CostOfRevenue", "us-gaap", "USD", "duration"),
    ("CostOfGoodsAndServicesSold", "us-gaap", "USD", "duration"),
    ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "us-gaap", "USD", "instant"),
)
DAILY_FETCH_CONCEPTS: tuple[ConceptSpec, ...] = BASE_CONCEPTS + FALLBACK_CONCEPTS
BOTH_FETCH_CONCEPTS: tuple[ConceptSpec, ...] = EXTENDED_CONCEPTS + FALLBACK_CONCEPTS

# Per-ratio tag chains — FIRST finite value wins and the PRIMARY (legacy) tag
# leads every chain, so any (ticker, period) already served by the primary tag
# keeps its exact pre-fix value (behavior-additive by construction). The
# weighted-average tags approximate point-in-time shares outstanding for
# market cap; they are last in the chain and only used when no point-in-time
# share count is tagged non-dimensionally at all. ``dei`` cover-page shares
# are deliberately NOT used: their instant dates are filing-cover dates, not
# fiscal-period ends, and would smear the fiscal provenance columns.
SHARES_TAG_CHAIN = (
    "CommonStockSharesOutstanding",
    "CommonStockSharesIssued",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)
EQUITY_TAG_CHAIN = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
REVENUE_TAG_CHAIN = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
)
COST_OF_REVENUE_TAG_CHAIN = (
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
)

RAW_VALUE_COLS = tuple(concept for concept, *_ in DAILY_FETCH_CONCEPTS)
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

# IMPUTED-SHARE guard — the daily feed build stamps per-feature finite-coverage
# fractions into a fingerprinted per-run ingestion manifest (the
# crypto_bars / sleeve_bars pattern: ONE ``manifest_fingerprint`` impl), and
# ``--verify`` fails when coverage drops below the configured floor. Field
# vocabulary (``coverage`` / ``n_have`` / ``n_expected`` / ``min_coverage``)
# is aligned with the renquant-pipeline DataAvailabilityGate
# ``data_contracts.v1`` axis contracts so the gate can consume this manifest
# without translation. The BUILD itself only WARNS below floor
# (``degrade_with_alarm`` — a coverage regression must alarm, not kill the
# weekly refresh); the verify CLI is the fail-closed surface.
DAILY_DATASET_ID = "sec-fundamentals-daily"
DAILY_MANIFEST_SCHEMA_VERSION = "sec-fundamentals-manifest-v1"
DAILY_MANIFEST_FILENAME = "ingestion_manifest_sec_fundamentals_daily.json"
DAILY_PROVIDER = "sec-edgar-frames"
# earnings_yield / book_to_price need a SAME-DAY close (market cap), so their
# coverage is measured against the PRICED tickers on the last serving date —
# an OHLCV price-cache outage is the ohlcv dataset's own contract
# (renquant-pipeline data_contracts ``ohlcv_bars`` axis), not a fundamentals
# regression. The price-independent features measure against every served
# ticker.
PRICE_DEPENDENT_FEATURE_COLS = ("earnings_yield", "book_to_price")
# Floors sit between the PRE-fix bug level and the POST-fix measured coverage
# (2026-07-11 local rebuild, last serving session 2026-07-10, 831 served /
# 131 priced tickers — pre -> post on each feature's own denominator:
#   earnings_yield       0.52 -> 0.91  (of priced)
#   book_to_price        0.54 -> 0.99  (of priced)
#   gross_profitability  0.38 -> 0.61  (of served)
#   roe                  0.84 -> 0.93  (of served)
#   asset_growth         1.00 -> 1.00  (of served, production feed)
# ) so the guard trips on a coverage REGRESSION, not on filing-season jitter.
# gross_profitability's floor is lowest because issuers with no gross-profit
# presentation and no cost-of-revenue tagging (banks/insurers) legitimately
# have none.
DEFAULT_FEATURE_COVERAGE_FLOORS: dict[str, float] = {
    "earnings_yield": 0.60,
    "book_to_price": 0.60,
    "gross_profitability": 0.50,
    "roe": 0.60,
    "asset_growth": 0.60,
}


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

    @property
    def cik_tickers(self) -> dict[int, tuple[str, ...]]:
        """CIK -> ALL universe tickers filed under it.

        Dual-class listings (GOOG/GOOGL, FOX/FOXA, NWS/NWSA, UA/UAA, …) share
        one SEC filer CIK. The scalar ``cik_ticker`` map is last-wins, which
        silently dropped one share class of every such pair from the feed
        (2026-07-11: 8 of 10 dual-class universe pairs had one class entirely
        absent). Sorted for determinism; the previously-winning ticker keeps
        byte-identical rows because both classes carry the SAME filer facts.
        """
        out: dict[int, list[str]] = {}
        for ticker, cik in self.ticker_cik.items():
            out.setdefault(cik, []).append(ticker)
        return {cik: tuple(sorted(tickers)) for cik, tickers in out.items()}


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


def _normalize_cik_map(
    cik_to_ticker: "dict[int, str | Sequence[str]]",
) -> dict[int, tuple[str, ...]]:
    """Normalize a CIK map to multi-ticker tuples (dual-class listings share
    one filer CIK; scalar values are the legacy single-ticker form)."""
    out: dict[int, tuple[str, ...]] = {}
    for cik, tickers in cik_to_ticker.items():
        if isinstance(tickers, str):
            out[int(cik)] = (tickers,)
        else:
            out[int(cik)] = tuple(tickers)
    return out


def build_quarterly_panel(
    raw: pd.DataFrame,
    cik_to_ticker: "dict[int, str | Sequence[str]] | None" = None,
    *,
    accepted_dates: dict[tuple[str, pd.Timestamp], pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Pivot SEC frames data to one PIT row per ticker and period end.

    ``cik_to_ticker`` may map a CIK to one ticker (legacy) or to every
    universe ticker filed under it; multi-ticker CIKs get one identical
    row set per ticker (dual-class share classes carry the same filer facts).

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
    cik_map = _normalize_cik_map(cik_to_ticker) if cik_to_ticker else None
    if "ticker" not in frame.columns:
        if not cik_map:
            raise ValueError("raw SEC frame requires ticker column or cik_to_ticker map")
        frame["ticker"] = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64").map(cik_map)
    elif cik_map and "cik" in frame.columns:
        mapped = pd.to_numeric(frame["cik"], errors="coerce").astype("Int64").map(cik_map)
        frame["ticker"] = frame["ticker"].fillna(mapped)
    # Multi-ticker CIKs (dual-class listings) fan out to one row per ticker;
    # scalar ticker values pass through ``explode`` unchanged.
    frame = frame.explode("ticker")
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
    carry_forward_within_ticker: bool = False,
) -> pd.DataFrame:
    """As-of forward-fill the quarterly panel onto the daily serving axis.

    Each daily row carries the values of the LATEST filing whose
    ``available_date`` is on/before that row's ``date`` (``merge_asof``
    backward — the PIT join). When the quarterly panel has ``end`` /
    ``available_date`` / ``available_source``, they are carried ADDITIVELY as
    the per-row provenance columns ``fiscal_period_end`` / ``available_at`` /
    ``available_source``, so every daily row states WHICH fiscal period it
    reflects and WHEN that filing became available. Rows before a ticker's
    first filing keep NaT provenance (nothing is available yet).

    ``carry_forward_within_ticker`` fixes the WHOLE-ROW-WIPE coverage bug
    (2026-07-11, orchestrator PR #475): the as-of join takes the entire latest
    filing row, so a concept the newest filing did NOT tag (e.g.
    ``CommonStockSharesOutstanding``, which most filers only tag in the 10-K)
    erased the previously-known value and NaN'd every ratio built on it —
    the dominant cause of the 67/826 ``earnings_yield`` coverage hole. With
    the flag on, each VALUE column carries its last known value forward
    across the ticker's filing history before the as-of join. PIT-safe by
    construction: rows are sorted by ``available_date``, so a carried value
    was available strictly BEFORE the row that inherits it. Cells that
    already had a value are untouched (behavior-additive); provenance
    columns always describe the LATEST filing and are never carried. The
    DAILY feature feed opts in; the extended z-scored feed does NOT (its
    train-window z-parameters must not move)."""
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
        if carry_forward_within_ticker and cols:
            # ffill BEFORE the same-date dedup so a kept row inherits values
            # from a dropped same-date sibling as well.
            updates[cols] = updates[cols].ffill()
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


def _coalesce_series(frame: pd.DataFrame, tag_chain: Sequence[str]) -> pd.Series:
    """Row-wise first-finite-wins across a fallback tag chain.

    The chain's FIRST (primary) tag wins wherever it has a value, so every
    row previously served by the primary tag is byte-identical post-fix;
    later tags only fill rows the earlier tags left NaN."""
    result = _numeric_series(frame, tag_chain[0])
    for name in tag_chain[1:]:
        result = result.where(result.notna(), _numeric_series(frame, name))
    return result


def compute_derived_features(daily_raw: pd.DataFrame, ohlcv_dir: str | Path) -> pd.DataFrame:
    """Compute market-cap-normalized daily fundamental features.

    Ratio inputs resolve through the per-concept fallback tag chains
    (``SHARES_TAG_CHAIN`` etc. — see the FALLBACK_CONCEPTS rationale):
    ``gross_profitability`` falls back to revenue − cost-of-revenue when the
    issuer never tags a ``GrossProfit`` subtotal. Rows fully served by the
    primary tags are unchanged."""
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
        revenue = _coalesce_series(merged, REVENUE_TAG_CHAIN)
        cost_of_revenue = _coalesce_series(merged, COST_OF_REVENUE_TAG_CHAIN)
        # revenue − cost is NaN unless BOTH legs are tagged (no partial math).
        gp = gp.where(gp.notna(), revenue - cost_of_revenue)
        assets = _numeric_series(merged, "Assets")
        equity = _coalesce_series(merged, EQUITY_TAG_CHAIN)
        shares = _coalesce_series(merged, SHARES_TAG_CHAIN)
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
        # ADDITIVE diagnostic column: the close used for market cap. Lets the
        # IMPUTED-SHARE guard separate a FUNDAMENTALS-side coverage regression
        # (this repo's contract) from an OHLCV price-cache outage: on
        # 2026-07-11 only ~150/2788 cached price files reached July (131 of
        # 831 served names priced on the last session), capping
        # earnings_yield/book_to_price on the full 1008-name watchlist
        # regardless of ratio-input coverage. Consumers select columns
        # explicitly (job_panel_scoring fund_cols; PROVENANCE_COLS precedent),
        # so the extra column is ignored downstream.
        result["price"] = _numeric_series(merged, "price")
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


def compute_feature_coverage(
    features: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = BASE_FEATURE_COLS,
    floors: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Per-feature FINITE coverage on the LAST serving date (what the panel
    scorer consumes; every non-finite cell is silently median-imputed
    downstream, so this fraction IS the imputed-share complement).

    Field names (``coverage`` / ``n_have`` / ``n_expected`` /
    ``min_coverage``) follow the renquant-pipeline DataAvailabilityGate
    ``data_contracts.v1`` axis vocabulary."""
    floors = DEFAULT_FEATURE_COVERAGE_FLOORS if floors is None else floors
    last_date = pd.to_datetime(features["date"]).max()
    last = features[pd.to_datetime(features["date"]) == last_date]
    n_served = int(last["ticker"].nunique())
    if "price" in last.columns:
        n_priced = int(np.isfinite(pd.to_numeric(last["price"], errors="coerce")).sum())
    else:
        # Pre-guard feeds carry no price column; fall back to the served
        # denominator (strictly larger, so this only makes the check STRICTER).
        n_priced = n_served
    per_feature: dict[str, Any] = {}
    for col in feature_cols:
        values = pd.to_numeric(last.get(col), errors="coerce") if col in last.columns \
            else pd.Series(dtype="float64")
        n_have = int(np.isfinite(values).sum())
        price_dependent = col in PRICE_DEPENDENT_FEATURE_COLS
        n_expected = n_priced if price_dependent else n_served
        coverage = (n_have / n_expected) if n_expected else 0.0
        floor = floors.get(col)
        per_feature[col] = {
            "coverage": round(coverage, 6),
            "n_have": n_have,
            "n_expected": n_expected,
            "denominator": "priced_tickers" if price_dependent else "served_tickers",
            "min_coverage": floor,
            "ok": bool(coverage >= floor) if floor is not None else True,
        }
    return {
        "serving_axis_max_date": str(pd.Timestamp(last_date).date()),
        "n_served": n_served,
        "n_priced": n_priced,
        "features": per_feature,
        "coverage_ok": all(entry["ok"] for entry in per_feature.values()),
    }


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_daily_ingestion_manifest(
    features: pd.DataFrame,
    *,
    output_path: Path,
    universe: Sequence[str],
    floors: dict[str, float] | None = None,
) -> Path:
    """Stamp the fingerprinted per-run ingestion manifest next to the daily
    feed (crypto_bars / sleeve_bars pattern; ONE ``manifest_fingerprint``
    impl). Coverage below a floor WARNS here — ``degrade_with_alarm``, the
    day-one DataAvailabilityGate default — and FAILS in ``verify_daily_feed``
    (the validation command declared by the registry manifest)."""
    import hashlib

    from .crypto_bars import manifest_fingerprint

    coverage = compute_feature_coverage(features, floors=floors)
    expected_universe = sorted(str(symbol).upper() for symbol in universe)
    expected_universe_hash = "sha256:" + hashlib.sha256(
        json.dumps(expected_universe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "dataset_id": DAILY_DATASET_ID,
        "schema_version": DAILY_MANIFEST_SCHEMA_VERSION,
        "asset_class": "us_equity",
        "provider": DAILY_PROVIDER,
        "uri": f"store://{output_path.name}",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "output_name": output_path.name,
        "content_sha256": _file_sha256(output_path),
        "expected_universe": expected_universe,
        "expected_universe_hash": expected_universe_hash,
        "n_expected_universe": len(expected_universe),
        "n_served": coverage["n_served"],
        "n_priced": coverage["n_priced"],
        "serving_axis_max_date": coverage["serving_axis_max_date"],
        "feature_coverage": coverage["features"],
        "coverage_ok": coverage["coverage_ok"],
        "coverage_policy": "degrade_with_alarm",
    }
    payload["fingerprint"] = manifest_fingerprint(payload)
    for name, entry in coverage["features"].items():
        if not entry["ok"]:
            log.warning(
                "IMPUTED-SHARE guard: %s finite coverage %.3f (%d/%d) is BELOW "
                "min_coverage %.2f on %s — the panel scorer median-imputes every "
                "missing cell silently; investigate the ratio inputs",
                name, entry["coverage"], entry["n_have"], entry["n_expected"],
                entry["min_coverage"], coverage["serving_axis_max_date"],
            )
    manifest_path = output_path.parent / DAILY_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def verify_daily_feed(
    *,
    data_dir: str | Path,
    daily_output: str | Path | None = None,
    floors: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Validation command for the ``sec-fundamentals-daily`` registry manifest.

    No network. FAILS (``ok=False``) on: missing feed/manifest, manifest
    fingerprint mismatch (tamper), content sha mismatch (manifest describes a
    different parquet), or any feature's finite coverage below its
    ``min_coverage`` floor."""
    from .crypto_bars import manifest_fingerprint

    data_dir = Path(data_dir).expanduser().resolve()
    feed_path = Path(daily_output).expanduser().resolve() if daily_output \
        else data_dir / DEFAULT_DAILY_OUTPUT
    manifest_path = feed_path.parent / DAILY_MANIFEST_FILENAME
    report: dict[str, Any] = {
        "dataset_id": DAILY_DATASET_ID,
        "feed": str(feed_path),
        "manifest": str(manifest_path),
        "checks": {},
        "ok": False,
    }
    if not feed_path.exists():
        report["error"] = "feed parquet missing"
        return report
    if not manifest_path.exists():
        report["error"] = "ingestion manifest missing (feed built pre-guard? rebuild stamps it)"
        return report
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint_ok = payload.get("fingerprint") == manifest_fingerprint(payload)
    report["checks"]["fingerprint_ok"] = fingerprint_ok
    content_ok = payload.get("content_sha256") == _file_sha256(feed_path)
    report["checks"]["content_sha256_ok"] = content_ok
    features = pd.read_parquet(feed_path)
    coverage = compute_feature_coverage(features, floors=floors)
    report["serving_axis_max_date"] = coverage["serving_axis_max_date"]
    report["feature_coverage"] = coverage["features"]
    report["checks"]["coverage_ok"] = coverage["coverage_ok"]
    report["ok"] = bool(fingerprint_ok and content_ok and coverage["coverage_ok"])
    return report


def build_daily_fundamentals(
    *,
    raw: pd.DataFrame,
    universe: Sequence[str],
    cik_to_ticker: "dict[int, str | Sequence[str]] | None",
    data_dir: str | Path,
    alpha_path: str | Path | None = None,
    output_path: str | Path | None = None,
    fmp_harvest_dir: str | Path | None = None,
    coverage_floors: dict[str, float] | None = None,
) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_DAILY_OUTPUT
    accepted_dates = load_fmp_accepted_dates(resolve_fmp_harvest_dir(data_dir, fmp_harvest_dir))
    quarterly = build_quarterly_panel(raw, cik_to_ticker, accepted_dates=accepted_dates)
    daily_index = resolve_serving_daily_index(
        data_dir=data_dir, universe=universe, alpha_path=alpha_path
    )
    daily_raw = forward_fill_to_daily(
        quarterly,
        daily_index,
        universe,
        value_cols=RAW_VALUE_COLS,
        carry_forward_within_ticker=True,
    )
    features = compute_derived_features(daily_raw, data_dir / "ohlcv")
    if features.empty:
        raise RuntimeError("SEC daily fundamentals produced no feature rows")
    validate_pit_provenance(features)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    write_daily_ingestion_manifest(
        features, output_path=out, universe=universe, floors=coverage_floors
    )
    return out


def build_extended_fundamentals(
    *,
    raw: pd.DataFrame,
    universe: Sequence[str],
    cik_to_ticker: "dict[int, str | Sequence[str]] | None",
    data_dir: str | Path,
    alpha_path: str | Path | None = None,
    output_path: str | Path | None = None,
    train_end: str | pd.Timestamp = "2022-11-01",
    fmp_harvest_dir: str | Path | None = None,
) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_EXTENDED_OUTPUT
    accepted_dates = load_fmp_accepted_dates(resolve_fmp_harvest_dir(data_dir, fmp_harvest_dir))
    # The extended z-scored feed stays PINNED to the pre-fix concept set: the
    # shared mode=both fetch now also carries FALLBACK_CONCEPTS rows, and any
    # extra (ticker, end) row or concept would move the train-window z-score
    # parameters (median/MAD) and thereby EVERY value in this feed. Filtering
    # keeps it byte-identical; the coverage fix is a daily-feed concern.
    if "concept" in raw.columns:
        extended_names = {concept for concept, *_ in EXTENDED_CONCEPTS}
        raw = raw[raw["concept"].isin(extended_names)]
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
        concepts = BOTH_FETCH_CONCEPTS if ctx.config.mode == "both" else DAILY_FETCH_CONCEPTS
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
            cik_to_ticker=ctx.cik_tickers,
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
            # Deliberately the LEGACY scalar map (not ``cik_tickers``): extra
            # dual-class rows in the z-scored feed would move the train-window
            # median/MAD and re-price every value. Byte-identity > coverage
            # here; the coverage fix targets the daily feature feed.
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
    parser.add_argument(
        "--verify", action="store_true",
        help="audit the existing daily feed + ingestion manifest (fingerprint, "
             "content sha256, per-feature finite-coverage floors); no network; "
             "exit 1 on any failed check — the registry manifest's "
             "validation_command",
    )
    parser.add_argument(
        "--coverage-floor", nargs="*", default=None, metavar="FEATURE=FRACTION",
        help="override per-feature min_coverage floors for --verify "
             "(e.g. earnings_yield=0.6); defaults: "
             f"{DEFAULT_FEATURE_COVERAGE_FLOORS}",
    )
    return parser


def parse_coverage_floors(pairs: Sequence[str] | None) -> dict[str, float] | None:
    if pairs is None:
        return None
    floors = dict(DEFAULT_FEATURE_COVERAGE_FLOORS)
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not _ or not name:
            raise SystemExit(f"--coverage-floor expects FEATURE=FRACTION, got {pair!r}")
        floors[name] = float(value)
    return floors


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    if args.verify:
        report = verify_daily_feed(
            data_dir=args.data_dir,
            daily_output=args.daily_output,
            floors=parse_coverage_floors(args.coverage_floor),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
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
