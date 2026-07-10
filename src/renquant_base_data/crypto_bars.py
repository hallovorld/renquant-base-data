"""Alpaca crypto bars ingestion with UTC-session watermarks (crypto RFC D-C2).

Implements the base-data slice of the merged crypto RFC
(orchestrator ``doc/design/2026-07-10-crypto-trading-rfc.md`` §2.3 B1/B2/B3/B5/
B6, §3.3, §3.5):

* **Provider seam (B1/B2)** — ``fetch_crypto_bars`` uses alpaca-py's
  ``CryptoHistoricalDataClient`` (api_version ``v1beta3``) +
  ``CryptoBarsRequest``. There is NO feed argument (crypto has no IEX/SIP
  split; the SDK defaults to ``CryptoFeed.US``). Credentials are the same
  ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` env vars as the equity intraday
  path, but are OPTIONAL — the crypto data API works unauthenticated (auth
  only raises the rate limit). The equity daily path
  (``loaders.data.fetch_ohlcv`` provider ``yfinance``) is untouched; the new
  provider is additive.

* **Symbol policy (B5, RFC §3.0)** — canonical PAIR form ``"BTC/USD"`` in
  configs and all broker/data API calls; canonical SLUG form ``"BTC-USD"``
  (slash→dash) for every file path, directory and cache key. The slug form
  coincides with yfinance's crypto ticker format, so the two-source
  cross-check needs no third form. ``pair_slug``/``slug_pair`` here are the
  repo-local stand-in for the shared helper the RFC places in
  renquant-common (deliverable D-C1, not yet merged); when D-C1 lands this
  module repoints to it — semantics are frozen by tests to be identical.

* **24/7 session semantics + watermark (B3, RFC §3.5)** — bars are keyed by
  UTC calendar day. Session D's signal may consume ONLY bars whose close
  timestamp is ≤ ``D 00:00:00 UTC`` (i.e. day D-1's full UTC day), and a bar
  is eligible only once this ingestion job has marked it **closed AND
  fetched** — not merely "timestamp has passed" — so a late-arriving vendor
  bar can never silently backfill into an already-frozen signal. Ingestion
  therefore stamps ``bar_close_utc`` on every stored row, drops in-progress
  bars (close > fetch time), and writes a fingerprinted ingestion manifest
  (sha256 over sorted-keys JSON; per-symbol content sha256 over canonical
  bar bytes) that the orchestrator's sealed-manifest contract (D-C11)
  consumes and re-verifies.

* **UTC-day keying is verified, not assumed** — Alpaca's 1Day aggregation
  boundary for crypto is not contractually documented as UTC midnight (the
  equity daily bars, for instance, are keyed to America/New_York). If the
  vendor's daily bars are not exactly UTC-midnight aligned,
  ``ingest_crypto_bars`` FAILS CLOSED on the vendor daily path and rebuilds
  daily bars deterministically by resampling vendor 1Hour bars into UTC
  calendar days. Either way the stored daily grid is UTC-midnight keyed by
  construction.

* **Freshness clock (B3)** — crypto freshness uses "last completed UTC day",
  the ALWAYS_OPEN stand-in until renquant-common ships the canonical
  always-open calendar (D-C1/M2). NYSE freshness logic is never consulted.

Boundaries: ingestion + feature groundwork ONLY. No labels (SPY-excess label
stays equity-only; crypto labels are D-C4/D-C3 scope), no decision logic, no
broker/order logic.

.. admonition:: TODO(D-C1 dependency, tracked — Codex review round 1 on #41)

   ``pair_slug``/``slug_pair``/``last_completed_utc_session`` below are a
   REPO-LOCAL DUPLICATE of the canonical ``renquant-common`` primitives the
   RFC assigns to deliverable D-C1 (``doc/design/2026-07-10-crypto-trading-
   rfc.md`` §7: "ALWAYS_OPEN calendar mode in ``market_calendar`` +
   ``pair_slug`` symbol helper"). D-C1 does not exist yet as a merged PR in
   any repo. This PR stays in DRAFT until it does, per Codex's explicit
   instruction — merging a "repoint later" duplicate would recreate exactly
   the duplicated-contract class the architecture-compliance audit
   (orchestrator#454) was merged to eliminate.

   **Required follow-up once D-C1 lands** (not optional cleanup):
   1. Delete the local ``pair_slug``/``slug_pair``/``last_completed_utc_
      session`` implementations here; import the ``renquant-common``
      versions instead.
   2. Add a cross-repo parity test asserting the two are behaviorally
      identical (or, once deleted, that this module's own tests still pass
      unchanged against the imported common helper — proving the semantics
      really were frozen, not just documented as such).
   3. Only then may this PR come out of draft.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .loaders.data import LocalStore

log = logging.getLogger("renquant_base_data.crypto_bars")

CRYPTO_OHLCV_DIRNAME = "crypto_ohlcv"
MANIFEST_SCHEMA_VERSION = "crypto-ohlcv-manifest-v1"
PROVIDER = "alpaca:v1beta3"

# Repo-root-anchored default store (same rationale as loaders.data
# _REPO_ROOT_OHLCV: cwd-relative defaults silently point at truncated stores).
_REPO_ROOT_CRYPTO_OHLCV = Path(__file__).resolve().parents[3] / "data" / CRYPTO_OHLCV_DIRNAME

# Alpaca timeframe string -> store filename stem. Daily matches the equity
# "1d.parquet" convention; intraday stems are new (equities never persisted
# intraday bars).
TIMEFRAME_FILES = {
    "1Day": "1d",
    "1Hour": "1h",
    "30Min": "30min",
    "15Min": "15min",
    "5Min": "5min",
    "1Min": "1min",
}

_TIMEFRAME_DELTAS = {
    "1Day": pd.Timedelta(days=1),
    "1Hour": pd.Timedelta(hours=1),
    "30Min": pd.Timedelta(minutes=30),
    "15Min": pd.Timedelta(minutes=15),
    "5Min": pd.Timedelta(minutes=5),
    "1Min": pd.Timedelta(minutes=1),
}

BAR_CLOSE_COL = "bar_close_utc"
_CONTENT_SHA_COLS = ["open", "high", "low", "close", "volume", BAR_CLOSE_COL]


# ---------------------------------------------------------------------------
# Symbol policy (RFC §3.0) — local stand-in for renquant-common D-C1
# ---------------------------------------------------------------------------

def pair_slug(pair: str) -> str:
    """Canonical pair form -> canonical slug form: ``"BTC/USD"`` -> ``"BTC-USD"``.

    Strict by design: the input must be pair form (exactly one ``/``, both
    sides non-empty, no ``-``). Malformed symbols raise instead of silently
    producing a colliding cache key (gap B5: ``"BTC/USD"`` used as a path
    creates a nested ``BTC/USD/`` directory).
    """
    p = str(pair).strip().upper()
    if p.count("/") != 1 or "-" in p:
        raise ValueError(f"not a canonical crypto pair (expected 'BASE/QUOTE'): {pair!r}")
    base, _, quote = p.partition("/")
    if not base or not quote:
        raise ValueError(f"not a canonical crypto pair (expected 'BASE/QUOTE'): {pair!r}")
    return f"{base}-{quote}"


def slug_pair(slug: str) -> str:
    """Canonical slug form -> canonical pair form: ``"BTC-USD"`` -> ``"BTC/USD"``.

    Exact inverse of :func:`pair_slug`; round-trip is pinned by tests.
    """
    s = str(slug).strip().upper()
    if s.count("-") != 1 or "/" in s:
        raise ValueError(f"not a canonical crypto slug (expected 'BASE-QUOTE'): {slug!r}")
    base, _, quote = s.partition("-")
    if not base or not quote:
        raise ValueError(f"not a canonical crypto slug (expected 'BASE-QUOTE'): {slug!r}")
    return f"{base}/{quote}"


def _as_pair(symbol: str) -> str:
    """Accept either canonical form, return validated pair form."""
    s = str(symbol).strip().upper()
    # Round-trip through the strict helpers so malformed symbols
    # (e.g. "BTC/USD/X") are rejected here, not deep in the store layer.
    return slug_pair(pair_slug(s)) if "/" in s else slug_pair(s)


def _as_slug(symbol: str) -> str:
    """Accept either canonical form, return slug form."""
    s = str(symbol).strip().upper()
    return pair_slug(s) if "/" in s else pair_slug(slug_pair(s))


# ---------------------------------------------------------------------------
# 24/7 session clock (RFC §3.5 / B3) — ALWAYS_OPEN stand-in until D-C1
# ---------------------------------------------------------------------------

def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _to_utc(ts) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def session_watermark_utc(session_date: "date | str | pd.Timestamp") -> pd.Timestamp:
    """The RFC §3.5 watermark for session D: ``D 00:00:00 UTC``.

    Session D's daily signal may consume price bars ONLY up through the bar
    that CLOSES at this instant (day D-1's full UTC day).
    """
    d = pd.Timestamp(session_date)
    if d.tzinfo is not None:
        d = d.tz_convert("UTC")
        if d != d.normalize():
            raise ValueError(f"session_date must be a calendar date, got {session_date!r}")
        return d
    if d != d.normalize():
        raise ValueError(f"session_date must be a calendar date, got {session_date!r}")
    return d.tz_localize("UTC")


def last_completed_utc_session(ref=None) -> date:
    """Last fully-completed UTC calendar day as of ``ref`` (default: now).

    ALWAYS_OPEN semantics: every UTC day is a session; day X is complete once
    ``X+1 00:00 UTC`` has been reached. Local stand-in for the canonical
    always-open calendar (renquant-common D-C1/M2).
    """
    ref_ts = _to_utc(ref) if ref is not None else _utc_now()
    return (ref_ts.normalize() - pd.Timedelta(days=1)).date()


def bars_eligible_for_session(df: pd.DataFrame, session_date) -> pd.DataFrame:
    """Filter bars to those consumable by session D per the RFC watermark.

    Eligible: ``bar_close_utc <= D 00:00:00 UTC``. A daily bar that closes
    exactly AT the watermark (day D-1's bar) IS eligible; any bar closing
    after it is not. Requires the ``bar_close_utc`` stamp this module's
    ingestion writes — being fetched/stamped is what "closed and fetched"
    means; a bar missing from the store is by definition ineligible.
    """
    if BAR_CLOSE_COL not in df.columns:
        raise ValueError(
            f"bars frame lacks the {BAR_CLOSE_COL!r} ingestion stamp; "
            "only stamped (closed-and-fetched) bars are session-eligible"
        )
    watermark = session_watermark_utc(session_date)
    closes = pd.to_datetime(df[BAR_CLOSE_COL])
    if closes.dt.tz is None:
        closes = closes.dt.tz_localize("UTC")
    else:
        closes = closes.dt.tz_convert("UTC")
    return df.loc[(closes <= watermark).to_numpy()]


class ManifestNotSignalEligibleError(RuntimeError):
    """A crypto ingestion manifest failed the RFC §3.5 signal-eligibility
    contract and must not back any session's Class-A (entry) decisions.
    Raised, never silently swallowed — callers must fail closed."""


#: RFC §3.5 frozen quiet-interval width: signal computation must consume a
#: manifest generated within [D 00:00, D 00:15) UTC, never later.
SIGNAL_FREEZE_CUTOFF_MINUTES = 15


def manifest_eligible_for_session(
    manifest: dict,
    session_date: "date | str | pd.Timestamp",
    df: "pd.DataFrame | None" = None,
    symbol: "str | None" = None,
    store: "CryptoLocalStore | None" = None,
) -> "pd.DataFrame | None":
    """RFC §3.5 manifest-BOUND eligibility (Codex review: bar-close
    timestamps alone are not a leakage proof).

    A raw dataframe's ``bar_close_utc`` column can look eligible for session
    D purely by chance (or by a late fetch that happens to land on the right
    calendar boundary) even when the manifest that produced it is stale,
    tampered, or missing part of the requested universe. This function is
    the mandatory gate a consumer (D-C11's session scheduler, not yet
    built) must pass THROUGH before touching bars at all — it verifies the
    manifest identity itself, not just row timestamps, and fails closed
    (raises :class:`ManifestNotSignalEligibleError`) on any violation:

    1. **Fingerprint integrity** — :func:`verify_crypto_manifest`; a
       tampered or corrupt manifest is never eligible.
    2. **Universe completeness** — ``manifest["universe_complete"]``; a
       partial fetch (any expected pair missing/failed) never backs a
       session regardless of what DID succeed.
    3. **Non-null watermark** — an empty or all-failed universe has
       nothing to certify.
    4. **Watermark match** — ``manifest["watermark_utc"]`` must equal
       session D's frozen watermark (``session_watermark_utc(D)``); a
       manifest certifying a different watermark is not this session's
       manifest, however recent it is.
    5. **Signal-freeze cutoff (RFC §3.5)** — ``generated_at_utc`` must fall
       in the half-open window ``[D 00:00:00 UTC, D 00:15:00 UTC)``. A
       manifest generated after the window closed is the exact "late fetch
       at noon labelled eligible because its bar closed at D 00:00"
       scenario Codex flagged — rejected regardless of bar timestamps,
       fingerprint validity, or universe completeness.
    6. **Content binding (Codex review, r2)** — checks 1-5 only prove the
       MANIFEST is internally consistent; they say nothing about whether
       the bars a caller consumes alongside it are the data the manifest
       actually sealed. An intact, untampered manifest could otherwise be
       paired with modified rows, or even another symbol's frame, and this
       function would wave it through. Bar consumption therefore requires a
       declared ``symbol`` (which manifest-sealed pair the bars claim to
       be; pair or slug form) plus EITHER the sealed store artifact
       (``store=`` — this function loads ``{SLUG}/{tf}.parquet`` itself)
       OR an explicit ``df``; either way the frame's canonical
       :func:`_content_sha256` must equal
       ``manifest["symbols"][pair]["content_sha256"]`` exactly — a
       tampered-row or wrong-symbol frame fails closed here, regardless of
       how it scores on checks 1-5.

    In bar-consumption mode (``symbol`` + ``df``/``store``), returns the
    content-verified bars eligible for session D (delegates to
    :func:`bars_eligible_for_session`) — but only once every check above
    has passed. With neither ``df`` nor ``store``, returns ``None``
    (callers that only need the eligibility verdict itself, e.g. to decide
    whether to proceed at all before loading any bars).
    """
    if not verify_crypto_manifest(manifest):
        raise ManifestNotSignalEligibleError(
            f"manifest fingerprint mismatch (tampered or corrupt): "
            f"{manifest.get('dataset_id')!r}"
        )
    if not manifest.get("universe_complete"):
        raise ManifestNotSignalEligibleError(
            "manifest universe incomplete — not every expected pair sealed "
            f"an 'ok' bar: expected={manifest.get('expected_universe')} "
            f"symbols={manifest.get('symbols')}"
        )
    watermark_raw = manifest.get("watermark_utc")
    if watermark_raw is None:
        raise ManifestNotSignalEligibleError(
            f"manifest has no watermark (empty or all-failed universe): "
            f"{manifest.get('dataset_id')!r}"
        )
    session_watermark = session_watermark_utc(session_date)
    manifest_watermark = _to_utc(watermark_raw)
    if manifest_watermark != session_watermark:
        raise ManifestNotSignalEligibleError(
            f"manifest watermark {manifest_watermark.isoformat()} does not match "
            f"session {session_date}'s frozen watermark {session_watermark.isoformat()}"
        )
    generated_at = manifest.get("generated_at_utc")
    if generated_at is None:
        raise ManifestNotSignalEligibleError(
            f"manifest lacks generated_at_utc: {manifest.get('dataset_id')!r}"
        )
    generated_at = _to_utc(generated_at)
    window_start = session_watermark
    window_end = session_watermark + pd.Timedelta(minutes=SIGNAL_FREEZE_CUTOFF_MINUTES)
    if not (window_start <= generated_at < window_end):
        raise ManifestNotSignalEligibleError(
            f"manifest generated_at_utc {generated_at.isoformat()} is outside the "
            f"frozen Class-A signal cutoff window [{window_start.isoformat()}, "
            f"{window_end.isoformat()}) for session {session_date} — a late-generated "
            "manifest cannot back this session's entries regardless of its bar "
            "timestamps (RFC §3.5)"
        )
    if df is None and store is None:
        return None
    if not symbol:
        raise ValueError(
            "manifest_eligible_for_session(df=/store=) requires symbol= too — "
            "the caller must declare which manifest-sealed pair it consumes, "
            "or content-binding cannot be checked at all"
        )
    pair = _as_pair(symbol)
    symbol_entry = manifest.get("symbols", {}).get(pair)
    if symbol_entry is None or symbol_entry.get("status") != "ok":
        raise ManifestNotSignalEligibleError(
            f"manifest has no sealed 'ok' entry for symbol {pair!r}: "
            f"{manifest.get('dataset_id')!r}"
        )
    if df is None:
        tf_file = TIMEFRAME_FILES.get(manifest.get("timeframe"))
        if tf_file is None:
            raise ManifestNotSignalEligibleError(
                f"manifest carries an unknown timeframe "
                f"{manifest.get('timeframe')!r}; cannot locate the sealed "
                "store artifact"
            )
        df = store.load(pair_slug(pair), tf_file)
        if df is None:
            raise ManifestNotSignalEligibleError(
                f"sealed store artifact missing or empty for {pair!r} "
                f"({pair_slug(pair)}/{tf_file}.parquet under {store.data_dir}) — "
                "the manifest certifies content this store does not hold"
            )
    sealed_hash = symbol_entry.get("content_sha256")
    actual_hash = _content_sha256(df)
    if not sealed_hash or actual_hash != sealed_hash:
        raise ManifestNotSignalEligibleError(
            f"bar content for {pair!r} does not match the manifest's "
            f"sealed content_sha256 (sealed={sealed_hash!r}, "
            f"actual={actual_hash!r}) — this is either a tampered/modified "
            f"frame or the wrong symbol's data, and cannot back this "
            f"session's signal regardless of its timestamps"
        )
    return bars_eligible_for_session(df, session_date)


# ---------------------------------------------------------------------------
# Store — slug paths under data/crypto_ohlcv (B5), UTC freshness (B3)
# ---------------------------------------------------------------------------

class CryptoLocalStore(LocalStore):
    """Parquet OHLCV store for crypto pairs.

    Layout: ``{data_dir}/{SLUG}/{timeframe}.parquet`` (e.g.
    ``data/crypto_ohlcv/BTC-USD/1d.parquet``). Accepts pair or slug symbol
    form; always writes slug paths, so the B5 nested-directory break is
    structurally impossible. Freshness is judged against the last completed
    UTC calendar day, never the NYSE calendar.
    """

    def __init__(self, data_dir: "Path | str | None" = None):
        super().__init__(data_dir if data_dir is not None else _REPO_ROOT_CRYPTO_OHLCV)

    def _path(self, symbol: str, timeframe: str = "1d") -> Path:
        return self.data_dir / _as_slug(symbol) / f"{timeframe}.parquet"

    def has_range(
        self,
        symbol: str,
        timeframe: str = "1d",
        start: "str | None" = None,
        end: "str | None" = None,
        tolerance_days: "int | None" = None,
    ) -> bool:
        """Coverage + freshness on the ALWAYS_OPEN (UTC-day) clock."""
        path = self._path(symbol, timeframe)
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        if df.empty:
            return False
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        idx_min = _to_utc(df.index.min())
        idx_max = _to_utc(df.index.max())
        if start is not None and idx_min > _to_utc(start):
            return False
        ref = _to_utc(end) if end is not None else _utc_now()
        if tolerance_days is not None:
            return idx_max >= ref - pd.Timedelta(days=tolerance_days)
        return idx_max.date() >= last_completed_utc_session(ref)


_default_crypto_store = CryptoLocalStore()


# ---------------------------------------------------------------------------
# Provider seam (B1/B2): Alpaca CryptoHistoricalDataClient, v1beta3
# ---------------------------------------------------------------------------

def fetch_crypto_bars(
    symbols: "list[str] | str",
    *,
    timeframe: str = "1Day",
    start: "datetime | None" = None,
    end: "datetime | None" = None,
    limit: "int | None" = None,
    timeout_sec: float = 30.0,
    client=None,
) -> dict[str, pd.DataFrame]:
    """Fetch crypto bars from Alpaca's v1beta3 crypto market-data API.

    ``timeframe`` is an Alpaca string (see ``TIMEFRAME_FILES``). Symbols may
    be pair ("BTC/USD") or slug ("BTC-USD") form; the API is always called
    with the pair form and the result dict is keyed by pair form. Returned
    frames have a tz-aware UTC ``DatetimeIndex`` of bar OPEN timestamps —
    raw vendor bars, no close stamp yet (that is ingestion's job).

    No feed argument exists on purpose: crypto has a single US feed (SDK
    default), unlike the equity intraday path's forced IEX. Credentials come
    from ``ALPACA_API_KEY``/``ALPACA_SECRET_KEY`` when present but are
    optional for crypto data. ``client`` is injectable for tests (recorded/
    fake payloads; CI never hits the live API).
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    pairs = [_as_pair(s) for s in symbols]
    if not pairs:
        return {}
    if timeframe not in TIMEFRAME_FILES:
        raise ValueError(
            f"Unknown crypto timeframe {timeframe!r}. Supported: {list(TIMEFRAME_FILES)}"
        )

    try:
        from alpaca.data.historical import CryptoHistoricalDataClient  # noqa: PLC0415
        from alpaca.data.requests import CryptoBarsRequest  # noqa: PLC0415
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without alpaca-py
        raise RuntimeError("alpaca-py not installed") from exc

    tf_map = {
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    }

    if client is None:
        client = CryptoHistoricalDataClient(
            api_key=os.environ.get("ALPACA_API_KEY"),
            secret_key=os.environ.get("ALPACA_SECRET_KEY"),
        )

    req = CryptoBarsRequest(
        symbol_or_symbols=pairs,
        timeframe=tf_map[timeframe],
        start=start,
        end=end,
        limit=limit,
    )

    from renquant_common.net_safety import call_with_timeout  # noqa: PLC0415

    bars = call_with_timeout(
        lambda: client.get_crypto_bars(req),
        timeout_sec=timeout_sec,
        label=f"alpaca.get_crypto_bars(n={len(pairs)}, tf={timeframe})",
    )
    if bars is None:
        log.warning("fetch_crypto_bars: Alpaca timeout after %.0fs — returning empty", timeout_sec)
        return {}
    df_all = bars.df

    out: dict[str, pd.DataFrame] = {}
    if df_all is None or df_all.empty:
        return out
    for pair in pairs:
        if pair in df_all.index.get_level_values(0):
            sub = df_all.xs(pair, level=0).copy()
            sub.index = pd.DatetimeIndex(
                [_to_utc(ts) for ts in sub.index], name=sub.index.name or "timestamp"
            )
            out[pair] = sub.sort_index()
    return out


# ---------------------------------------------------------------------------
# UTC-day keying (RFC §3.5): verify vendor alignment or rebuild from hourly
# ---------------------------------------------------------------------------

class VendorDailyNotUtcAlignedError(RuntimeError):
    """Vendor 1Day bars are not keyed at UTC midnight — the caller must not
    treat them as UTC-calendar-day bars (silent mis-keying would corrupt the
    §3.5 watermark contract). ``ingest_crypto_bars`` catches this and rebuilds
    daily bars from vendor 1Hour bars instead."""


def normalize_daily_bars_utc(
    df: pd.DataFrame, *, fetched_through_utc: pd.Timestamp
) -> pd.DataFrame:
    """Validate + stamp vendor daily bars as UTC-calendar-day bars.

    Requires every bar OPEN timestamp to sit exactly on UTC midnight
    (raises :class:`VendorDailyNotUtcAlignedError` otherwise — fail closed,
    never re-key silently). Stamps ``bar_close_utc = open + 1 day`` and drops
    any bar whose close is after ``fetched_through_utc`` (an in-progress
    vendor bar for the current UTC day is not "closed and fetched").
    """
    fetched_through = _to_utc(fetched_through_utc)
    if df.empty:
        out = df.copy()
        out[BAR_CLOSE_COL] = pd.Series(dtype="datetime64[ns, UTC]")
        return out
    out = df.copy()
    out.index = pd.DatetimeIndex([_to_utc(ts) for ts in out.index], name="timestamp")
    out = out.sort_index()
    misaligned = out.index[out.index != out.index.normalize()]
    if len(misaligned) > 0:
        raise VendorDailyNotUtcAlignedError(
            "vendor 1Day bars are not UTC-midnight aligned "
            f"(first offender: {misaligned[0].isoformat()}); refusing to key "
            "them as UTC calendar days — rebuild from 1Hour bars instead"
        )
    out[BAR_CLOSE_COL] = out.index + pd.Timedelta(days=1)
    return out.loc[out[BAR_CLOSE_COL] <= fetched_through]


def resample_hourly_to_utc_daily(
    hourly: pd.DataFrame, *, fetched_through_utc: pd.Timestamp
) -> pd.DataFrame:
    """Deterministically aggregate vendor 1Hour bars into UTC-calendar-day bars.

    open = first hourly open, high = max, low = min, close = last hourly
    close, volume = sum, ``n_source_bars`` = hourly bar count. Only days
    whose full window has elapsed AND been fetched (``day end <=
    fetched_through_utc``) are candidates — the current partial UTC day is
    never written to the daily store.

    **Completeness (Codex review, r1+r2)**: a day is emitted ONLY if it has
    BOTH (a) exactly 24 rows AND (b) its hour-of-day set equals all 24
    distinct UTC slots (``00:00``..``23:00``). Neither condition alone is
    sufficient: the row-count check alone misses a real gap padded by a
    duplicate row for some other hour, and the set-equality check alone
    (r1's fix) misses a duplicate hour that inflates the row count to 25
    while the SET of hours still covers all 24 — that duplicate would
    silently double-count its hour's volume in the aggregation above,
    reporting inflated volume for an otherwise-real day (Codex r2). "The
    fetch window has elapsed" is not a third substitute either. A daily
    candle is a scoring input; anything short of exactly-24-contiguous-
    hours is incomplete or duplicated market data, not a valid sealed day,
    regardless of the ``fetched_through`` cutoff. Incomplete/duplicated days
    are silently dropped from the output (never emitted with a misleadingly
    -low ``n_source_bars`` stamp, and never emitted with inflated volume
    from a masked duplicate) — the caller sees them disappear the same way
    a ``no_data``/``no_sealed_bars`` pair does elsewhere in this module's
    ``ingest_crypto_bars`` status reporting.
    """
    fetched_through = _to_utc(fetched_through_utc)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in hourly.columns]
    if missing:
        raise ValueError(f"hourly bars missing columns: {missing}")
    if hourly.empty:
        out = pd.DataFrame(columns=[*required, "n_source_bars", BAR_CLOSE_COL])
        out.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return out
    h = hourly.copy()
    h.index = pd.DatetimeIndex([_to_utc(ts) for ts in h.index], name="timestamp")
    h = h.sort_index()
    day = h.index.normalize()
    grouped = h.groupby(day)
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "n_source_bars": grouped["close"].count().astype("int64"),
        }
    )
    out.index.name = "timestamp"
    out[BAR_CLOSE_COL] = out.index + pd.Timedelta(days=1)
    out = out.loc[out[BAR_CLOSE_COL] <= fetched_through]

    full_utc_day_hours = frozenset(range(24))
    hour_sets = grouped.apply(lambda g: frozenset(g.index.hour))
    row_counts = grouped.size()
    # Both conditions required (Codex r2): the hour-SET check alone cannot
    # tell a genuinely-complete day from one where a duplicate row for hour
    # H masks a real gap at hour K (24 rows, but hour K is missing and H is
    # doubled) UNLESS combined with an exact row-count check — and the
    # row-count check alone cannot tell "24 distinct hours" from "24 rows
    # that happen to skip an hour and double another". Together they leave
    # no gap: exactly 24 rows AND all 24 distinct hours present.
    complete_days = hour_sets.index[
        (hour_sets == full_utc_day_hours) & (row_counts == 24)
    ]
    return out.loc[out.index.isin(complete_days)]


def _stamp_intraday_bars(
    df: pd.DataFrame, *, timeframe: str, fetched_through_utc: pd.Timestamp
) -> pd.DataFrame:
    """Stamp ``bar_close_utc = open + timeframe`` on intraday bars and drop
    bars not fully closed-and-fetched."""
    fetched_through = _to_utc(fetched_through_utc)
    delta = _TIMEFRAME_DELTAS[timeframe]
    if df.empty:
        out = df.copy()
        out[BAR_CLOSE_COL] = pd.Series(dtype="datetime64[ns, UTC]")
        return out
    out = df.copy()
    out.index = pd.DatetimeIndex([_to_utc(ts) for ts in out.index], name="timestamp")
    out = out.sort_index()
    out[BAR_CLOSE_COL] = out.index + delta
    return out.loc[out[BAR_CLOSE_COL] <= fetched_through]


# ---------------------------------------------------------------------------
# Ingestion manifest (RFC §3.5 sealed-data contract, B6)
# ---------------------------------------------------------------------------

def _content_sha256(df: pd.DataFrame) -> str:
    """Deterministic sha256 over the canonical bar content.

    Canonical form: sorted-by-index CSV of the OHLCV + bar-close columns with
    ISO-8601 UTC timestamps. Independent of parquet encoding, so a byte-level
    parquet rewrite with identical bars keeps the same fingerprint.
    """
    cols = [c for c in _CONTENT_SHA_COLS if c in df.columns]
    canon = df.sort_index()[cols].copy()
    canon.index = pd.DatetimeIndex([_to_utc(ts) for ts in canon.index]).strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    if BAR_CLOSE_COL in canon.columns:
        closes = pd.to_datetime(canon[BAR_CLOSE_COL])
        closes = closes.dt.tz_localize("UTC") if closes.dt.tz is None else closes.dt.tz_convert("UTC")
        canon[BAR_CLOSE_COL] = closes.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return hashlib.sha256(canon.to_csv().encode("utf-8")).hexdigest()


def manifest_fingerprint(payload: dict) -> str:
    """sha256 over the sorted-keys JSON of everything except ``fingerprint``."""
    body = {k: v for k, v in payload.items() if k != "fingerprint"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def verify_crypto_manifest(payload: dict) -> bool:
    """Re-derive the manifest fingerprint; ``False`` on any tamper/mismatch."""
    fp = payload.get("fingerprint")
    return bool(fp) and fp == manifest_fingerprint(payload)


def load_crypto_ingestion_manifest(path: "str | Path") -> dict:
    """Load an ingestion manifest, failing closed on fingerprint mismatch."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not verify_crypto_manifest(payload):
        raise ValueError(f"crypto ingestion manifest fingerprint mismatch: {path}")
    return payload


def crypto_manifest_path(store: CryptoLocalStore, timeframe: str) -> Path:
    return store.data_dir / f"ingestion_manifest_{TIMEFRAME_FILES[timeframe]}.json"


def _write_manifest_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Ingestion job
# ---------------------------------------------------------------------------

def ingest_crypto_bars(
    symbols: "list[str] | str",
    *,
    timeframe: str = "1Day",
    start: "datetime | None" = None,
    end: "datetime | None" = None,
    store: "CryptoLocalStore | None" = None,
    fetch_fn: "Callable[..., dict[str, pd.DataFrame]] | None" = None,
    timeout_sec: float = 30.0,
    write_manifest: bool = True,
    now_fn: "Callable[[], pd.Timestamp] | None" = None,
) -> dict:
    """Ingest crypto bars into the slug store and stamp the sealed manifest.

    Daily (``1Day``): vendor daily bars are used only if UTC-midnight
    aligned; otherwise daily bars are rebuilt from vendor 1Hour bars
    (:class:`VendorDailyNotUtcAlignedError` fallback). Intraday: bars get
    ``bar_close_utc = open + timeframe``. In both cases only bars fully
    closed and fetched (close ≤ fetch time) are stored.

    Returns a summary dict (also persisted as the ingestion manifest when
    ``write_manifest`` — the artifact the orchestrator's §3.5 sealed-data
    contract consumes: per-symbol last bar-close stamps + content sha256,
    a global ``watermark_utc`` = min over symbols of the last sealed bar
    close, and a manifest-level fingerprint).
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    pairs = [_as_pair(s) for s in symbols]
    if not pairs:
        raise ValueError("ingest_crypto_bars: no symbols given")
    if timeframe not in TIMEFRAME_FILES:
        raise ValueError(
            f"Unknown crypto timeframe {timeframe!r}. Supported: {list(TIMEFRAME_FILES)}"
        )
    store = store or _default_crypto_store
    fetch = fetch_fn or fetch_crypto_bars
    now = now_fn() if now_fn is not None else _utc_now()
    fetched_through = min(_to_utc(end), _to_utc(now)) if end is not None else _to_utc(now)
    tf_file = TIMEFRAME_FILES[timeframe]

    raw = fetch(pairs, timeframe=timeframe, start=start, end=end, timeout_sec=timeout_sec)

    per_symbol: dict[str, dict] = {}
    hourly_fallback_pairs: list[str] = []
    normalized: dict[str, pd.DataFrame] = {}

    for pair in pairs:
        df = raw.get(pair)
        if df is None or df.empty:
            per_symbol[pair] = {"status": "no_data", "rows_ingested": 0}
            continue
        if timeframe == "1Day":
            try:
                normalized[pair] = normalize_daily_bars_utc(
                    df, fetched_through_utc=fetched_through
                )
            except VendorDailyNotUtcAlignedError as exc:
                log.warning("%s: %s", pair, exc)
                hourly_fallback_pairs.append(pair)
        else:
            normalized[pair] = _stamp_intraday_bars(
                df, timeframe=timeframe, fetched_through_utc=fetched_through
            )

    if hourly_fallback_pairs:
        log.info(
            "rebuilding UTC daily bars from 1Hour for %d pair(s): %s",
            len(hourly_fallback_pairs),
            hourly_fallback_pairs,
        )
        hourly_raw = fetch(
            hourly_fallback_pairs,
            timeframe="1Hour",
            start=start,
            end=end,
            timeout_sec=timeout_sec,
        )
        for pair in hourly_fallback_pairs:
            hdf = hourly_raw.get(pair)
            if hdf is None or hdf.empty:
                per_symbol[pair] = {"status": "no_data", "rows_ingested": 0}
                continue
            normalized[pair] = resample_hourly_to_utc_daily(
                hdf, fetched_through_utc=fetched_through
            )

    for pair, df in normalized.items():
        slug = pair_slug(pair)
        if df.empty:
            per_symbol[pair] = {"status": "no_sealed_bars", "rows_ingested": 0}
            continue
        store.save(df, slug, tf_file)
        stored = store.load(slug, tf_file)
        closes = pd.to_datetime(stored[BAR_CLOSE_COL])
        closes = closes.dt.tz_localize("UTC") if closes.dt.tz is None else closes.dt.tz_convert("UTC")
        per_symbol[pair] = {
            "status": "ok",
            "slug": slug,
            "path": f"{slug}/{tf_file}.parquet",
            "rows_ingested": int(len(df)),
            "rows_total": int(len(stored)),
            "first_bar_open_utc": _to_utc(stored.index.min()).isoformat(),
            "last_bar_open_utc": _to_utc(stored.index.max()).isoformat(),
            "last_bar_close_utc": closes.max().isoformat(),
            "content_sha256": _content_sha256(stored),
            "source": (
                "alpaca-1hour-resampled-utc"
                if pair in hourly_fallback_pairs
                else f"alpaca-{timeframe.lower()}"
            ),
        }

    sealed_closes = [
        pd.Timestamp(info["last_bar_close_utc"])
        for info in per_symbol.values()
        if info.get("status") == "ok"
    ]
    watermark = min(sealed_closes).isoformat() if sealed_closes else None

    # Universe completeness (Codex review: a partial-universe fetch must not
    # silently advance a signal-eligible watermark). `pairs` is the FULL
    # requested/expected universe for this ingestion call, known up front —
    # persist it (+ a content hash) so a manifest's completeness claim is
    # checkable against an immutable snapshot, not re-derived from whatever
    # happens to be in `symbols` after the fact.
    expected_universe = sorted(pairs)
    expected_universe_hash = "sha256:" + hashlib.sha256(
        json.dumps(expected_universe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    universe_complete = bool(expected_universe) and all(
        per_symbol.get(p, {}).get("status") == "ok" for p in expected_universe
    )
    # signal_eligible is the single field a consumer must check before
    # treating this manifest as fit to back a session's signal: complete
    # universe AND a real (non-None) watermark. `watermark_utc` above stays
    # populated from the ok-subset regardless, for ops visibility into a
    # partial fetch -- it is diagnostic only and must NOT be read as an
    # eligibility signal on its own (see `manifest_eligible_for_session`).
    signal_eligible = bool(universe_complete and watermark is not None)

    payload: dict = {
        "dataset_id": f"crypto-ohlcv-{tf_file}",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asset_class": "crypto",
        "provider": PROVIDER,
        "timeframe": timeframe,
        "uri": f"store://{CRYPTO_OHLCV_DIRNAME}/{tf_file}",
        "generated_at_utc": _to_utc(now).isoformat(),
        "fetched_through_utc": fetched_through.isoformat(),
        "expected_universe": expected_universe,
        "expected_universe_hash": expected_universe_hash,
        "universe_complete": universe_complete,
        "watermark_utc": watermark,
        "signal_eligible": signal_eligible,
        "symbols": per_symbol,
    }
    payload["fingerprint"] = manifest_fingerprint(payload)

    if write_manifest:
        from .validation import validate_data_manifest  # noqa: PLC0415

        validate_data_manifest(payload)
        _write_manifest_atomic(crypto_manifest_path(store, timeframe), payload)

    return payload


def fetch_crypto_daily_cached(
    symbol: str,
    *,
    start: "str | None" = None,
    end: "str | None" = None,
    cache: bool = True,
    timeout_sec: float = 30.0,
    store: "CryptoLocalStore | None" = None,
    fetch_fn: "Callable[..., dict[str, pd.DataFrame]] | None" = None,
) -> pd.DataFrame:
    """Cache-first daily crypto bars — the ``provider="alpaca_crypto"`` seam
    behind :func:`renquant_base_data.loaders.data.fetch_ohlcv`.

    Same shape as the equity path: return the cached UTC-daily frame when it
    covers [start, end] and is fresh on the UTC-day clock; otherwise run
    :func:`ingest_crypto_bars` (which also refreshes the sealed manifest)
    and serve from the store.
    """
    store = store or _default_crypto_store
    slug = _as_slug(symbol)
    if cache and store.has_range(slug, start=start, end=end):
        cached = store.load(slug, start=start, end=end)
        if cached is not None:
            return cached
    fetch_start = None
    if start is not None:
        fetch_start = pd.Timestamp(start).to_pydatetime()
    ingest_crypto_bars(
        [symbol],
        timeframe="1Day",
        start=fetch_start,
        store=store,
        fetch_fn=fetch_fn,
        timeout_sec=timeout_sec,
        write_manifest=cache,
    )
    df = store.load(slug, start=start, end=end)
    if df is None:
        raise RuntimeError(
            f"fetch_crypto_daily_cached({symbol!r}): no sealed bars available "
            f"after ingestion; check {store.data_dir / slug}"
        )
    return df


# ---------------------------------------------------------------------------
# Feature groundwork (B7): alpha158 price/volume subset, reused not forked
# ---------------------------------------------------------------------------

def build_crypto_features_for_pair(
    symbol: str, crypto_ohlcv_dir: "str | Path | None" = None
) -> "pd.DataFrame | None":
    """alpha158 price/volume features for one crypto pair.

    Reuses :func:`renquant_base_data.alpha158_qlib_panel.build_features_for_ticker`
    directly — VERIFIED asset-agnostic: it consumes only the shared
    ``alpha158_ops`` kbar/price/rolling operators over OHLCV columns and has
    no fundamentals dependency (fundamentals live in the separate
    ``alpha158_fund_panel`` module, which crypto never touches). This is the
    documented price/volume-only mode; no fork, no new operators.

    The returned frame's ``ticker`` column carries the SLUG form (matching
    the store directory). Labels are explicitly out of scope here (D-C3/D-C4
    per the RFC — the SPY-excess label sidecar stays equity-only).
    """
    from .alpha158_qlib_panel import build_features_for_ticker  # noqa: PLC0415

    slug = _as_slug(symbol)
    ohlcv_dir = Path(crypto_ohlcv_dir) if crypto_ohlcv_dir is not None else _REPO_ROOT_CRYPTO_OHLCV
    return build_features_for_ticker(slug, ohlcv_dir)


# ---------------------------------------------------------------------------
# Two-source parity (B-audit): Alpaca vs yfinance daily closes
# ---------------------------------------------------------------------------

def crosscheck_daily_close(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    rel_tol: float = 0.01,
) -> dict:
    """Compare two daily close series by UTC calendar date.

    ``primary`` is the Alpaca UTC-daily store frame; ``secondary`` a vendor
    frame (yfinance slug ticker). Pure function — no network. Returns a
    parity report: overlap size, max relative close delta, breach dates
    (relative delta > ``rel_tol``), and ``ok`` (non-empty overlap, zero
    breaches). Vendor day-boundary differences are expected to appear here
    as systematic deltas — that is the point of the check: two-source parity
    must pass before any training run consumes this store (RFC §3.3).
    """
    def _daily_close(df: pd.DataFrame) -> pd.Series:
        if "close" not in df.columns:
            raise ValueError("crosscheck frame lacks a 'close' column")
        s = df["close"].copy()
        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        s.index = idx.normalize().date
        return s[~pd.Index(s.index).duplicated(keep="last")]

    a = _daily_close(primary)
    b = _daily_close(secondary)
    joint = pd.concat({"primary": a, "secondary": b}, axis=1, join="inner").dropna()
    if joint.empty:
        return {
            "n_overlap": 0,
            "max_rel_close_delta": None,
            "n_breaches": 0,
            "breach_dates": [],
            "rel_tol": float(rel_tol),
            "ok": False,
        }
    mid = (joint["primary"].abs() + joint["secondary"].abs()) / 2.0
    rel = (joint["primary"] - joint["secondary"]).abs() / mid.where(mid > 0.0)
    breaches = rel[rel > rel_tol]
    return {
        "n_overlap": int(len(joint)),
        "max_rel_close_delta": float(rel.max()),
        "n_breaches": int(len(breaches)),
        "breach_dates": [d.isoformat() for d in breaches.index],
        "rel_tol": float(rel_tol),
        "ok": bool(len(joint) > 0 and len(breaches) == 0),
    }


def _fetch_yfinance_daily(slug: str, *, timeout_sec: float = 30.0) -> pd.DataFrame:
    """Default secondary-source fetcher: yfinance crypto daily bars via
    OpenBB (the slug form IS yfinance's crypto ticker — RFC §3.0)."""
    def _fetch():
        from openbb import obb  # noqa: PLC0415

        return obb.crypto.price.historical(symbol=slug, provider="yfinance").to_df()

    from renquant_common.net_safety import call_with_timeout  # noqa: PLC0415

    df = call_with_timeout(
        _fetch, timeout_sec=timeout_sec, label=f"yfinance_crypto_daily({slug})"
    )
    if df is None:
        raise RuntimeError(f"yfinance crypto fetch for {slug} timed out after {timeout_sec}s")
    return df


def run_yfinance_crosscheck(
    symbol: str,
    *,
    store: "CryptoLocalStore | None" = None,
    fetch_secondary: "Callable[[str], pd.DataFrame] | None" = None,
    rel_tol: float = 0.01,
    timeout_sec: float = 30.0,
) -> dict:
    """Run the two-source parity check for one pair against yfinance."""
    store = store or _default_crypto_store
    slug = _as_slug(symbol)
    primary = store.load(slug)
    if primary is None:
        raise RuntimeError(f"no Alpaca daily bars stored for {slug}; ingest first")
    if fetch_secondary is not None:
        secondary = fetch_secondary(slug)
    else:
        secondary = _fetch_yfinance_daily(slug, timeout_sec=timeout_sec)
    report = crosscheck_daily_close(primary, secondary, rel_tol=rel_tol)
    report["pair"] = slug_pair(slug)
    report["slug"] = slug
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest Alpaca crypto bars (v1beta3) into the slug store "
        "and stamp the UTC-session watermark manifest (crypto RFC D-C2)."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="crypto pairs, pair or slug form (e.g. BTC/USD ETH-USD)",
    )
    parser.add_argument(
        "--timeframe",
        default="1Day",
        choices=sorted(TIMEFRAME_FILES),
        help="Alpaca timeframe (default 1Day)",
    )
    parser.add_argument("--start", default=None, help="ISO start (default: vendor max depth)")
    parser.add_argument("--end", default=None, help="ISO end (default: now)")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=f"store root (default: repo data/{CRYPTO_OHLCV_DIRNAME})",
    )
    parser.add_argument(
        "--crosscheck",
        action="store_true",
        help="after daily ingestion, run the yfinance two-source parity check",
    )
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser


def main(argv: "list[str] | None" = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    store = CryptoLocalStore(args.data_dir) if args.data_dir else _default_crypto_store
    start = pd.Timestamp(args.start).to_pydatetime() if args.start else None
    end = pd.Timestamp(args.end).to_pydatetime() if args.end else None
    summary = ingest_crypto_bars(
        args.pairs,
        timeframe=args.timeframe,
        start=start,
        end=end,
        store=store,
        timeout_sec=args.timeout_sec,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.crosscheck and args.timeframe == "1Day":
        for pair in args.pairs:
            report = run_yfinance_crosscheck(pair, store=store, timeout_sec=args.timeout_sec)
            print(json.dumps(report, indent=2, sort_keys=True))
    statuses = {info.get("status") for info in summary["symbols"].values()}
    return 0 if statuses <= {"ok"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
