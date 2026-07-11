"""Parking-sleeve equity daily bars (SGOV + SPY) — ingestion, fingerprinted
manifest, and serving contract for renquant-pipeline#185 (RS-1 SGOV floor).

Why this lives HERE
-------------------
renquant-pipeline#185 (parking-sleeve ``mode="live"``, RS-1 SGOV floor
variant) correctly FAIL-CLOSES on a missing SGOV price, and its
data-availability finding stands `[VERIFIED 2026-07-10]`: SGOV daily bars
exist in no production OHLCV store. Codex review of the first attempt at
this dependency (umbrella RenQuant#459, now closed) ruled that the umbrella
is being deprecated and must not regain runtime data ownership: SGOV
daily-bar ingestion, storage, freshness/manifest fingerprinting, and the
serving contract belong in renquant-base-data, and pipeline #185 must
consume this repo's pinned artifact through the multi-repo run manifest.
This module is that base-data slice, patterned on the crypto RFC D-C2
ingestion (``crypto_bars.py``) — the repo's most recent fingerprinted
bar-ingestion precedent — with NYSE-session freshness instead of UTC-day
watermarks.

Symbol source of truth (finding kept from RenQuant#459)
-------------------------------------------------------
The normalization authority for the sleeve legs is the umbrella's
``adapters/sleeve_prices.parking_sleeve_leg_tickers`` (strategy-104#39
follow-up): read ``sleeve.spy_symbol`` / ``sleeve.sgov_symbol`` regardless
of ``sleeve.enabled``, strip/upcase, blank-falls-back-to-``SPY``/``SGOV``,
dedupe. base-data sits UPSTREAM of every consumer, so importing that
helper here would invert the dependency graph — instead
:func:`sleeve_leg_tickers` mirrors the normalization EXACTLY and is
mirror-pinned by test (the same convention the ``sleeve`` config section
uses against the pipeline task's reads, and ``crypto_bars`` used for its
pre-canonical helpers). Consumers that hold a strategy config should keep
resolving legs via ``parking_sleeve_leg_tickers`` and PASS the result in;
the defaults here only guarantee the two resolutions agree.

Watchlist / P-CONFIG-FP non-coupling (finding kept from RenQuant#459)
---------------------------------------------------------------------
SGOV joins PRICE COVERAGE only — never the tradable watchlist, panel
scoring, or cross-sectional admission stats. The panel config fingerprint
(``renquant_common.config_consistency._model_relevant_fields``) hashes
only watchlist / ``panel_ltr`` flags / sector maps; neither the ``sleeve``
config section nor any bar store enters the hash, so this dataset cannot
trip P-CONFIG-FP `[VERIFIED against the real impl in RenQuant#459's test
run]`. Fail-closed here too: the CLI refuses to ingest when a provided
strategy config has the SGOV leg in its watchlist (st104#39 violation —
fix the config, don't mask it).

Serving contract
----------------
* Registry entry: ``manifests/sleeve-ohlcv-1d.json``
  (``dataset_id="sleeve-ohlcv-1d"``, resolved via
  ``renquant_base_data.registry.resolve_data_manifest``).
* Authoritative per-run artifact: ``ingestion_manifest_sleeve_1d.json`` in
  the OHLCV store root — fingerprinted (sha256 over sorted-keys JSON,
  reusing ``crypto_bars.manifest_fingerprint``: ONE fingerprint impl on
  purpose, the calibrator triple-impl bug is what hand-copied hashing
  buys), with per-symbol content sha256 over canonical bar bytes and a
  ``serving_eligible`` completeness+freshness verdict.
* Consumer entrypoint: :func:`resolve_sleeve_leg_bars` — resolves one
  leg's bars FROM the pinned artifact, failing closed on manifest
  fingerprint mismatch, run-manifest pin mismatch, per-symbol status,
  and parquet-vs-manifest content sha256 drift.

Boundaries: ingestion + serving contract ONLY. Wiring the daily runner
(which today reads its own store) onto this artifact is a follow-up in the
consuming repos; nothing here imports or references the umbrella.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .crypto_bars import (
    _content_sha256,
    _to_utc,
    _utc_now,
    _write_manifest_atomic,
    manifest_fingerprint,
)
from .loaders import data as _ohlcv

log = logging.getLogger("renquant_base_data.sleeve_bars")

SLEEVE_OHLCV_DATASET_ID = "sleeve-ohlcv-1d"
MANIFEST_SCHEMA_VERSION = "sleeve-ohlcv-manifest-v1"
PROVIDER = "yfinance"
OHLCV_DIRNAME = "ohlcv"
MANIFEST_FILENAME = "ingestion_manifest_sleeve_1d.json"

_DEFAULT_SPY = "SPY"
_DEFAULT_SGOV = "SGOV"


# ---------------------------------------------------------------------------
# Symbol resolution (mirror-pinned against the umbrella normalization)
# ---------------------------------------------------------------------------

def _sleeve_section(config: Any) -> dict:
    cfg = config if isinstance(config, dict) else (getattr(config, "config", None) or {})
    if not isinstance(cfg, dict):
        return {}
    sleeve = cfg.get("sleeve")
    return sleeve if isinstance(sleeve, dict) else {}


def sleeve_leg_tickers(config: Any = None) -> list[str]:
    """Normalized sleeve legs ``[spy_symbol, sgov_symbol]``, IGNORING ``enabled``.

    EXACT mirror of the umbrella's
    ``adapters/sleeve_prices.parking_sleeve_leg_tickers`` normalization
    (strip/upcase, blank→default, dedupe) — see module docstring for why
    this is a mirror-pin rather than an import. Any change here must land
    in lockstep with that helper or the two resolutions diverge.
    """
    sleeve = _sleeve_section(config)
    spy_symbol = str(sleeve.get("spy_symbol", _DEFAULT_SPY)).strip().upper() or _DEFAULT_SPY
    sgov_symbol = str(sleeve.get("sgov_symbol", _DEFAULT_SGOV)).strip().upper() or _DEFAULT_SGOV
    return list(dict.fromkeys([spy_symbol, sgov_symbol]))


def sleeve_sgov_ticker(config: Any = None) -> str:
    """The T-bill leg alone (watchlist-guard subject; same normalization)."""
    sleeve = _sleeve_section(config)
    return str(sleeve.get("sgov_symbol", _DEFAULT_SGOV)).strip().upper() or _DEFAULT_SGOV


# ---------------------------------------------------------------------------
# Manifest helpers (fingerprint impl reused from crypto_bars — single impl)
# ---------------------------------------------------------------------------

def verify_sleeve_manifest(payload: dict) -> bool:
    """Re-derive the manifest fingerprint; ``False`` on any tamper/mismatch."""
    fp = payload.get("fingerprint")
    return bool(fp) and fp == manifest_fingerprint(payload)


def load_sleeve_ingestion_manifest(path: "str | Path") -> dict:
    """Load an ingestion manifest, failing closed on fingerprint mismatch."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not verify_sleeve_manifest(payload):
        raise ValueError(f"sleeve ingestion manifest fingerprint mismatch: {path}")
    return payload


def sleeve_manifest_path(store: "_ohlcv.LocalStore") -> Path:
    return store.data_dir / MANIFEST_FILENAME


def _leg_state(store: "_ohlcv.LocalStore", symbol: str) -> dict:
    """Store-side state for one leg (no network)."""
    df = store.load(symbol)
    fresh = False
    if df is not None:
        try:
            fresh = bool(store.has_range(symbol))  # NYSE-session freshness
        except Exception as exc:  # pragma: no cover - calendar lib edge
            log.warning("has_range(%s) failed: %s", symbol, exc)
    return {
        "symbol": symbol,
        "path": f"{symbol}/1d.parquet",
        "exists": df is not None,
        "rows_total": 0 if df is None else int(len(df)),
        "first_bar": None if df is None else str(df.index.min().date()),
        "last_bar": None if df is None else str(df.index.max().date()),
        "fresh": fresh,
    }


# ---------------------------------------------------------------------------
# Ingestion (warm-up backfill + daily refresh share this one entrypoint)
# ---------------------------------------------------------------------------

def ingest_sleeve_bars(
    symbols: "list[str] | None" = None,
    *,
    store: "_ohlcv.LocalStore | None" = None,
    fetch_fn: "Callable[[str], object] | None" = None,
    timeout_sec: float = 120.0,
    write_manifest: bool = True,
    now_fn: "Callable[[], pd.Timestamp] | None" = None,
) -> dict:
    """Ingest the sleeve legs' daily bars and stamp the sealed manifest.

    Idempotent: routes through ``loaders.data.fetch_ohlcv_incremental``
    (cache-first, incremental delta, ~10y cold start, timeout-protected),
    then re-reads the store and stamps per-symbol content sha256 + a
    NYSE-session freshness verdict. ``serving_eligible`` is the single
    field a consumer must check: full expected universe ingested AND every
    leg session-fresh.
    """
    legs = [s.strip().upper() for s in (symbols or sleeve_leg_tickers())]
    legs = [s for s in dict.fromkeys(legs) if s]
    if not legs:
        raise ValueError("ingest_sleeve_bars: no symbols given")
    store = store or _ohlcv.LocalStore()
    now = now_fn() if now_fn is not None else _utc_now()

    if fetch_fn is None:
        def fetch_fn(symbol: str):  # noqa: F811 - default fetcher
            return _ohlcv.fetch_ohlcv_incremental(
                symbol, store=store, timeout_sec=timeout_sec)

    per_symbol: dict[str, dict] = {}
    for symbol in legs:
        try:
            fetch_fn(symbol)
        except Exception as exc:
            log.error("fetch failed for %s: %s", symbol, exc)
        stored = store.load(symbol)
        if stored is None or stored.empty:
            per_symbol[symbol] = {"status": "no_data", "rows_total": 0}
            continue
        state = _leg_state(store, symbol)
        per_symbol[symbol] = {
            "status": "ok" if state["fresh"] else "stale",
            "path": state["path"],
            "rows_total": state["rows_total"],
            "first_bar": state["first_bar"],
            "last_bar": state["last_bar"],
            "fresh": state["fresh"],
            "content_sha256": _content_sha256(stored),
            "source": PROVIDER,
        }

    # Universe completeness (crypto_bars precedent: a partial fetch must
    # not silently advance a serving-eligible verdict). `legs` is the FULL
    # expected universe for this run, persisted with a content hash so the
    # completeness claim is checkable against an immutable snapshot.
    expected_universe = sorted(legs)
    expected_universe_hash = "sha256:" + hashlib.sha256(
        json.dumps(expected_universe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    universe_complete = all(
        per_symbol.get(s, {}).get("status") == "ok" for s in expected_universe
    )
    serving_eligible = bool(expected_universe) and universe_complete

    payload: dict = {
        "dataset_id": SLEEVE_OHLCV_DATASET_ID,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asset_class": "us_equity",
        "provider": PROVIDER,
        "timeframe": "1d",
        "uri": f"store://{OHLCV_DIRNAME}/1d",
        "generated_at_utc": _to_utc(now).isoformat(),
        "expected_universe": expected_universe,
        "expected_universe_hash": expected_universe_hash,
        "universe_complete": universe_complete,
        "serving_eligible": serving_eligible,
        "freshness_rule": "last-completed-nyse-session",
        "symbols": per_symbol,
    }
    payload["fingerprint"] = manifest_fingerprint(payload)

    if write_manifest:
        from .validation import validate_data_manifest  # noqa: PLC0415

        validate_data_manifest(payload)
        _write_manifest_atomic(sleeve_manifest_path(store), payload)

    return payload


# ---------------------------------------------------------------------------
# Serving contract (pinned-artifact consumption)
# ---------------------------------------------------------------------------

def resolve_sleeve_leg_bars(
    symbol: str,
    *,
    store: "_ohlcv.LocalStore | None" = None,
    manifest_path: "str | Path | None" = None,
    pinned_fingerprint: "str | None" = None,
) -> pd.DataFrame:
    """Resolve one sleeve leg's daily bars FROM the pinned artifact.

    This is the consumer entrypoint the multi-repo run manifest pins
    (pipeline #185 / orchestrator side): given the ingestion manifest and
    optionally the run-manifest's pinned fingerprint, return the leg's
    bars. FAIL-CLOSED on: manifest fingerprint mismatch (tamper), pinned
    fingerprint mismatch (stale/foreign artifact), symbol absent or not
    ``status="ok"``, and parquet-vs-manifest content sha256 drift (the
    store was rewritten after sealing).
    """
    symbol = symbol.strip().upper()
    store = store or _ohlcv.LocalStore()
    path = Path(manifest_path) if manifest_path is not None else sleeve_manifest_path(store)
    payload = load_sleeve_ingestion_manifest(path)  # fingerprint-verified
    if pinned_fingerprint is not None and payload.get("fingerprint") != pinned_fingerprint:
        raise ValueError(
            "sleeve ingestion manifest fingerprint does not match the pinned "
            f"run-manifest fingerprint: manifest={payload.get('fingerprint')} "
            f"pinned={pinned_fingerprint}"
        )
    info = (payload.get("symbols") or {}).get(symbol)
    if not info or info.get("status") != "ok":
        raise ValueError(
            f"sleeve leg {symbol} not serving-eligible in manifest {path}: "
            f"{info!r}"
        )
    df = store.load(symbol)
    if df is None or df.empty:
        raise ValueError(f"sleeve leg {symbol} has no bars in store {store.data_dir}")
    actual_sha = _content_sha256(df)
    if actual_sha != info.get("content_sha256"):
        raise ValueError(
            f"sleeve leg {symbol} content sha256 drifted from sealed manifest: "
            f"store={actual_sha} manifest={info.get('content_sha256')}"
        )
    return df


# ---------------------------------------------------------------------------
# CLI — dry-run by default; --write ingests; --verify audits
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        log.warning("strategy config %s missing — using SPY/SGOV defaults", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Warm-up backfill + refresh of the parking-sleeve equity "
        "daily bars (sleeve-ohlcv-1d). DRY-RUN by default: reports store "
        "state, no network, no writes. --write ingests and stamps the "
        "fingerprinted ingestion manifest; --verify audits an existing "
        "manifest against the store."
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="sleeve legs (default: SPY SGOV, or resolved from "
             "--strategy-config's sleeve section)")
    parser.add_argument(
        "--strategy-config", type=Path, default=None,
        help="optional strategy-config-shaped JSON: resolves "
             "sleeve.spy_symbol/sgov_symbol and arms the "
             "sgov-in-watchlist refusal guard")
    parser.add_argument(
        "--data-dir", default=None,
        help=f"store root (default: repo data/{OHLCV_DIRNAME})")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--write", action="store_true",
        help="fetch + persist + stamp the ingestion manifest")
    parser.add_argument(
        "--verify", action="store_true",
        help="audit the existing ingestion manifest against the store "
             "(fingerprint, content sha256, freshness); no network")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    store = _ohlcv.LocalStore(args.data_dir) if args.data_dir else _ohlcv.LocalStore()

    config = _load_json(args.strategy_config) if args.strategy_config else None
    legs = ([s.strip().upper() for s in args.symbols]
            if args.symbols else sleeve_leg_tickers(config))
    legs = [s for s in dict.fromkeys(legs) if s]

    # FAIL-CLOSED guard (st104#39): the T-bill leg must never be in the
    # tradable watchlist — refuse rather than mask the config violation.
    if config is not None:
        sgov = sleeve_sgov_ticker(config)
        if sgov in set(config.get("watchlist") or []):
            print(
                f"REFUSING: sleeve sgov_symbol {sgov} is in the watchlist — "
                "st104#39 violation; fix the strategy config first.",
                flush=True,
            )
            return 2

    print(f"store: {store.data_dir}")
    print(f"legs: {legs}")

    if args.verify:
        path = sleeve_manifest_path(store)
        try:
            payload = load_sleeve_ingestion_manifest(path)
            for symbol in payload.get("expected_universe") or legs:
                resolve_sleeve_leg_bars(symbol, store=store, manifest_path=path)
            fresh = all(_leg_state(store, s)["fresh"]
                        for s in payload.get("expected_universe") or legs)
        except (OSError, ValueError) as exc:
            print(f"VERIFY FAIL: {exc}")
            return 1
        if not fresh:
            print("VERIFY FAIL: manifest is sealed+consistent but a leg is "
                  "no longer session-fresh — re-run with --write")
            return 1
        print(f"VERIFY OK: {path} sealed, content-consistent, fresh [VERIFIED]")
        return 0

    for symbol in legs:
        state = _leg_state(store, symbol)
        status = ("[VERIFIED fresh]" if state["fresh"]
                  else ("[STALE]" if state["exists"] else "[MISSING]"))
        print(f"before {symbol}: {status} rows={state['rows_total']} "
              f"span={state['first_bar']}..{state['last_bar']}")

    if not args.write:
        print("DRY-RUN (default): no fetch performed. Re-run with --write to "
              "ingest via loaders.data.fetch_ohlcv_incremental and stamp "
              f"{MANIFEST_FILENAME}.")
        return 0

    summary = ingest_sleeve_bars(legs, store=store, timeout_sec=args.timeout_sec)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["serving_eligible"]:
        print("FAIL: ingestion completed but the dataset is NOT "
              "serving-eligible (missing/stale leg) — see symbols above.")
        return 1
    print(f"OK: {MANIFEST_FILENAME} stamped fingerprint="
          f"{summary['fingerprint']} [VERIFIED]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
