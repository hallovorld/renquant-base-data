"""Backfill PIT snapshots from FMP ``grades-historical`` endpoint.

The ``grades-historical`` endpoint returns MONTHLY rating distributions going
back ~7 years (91+ monthly records per ticker as of 2026-07).  Each record has
the analyst buy/hold/sell distribution AS OF that month -- genuine point-in-time
data that FMP aggregated and stored at that time.

This script fetches the full history for each ticker in the universe and writes
one ``grades_consensus.parquet`` per month into the same directory layout the
forward collector uses (``<out>/<YYYY-MM-DD>/grades_consensus.parquet``), using
the first of each month as the snapshot date.

The ``pit_feature_builder`` then picks these up and can compute
``revision_breadth`` (net-UP fraction drift) across the full history -- no 90-day
forward wait required.

PIT PROVENANCE: the grades-historical data is AGGREGATED BY FMP on each month.
It is NOT fabricated by us.  Each record's date is the month FMP computed it.
The ``snapshot_as_of`` stamp in the manifest reflects that month, and
``pit_source=grades_historical_backfill`` distinguishes these from live forward
snapshots.

SAFE BY CONSTRUCTION:
  * Never overwrites an existing snapshot directory that has a grades_consensus
  * Only writes grades_consensus.parquet (not analyst_estimates or price_target)
  * Dry-run mode by default (pass --execute to actually write)
  * Respects FMP rate limits (300/min on Starter)

Usage::

    # preview what would be written
    python -m renquant_base_data.backfill_grades_historical --out /tmp/snap_test

    # execute into the real snapshot path
    python -m renquant_base_data.backfill_grades_historical \\
        --out data/estimate_snapshots --execute
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("renquant_base_data.backfill_grades_historical")

FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
GRADES_HISTORICAL_EP = "grades-historical"

DEFAULT_OUT = "data/estimate_snapshots"
DEFAULT_ENV = Path("/Users/renhao/git/github/RenQuant/.env")
DEFAULT_UNIVERSE_CONFIG = Path(
    "/Users/renhao/git/github/RenQuant/backtesting/renquant_104/strategy_config.golden.json"
)
REQUEST_TIMEOUT_S = 30
THROTTLE_S = 0.25
DEFAULT_MIN_COVERAGE = 0.80

_COLUMN_MAP = {
    "analystRatingsStrongBuy": "strongBuy",
    "analystRatingsBuy": "buy",
    "analystRatingsHold": "hold",
    "analystRatingsSell": "sell",
    "analystRatingsStrongSell": "strongSell",
}


def _read_env_value(env_path: Path, key: str) -> str | None:
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def load_api_key(env_path: Path) -> str | None:
    return os.environ.get("FMP_API_KEY") or _read_env_value(env_path, "FMP_API_KEY")


def load_universe(universe_arg: str | None) -> list[str]:
    path = Path(universe_arg) if universe_arg else DEFAULT_UNIVERSE_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"universe source not found: {path}")
    text = path.read_text()
    if path.suffix == ".json":
        cfg = json.loads(text)
        if "watchlist" in cfg:
            return sorted(set(cfg["watchlist"]))
        raise ValueError(f"{path} has no 'watchlist' key")
    tickers = []
    for line in text.splitlines():
        t = line.strip().split("#")[0].split(",")[0].strip()
        if t:
            tickers.append(t.upper())
    return sorted(set(tickers))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_grades_historical(
    ticker: str, api_key: str, *, timeout: int = REQUEST_TIMEOUT_S
) -> list[dict[str, Any]]:
    import requests

    url = f"{FMP_STABLE_BASE}/{GRADES_HISTORICAL_EP}?symbol={ticker}&apikey={api_key}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return []


def transform_grades_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"symbol": row["symbol"]}
    for src, dst in _COLUMN_MAP.items():
        out[dst] = row.get(src, 0)
    return out


def fetch_all_tickers(
    tickers: list[str],
    api_key: str,
    *,
    throttle: float = THROTTLE_S,
) -> dict[str, list[dict[str, Any]]]:
    all_data: dict[str, list[dict[str, Any]]] = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            records = fetch_grades_historical(ticker, api_key)
            if records:
                all_data[ticker] = records
                log.debug("%d/%d %s: %d months", i, len(tickers), ticker, len(records))
            else:
                log.debug("%d/%d %s: no data", i, len(tickers), ticker)
        except Exception as exc:
            log.warning("%d/%d %s: fetch error: %s", i, len(tickers), ticker, exc)
        if i < len(tickers):
            time.sleep(throttle)
    return all_data


def group_by_month(
    all_data: dict[str, list[dict[str, Any]]]
) -> dict[str, pd.DataFrame]:
    rows_by_month: dict[str, list[dict[str, Any]]] = {}
    for ticker, records in all_data.items():
        for rec in records:
            month_date = rec.get("date", "")
            if not month_date:
                continue
            transformed = transform_grades_row(rec)
            rows_by_month.setdefault(month_date, []).append(transformed)

    result: dict[str, pd.DataFrame] = {}
    for month_date, rows in sorted(rows_by_month.items()):
        df = pd.DataFrame(rows)
        result[month_date] = df
    return result


def write_snapshot(
    out_root: Path,
    snapshot_date: str,
    df: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> dict[str, Any] | None:
    snap_dir = out_root / snapshot_date
    grades_path = snap_dir / "grades_consensus.parquet"
    manifest_path = snap_dir / "grades_consensus.manifest.json"

    if grades_path.exists() and not overwrite:
        log.debug("skip %s: grades_consensus already exists", snapshot_date)
        return None

    snap_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(grades_path, index=False)

    manifest = {
        "schema_version": 1,
        "endpoint": "grades_consensus",
        "snapshot_as_of": snapshot_date,
        "pit_source": "grades_historical_backfill",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "n_symbols": int(df["symbol"].nunique()),
        "rows": len(df),
        "sha256": _sha256_file(grades_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def backfill(
    tickers: list[str],
    api_key: str,
    out_root: Path,
    *,
    execute: bool = False,
    overwrite: bool = False,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    log.info(
        "fetching grades-historical for %d tickers (execute=%s)",
        len(tickers),
        execute,
    )
    all_data = fetch_all_tickers(tickers, api_key)
    coverage = len(all_data) / len(tickers) if tickers else 0
    log.info(
        "coverage: %d/%d (%.1f%%)",
        len(all_data),
        len(tickers),
        coverage * 100,
    )

    if coverage < min_coverage:
        log.error(
            "coverage %.1f%% < floor %.1f%%; aborting",
            coverage * 100,
            min_coverage * 100,
        )
        return {
            "status": "error",
            "reason": "below_coverage_floor",
            "coverage": coverage,
            "tickers_ok": len(all_data),
            "tickers_total": len(tickers),
        }

    by_month = group_by_month(all_data)
    log.info("grouped into %d monthly snapshots", len(by_month))

    written = 0
    skipped = 0
    months_written: list[str] = []

    for month_date, df in by_month.items():
        if execute:
            result = write_snapshot(out_root, month_date, df, overwrite=overwrite)
            if result:
                written += 1
                months_written.append(month_date)
            else:
                skipped += 1
        else:
            snap_dir = out_root / month_date
            grades_path = snap_dir / "grades_consensus.parquet"
            if grades_path.exists() and not overwrite:
                skipped += 1
            else:
                written += 1
                months_written.append(month_date)

    return {
        "status": "ok" if execute else "dry_run",
        "coverage": coverage,
        "tickers_ok": len(all_data),
        "tickers_total": len(tickers),
        "months_total": len(by_month),
        "months_written": written,
        "months_skipped": skipped,
        "date_range": (
            [min(by_month.keys()), max(by_month.keys())] if by_month else []
        ),
        "months_written_list": months_written,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill grades-historical PIT snapshots from FMP"
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="snapshot output root")
    parser.add_argument("--universe", help="watchlist file or strategy_config.json")
    parser.add_argument("--env", default=str(DEFAULT_ENV), help=".env file for FMP key")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually write (default=dry run)",
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    api_key = load_api_key(Path(args.env))
    if not api_key:
        print("error: FMP_API_KEY not found", file=sys.stderr)
        return 1

    tickers = load_universe(args.universe)
    if not tickers:
        print("error: empty universe", file=sys.stderr)
        return 1

    result = backfill(
        tickers,
        api_key,
        Path(args.out),
        execute=args.execute,
        overwrite=args.overwrite,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Status: {result['status']}")
        print(f"Coverage: {result['tickers_ok']}/{result['tickers_total']} "
              f"({result['coverage']:.1%})")
        print(f"Months: {result['months_total']} total, "
              f"{result['months_written']} {'written' if args.execute else 'would write'}, "
              f"{result['months_skipped']} skipped")
        if result.get("date_range"):
            print(f"Range: {result['date_range'][0]} -> {result['date_range'][1]}")
        if not args.execute and result["months_written"] > 0:
            print("\nDry run. Pass --execute to write.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
