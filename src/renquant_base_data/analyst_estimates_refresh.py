"""CLI for refreshing analyst estimate/revision snapshots (yfinance, no key).

Appends one point-in-time snapshot row per watchlist ticker, stamped with today
(``asof``), to ``data/analyst_estimates.parquet``. Run daily/weekly: the panel
accumulates a clean PIT history forward so a downstream estimate-revision
feature can be validated without look-ahead. The richest PIT-clean signals
(upgrades/downgrades, EPS-estimate trend) come for free from yfinance — no API
key, unlike the Finnhub free tier (recommendations only).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from renquant_base_data.fetchers.analyst_estimates import (
    AnalystEstimatesStore,
    fetch_analyst_snapshot,
)

log = logging.getLogger("renquant_base_data.analyst_estimates_refresh")


def load_watchlist(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        wl = payload
    else:
        wl = payload.get("watchlist") or payload.get("data", {}).get("watchlist")
    if not wl:
        raise ValueError(f"watchlist is empty or missing in {path}")
    return [str(s).upper() for s in wl if s and str(s) != "-"]


def refresh_analyst_estimates(
    *,
    watchlist: list[str],
    output: str | Path,
    asof: pd.Timestamp | None = None,
    sleep_sec: float = 0.4,
    ticker_factory=None,
) -> dict[str, object]:
    """Fetch a snapshot per ticker and append to the store. Defensive: a
    per-ticker failure is recorded as a skip, never aborts the batch."""
    asof = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.today().normalize()
    rows: list[dict] = []
    ok = skipped = 0
    for i, t in enumerate(watchlist):
        try:
            row = fetch_analyst_snapshot(t, asof, ticker_factory=ticker_factory)
            # require at least one real analyst signal, else count as a skip
            if row.get("n_analysts", 0) or pd.notna(row.get("eps_rev_30d")):
                rows.append(row); ok += 1
            else:
                skipped += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("analyst snapshot failed for %s: %s", t, exc)
            skipped += 1
        if sleep_sec and i < len(watchlist) - 1:
            time.sleep(sleep_sec)
    store = AnalystEstimatesStore(Path(output))
    df = store.upsert(rows)
    summary = {"asof": asof.date().isoformat(), "ok": ok, "skipped": skipped,
               "total_rows": int(len(df)), "output": str(output)}
    log.info("analyst refresh: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--watchlist", required=True,
                   help="JSON file: a list, or {watchlist:[...]} / strategy_config")
    p.add_argument("--output", default="data/analyst_estimates.parquet")
    p.add_argument("--sleep-sec", type=float, default=0.4)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    summary = refresh_analyst_estimates(
        watchlist=load_watchlist(args.watchlist),
        output=args.output, sleep_sec=args.sleep_sec,
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
