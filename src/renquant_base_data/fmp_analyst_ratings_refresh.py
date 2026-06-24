"""CLI to refresh FMP historical analyst rating distributions (weekly cron).

Pulls ``grades-historical`` per watchlist ticker (full ~7.5y history each),
throttled to respect the free-tier per-minute cap, into
``data/analyst_ratings_fmp.parquet``. ~142 names fit the 250-calls/day free
limit; ratings update monthly so a WEEKLY cron is ample. The key comes from
``FMP_API_KEY`` (in .env, gitignored — never committed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from renquant_base_data.fetchers.fmp_analyst_ratings import (
    FmpRatingsStore,
    fetch_grades_historical,
)

log = logging.getLogger("renquant_base_data.fmp_analyst_ratings_refresh")


def load_watchlist(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    wl = payload if isinstance(payload, list) else (
        payload.get("watchlist") or payload.get("data", {}).get("watchlist"))
    if not wl:
        raise ValueError(f"watchlist missing/empty in {path}")
    return [str(s).upper() for s in wl if s and str(s) != "-"]


def select_to_refresh(watchlist: list[str], existing, max_pull: int | None,
                      today) -> list[str]:
    """Pick which tickers to pull THIS run — incremental, staleness-first.

    Per the 'many small batches, never burst the rate limit' design: rank
    never-fetched tickers first, then by oldest ``fetched_at``, and take the top
    ``max_pull``. A daily cron with max_pull≈40 rotates through the ~142-name
    watchlist every few days, always under the free 250/day + per-minute caps.
    ``max_pull`` None/0 → the whole watchlist (one-shot/backfill)."""
    if not max_pull:
        return list(watchlist)
    import pandas as pd  # noqa: PLC0415
    last: dict[str, "pd.Timestamp"] = {}
    if existing is not None and len(existing) and "fetched_at" in existing.columns:
        last = (existing.groupby("ticker")["fetched_at"].max()).to_dict()
    floor = pd.Timestamp("1900-01-01")
    ranked = sorted(watchlist, key=lambda t: pd.Timestamp(last.get(t, floor)))
    return ranked[:max_pull]


def refresh_fmp_ratings(*, watchlist: list[str], output: str | Path, api_key: str,
                        sleep_sec: float = 1.0, max_pull: int | None = None,
                        asof=None, getter=None) -> dict:
    import pandas as pd  # noqa: PLC0415
    asof = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.today().normalize()
    store = FmpRatingsStore(Path(output))
    todo = select_to_refresh(watchlist, store.load(), max_pull, asof)
    frames, ok, empty = [], 0, 0
    for i, t in enumerate(todo):
        f = fetch_grades_historical(t, api_key, asof=asof, getter=getter)
        if len(f):
            frames.append(f); ok += 1
        else:
            empty += 1
        if sleep_sec and i < len(todo) - 1:
            time.sleep(sleep_sec)
    df = store.upsert(frames)
    summary = {"watchlist": len(watchlist), "pulled_this_run": len(todo),
               "with_data": ok, "empty": empty, "total_rows": int(len(df)),
               "tickers_in_store": int(df["ticker"].nunique()) if len(df) else 0,
               "output": str(output)}
    log.info("fmp ratings refresh: %s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--watchlist", required=True)
    p.add_argument("--output", default="data/analyst_ratings_fmp.parquet")
    p.add_argument("--sleep-sec", type=float, default=1.0,
                   help="throttle between calls (avoid the free per-minute cap)")
    p.add_argument("--max-pull", type=int, default=40,
                   help="incremental: pull only the N most-stale/missing tickers "
                        "this run (daily cron rotates the watchlist). 0 = all.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    key = os.environ.get("FMP_API_KEY")
    if not key:
        log.error("FMP_API_KEY not set"); return 1
    summary = refresh_fmp_ratings(
        watchlist=load_watchlist(args.watchlist), output=args.output,
        api_key=key, sleep_sec=args.sleep_sec, max_pull=args.max_pull)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
