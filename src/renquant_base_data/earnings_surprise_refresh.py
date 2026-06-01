"""CLI for refreshing PEAD/SUE earnings-surprise caches from base-data."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from renquant_common.net_safety import FetchBudget, call_with_timeout

from renquant_base_data.fetchers.earnings_surprise import (
    EarningsSurpriseStore,
    SURPRISE_COLS,
    fetch_earnings_surprise,
)


log = logging.getLogger("renquant_base_data.earnings_surprise_refresh")


def load_watchlist(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    watchlist = payload.get("watchlist") or payload.get("data", {}).get("watchlist")
    if not watchlist:
        raise ValueError(f"watchlist is empty or missing in {path}")
    return [str(symbol).upper() for symbol in watchlist]


def refresh_earnings_surprise(
    *,
    watchlist: list[str],
    data_dir: str | Path,
    cache: bool = True,
    provider_fn: Callable[[str], pd.DataFrame] | None = None,
    total_budget_sec: float = 120.0,
    per_ticker_sec: float = 25.0,
) -> dict[str, object]:
    data_dir = Path(data_dir).expanduser().resolve()
    store = EarningsSurpriseStore(data_dir=data_dir / "earnings_surprise")
    budget = FetchBudget(total_sec=float(total_budget_sec), label="earnings_surprise_refresh")
    per_symbol: dict[str, dict[str, object]] = {}

    for symbol in watchlist:
        if budget.exhausted():
            per_symbol[symbol] = {"rows": 0, "status": "skipped_budget"}
            continue
        frame = call_with_timeout(
            fetch_earnings_surprise,
            symbol,
            timeout_sec=float(per_ticker_sec),
            label=f"earnings.refresh({symbol})",
            budget=budget,
            cache=cache,
            store=store,
            provider_fn=provider_fn,
        )
        if frame is None:
            frame = pd.DataFrame(columns=SURPRISE_COLS)
            status = "timeout_or_empty"
        else:
            status = "ok" if not frame.empty else "empty"
        per_symbol[symbol] = {
            "rows": int(len(frame)),
            "status": status,
            "path": str(store._path(symbol)),
        }

    non_empty = sum(1 for item in per_symbol.values() if int(item["rows"]) > 0)
    summary: dict[str, object] = {
        "ok": True,
        "n_symbols": int(len(watchlist)),
        "non_empty": int(non_empty),
        "data_dir": str(data_dir),
        "per_symbol": per_symbol,
    }
    log.info("earnings refresh done: %d/%d symbols non-empty", non_empty, len(watchlist))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--total-budget-sec", type=float, default=120.0)
    parser.add_argument("--per-ticker-sec", type=float, default=25.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    watchlist = (
        [s.upper() for s in args.symbols]
        if args.symbols is not None
        else load_watchlist(args.strategy_config)
    )
    summary = refresh_earnings_surprise(
        watchlist=watchlist,
        data_dir=args.data_dir,
        cache=not args.no_cache,
        total_budget_sec=args.total_budget_sec,
        per_ticker_sec=args.per_ticker_sec,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
