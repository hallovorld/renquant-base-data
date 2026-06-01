"""CLI for refreshing Alpaca news parquet caches from base-data."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from renquant_base_data.alpaca_common import TokenBucket, load_strategy_watchlist


log = logging.getLogger("renquant_base_data.alpaca_news_refresh")

NEWS_DIRNAME = "news_alpaca"


def iter_chunks(start: datetime, end: datetime, *, days: int = 14):
    step = timedelta(days=int(days))
    current = start
    while current < end:
        nxt = min(current + step, end)
        yield current, nxt
        current = nxt


def _news_item_to_row(item, symbol: str) -> dict[str, object]:
    data = item.model_dump() if hasattr(item, "model_dump") else (
        item.dict() if hasattr(item, "dict") else (item if isinstance(item, dict) else {})
    )
    return {
        "symbol": symbol,
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "headline": data.get("headline"),
        "summary": data.get("summary"),
        "author": data.get("author"),
        "url": data.get("url"),
        "all_symbols": ",".join(data.get("symbols", []) or []),
    }


def fetch_news_window(
    client,
    bucket: TokenBucket,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    max_per_request: int = 50,
) -> list[dict[str, object]]:
    """Fetch one Alpaca News window with server pagination and backoff."""
    from alpaca.data.requests import NewsRequest

    rows: list[dict[str, object]] = []
    page_token: str | None = None
    backoff = 1.0
    while True:
        bucket.acquire()
        request = NewsRequest(
            symbols=symbol,
            start=start,
            end=end,
            limit=max_per_request,
            page_token=page_token,
            include_content=False,
            sort="asc",
        )
        try:
            response = client.get_news(request)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "rate" in message.lower() or "429" in message:
                log.warning("rate-limited on %s; backoff %.1fs", symbol, backoff)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            raise
        backoff = 1.0

        if isinstance(response, dict):
            news_list = response.get("news", []) or response.get("data", {}).get("news", [])
            next_token = response.get("next_page_token")
        else:
            data = getattr(response, "data", {}) or {}
            news_list = data.get("news", [])
            next_token = getattr(response, "next_page_token", None)
        for item in news_list or []:
            rows.append(_news_item_to_row(item, symbol))
        if not next_token:
            break
        page_token = next_token
    return rows


def fetch_symbol_news(
    client,
    bucket: TokenBucket,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    max_per_request: int = 50,
    chunk_days: int = 14,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chunk_start, chunk_end in iter_chunks(start, end, days=chunk_days):
        chunk_rows = fetch_news_window(
            client,
            bucket,
            symbol,
            chunk_start,
            chunk_end,
            max_per_request=max_per_request,
        )
        rows.extend(chunk_rows)
        if len(chunk_rows) >= max_per_request and chunk_days > 1:
            midpoint = chunk_start + (chunk_end - chunk_start) / 2
            rows.extend(fetch_news_window(client, bucket, symbol, chunk_start, midpoint, max_per_request=max_per_request))
            rows.extend(fetch_news_window(client, bucket, symbol, midpoint, chunk_end, max_per_request=max_per_request))
    return normalize_news_frame(pd.DataFrame(rows))


def normalize_news_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[
            "symbol",
            "created_at",
            "updated_at",
            "headline",
            "summary",
            "author",
            "url",
            "all_symbols",
        ])
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True)
    out["updated_at"] = pd.to_datetime(out["updated_at"], utc=True, errors="coerce")
    return out.drop_duplicates(subset=["symbol", "created_at", "headline"]).sort_values("created_at").reset_index(drop=True)


def merge_news(prior: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    new = normalize_news_frame(new)
    if prior is None or prior.empty:
        return new
    prior = normalize_news_frame(prior)
    merged = pd.concat([prior, new], ignore_index=True)
    return merged.drop_duplicates(subset=["symbol", "created_at", "headline"]).sort_values("created_at").reset_index(drop=True)


def refresh_alpaca_news(
    *,
    symbols: list[str],
    data_dir: str | Path,
    start: datetime,
    end: datetime,
    client=None,
    fetch_symbol_fn: Callable[..., pd.DataFrame] = fetch_symbol_news,
    rate_limit: int = 180,
    max_per_request: int = 50,
    chunk_days: int = 14,
) -> dict[str, object]:
    out_dir = Path(data_dir).expanduser().resolve() / NEWS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = TokenBucket(max_calls=rate_limit, window_seconds=60.0)
    per_symbol: dict[str, dict[str, object]] = {}
    total_rows = 0

    for symbol in [s.upper() for s in symbols]:
        out_path = out_dir / f"{symbol}.parquet"
        try:
            new_frame = fetch_symbol_fn(
                client,
                bucket,
                symbol,
                start,
                end,
                max_per_request=max_per_request,
                chunk_days=chunk_days,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: news fetch failed: %s", symbol, exc)
            per_symbol[symbol] = {"status": "failed", "rows": 0, "path": str(out_path)}
            continue
        prior = pd.read_parquet(out_path) if out_path.exists() else None
        merged = merge_news(prior, new_frame)
        if not merged.empty:
            merged.to_parquet(out_path, index=False)
        total_rows += len(merged)
        per_symbol[symbol] = {"status": "ok", "rows": int(len(merged)), "path": str(out_path)}

    return {
        "ok": True,
        "n_symbols": int(len(symbols)),
        "total_rows": int(total_rows),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "per_symbol": per_symbol,
    }


def _default_dates(since: date | None, until: date | None) -> tuple[datetime, datetime]:
    until_date = until or date.today() + timedelta(days=1)
    since_date = since or until_date - timedelta(days=1)
    if since_date >= until_date:
        raise ValueError("--since must be before --until")
    start = datetime.combine(since_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(until_date, datetime.min.time(), tzinfo=timezone.utc)
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy-config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--since", type=lambda value: date.fromisoformat(value), default=None)
    parser.add_argument("--until", type=lambda value: date.fromisoformat(value), default=None)
    parser.add_argument("--max-per-request", type=int, default=50)
    parser.add_argument("--rate-limit", type=int, default=180)
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    if args.symbols is None and args.strategy_config is None:
        raise SystemExit("--symbols or --strategy-config is required")
    symbols = (
        [symbol.upper() for symbol in args.symbols]
        if args.symbols is not None
        else load_strategy_watchlist(args.strategy_config)
    )
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not in env")
        return 3
    from alpaca.data.historical.news import NewsClient

    start, end = _default_dates(args.since, args.until)
    summary = refresh_alpaca_news(
        symbols=symbols,
        data_dir=args.data_dir,
        start=start,
        end=end,
        client=NewsClient(api_key=key, secret_key=secret),
        rate_limit=args.rate_limit,
        max_per_request=args.max_per_request,
        chunk_days=args.chunk_days,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
