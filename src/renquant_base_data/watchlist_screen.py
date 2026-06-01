"""Watchlist screening report CLI for RenQuant OHLCV caches."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


log = logging.getLogger("renquant_base_data.watchlist_screen")

TRADING_DAYS_PER_YEAR = 252
NTFY_TOPIC = "renquant"


@dataclass(frozen=True)
class ScreenResult:
    report_path: Path
    drops: list[dict[str, Any]]
    adds: list[dict[str, Any]]
    watchlist_size: int
    valid_watchlist_size: int
    median_sharpe: float
    spy_return: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "report_path": str(self.report_path),
            "drops": self.drops,
            "adds": self.adds,
            "watchlist_size": self.watchlist_size,
            "valid_watchlist_size": self.valid_watchlist_size,
            "median_sharpe": self.median_sharpe,
            "spy_return": self.spy_return,
        }


def perf_stats(closes: pd.Series) -> dict[str, float]:
    returns = closes.pct_change().dropna()
    if len(returns) < 20:
        return {}
    total_return = float(closes.iloc[-1] / closes.iloc[0] - 1)
    ann_vol = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float((returns.mean() * TRADING_DAYS_PER_YEAR) / ann_vol) if ann_vol > 0 else 0.0
    drawdown = (closes - closes.cummax()) / closes.cummax()
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "ann_vol": ann_vol,
        "max_dd": float(drawdown.min()),
        "final_price": float(closes.iloc[-1]),
        "n_days": float(len(returns)),
    }


def load_ticker_series(ticker: str, cache_root: str | Path, lookback_days: int) -> pd.Series | None:
    path = Path(cache_root) / ticker / "1d.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "close" not in frame.columns:
        return None
    if "date" in frame.columns:
        frame.index = pd.to_datetime(frame["date"])
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    cutoff = frame.index.max() - pd.Timedelta(days=lookback_days)
    closes = pd.to_numeric(frame.loc[frame.index >= cutoff, "close"], errors="coerce").dropna()
    return closes if len(closes) >= 20 else None


def correlation_to_spy(closes: pd.Series, spy_closes: pd.Series) -> float:
    returns = closes.pct_change().dropna()
    spy_returns = spy_closes.pct_change().dropna()
    common = returns.index.intersection(spy_returns.index)
    if len(common) < 20:
        return float("nan")
    return float(returns.loc[common].corr(spy_returns.loc[common]))


def load_strategy_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def notify(title: str, body: str, *, topic: str = NTFY_TOPIC) -> None:
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        log.info("[ntfy suppressed] %s: %s", title, body)
        return
    try:
        import urllib.request

        request = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title},
        )
        urllib.request.urlopen(request, timeout=5).close()
    except Exception as exc:  # noqa: BLE001
        log.warning("ntfy failed: %s", exc)


def screen_watchlist(
    *,
    strategy_config: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    lookback_days: int = 180,
    top_add_candidates: int = 10,
    spy_symbol: str = "SPY",
    send_notify: bool = True,
) -> ScreenResult:
    cfg = load_strategy_config(strategy_config)
    watchlist = [str(symbol).upper() for symbol in cfg.get("watchlist", [])]
    defensive = {str(symbol).upper() for symbol in cfg.get("defensive_tickers", [])}
    cache_root = Path(data_dir).expanduser().resolve() / "ohlcv"

    spy_closes = load_ticker_series(spy_symbol, cache_root, lookback_days)
    if spy_closes is None:
        raise RuntimeError(f"{spy_symbol} parquet missing or too short")
    spy_stats = perf_stats(spy_closes)
    spy_return = spy_stats["total_return"]

    watchlist_stats: dict[str, dict[str, float | bool]] = {}
    for ticker in watchlist:
        closes = load_ticker_series(ticker, cache_root, lookback_days)
        if closes is None:
            log.warning("%s parquet missing or too short", ticker)
            continue
        stats = perf_stats(closes)
        stats["corr_spy"] = correlation_to_spy(closes, spy_closes)
        stats["is_defensive"] = ticker in defensive
        watchlist_stats[ticker] = stats
    if not watchlist_stats:
        raise RuntimeError("no valid watchlist stats computed")

    drops = [
        {"ticker": ticker, **stats}
        for ticker, stats in watchlist_stats.items()
        if stats["sharpe"] < 0 and stats["total_return"] < spy_return and not stats["is_defensive"]
    ]
    drops.sort(key=lambda item: item["sharpe"])

    sharpe_values = sorted(float(stats["sharpe"]) for stats in watchlist_stats.values())
    median_sharpe = sharpe_values[len(sharpe_values) // 2]
    mean_sharpe = sum(sharpe_values) / len(sharpe_values)
    sigma_sharpe = math.sqrt(sum((value - mean_sharpe) ** 2 for value in sharpe_values) / max(1, len(sharpe_values) - 1))
    add_threshold = median_sharpe + 0.5 * sigma_sharpe

    adds: list[dict[str, Any]] = []
    if cache_root.exists():
        watchlist_set = set(watchlist)
        for ticker_dir in cache_root.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name.upper()
            if ticker in watchlist_set or ticker == spy_symbol:
                continue
            closes = load_ticker_series(ticker, cache_root, lookback_days)
            if closes is None:
                continue
            stats = perf_stats(closes)
            if stats.get("sharpe", -99.0) > add_threshold:
                stats["corr_spy"] = correlation_to_spy(closes, spy_closes)
                adds.append({"ticker": ticker, **stats})
    adds.sort(key=lambda item: -item["sharpe"])
    adds = adds[:top_add_candidates]

    run_date = date.today().isoformat()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{run_date}.md"
    report_path.write_text(
        build_report(
            run_date=run_date,
            lookback_days=lookback_days,
            spy_stats=spy_stats,
            watchlist=watchlist,
            watchlist_stats=watchlist_stats,
            drops=drops,
            adds=adds,
            add_threshold=add_threshold,
            median_sharpe=median_sharpe,
        ),
        encoding="utf-8",
    )
    body = (
        f"drops={len(drops)} adds={len(adds)} "
        f"median_sharpe={median_sharpe:.2f} SPY_ret={spy_return:+.1%}"
    )
    if send_notify:
        notify(f"RenQuant watchlist screen {run_date}", body)
    log.info("watchlist screen report written: %s", report_path)
    return ScreenResult(
        report_path=report_path,
        drops=drops,
        adds=adds,
        watchlist_size=len(watchlist),
        valid_watchlist_size=len(watchlist_stats),
        median_sharpe=median_sharpe,
        spy_return=spy_return,
    )


def build_report(
    *,
    run_date: str,
    lookback_days: int,
    spy_stats: dict[str, float],
    watchlist: list[str],
    watchlist_stats: dict[str, dict[str, Any]],
    drops: list[dict[str, Any]],
    adds: list[dict[str, Any]],
    add_threshold: float,
    median_sharpe: float,
) -> str:
    lines: list[str] = [
        f"# Watchlist screen - {run_date}",
        "",
        f"**Lookback:** {lookback_days} days  ",
        f"**SPY baseline:** return={spy_stats['total_return']:+.2%}, sharpe={spy_stats['sharpe']:.2f}",
        "",
        "## Watchlist metrics",
        "",
        "| Ticker | Return | Sharpe | Vol (ann) | Max DD | Corr(SPY) | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for ticker, stats in sorted(watchlist_stats.items(), key=lambda item: -item[1]["sharpe"]):
        note: list[str] = []
        if stats["is_defensive"]:
            note.append("defensive")
        if stats["sharpe"] < 0:
            note.append("negative Sharpe")
        if stats["total_return"] < spy_stats["total_return"] and not stats["is_defensive"]:
            note.append("below SPY")
        lines.append(
            f"| {ticker} | {stats['total_return']:+.1%} | {stats['sharpe']:.2f} | "
            f"{stats['ann_vol']:.1%} | {stats['max_dd']:.1%} | {stats['corr_spy']:.2f} | "
            f"{', '.join(note) or '-'} |"
        )
    lines.extend(["", "## Drop candidates", ""])
    if drops:
        for item in drops:
            lines.append(
                f"- **{item['ticker']}** - Sharpe={item['sharpe']:.2f} "
                f"return={item['total_return']:+.1%}"
            )
    else:
        lines.append("*None.*")
    lines.extend(["", f"## Add candidates (Sharpe > {add_threshold:.2f})", ""])
    if adds:
        lines.extend(["| Ticker | Return | Sharpe | Vol | Corr(SPY) |", "|---|---:|---:|---:|---:|"])
        for item in adds:
            lines.append(
                f"| {item['ticker']} | {item['total_return']:+.1%} | {item['sharpe']:.2f} | "
                f"{item['ann_vol']:.1%} | {item['corr_spy']:.2f} |"
            )
    else:
        lines.append("*No non-watchlist names above threshold.*")
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Watchlist size: {len(watchlist_stats)} / {len(watchlist)}",
        f"- Median Sharpe: {median_sharpe:.2f}",
        f"- Drop suggestions: {len(drops)}",
        f"- Add suggestions: {len(adds)}",
        "",
        f"_Generated by `renquant_base_data.watchlist_screen` on {run_date}._",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy-config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/watchlist_screen"))
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--top-add-candidates", type=int, default=10)
    parser.add_argument("--spy-symbol", default="SPY")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    result = screen_watchlist(
        strategy_config=args.strategy_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lookback_days=args.lookback_days,
        top_add_candidates=args.top_add_candidates,
        spy_symbol=args.spy_symbol,
        send_notify=not args.no_notify,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
