"""Crypto vol-rank signal — low-volatility factor selection (G2 simplified).

Replaces the full alpha158 → XGB pipeline after the 2026-07-13 signal
viability check found that alpha158 cross-sectional IC on crypto is
concentrated in the low-volatility anomaly (IC = -0.13, t = -4.3,
stable across 2y halves, all 10 key features same-sign stable).

The strategy: rank crypto pairs by trailing realized volatility,
select the lowest-vol subset, equal-weight, rebalance periodically.

This module computes the signal; deployment/execution is orchestrator's
responsibility.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .crypto_bars import CryptoLocalStore, CRYPTO_OHLCV_DIRNAME

log = logging.getLogger("renquant_base_data.crypto_vol_rank")

DEFAULT_VOL_WINDOW = 20
DEFAULT_TOP_N = 5
DEFAULT_MIN_BARS = 60
ANNUALIZATION_FACTOR = np.sqrt(365.0)


@dataclass
class VolRankConfig:
    vol_window: int = DEFAULT_VOL_WINDOW
    top_n: int = DEFAULT_TOP_N
    min_bars: int = DEFAULT_MIN_BARS
    crypto_ohlcv_dir: Path | None = None
    exclude_pairs: list[str] = field(default_factory=list)


@dataclass
class VolRankResult:
    as_of_date: date
    rankings: list[dict]
    selected: list[str]
    weights: dict[str, float]
    n_pairs_scored: int
    n_pairs_excluded: int
    vol_window: int
    top_n: int

    def to_dict(self) -> dict:
        return {
            "as_of_date": str(self.as_of_date),
            "rankings": self.rankings,
            "selected": self.selected,
            "weights": self.weights,
            "n_pairs_scored": self.n_pairs_scored,
            "n_pairs_excluded": self.n_pairs_excluded,
            "vol_window": self.vol_window,
            "top_n": self.top_n,
        }


def compute_trailing_vol(
    close: pd.Series, window: int = DEFAULT_VOL_WINDOW,
) -> float | None:
    """Annualized trailing realized volatility from daily log returns."""
    if len(close) < window + 1:
        return None
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < window:
        return None
    trailing = log_ret.iloc[-window:]
    vol = float(trailing.std()) * ANNUALIZATION_FACTOR
    return vol if np.isfinite(vol) and vol > 0 else None


def rank_by_vol(
    cfg: VolRankConfig,
    as_of: date | datetime | None = None,
) -> VolRankResult:
    """Rank all available crypto pairs by trailing realized vol (ascending).

    Returns the full ranking plus the top-N lowest-vol selection with
    equal weights.
    """
    store_dir = cfg.crypto_ohlcv_dir or (
        Path(__file__).resolve().parents[3] / "data" / CRYPTO_OHLCV_DIRNAME
    )
    store = CryptoLocalStore(store_dir)

    if not store_dir.is_dir():
        raise FileNotFoundError(f"Crypto OHLCV store not found: {store_dir}")

    slugs = sorted(
        d.name
        for d in store_dir.iterdir()
        if d.is_dir() and (d / "1d.parquet").exists()
    )
    if not slugs:
        raise RuntimeError(f"No crypto pairs in {store_dir}")

    exclude_set = set(cfg.exclude_pairs)
    n_excluded = 0

    pair_vols: list[tuple[str, float, int]] = []
    for slug in slugs:
        if slug in exclude_set:
            n_excluded += 1
            continue
        df = store.load(slug, "1d")
        if df is None or df.empty:
            continue
        if "close" not in df.columns:
            continue
        close = df["close"].sort_index()
        if as_of is not None:
            close = close.loc[:str(as_of)]
        if len(close) < cfg.min_bars:
            log.info("%s: only %d bars (need %d), skipping", slug, len(close), cfg.min_bars)
            continue
        vol = compute_trailing_vol(close, window=cfg.vol_window)
        if vol is None:
            continue
        pair_vols.append((slug, vol, len(close)))

    if not pair_vols:
        raise RuntimeError("No pairs with sufficient data for vol ranking")

    pair_vols.sort(key=lambda x: x[1])

    rankings = [
        {
            "rank": i + 1,
            "pair": slug,
            "annualized_vol": round(vol, 4),
            "n_bars": n_bars,
        }
        for i, (slug, vol, n_bars) in enumerate(pair_vols)
    ]

    top_n = min(cfg.top_n, len(pair_vols))
    selected = [pair_vols[i][0] for i in range(top_n)]
    weight = round(1.0 / top_n, 6)
    weights = {s: weight for s in selected}

    ref_date = as_of if isinstance(as_of, date) else (
        as_of.date() if isinstance(as_of, datetime) else
        date.today()
    )

    result = VolRankResult(
        as_of_date=ref_date,
        rankings=rankings,
        selected=selected,
        weights=weights,
        n_pairs_scored=len(pair_vols),
        n_pairs_excluded=n_excluded,
        vol_window=cfg.vol_window,
        top_n=top_n,
    )

    log.info(
        "Vol rank as of %s: %d pairs scored, top-%d selected: %s",
        ref_date, len(pair_vols), top_n, selected,
    )
    return result


def backtest_vol_rank(
    cfg: VolRankConfig,
    rebalance_days: int = 20,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Simple backtest: rebalance every N calendar days, equal-weight top-N
    lowest-vol pairs. Returns a DataFrame with daily portfolio returns.
    """
    store_dir = cfg.crypto_ohlcv_dir or (
        Path(__file__).resolve().parents[3] / "data" / CRYPTO_OHLCV_DIRNAME
    )
    store = CryptoLocalStore(store_dir)

    slugs = sorted(
        d.name
        for d in store_dir.iterdir()
        if d.is_dir() and (d / "1d.parquet").exists()
    )
    exclude_set = set(cfg.exclude_pairs)

    all_close: dict[str, pd.Series] = {}
    for slug in slugs:
        if slug in exclude_set:
            continue
        df = store.load(slug, "1d")
        if df is not None and "close" in df.columns and len(df) > cfg.min_bars:
            all_close[slug] = df["close"].sort_index()

    if not all_close:
        raise RuntimeError("No pairs with sufficient data for backtest")

    close_panel = pd.DataFrame(all_close)
    if start:
        close_panel = close_panel.loc[start:]
    if end:
        close_panel = close_panel.loc[:end]

    daily_ret = close_panel.pct_change()
    dates = close_panel.index
    warmup = max(cfg.vol_window + 1, cfg.min_bars)

    portfolio_returns = []
    current_selection: list[str] = []
    days_since_rebalance = rebalance_days

    for i, d in enumerate(dates):
        if i < warmup:
            portfolio_returns.append({"date": d, "port_return": 0.0, "rebalance": False})
            continue

        if days_since_rebalance >= rebalance_days:
            vols = {}
            for slug in all_close:
                hist = close_panel[slug].iloc[:i+1].dropna()
                if len(hist) < cfg.vol_window + 1:
                    continue
                v = compute_trailing_vol(hist, cfg.vol_window)
                if v is not None:
                    vols[slug] = v
            if vols:
                sorted_pairs = sorted(vols.items(), key=lambda x: x[1])
                current_selection = [p for p, _ in sorted_pairs[:cfg.top_n]]
                days_since_rebalance = 0

        if current_selection:
            day_rets = [daily_ret.loc[d, s] for s in current_selection
                        if s in daily_ret.columns and np.isfinite(daily_ret.loc[d, s])]
            port_ret = np.mean(day_rets) if day_rets else 0.0
        else:
            port_ret = 0.0

        portfolio_returns.append({
            "date": d, "port_return": port_ret,
            "rebalance": days_since_rebalance == 0,
        })
        days_since_rebalance += 1

    return pd.DataFrame(portfolio_returns).set_index("date")
