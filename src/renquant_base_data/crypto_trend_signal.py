"""Crypto SMA feature computation (G2 data layer).

Computes rolling simple moving averages for crypto pairs from locally
stored OHLCV bars. This is a NEUTRAL data-layer primitive: it produces
the SMA value and the current close — it does NOT decide whether to go
long or stay in cash. That policy belongs in the strategy repo.

Consumers (strategy/pipeline) compare close vs SMA themselves and apply
whatever threshold or rule they own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .crypto_bars import CryptoLocalStore, CRYPTO_OHLCV_DIRNAME, _as_slug

log = logging.getLogger("renquant_base_data.crypto_trend_signal")

DEFAULT_SMA_PERIOD = 50
DEFAULT_MIN_BARS = 60


@dataclass(frozen=True)
class SMAConfig:
    sma_period: int = DEFAULT_SMA_PERIOD
    min_bars: int = DEFAULT_MIN_BARS
    crypto_ohlcv_dir: Path | None = None


@dataclass(frozen=True)
class PairSMA:
    pair: str
    close: float
    sma: float
    bar_date: date


def compute_sma_for_pair(
    close: pd.Series,
    sma_period: int = DEFAULT_SMA_PERIOD,
) -> tuple[float, float] | None:
    """Compute rolling SMA for one pair's close series.

    Returns (last_close, sma_value) or None if insufficient data.
    """
    if len(close) < sma_period:
        return None
    sma = float(close.rolling(sma_period).mean().iloc[-1])
    if not np.isfinite(sma) or sma <= 0:
        return None
    last_close = float(close.iloc[-1])
    if not np.isfinite(last_close) or last_close <= 0:
        return None
    return last_close, sma


def compute_sma_features(
    pairs: list[str],
    cfg: SMAConfig | None = None,
    as_of: date | datetime | None = None,
) -> list[PairSMA]:
    """Compute rolling SMA features for each pair in the universe.

    Returns a list of PairSMA with close and sma values — no policy
    decisions (long/cash/signal). Strategy layer owns the comparison.
    """
    if cfg is None:
        cfg = SMAConfig()

    store_dir = cfg.crypto_ohlcv_dir or (
        Path(__file__).resolve().parents[3] / "data" / CRYPTO_OHLCV_DIRNAME
    )
    store = CryptoLocalStore(store_dir)

    ref_date: date
    if isinstance(as_of, datetime):
        ref_date = as_of.date()
    elif isinstance(as_of, date):
        ref_date = as_of
    else:
        ref_date = date.today()

    results: list[PairSMA] = []
    for pair in pairs:
        slug = _as_slug(pair)
        df = store.load(slug, "1d")
        if df is None or df.empty:
            log.info("%s: no bars available, skipping", pair)
            continue
        if "close" not in df.columns:
            log.info("%s: no close column, skipping", pair)
            continue

        close = df["close"].sort_index()
        if as_of is not None:
            close = close.loc[:str(as_of)]
        if len(close) < cfg.min_bars:
            log.info("%s: only %d bars (need %d), skipping", pair, len(close), cfg.min_bars)
            continue

        result = compute_sma_for_pair(close, cfg.sma_period)
        if result is None:
            log.info("%s: SMA computation returned None, skipping", pair)
            continue

        last_close, sma = result
        bar_dt = close.index[-1]
        if hasattr(bar_dt, "date"):
            bar_d = bar_dt.date()
        else:
            bar_d = pd.Timestamp(bar_dt).date()

        results.append(PairSMA(
            pair=pair,
            close=round(last_close, 8),
            sma=round(sma, 8),
            bar_date=bar_d,
        ))

    log.info(
        "SMA features as of %s: %d/%d pairs computed (period=%d)",
        ref_date, len(results), len(pairs), cfg.sma_period,
    )
    return results
