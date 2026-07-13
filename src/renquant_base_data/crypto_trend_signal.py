"""Crypto SMA50 trend-following signal (G2 v3).

Fixed SMA50 trend filter: signal = 1 (LONG) if close > SMA50, else 0 (CASH).
Walk-forward validated on BTC (11.8y, Sharpe +1.36 [0.71, 1.85]) and ETH
(8.7y, Sharpe +0.60 [0.12, 1.47]). Adaptive per-pair strategy selection was
tested and disproven — fixed SMA50 beats it (+1.53 vs +1.31 on BTC).

This module computes signals; scheduling and execution are orchestrator's
responsibility. The SignalSnapshot.digest integrates with crypto_session.py's
gate #7/#10.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .crypto_bars import CryptoLocalStore, CRYPTO_OHLCV_DIRNAME, _as_slug

log = logging.getLogger("renquant_base_data.crypto_trend_signal")

DEFAULT_SMA_PERIOD = 50
DEFAULT_MIN_BARS = 60


@dataclass(frozen=True)
class TrendSignalConfig:
    sma_period: int = DEFAULT_SMA_PERIOD
    min_bars: int = DEFAULT_MIN_BARS
    crypto_ohlcv_dir: Path | None = None


@dataclass(frozen=True)
class PairSignal:
    pair: str
    signal: int
    close: float
    sma: float
    bar_date: date


@dataclass(frozen=True)
class SignalSnapshot:
    as_of_date: date
    signals: tuple[PairSignal, ...]
    universe_hash: str
    n_long: int
    n_cash: int
    digest: str

    def to_dict(self) -> dict:
        return {
            "as_of_date": str(self.as_of_date),
            "signals": [
                {
                    "pair": s.pair,
                    "signal": s.signal,
                    "close": s.close,
                    "sma": s.sma,
                    "bar_date": str(s.bar_date),
                }
                for s in self.signals
            ],
            "universe_hash": self.universe_hash,
            "n_long": self.n_long,
            "n_cash": self.n_cash,
            "digest": self.digest,
        }


def _universe_hash(pairs: list[str]) -> str:
    payload = json.dumps(sorted(pairs), separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _snapshot_digest(as_of: date, signals: list[PairSignal], universe_hash: str) -> str:
    parts = [str(as_of), universe_hash]
    for s in sorted(signals, key=lambda x: x.pair):
        parts.append(f"{s.pair}:{s.signal}:{s.close:.8f}:{s.sma:.8f}")
    payload = "|".join(parts)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def compute_signal_for_pair(
    close: pd.Series,
    sma_period: int = DEFAULT_SMA_PERIOD,
) -> tuple[int, float, float] | None:
    """Compute SMA50 trend signal for one pair's close series.

    Returns (signal, last_close, sma_value) or None if insufficient data.
    """
    if len(close) < sma_period:
        return None
    sma = float(close.rolling(sma_period).mean().iloc[-1])
    if not np.isfinite(sma) or sma <= 0:
        return None
    last_close = float(close.iloc[-1])
    if not np.isfinite(last_close) or last_close <= 0:
        return None
    signal = 1 if last_close > sma else 0
    return signal, last_close, sma


def compute_signals(
    pairs: list[str],
    cfg: TrendSignalConfig | None = None,
    as_of: date | datetime | None = None,
) -> SignalSnapshot:
    """Compute fixed SMA50 trend signal for each pair in the universe."""
    if cfg is None:
        cfg = TrendSignalConfig()

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

    pair_signals: list[PairSignal] = []
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

        result = compute_signal_for_pair(close, cfg.sma_period)
        if result is None:
            log.info("%s: signal computation returned None, skipping", pair)
            continue

        signal, last_close, sma = result
        bar_dt = close.index[-1]
        if hasattr(bar_dt, "date"):
            bar_d = bar_dt.date()
        else:
            bar_d = pd.Timestamp(bar_dt).date()

        pair_signals.append(PairSignal(
            pair=pair,
            signal=signal,
            close=round(last_close, 8),
            sma=round(sma, 8),
            bar_date=bar_d,
        ))

    u_hash = _universe_hash(pairs)
    digest = _snapshot_digest(ref_date, pair_signals, u_hash)
    n_long = sum(1 for s in pair_signals if s.signal == 1)
    n_cash = sum(1 for s in pair_signals if s.signal == 0)

    snapshot = SignalSnapshot(
        as_of_date=ref_date,
        signals=tuple(pair_signals),
        universe_hash=u_hash,
        n_long=n_long,
        n_cash=n_cash,
        digest=digest,
    )

    log.info(
        "Trend signal as of %s: %d pairs, %d LONG, %d CASH, digest=%s",
        ref_date, len(pair_signals), n_long, n_cash, digest[:24],
    )
    return snapshot
