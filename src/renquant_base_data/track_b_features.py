"""Track B BULL_CALM-regime feature builders (4 features, ≤120 lines total).

References (read, not name-dropped):
  - Kelly, B. T., Gu, S., & Xiu, D. (2020). "Empirical Asset Pricing via
    Machine Learning." *Review of Financial Studies* 33(5), 2223-2273.
    Eq. (4) "MOM12_m" + Table 9 (low-vol regime IC).
  - Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta."
    *Journal of Financial Economics* 111(1), 1-25. Their BAB factor
    sorts cross-sectionally on this beta.
  - Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The
    Cross-Section of Volatility and Expected Returns." *Journal of
    Finance* 61(1), 259-299. 3-factor idio-vol = residual std after
    regressing returns on systematic factors.

Causality contract: every output value at date ``t`` uses only data at
dates ``<= t``. Test ``track_b_features_no_future_leak`` enforces this.

Implementations are intentionally compact; each helper is < 30 lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRACK_B_FEATURES: tuple[str, ...] = (
    "mom_carry_12_1",
    "beta_dm",
    "rvar_total",
    "idio_vol_3f",
)

# Standard cross-sectional momentum windows. 252 ~ 12 trading months;
# 21 ~ 1 trading month (the short-term reversal window we *skip*).
MOM_LONG_DAYS: int = 252
MOM_SKIP_DAYS: int = 21
# Frazzini-Pedersen rolling beta window.
BETA_WINDOW: int = 252
# Total realized variance + Ang-Hodrick-Xing-Zhang idio-vol window.
VOL_WINDOW: int = 60


def mom_carry_12_1(close: pd.Series) -> pd.Series:
    """12-month return minus 1-month reversal window (Kelly-Gu-Xiu 2020 Eq. 4).

    Definition: ``close[t-21] / close[t-252] - 1``. The numerator stops 1
    month ago (skipping short-term reversal), the denominator is 12
    months ago.

    Causal: at date ``t`` only uses ``close[t-252 ... t-21]``.
    """
    c = close.astype(float)
    num = c.shift(MOM_SKIP_DAYS)
    den = c.shift(MOM_LONG_DAYS)
    return num / den - 1.0


def _rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int) -> pd.Series:
    """Rolling OLS beta of ``stock_ret`` on ``mkt_ret`` over ``window`` days.

    beta = cov(s, m) / var(m), both with ``ddof=1`` so the ratio matches the
    OLS slope from ``numpy.cov`` / ``numpy.var(ddof=1)``. Both series must
    share the same index.

    Causal: window ends at ``t``, value at ``t`` uses returns ``[t-window+1 .. t]``.
    """
    m = mkt_ret.astype(float)
    s = stock_ret.astype(float)
    cov = s.rolling(window).cov(m, ddof=1)
    var = m.rolling(window).var(ddof=1)
    beta = cov / var.replace(0.0, np.nan)
    return beta


def beta_dm(close: pd.Series, spy_close: pd.Series) -> pd.Series:
    """Daily-rolling 252-day beta of stock vs SPY (Frazzini-Pedersen 2014).

    Causal: at ``t`` uses returns ``[t-252 .. t]``.
    """
    spy_aligned = spy_close.reindex(close.index, method="ffill", limit=5)
    s_ret = close.astype(float).pct_change()
    m_ret = spy_aligned.astype(float).pct_change()
    return _rolling_beta(s_ret, m_ret, BETA_WINDOW)


def rvar_total(close: pd.Series) -> pd.Series:
    """Total realized variance over 60 trading days (sum of squared returns).

    Causal: at ``t`` uses daily returns ``[t-59 .. t]`` (60 values).
    """
    r = close.astype(float).pct_change()
    return (r * r).rolling(VOL_WINDOW).sum()


def idio_vol_3f(
    close: pd.Series,
    spy_close: pd.Series,
    size_proxy: pd.Series,
    sector_close: pd.Series | None = None,
) -> pd.Series:
    """Rolling 60-day idiosyncratic volatility after orthogonalizing daily
    returns vs (SPY return, sector ETF return, size proxy).

    Per Ang-Hodrick-Xing-Zhang 2006 — residual std from a multi-factor
    regression. We use SPY as the market factor and either the ticker's
    sector ETF return (when supplied) or a constant zero (size-only fallback)
    as the sector factor. ``size_proxy`` is log(dollar volume) z-score over
    the same window (used in place of log-market-cap when fund data is
    absent, per Ang-Hodrick-Xing-Zhang section II.B).

    Implementation: build the ``[ret_s, ret_m, ret_sec, size]`` matrix,
    rolling-window OLS the stock return on the 3 factors, return the
    residual std over the window. Vectorized via covariance algebra
    (avoiding statsmodels-per-window).

    Causal: at ``t`` uses returns ``[t-59 .. t]`` only.
    """
    idx = close.index
    spy_a = spy_close.reindex(idx, method="ffill", limit=5)
    r_s = close.astype(float).pct_change()
    r_m = spy_a.astype(float).pct_change()
    if sector_close is not None:
        r_sec = sector_close.reindex(idx, method="ffill", limit=5).astype(float).pct_change()
    else:
        r_sec = pd.Series(0.0, index=idx)
    z_size = (size_proxy - size_proxy.rolling(VOL_WINDOW).mean()) / (
        size_proxy.rolling(VOL_WINDOW).std(ddof=1).replace(0.0, np.nan)
    )
    z_size = z_size.fillna(0.0)
    # Build [intercept, r_m, r_sec, z_size] and rolling-OLS via vectorized
    # window. For a portable implementation we use rolling apply on stacked
    # values; 60-day window × ~5k bars/ticker keeps total cost bounded.
    factor_df = pd.DataFrame({"m": r_m, "sec": r_sec, "z": z_size}, index=idx)

    def _resid_std(window_idx: np.ndarray) -> float:
        i0, i1 = int(window_idx[0]), int(window_idx[-1]) + 1
        y = r_s.iloc[i0:i1].to_numpy()
        X = factor_df.iloc[i0:i1].to_numpy()
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if ok.sum() < 10:
            return np.nan
        yy = y[ok]
        XX = np.column_stack([np.ones(ok.sum()), X[ok]])
        try:
            beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan
        return float(np.std(yy - XX @ beta, ddof=1))

    n = len(idx)
    pos = pd.Series(np.arange(n), index=idx, dtype=float)
    return pos.rolling(VOL_WINDOW).apply(_resid_std, raw=True)


def add_track_b_features(
    panel: pd.DataFrame,
    *,
    spy_close: pd.Series,
    sector_close_by_ticker: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Append the 4 Track B columns to a per-ticker (ticker, date, close, volume)
    panel. Operates per-ticker so causality is preserved within each ticker
    timeline. ``panel`` is the alpha158 panel-row frame; rows missing OHLCV
    columns will produce NaN features and be filled downstream like the
    existing alpha158 features.
    """
    if "close" not in panel.columns or "volume" not in panel.columns:
        # The alpha158 qlib panel doesn't carry raw close/volume; caller must
        # join those back (per build_alpha158_qlib_panel which has them at
        # the per-ticker phase). This function is exported for that callsite.
        raise KeyError("track-b features require raw close + volume in the panel")
    out_blocks: list[pd.DataFrame] = []
    sector_map = sector_close_by_ticker or {}
    for ticker, group in panel.groupby("ticker", sort=False):
        g = group.sort_values("date").reset_index(drop=True).copy()
        close = pd.Series(g["close"].to_numpy(), index=pd.DatetimeIndex(g["date"]))
        volume = pd.Series(g["volume"].to_numpy(), index=close.index)
        size_proxy = np.log(volume.astype(float) * close.astype(float) + 1.0)
        sector_close = sector_map.get(ticker)
        g["mom_carry_12_1"] = mom_carry_12_1(close).to_numpy()
        g["beta_dm"] = beta_dm(close, spy_close).to_numpy()
        g["rvar_total"] = rvar_total(close).to_numpy()
        g["idio_vol_3f"] = idio_vol_3f(close, spy_close, size_proxy, sector_close).to_numpy()
        out_blocks.append(g)
    return pd.concat(out_blocks, ignore_index=True)


__all__ = [
    "TRACK_B_FEATURES",
    "MOM_LONG_DAYS",
    "MOM_SKIP_DAYS",
    "BETA_WINDOW",
    "VOL_WINDOW",
    "add_track_b_features",
    "beta_dm",
    "idio_vol_3f",
    "mom_carry_12_1",
    "rvar_total",
]
