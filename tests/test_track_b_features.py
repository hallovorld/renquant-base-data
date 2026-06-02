"""Tests for Track B BULL_CALM-regime feature builders.

Pins each feature's hand-computed value on a small fixture and asserts
causality: changes to data at any date ``t' > t`` must not change the
feature value at ``t``.

Canonical references (see ``src/renquant_base_data/track_b_features.py``):
  - Kelly-Gu-Xiu 2020 RFS Eq. (4) — mom_carry_12_1
  - Frazzini-Pedersen 2014 JFE — beta_dm
  - Ang-Hodrick-Xing-Zhang 2006 JF — idio_vol_market (market+size 2-factor residual)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.track_b_features import (
    BETA_WINDOW,
    MOM_LONG_DAYS,
    MOM_SKIP_DAYS,
    TRACK_B_FEATURES,
    VOL_WINDOW,
    add_track_b_features,
    beta_dm,
    idio_vol_market,
    mom_carry_12_1,
    rvar_total,
)


def _make_close(seed: int, n: int = 400, start: float = 100.0) -> pd.Series:
    """Smooth synthetic close with mild drift + small noise; enough history
    for the 252-day momentum / beta windows.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n)
    px = start * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(px, index=idx, name="close")


# ── 1. mom_carry_12_1 ──────────────────────────────────────────────────────


def test_mom_carry_12_1_matches_hand_computation():
    """Pin ``mom_carry_12_1`` against literal hand-computed values from the
    REAL helper (not a re-expression of the formula). An off-by-one in
    ``MOM_SKIP_DAYS`` / ``MOM_LONG_DAYS`` or a flipped numerator/denominator
    in the helper would fail this. Fixes codex PR #16 LOW finding.
    """
    # Deterministic linear close: close[i] = 100 + i. Long enough for the
    # production 252-day window plus a small inspection tail.
    n = MOM_LONG_DAYS + 6
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series([100.0 + i for i in range(n)], index=idx, name="close")
    actual = mom_carry_12_1(close)

    # Pre-warmup must be NaN (first MOM_LONG_DAYS values lack history).
    assert actual.iloc[MOM_LONG_DAYS - 1] != actual.iloc[MOM_LONG_DAYS - 1]  # NaN
    # Just past warmup (t = MOM_LONG_DAYS): num = close[t - skip], den = close[t - long].
    t0 = MOM_LONG_DAYS  # = 252
    expected_t0 = (100.0 + (t0 - MOM_SKIP_DAYS)) / (100.0 + (t0 - MOM_LONG_DAYS)) - 1.0
    assert actual.iloc[t0] == pytest.approx(expected_t0, rel=1e-12, abs=1e-15)
    # Three more sample points to lock the window arithmetic.
    for offset in (1, 3, 5):
        t = t0 + offset
        num_val = 100.0 + (t - MOM_SKIP_DAYS)  # close at t - 21
        den_val = 100.0 + (t - MOM_LONG_DAYS)  # close at t - 252
        expected = num_val / den_val - 1.0
        assert actual.iloc[t] == pytest.approx(expected, rel=1e-12, abs=1e-15), (
            f"mom_carry_12_1 mismatch at t={t}: got {actual.iloc[t]}, "
            f"hand-computed {expected}"
        )


def test_mom_carry_12_1_production_windows_no_nan_after_warmup():
    close = _make_close(seed=1)
    out = mom_carry_12_1(close)
    # First MOM_LONG_DAYS values must be NaN (insufficient history for the
    # 252-day denominator). After that the feature is fully populated.
    assert out.iloc[: MOM_LONG_DAYS].isna().all()
    assert out.iloc[MOM_LONG_DAYS:].isna().sum() == 0


def test_mom_carry_12_1_causal_at_t_does_not_use_future():
    close = _make_close(seed=2)
    t_idx = 300  # well past MOM_LONG_DAYS
    baseline = mom_carry_12_1(close).iloc[t_idx]

    perturbed = close.copy()
    perturbed.iloc[t_idx + 1 :] *= 1.5  # mutate every future date
    perturbed_val = mom_carry_12_1(perturbed).iloc[t_idx]

    assert np.isfinite(baseline)
    assert baseline == pytest.approx(perturbed_val), (
        "mom_carry_12_1 leaked future data: changing close[t+1:] altered the value at t"
    )


# ── 2. beta_dm ─────────────────────────────────────────────────────────────


def test_beta_dm_matches_rolling_cov_var_definition():
    rng = np.random.default_rng(42)
    n = 350
    idx = pd.bdate_range("2024-01-02", periods=n)
    market = pd.Series(np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n))), index=idx)
    # Stock = 1.7 * market_ret + ε; beta should converge to ~1.7.
    mret = market.pct_change().fillna(0.0)
    eps = rng.normal(0, 0.003, size=n)
    sret = 1.7 * mret + eps
    stock = (1.0 + sret).cumprod() * 100.0

    out = beta_dm(stock, market)
    # Hand-compute beta on the last window and match.
    last = pd.DataFrame({"s": stock.pct_change(), "m": market.pct_change()}).dropna().iloc[-BETA_WINDOW:]
    expected = float(np.cov(last["s"], last["m"], ddof=1)[0, 1] / np.var(last["m"], ddof=1))
    assert out.iloc[-1] == pytest.approx(expected, rel=1e-6, abs=1e-9)
    # And the convergence sanity: high-beta synthetic should yield ~1.7.
    assert abs(out.iloc[-1] - 1.7) < 0.1


def test_beta_dm_causal_at_t_does_not_use_future():
    close = _make_close(seed=10)
    spy = _make_close(seed=20)
    t = 300
    baseline = beta_dm(close, spy).iloc[t]
    perturbed_close = close.copy()
    perturbed_close.iloc[t + 1 :] *= 2.0
    perturbed_spy = spy.copy()
    perturbed_spy.iloc[t + 1 :] *= 2.0
    perturbed = beta_dm(perturbed_close, perturbed_spy).iloc[t]
    assert np.isfinite(baseline)
    assert baseline == pytest.approx(perturbed, rel=1e-9, abs=1e-12)


# ── 3. rvar_total ──────────────────────────────────────────────────────────


def test_rvar_total_matches_hand_sum_of_squared_returns():
    """Pin ``rvar_total`` against literal hand-computed sum-of-squared returns
    from the REAL helper. A wrong window size (e.g. 30 instead of 60) or a
    swapped-sign return would fail this. Fixes codex PR #16 LOW finding.
    """
    # Geometric-progression close with a constant per-bar return r so the
    # arithmetic is exact: close[i] = base * (1 + r) ** i  =>  pct_change == r.
    # rvar_total at index i is sum_{k=0..VOL_WINDOW-1} r^2 = VOL_WINDOW * r^2
    # for every i >= VOL_WINDOW (because pct_change[0] is NaN, the first
    # window with VOL_WINDOW valid returns lands at i = VOL_WINDOW).
    n = VOL_WINDOW + 10
    r = 0.0125  # 1.25 bp / day
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series([100.0 * (1.0 + r) ** i for i in range(n)], index=idx)
    actual = rvar_total(close)

    # Pre-warmup must be NaN: at i = VOL_WINDOW - 1 there are only
    # VOL_WINDOW - 1 valid returns inside the window (since pct_change[0] is NaN).
    assert actual.iloc[VOL_WINDOW - 1] != actual.iloc[VOL_WINDOW - 1]  # NaN
    # At i = VOL_WINDOW, all 60 returns inside the window are valid and equal r.
    expected_at_window_start = float(VOL_WINDOW) * (r ** 2)
    assert actual.iloc[VOL_WINDOW] == pytest.approx(
        expected_at_window_start, rel=1e-12, abs=1e-15
    )
    # The constant-return property means rvar_total is flat across the tail.
    tail = actual.iloc[VOL_WINDOW : VOL_WINDOW + 5]
    for val in tail:
        assert val == pytest.approx(expected_at_window_start, rel=1e-12, abs=1e-15)


def test_rvar_total_production_window_causal():
    close = _make_close(seed=3)
    t = 300
    baseline = rvar_total(close).iloc[t]
    perturbed = close.copy()
    perturbed.iloc[t + 1 :] *= 10.0
    after = rvar_total(perturbed).iloc[t]
    assert np.isfinite(baseline)
    assert baseline == pytest.approx(after, rel=1e-12, abs=1e-15)


def test_rvar_total_is_non_negative():
    close = _make_close(seed=4)
    out = rvar_total(close).dropna()
    assert (out >= 0).all()


# ── 4. idio_vol_market ─────────────────────────────────────────────────────


def test_idio_vol_market_returns_zero_resid_for_perfect_linear_combo():
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.bdate_range("2024-01-02", periods=n)
    spy = pd.Series(np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=idx)
    spy_ret = spy.pct_change().fillna(0.0)
    # Stock_ret = 1.0 * spy_ret exactly (no idio noise + no sector / size loading).
    # Construct close so that close.pct_change() == spy_ret elementwise.
    close = (1.0 + spy_ret).cumprod() * 50.0
    size = pd.Series(np.log1p(1_000_000.0), index=idx)
    out = idio_vol_market(close, spy, size, sector_close=None)
    # The residual std should be essentially zero (~ floating-point noise).
    tail = out.dropna().iloc[-50:]
    assert (tail.abs() < 1e-8).all(), f"expected ~0 residual std, got {tail.describe()}"


def test_idio_vol_market_causal_at_t_does_not_use_future():
    rng = np.random.default_rng(13)
    n = 200
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))) * 100, index=idx)
    spy = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, n))) * 300, index=idx)
    size = np.log1p(pd.Series(rng.uniform(1e5, 1e7, n), index=idx))
    sector = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.012, n))) * 80, index=idx)
    t = 150
    baseline = idio_vol_market(close, spy, size, sector).iloc[t]
    perturbed_close = close.copy(); perturbed_close.iloc[t + 1 :] *= 5
    perturbed_spy = spy.copy(); perturbed_spy.iloc[t + 1 :] *= 5
    perturbed_size = size.copy(); perturbed_size.iloc[t + 1 :] *= 5
    perturbed_sec = sector.copy(); perturbed_sec.iloc[t + 1 :] *= 5
    after = idio_vol_market(perturbed_close, perturbed_spy, perturbed_size, perturbed_sec).iloc[t]
    assert np.isfinite(baseline)
    assert baseline == pytest.approx(after, rel=1e-9, abs=1e-12)


# ── add_track_b_features end-to-end ────────────────────────────────────────


def test_add_track_b_features_appends_all_four_columns():
    n = 400
    idx = pd.bdate_range("2024-01-02", periods=n)
    spy_close = _make_close(seed=99)
    panel_rows = []
    for ticker, seed in (("AAA", 1), ("BBB", 2)):
        close = _make_close(seed=seed, n=n)
        vol = pd.Series(np.linspace(1e6, 2e6, n), index=idx)
        for d, c, v in zip(idx, close.values, vol.values):
            panel_rows.append(
                {"ticker": ticker, "date": d, "KMID": 0.0, "close": c, "volume": v}
            )
    panel = pd.DataFrame(panel_rows)
    out = add_track_b_features(panel, spy_close=spy_close)
    assert set(TRACK_B_FEATURES).issubset(out.columns)
    # After the warmup, every Track B column must have non-NaN values.
    tail = out.groupby("ticker").tail(50)
    for col in TRACK_B_FEATURES:
        assert tail[col].notna().any(), f"Track B feature {col} all-NaN at tail"


def test_add_track_b_features_requires_close_and_volume():
    panel = pd.DataFrame({"ticker": ["A"], "date": [pd.Timestamp("2025-01-02")], "KMID": [0.1]})
    with pytest.raises(KeyError, match="close \\+ volume"):
        add_track_b_features(panel, spy_close=_make_close(seed=1))


def test_track_b_feature_count_constants_match_canonical_references():
    # Lock the canonical-reference windows so a refactor can't silently swap
    # 252→200 or 60→30 without a test failure.
    assert MOM_LONG_DAYS == 252
    assert MOM_SKIP_DAYS == 21
    assert BETA_WINDOW == 252
    assert VOL_WINDOW == 60
    assert TRACK_B_FEATURES == ("mom_carry_12_1", "beta_dm", "rvar_total", "idio_vol_market")
