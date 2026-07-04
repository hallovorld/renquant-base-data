"""Anti-skew enforcement for the shared alpha158 operator module (campaign B8).

Three layers:
1. IDENTITY — the production panel builder's operators ARE the shared
   ``alpha158_ops`` objects (no re-implementation can drift silently).
2. CROSS-GRAIN LOCKSTEP — train grain (``rolling_features`` et al.) and serve
   grain (``compute_alpha158_at``) agree on the same OHLCV: exactly for the
   order-identical families, within fp-accumulation tolerance for the
   pandas-vs-numpy families.
3. DOCUMENTED DIVERGENCE — the RANK tie-handling skew is pinned AS IS
   (train=average-rank, serve=max-rank). If either side changes, this test
   trips: convergence is a model-lifecycle decision (retrain + gate), not a
   refactor. See KNOWN_TRAIN_SERVE_DIVERGENCES.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renquant_base_data import alpha158_ops as ops
from renquant_base_data import alpha158_qlib_panel as panel


# Families whose train/serve arithmetic is order-identical → must match
# EXACTLY (0.0, NaN==NaN).
EXACT_FAMILIES = (
    "ROC", "BETA", "RSQR", "RESI", "MAX", "MIN", "RSV",
    "IMAX", "IMIN", "IMXD", "CNTP", "CNTN", "CNTD",
)
EXACT_SINGLES = (
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
)
# Families computed via pandas rolling (moving-window accumulation) on the
# train grain vs fresh numpy windows on the serve grain → fp noise only.
# Measured max|delta| on 1600 real prod rows: 7.1e-10 (CORR10); bar 1e-8.
ACCUM_FAMILIES = (
    "MA", "STD", "QTLU", "QTLD", "CORR", "CORD",
    "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA",
    "VSUMP", "VSUMN", "VSUMD",
)
ACCUM_ATOL = 1e-8


def _synthetic_ohlcv(n_bars: int = 220, seed: int = 7) -> pd.DataFrame:
    """Deterministic, strictly positive OHLCV with continuous closes
    (no ties) and positive volume."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_bars)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_bars)))
    spread = np.abs(rng.normal(0.0, 0.008, n_bars)) + 1e-4
    open_ = close * (1 + rng.normal(0, 0.004, n_bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_00, 5_000_000, n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _train_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    feats.update(panel.kbar_features(ohlcv))
    feats.update(panel.price_features(ohlcv))
    feats.update(panel.rolling_features(ohlcv))
    return pd.DataFrame(feats, index=ohlcv.index)


def _classify(name: str) -> str:
    if name in EXACT_SINGLES:
        return "exact"
    fam = name.rstrip("0123456789")
    if fam == "RANK":
        return "rank"
    if fam in EXACT_FAMILIES:
        return "exact"
    if fam in ACCUM_FAMILIES:
        return "accum"
    raise AssertionError(f"unclassified feature {name}")


# ── 1. identity ─────────────────────────────────────────────────────────────

def test_panel_builder_uses_shared_ops_by_identity():
    """The builder's operators must BE the shared module's objects."""
    assert panel.kbar_features is ops.kbar_features
    assert panel.price_features is ops.price_features
    assert panel.rolling_features is ops.rolling_features
    assert panel._slope is ops.slope
    assert panel._rsquare is ops.rsquare
    assert panel._resi is ops.resi
    assert panel._idx_max_n is ops.idx_max_n
    assert panel._idx_min_n is ops.idx_min_n
    assert panel._greater is ops.greater
    assert panel._less is ops.less
    assert panel.WINDOWS is ops.WINDOWS
    assert panel.EPS == ops.EPS
    assert panel.STD_DDOF == ops.STD_DDOF


def test_feature_names_canonical():
    names = ops.alpha158_feature_names()
    assert len(names) == panel.EXPECTED_ALPHA158_FEATURES == 158
    assert len(set(names)) == 158
    assert names[:9] == list(EXACT_SINGLES[:9])       # KBAR block order
    assert names[9:13] == list(EXACT_SINGLES[9:])     # PRICE block order
    # every rolling family present for every window
    for n in ops.WINDOWS:
        for fam in ops.ROLLING_FAMILIES:
            assert f"{fam}{n}" in names
    # every name classified (guards this test against new unclassified ops)
    for name in names:
        _classify(name)


# ── 2. cross-grain lockstep ─────────────────────────────────────────────────

def test_train_serve_lockstep_cross_grain():
    ohlcv = _synthetic_ohlcv()
    assert not ohlcv["close"].duplicated().any()  # tie-free by construction
    train = _train_frame(ohlcv)
    names = ops.alpha158_feature_names()

    checked = 0
    for dt in ohlcv.index[-25:]:
        at = ops.compute_alpha158_at(ohlcv, dt)
        assert at and len(at) == 158
        t_row = train.loc[dt]
        for name in names:
            a, b = float(t_row[name]), float(at[name])
            if np.isnan(a) and np.isnan(b):
                continue
            cls = _classify(name)
            if cls == "exact":
                assert a == b, f"{name}@{dt.date()}: train={a!r} serve={b!r}"
            elif cls == "accum":
                assert a == pytest.approx(b, abs=ACCUM_ATOL), (
                    f"{name}@{dt.date()}: train={a!r} serve={b!r}")
            else:  # rank — tie-free fixture ⇒ conventions coincide
                assert a == b, f"{name}@{dt.date()} (tie-free): {a!r} vs {b!r}"
            checked += 1
    assert checked > 3500


def test_serve_frame_matches_serve_at():
    """The vectorized cache companion stays in lockstep with the at-bar path
    (this pins the invariant the old pipeline docstring cited a phantom test
    for)."""
    ohlcv = _synthetic_ohlcv(seed=11)
    frame = ops.compute_alpha158_frame(ohlcv)
    names = ops.alpha158_feature_names()
    assert list(frame.columns) == names
    for dt in ohlcv.index[-25:]:
        at = ops.compute_alpha158_at(ohlcv, dt)
        f_row = frame.loc[dt]
        for name in names:
            a, b = float(f_row[name]), float(at[name])
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b, abs=ACCUM_ATOL), (
                f"{name}@{dt.date()}: frame={a!r} at={b!r}")


# ── 3. documented divergences stay documented ───────────────────────────────

def test_rank_tie_divergence_pinned_as_documented():
    """RANK: train=average-rank / serve=max-rank on ties. This is LIVE
    behavior the prod model was trained/served on; it must not silently
    change on either side (KNOWN_TRAIN_SERVE_DIVERGENCES['RANK'])."""
    ohlcv = _synthetic_ohlcv(seed=3)
    # Force today's close to tie a value inside every window: repeat the
    # close from 2 bars back.
    ohlcv.iloc[-1, ohlcv.columns.get_loc("close")] = ohlcv["close"].iloc[-3]

    train = _train_frame(ohlcv)
    at = ops.compute_alpha158_at(ohlcv, ohlcv.index[-1])

    saw_divergence = False
    for n in ops.WINDOWS:
        t, s = float(train[f"RANK{n}"].iloc[-1]), float(at[f"RANK{n}"])
        # serve (max rank) >= train (average rank), always
        assert s >= t - 1e-15, f"RANK{n}: serve {s} < train {t}"
        # with a 2-way tie including today, they differ by 0.5/n
        if s > t:
            assert s - t == pytest.approx(0.5 / n, abs=1e-12)
            saw_divergence = True
    assert saw_divergence, "tie fixture failed to exercise the RANK skew"

    assert "RANK" in ops.KNOWN_TRAIN_SERVE_DIVERGENCES
    assert ops.KNOWN_TRAIN_SERVE_DIVERGENCES["RANK"]["severity"] == "material"


def test_divergence_registry_shape():
    for key, entry in ops.KNOWN_TRAIN_SERVE_DIVERGENCES.items():
        for field in ("severity", "train", "serve", "measured", "disposition"):
            assert entry.get(field), f"{key}: missing {field}"
