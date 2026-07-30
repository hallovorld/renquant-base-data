"""A near-zero denominator means NOT COMPUTABLE, and must encode as NaN.

Every derived fundamental ratio used `num / denom`. Measured on the
shipped 830-name panel before this fix: `book_to_price` reached **1.68e19** on
21,722 rows (1.62% of non-null, 26 tickers) and `earnings_yield` 7.82e17 on
19,736 rows, with `gross_profitability` and `roe` hit too. `book_to_price`
carries 2.0% of the production scorer's gain, so those rows were trained on.

Two defects, and the second is the one that reached the model:
  1. `+1e-9` does not prevent explosion — it multiplies the numerator by 1e9.
  2. For `denom == 0` exactly, `0 + 1e-9` is POSITIVE, so the ratio silently
     takes the numerator's sign as though that were information.

And in the EXTENDED path the subsequent z-score + clip to [-3, 3] turned the
explosion into a legitimate-looking +/-3.0, so the defect was HIDDEN, not absent
— which is why a test on the output distribution alone would have passed.

These tests drive the real helper with denominators chosen around the floor. A
test asserting only "no output exceeds 1e6" would pass on a clip, which is the
wrong fix: a clipped value still asserts a magnitude that was never computed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renquant_base_data.sec_fundamentals import MAX_ABS_RATIO, _safe_ratio


def _s(*vals):
    return pd.Series(list(vals), dtype="float64")


def test_zero_denominator_is_nan_not_a_giant_number():
    out = _safe_ratio(_s(1.0e9), _s(0.0))
    assert out.isna().all(), f"expected NaN, got {out.tolist()}"


def test_zero_denominator_does_not_invent_a_sign():
    """The `+1e-9` form returned +1e18 for a positive numerator and -1e18 for a
    negative one, from the SAME zero denominator."""
    pos = _safe_ratio(_s(+1.0e9), _s(0.0))
    neg = _safe_ratio(_s(-1.0e9), _s(0.0))
    assert pos.isna().all() and neg.isna().all()


def test_the_exact_shipped_failure_reproduces_as_nan():
    """Reconstructed shape of the real rows: a normal numerator over a
    market cap driven to ~0 by a missing share count."""
    ni = _s(1.0e9, 1.0e9, 1.0e9)
    mktcap = _s(1.0e-9, 1.0e-3, 1.0)          # all far below the floor
    out = _safe_ratio(ni, mktcap)
    assert out.isna().all()
    # The pre-fix expression on the same inputs produced ~1e18/1e12/1e9.
    legacy = ni / mktcap
    assert (legacy.abs() > 1.0e8).all(), "the pre-fix form really did explode"


def test_material_denominators_divide_normally():
    out = _safe_ratio(_s(2.0e9), _s(1.0e10))
    assert out.iloc[0] == pytest.approx(0.2)


def test_negative_equity_still_divides_and_keeps_its_sign():
    """Book equity is legitimately negative for heavily levered firms. The floor
    is on the MAGNITUDE, so those rows must not be silently dropped."""
    out = _safe_ratio(_s(1.0e8), _s(-5.0e8))
    assert out.iloc[0] == pytest.approx(-0.2)


def test_the_bound_is_on_the_OUTPUT_and_is_unit_free():
    """Small-magnitude but sane inputs must survive. An absolute denominator
    floor broke 10 existing tests for exactly this reason."""
    out = _safe_ratio(_s(1.0, 2.0e-6), _s(4.0, 1.0e-6))
    assert out.iloc[0] == pytest.approx(0.25)
    assert out.iloc[1] == pytest.approx(2.0)


def test_at_and_beyond_the_output_bound():
    at = _safe_ratio(_s(MAX_ABS_RATIO), _s(1.0))
    beyond = _safe_ratio(_s(MAX_ABS_RATIO * 10.0), _s(1.0))
    assert not at.isna().iloc[0], "exactly at the bound must survive"
    assert beyond.isna().iloc[0], "beyond the bound must be NaN"


def test_nan_and_inf_inputs_propagate_as_nan_not_as_values():
    out = _safe_ratio(_s(1.0e9, np.nan, 1.0e9),
                      _s(np.nan, 1.0e10, np.inf))
    assert out.isna().tolist() == [True, True, True]


def test_output_is_never_infinite_for_any_denominator_scale():
    denoms = _s(*[10.0 ** k for k in range(-12, 13)])
    out = _safe_ratio(pd.Series([1.0e9] * len(denoms)), denoms)
    assert np.isfinite(out.dropna()).all()
    assert out.dropna().abs().max() <= MAX_ABS_RATIO


def test_index_is_preserved():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    num = pd.Series([1.0e9, 1.0e9, 1.0e9], index=idx)
    den = pd.Series([1.0e10, 0.0, 2.0e10], index=idx)
    out = _safe_ratio(num, den)
    assert out.index.equals(idx)
    assert out.isna().tolist() == [False, True, False]


def test_no_ratio_site_still_uses_the_plus_epsilon_form():
    """The regression pin. The documented 2026-06-24 fix existed only in a dead
    umbrella script (`_safe_ratio` had ZERO matches in this package), so the live
    producer kept the unsafe form for another five weeks. This asserts the live
    file, not a copy."""
    from pathlib import Path

    import renquant_base_data.sec_fundamentals as mod

    # Only CODE lines count. The first version of this test matched the comment
    # that documents the old form, which is a false positive of exactly the kind
    # that makes a regression pin untrustworthy — so it is scoped to code.
    code = [ln for ln in Path(mod.__file__).read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("#")]
    offenders = [ln.strip() for ln in code if "+ 1e-9)" in ln]
    assert not offenders, (
        f"a `denom + 1e-9` ratio site survives in code: {offenders}. Every "
        f"derived ratio must go through _safe_ratio.")
