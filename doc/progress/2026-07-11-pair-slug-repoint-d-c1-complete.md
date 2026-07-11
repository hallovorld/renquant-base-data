# Pair/slug symbol helper repoint — D-C1 complete

**Date:** 2026-07-11
**PR:** base-data#41
**Spec:** merged crypto RFC — deliverable D-C1 (symbol-helper half), completing the
repoint alongside the calendar half already done (common#27).

## What

`common#29` (canonical `renquant_common.pair_slug`) merged as 0.13.0, completing D-C1.
`pair_slug`/`slug_pair`/`_as_pair`/`_as_slug` in `crypto_bars.py` now DELEGATE to the
canonical module instead of hand-rolling the same logic locally — the local
stand-in (explicitly labeled as such since the original PR) is gone. Same
structural-requirement pattern as the calendar repoint
(`_require_always_open_calendar`): `_require_common_pair_slug()` raises a loud,
named `RuntimeError` if the installed renquant-common predates common#29 (checked
both via `ImportError` on the module itself, and via `hasattr` on its expected
surface — no local fallback either way). Version floor bumped
`renquant-common>=0.11` → `>=0.13`.

## Tests

Added `test_pair_slug_matches_canonical_common_helper` (parity across the
representative pairs + the malformed-input battery) and
`test_pair_slug_fails_closed_on_pre29_common` (structural fail-closed, mirroring
the calendar's own fail-closed test). Caught and fixed two real bugs while writing
these:

1. The parity test's first draft tried `from renquant_common import
   pair_slug/slug_pair` (function-level import from the package root) — wrong,
   since `pair_slug` is submodule-scoped (not re-exported at root, same
   convention as `cost_model`/`model_fingerprint`); fixed to
   `from renquant_common.pair_slug import pair_slug, slug_pair`.
2. The fail-closed test's first draft tried `monkeypatch.setitem(sys.modules,
   "renquant_common.pair_slug", None)` to simulate a missing module — doesn't
   work, since the real file genuinely exists on disk in this test environment
   and Python re-imports it fresh once the cache entry is cleared. Fixed by
   adding a second, testable check inside `_require_common_pair_slug()` itself
   (`hasattr(ps, "as_pair")`), mirroring the calendar fail-closed test's own
   `monkeypatch.delattr` idiom, which IS meaningfully mockable.

Verified meaningful via stash-revert: `test_pair_slug_fails_closed_on_pre29_common`
correctly FAILS against the pre-repoint source (the local implementation never
calls `_require_common_pair_slug`, so the monkeypatch has no effect) — this is the
test that specifically proves real delegation happened, not just that two
independent-but-identical implementations agree (which the parity test alone
would not distinguish). Full suite: 365 passed, 1 skipped.

## D-C1 status

Complete. Both halves (calendar, common#27; symbol helper, common#29) are now
canonically consumed with no local stand-ins remaining.
