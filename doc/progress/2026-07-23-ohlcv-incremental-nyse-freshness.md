# OHLCV incremental fetch: NYSE-aware freshness (PR #50)

STATUS:    delivered
WHAT:      `_do_incremental_fetch` now gates cache-freshness on the module's
           existing NYSE-aware `_last_completed_nyse_session` helper (already
           used by the `is_fresh` path) instead of a raw 2-calendar-day cap,
           falling back to the calendar-day cap only when the calendar lib is
           unavailable. Follow-up fix (this revision): the freshness
           reference timestamp is now `_market_timestamp(end)` (real
           wall-clock-in-ET, the same helper `has_range` already uses for
           this decision) instead of `end_ts` — `end_ts` is
           `pd.Timestamp.now().normalize()` on the default `end=None` path,
           i.e. always midnight, which is always before NYSE close, so
           `_last_completed_nyse_session` never counted today's
           already-closed session on that path and a same-day-stale cache
           was served without refetching (Codex review, PR #50).
WHY/DIR:   Fixes a freshness deadlock (GOAL-5 daily-run reliability): the old
           2-calendar-day cache-fresh check let `fetch_ohlcv_incremental` serve
           a 2-session-stale cache without refetching, while
           `PanelUniverseFreshnessGuardTask` fail-closes at 1 session stale.
           The mismatch meant the panel feed could never advance, so the
           weekly WF promote could never pass — leaving the production
           panel-LTR artifact 32 days stale. This fix makes the fetch-side and
           guard-side freshness definitions agree (both session-based),
           unblocking the WF promote. The follow-up closes the same gap on
           the default `end=None` caller path (e.g. `sleeve_bars.py`), which
           the first revision left unfixed.
EVIDENCE:
  artifact:      src/renquant_base_data/loaders/data.py::_do_incremental_fetch
                 (fix) + tests/test_ohlcv_incremental_freshness.py (3 tests:
                 2 original + 1 new for the end=None after-close path)
  prod or exp:   prod — shared OHLCV primitive used by all callers (daily
                 fetch, panel retrain)
  existing data: live repro before the fix: fetch_ohlcv_incremental("XOM")
                 returned 07-21 while a direct yfinance call had 07-23, and
                 293/293 panel tickers were uniformly stuck at 07-21 (ruling
                 out vendor lag, confirming a fetch-side bug). Codex's
                 follow-up repro: cache seeded through 07-22, `pandas.Timestamp.now()`
                 patched to 2026-07-23 16:55 America/New_York (post-close),
                 `fetch_ohlcv_incremental('XOM', store=...)` with no `end` ->
                 network_calls=0, max_date=07-22 (still stale) on the
                 pre-follow-up code.
  best-known?:   only variant implemented; no alternative freshness-gate
                 design was compared
  scope:         "this is _do_incremental_fetch (prod primitive). Verified:
                 (1) explicit-end path, live (seeded 07-21 cache -> fixed
                 fetch returns 07-23; old code returned 07-21); (2) default
                 end=None path, new regression test
                 test_incremental_default_end_none_refetches_after_close —
                 fails against pre-follow-up code (0 network calls, stale
                 cache served) and passes against this revision (refetches,
                 reaches the completed session). Full
                 tests/test_ohlcv_incremental_freshness.py: 3/3 green."
NEXT:      Confirm the next scheduled weekly WF promote advances past the
           32d-stale artifact once this lands on main; no further code change
           anticipated from this PR.

## Blast radius
Shared primitive — all OHLCV callers now refetch when the cache trails the
completed session (marginally more network on the boundary day; no behavior
change when the cache is already current).
