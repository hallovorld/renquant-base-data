# OHLCV incremental fetch: NYSE-aware freshness (PR #50)

STATUS:    delivered
WHAT:      `_do_incremental_fetch` now gates cache-freshness on the module's
           existing NYSE-aware `_last_completed_nyse_session` helper (already
           used by the `is_fresh` path) instead of a raw 2-calendar-day cap,
           falling back to the calendar-day cap only when the calendar lib is
           unavailable.
WHY/DIR:   Fixes a freshness deadlock (GOAL-5 daily-run reliability): the old
           2-calendar-day cache-fresh check let `fetch_ohlcv_incremental` serve
           a 2-session-stale cache without refetching, while
           `PanelUniverseFreshnessGuardTask` fail-closes at 1 session stale.
           The mismatch meant the panel feed could never advance, so the
           weekly WF promote could never pass — leaving the production
           panel-LTR artifact 32 days stale. This fix makes the fetch-side and
           guard-side freshness definitions agree (both session-based),
           unblocking the WF promote.
EVIDENCE:
  artifact:      src/renquant_base_data/loaders/data.py::_do_incremental_fetch
                 (fix) + tests/test_ohlcv_incremental_freshness.py (new, 2 tests)
  prod or exp:   prod — shared OHLCV primitive used by all callers (daily
                 fetch, panel retrain)
  existing data: live repro before the fix: fetch_ohlcv_incremental("XOM")
                 returned 07-21 while a direct yfinance call had 07-23, and
                 293/293 panel tickers were uniformly stuck at 07-21 (ruling
                 out vendor lag, confirming a fetch-side bug)
  best-known?:   only variant implemented; no alternative freshness-gate
                 design was compared
  scope:         "this is _do_incremental_fetch (prod primitive), verified
                 live (seeded 07-21 cache -> fixed fetch returns 07-23; old
                 code returned 07-21) + tests/test_ohlcv_incremental_freshness.py
                 2/2 green (refetch-on-session-lag, no-network-when-fresh)"
NEXT:      Confirm the next scheduled weekly WF promote advances past the
           32d-stale artifact once this lands on main; no further code change
           anticipated from this PR.

## Blast radius
Shared primitive — all OHLCV callers now refetch when the cache trails the
completed session (marginally more network on the boundary day; no behavior
change when the cache is already current).
