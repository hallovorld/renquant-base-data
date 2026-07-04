# Re-point to the canonical NYSE market calendar (campaign B5)

Date: 2026-07-04
PR: fix(calendar): re-point to the common canonical

## What

`loaders/data.py::_last_completed_nyse_session` — the ORIGINAL copy the
orchestrator freshness guard hand-mirrored (audit #296 §4.1 row 3, XC-2) —
now delegates to the canonical
`renquant_common.market_calendar.last_completed_session`. The hand-rolled
schedule logic is deleted.

## Semantics / behavior

- Equivalence-proven: identical outputs on a 10-year daily fixture
  (2016-01-01..2026-12-31, 4 intraday probes per date incl. half-day closes
  and the exact-close boundary) against the old copy — run pre-re-point in
  the campaign workspace.
- The old 14-day lookback vs the canonical 30-day default: divergence class
  unreachable on real NYSE calendars (longest modern non-trading stretch ~6
  calendar days); proven immaterial by the fixture.
- Fail-mode: the canonical raises (fail-closed); this call-site keeps
  base-data's lenient contract EXPLICITLY — `except Exception -> None`, so
  the caller's conservative 2-calendar-day staleness cap takes over. A stale
  deployed renquant_common (predating market_calendar) degrades the same
  way; no crash.
- Dependency floor bumped: `renquant-common>=0.10,<1.0`.

## Merge order

renquant-common `feat(calendar): canonical NYSE market calendar (campaign
B5)` merges FIRST; this PR's CI is red until it lands (CI checks out
common@main).
