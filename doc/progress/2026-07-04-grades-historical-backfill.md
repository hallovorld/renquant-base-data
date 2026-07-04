# Grades-historical PIT backfill

**Date:** 2026-07-04
**PR:** TBD (base-data)
**Status:** ready for review

## What

Backfill script that pulls FMP `grades-historical` monthly data (~91 months,
2018-12 → 2026-07) and writes it into the `estimate_snapshots/` layout as
`grades_consensus.parquet` per month. This lets `pit_revision_features` compute
`grade_migration_1m` immediately over 7+ years of history — no 90-day forward
wait for C1 signal measurement.

## Changes

1. **`backfill_grades_historical.py`** — fetcher + transformer + writer:
   - Pulls `grades-historical` for the full universe (~140 tickers)
   - Maps FMP column names (`analystRatingsStrongBuy` → `strongBuy` etc.)
   - Writes one `grades_consensus.parquet` + manifest per month
   - Dry-run by default, `--execute` to write, `--overwrite` for re-runs
   - Coverage floor (80%) prevents partial fetches from publishing
   - `pit_source=grades_historical_backfill` in every manifest

2. **`pit_revision_features.py` (`load_lake`)** — graceful missing-file handling:
   - Each endpoint's parquet is now checked with `.is_file()` before reading
   - Grades-only snapshot days produce grade features; estimate/target features
     are NaN (correct: the data doesn't exist for those months)

3. **16 new tests** + all 21 existing PIT revision feature tests pass (37 total).

## Why

FMP Starter doesn't provide historical EPS/revenue consensus snapshots —
`analyst-estimates` only returns the CURRENT consensus per fiscal period. But
`grades-historical` gives genuine PIT monthly rating distributions going back
~7 years. This is enough to compute `grade_migration_1m` (the breadth feature
from C1), which is one of the strongest alpha signals in the revision-drift
literature (Womack 1996).

Without this backfill, C1's `grade_migration_1m` wouldn't have enough history
until ~2026-10 (90 forward-collected days). With it, M-SIG measurement on the
grade-migration feature can start immediately.

## Provenance

The grades-historical data is aggregated by FMP on each month — it is NOT
fabricated or reconstructed by us. Each record's date is the month FMP computed
it. Manifests stamp `pit_source=grades_historical_backfill` to distinguish from
live forward snapshots.

## Round 2 (Codex review — two operational bugs)

**Hard-coded machine-specific paths.** `DEFAULT_ENV`/`DEFAULT_UNIVERSE_CONFIG`
were hard-coded to `/Users/renhao/git/github/RenQuant/...`, making the CLI
non-portable to CI or any other checkout location. Fixed by removing the
hard-coded defaults entirely: `--universe` is now a required argument (no
silent fallback to one workstation's `strategy_config.golden.json`), and
`load_api_key()` checks the `FMP_API_KEY` env var first, falling back to
`--env <path>` only if explicitly given — no path is ever assumed.

**Exit code 0 on rejected backfill.** `main()` unconditionally returned 0 even
when `backfill()` returned `{"status": "error", "reason": "below_coverage_floor", ...}`
— a CI/cron caller would see shell-success on a hard safety rejection unless it
re-parsed stdout. Fixed: `main()` now returns 1 whenever `result["status"] ==
"error"`, in both the `--json` and human-readable output paths (the
human-readable branch also stopped unconditionally accessing
`result["months_total"]` etc. on the error path, where those keys don't exist).

(This exact pair of fixes was independently implemented twice — once here,
once by a concurrent session that pushed to this same branch first. Reconciled
by taking the concurrent session's implementation as canonical, since it's
equally correct and simpler — required `--universe` instead of introducing a
parallel env-var-derived resolver — rather than merging two solutions to the
same problem. See `test_cli_coverage_failure_returns_nonzero` for the
regression coverage.)
