# Per-entity fiscal provenance on the daily fundamentals feed

Date: 2026-07-03. Repo: renquant-base-data. Branch: `feat/fund-feed-provenance`.

## Problem

The S12 promote gate (RenQuant `scripts/promote_shadow_patchtst.py`,
`fundamentals_sla_verdict`) refused promotion:

> fundamentals ... QUARTERLY UNVERIFIABLE — no per-entity
> fiscal-period/available-at provenance (entity_col=ticker, fiscal_col=None,
> columns=['asset_growth','book_to_price','date','earnings_yield',
> 'gross_profitability','roe','ticker']); fail-closed until it exists

The gate's contract (mirrors renquant-pipeline P-FUND-FRESHNESS): the daily
feed must carry a real entity id (`ticker` — present) AND a per-entity
fiscal-period/available-at column (first present of `fiscal_period_end`,
`period_end`, `report_date`, `filed_date`, `acceptance_datetime`,
`available_at` — NONE present). Without it, per-entity quarterly coverage is
unverifiable and a single fresh issuer could otherwise certify a frozen panel
(the late-filer hole). Correctly fail-closed; the fix is data-side, here.

## Change (`src/renquant_base_data/sec_fundamentals.py`, ADDITIVE)

Every `(ticker, date)` row of `sec_fundamentals_daily.parquet` (and the
extended feed) now also carries:

- `fiscal_period_end` — fiscal-period end (`end`) of the latest filing whose
  values the row reflects (already tracked in the quarterly panel; previously
  dropped at the forward-fill step).
- `available_at` — the PIT date those values became available (the
  `available_date` merge_asof key; previously dropped).
- `available_source` — which availability tier stamped `available_at`.

Availability tiers (a tier is used only when every earlier tier has no genuine
timestamp):

1. `sec_filed` — max SEC `filed` date over contributing concepts. (The XBRL
   frames API returns no `filed` field, so in production this tier is empty.)
2. `fmp_accepted` — join from `data/fmp_harvest_5y/income_statement*.parquet`
   `acceptedDate` on `(ticker, fiscal_period_end)` — the C2 same-filing
   assumption (orchestrator m-sig spec): the income statement's EDGAR
   acceptance timestamp dates the whole filing. Day-granularity never-precedes
   rule: stamp = `max(date(acceptedDate), filingDate)` (post-close acceptances
   roll to FMP's next-day `filingDate`). Rows whose stamp precedes the period
   end (corrupt) are dropped, not clamped.
3. `expected_filing_lag` — period end + 45d (pre-existing conservative
   assumption; unchanged; never zero-lag).

PIT is enforced fail-closed at build time (`validate_pit_provenance`):
`available_at <= date` on every stamped row and
`fiscal_period_end <= available_at`; violations refuse to write the feed.
Rows before an entity's first filing keep NaT provenance (counted MISSING by
the gates, never fresh).

C2 spot-check (5 tickers, AAPL/MSFT/NVDA/JPM/XOM, latest 3 FYs each, live
`fmp_harvest_5y`): 15/15 rows have `acceptedDate >= period end` and
`filingDate >= date(acceptedDate)`, lags 24–59d (matches real 10-K timing).
Aggregate over 1,324 rows: 1,296 pass the >= period-end check; the 28
violating rows are exactly what the loader drops to tier 3. Median lag 48d —
note the 45d tier-3 assumption is mildly aggressive for 10-Ks (60d deadline);
acceptable because tier 3 only applies where no genuine timestamp exists and
predates this change.

## Gate verdict on a rebuilt fixture (actual gate code, read-only import)

- OLD schema: `on_sla=False` — reproduces the refusal verbatim.
- REBUILT feed, 60 entities all current: `on_sla=True` — `coverage n=60
  current=60 ... stale_frac=0.000 worst=0q OK`.
- REBUILT feed, one late filer (2q behind) while the global max fiscal date
  stays fresh: `on_sla=False` — `STALE-COVERAGE: worst=2q>1` (the exact
  guarded failure, now detectable).

## Consumers checked (all select columns explicitly -> unaffected)

base-data `alpha158_fund_panel` (keep-list; regression test added),
pipeline `job_panel_scoring`/`task_data_integrity` (hardcoded 5-col lists),
pipeline `fundamentals_freshness`/`_data_root` (reads `["date"]`/existence),
model `panel_data.FUND_COLS`, umbrella `production_runner`/
`train_production_model` (explicit col lists), promote gate (benefits).

## Tests

`tests/test_fund_feed_provenance.py` (12): additive regression (+ downstream
panel-builder ignores new cols), the late-filer-vs-global-max case, PIT
never-precedes (row-level + look-ahead/impossible fail-closed + NaT before
first filing), tier precedence, C2 join incl. corrupt-row drop and post-close
rollover, end-to-end FMP tier. Full suite: 218 passed.

## Landing (not done by this PR — machine landing is ask-first)

1. Merge; bump the base-data pin in umbrella `subrepos.lock.json`; sync the
   live sibling checkout of renquant-base-data on the daily-run machine.
2. Rebuild the real feed via the existing weekly job path
   (`scripts/weekly_fundamental_refresh.sh` → `python -m
   renquant_base_data.sec_fundamentals --mode both --data-dir RenQuant/data`);
   the `fmp_harvest_5y` default resolves under that data dir. No config
   change needed.
3. Re-run the promote gate; fundamentals axis should now report per-entity
   coverage instead of UNVERIFIABLE.
