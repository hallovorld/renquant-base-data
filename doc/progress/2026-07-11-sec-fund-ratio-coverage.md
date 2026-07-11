# 2026-07-11 — SEC fundamentals ratio coverage fix + IMPUTED-SHARE guard

**Finding (orchestrator PR #475 META attribution):** in the serving feed
`sec_fundamentals_daily.parquet`, `earnings_yield` / `book_to_price` /
`gross_profitability` were finite for only **67 / 70 / 317 of 826** served
universe members (META NEVER finite) — panel models were valuation-blind on
most of the universe, with every missing cell silently cross-sectional-median
imputed at scoring time, every day, unalerted.

## Root causes (quantified on the real feed + live SEC frames, read-only)

Fetch-budget exhaustion was ruled out first (2026-07-11 production log:
`SEC frames fetched: 476/476`). Three real causes, in impact order:

1. **Whole-row as-of wipe (dominant).** `forward_fill_to_daily` merge_asof
   takes the ENTIRE latest filing row; most filers tag
   `us-gaap:CommonStockSharesOutstanding` only in the 10-K, so every
   subsequent 10-Q row erased the known share count → market cap NaN →
   ey/b2p NaN for ~475 of the 759 non-finite names (JPM/NVDA/AMZN class:
   shares present ONLY on the FY row).
2. **Single-tag concept mappings.** Multi-class filers (META, GOOGL…) tag
   shares per-class WITH dimensions — excluded from the non-dimensional
   frames API → no share tag ever (~241 names). `GrossProfit` is only tagged
   by issuers that present the subtotal (400/992; META/AMZN/NFLX/banks never)
   → 472 of 509 gp holes. ASC 606 filers left plain `Revenues`.
   Some filers only tag equity including noncontrolling interest.
3. **Dual-class CIK collision.** The scalar `{cik: ticker}` map is last-wins:
   8 of 10 dual-class universe pairs (GOOG, FOX, NWS, UA, LBRDA, LBTYA,
   FWONA, LLYVA) were entirely absent from the feed.

A fourth, out-of-scope cap: only ~150/2788 cached OHLCV files reach July
(131/831 served names priced on the last session), which limits the
price-dependent ratios regardless of this fix — owned by the OHLCV refresh,
now made visible by the guard's `priced_tickers` denominator.

## Fix (behavior-ADDITIVE by construction)

- `carry_forward_within_ticker` in `forward_fill_to_daily` (daily feed only):
  per-column ffill across the filing history before the as-of join; PIT-safe
  (rows sorted by `available_date`); never touches non-NaN cells; provenance
  columns always describe the latest filing.
- Fallback tag chains (`sec_edgar_companyfacts_harvester.CANONICAL_CONCEPTS`
  precedent), primary tag first so already-served rows keep exact values:
  shares → issued → weighted-average diluted/basic; equity → incl-NCI;
  gross profit → revenue − cost-of-revenue (both legs required).
  7 fallback concepts fetched AFTER primaries (budget exhaustion degrades to
  pre-fix coverage). mode=both requests 476 → 952 (~530s at production pace,
  within the 900s default budget).
- Multi-ticker CIK fan-out (`cik_tickers`) for the daily feed; the extended
  z-scored feed deliberately keeps the legacy scalar map AND filters fetched
  fallback concepts out (its train-window z-params must not move —
  byte-identical, pinned by test).
- ADDITIVE `price` column persisted (consumers select columns explicitly).

## IMPUTED-SHARE guard

- Build stamps `ingestion_manifest_sec_fundamentals_daily.json`
  (crypto_bars/sleeve_bars fingerprint pattern) with per-feature
  `coverage` / `n_have` / `n_expected` / `min_coverage` / `ok` —
  the renquant-pipeline DataAvailabilityGate `data_contracts.v1` axis
  vocabulary; build WARNS below floor (degrade_with_alarm).
- `python -m renquant_base_data.sec_fundamentals --verify` fails closed on
  fingerprint/content-sha/coverage-floor violations (registry manifest
  `manifests/sec-fundamentals-daily.json` declares it as validation_command).
- Default floors: ey/b2p 0.60 (of priced), gp 0.50, roe/asset_growth 0.60
  (of served) — between pre-fix bug level and post-fix measured coverage.

## Evidence (local 10-session rebuild from identical cached frames, old vs new code)

| feature | pre-fix | post-fix | denominator |
|---|---|---|---|
| earnings_yield | 68 (0.52) | 119 (**0.91**) | 131 priced |
| book_to_price | 70 (0.54) | 130 (**0.99**) | 131 priced |
| gross_profitability | 317 (0.38) | 511 (**0.61**) | 831 served |
| roe | 691 (0.84) | 776 (**0.93**) | 831 served |

- META: all three ratios now finite (ey 0.0156, b2p 0.142, gp 0.117 on
  2026-07-10; shares via weighted-average diluted 2.564e9, quarter-aligned).
- **Additivity proven:** 0 of the cells finite under old code changed value
  under new code (full 10-session × 826-ticker frame compare).
- `validate_pit_provenance` passes post-fix (fail-closed retained).

## Tests

`tests/test_sec_fund_ratio_coverage.py` (11): primary-tag value pins
(pre-fix formulas), META-class fallback recovery, no-partial-math gp,
carry-forward wipe fix + opt-in default-off, dual-class fan-out, extended
feed byte-identity, manifest stamp + fingerprint, verify floor/tamper
failure modes, CLI exit codes, priced-denominator coverage.
Full suite: 399 passed +11 new (the only failure, `test_fetchers_lift`, is a
worktree-siting artifact — it resolves the umbrella repo relative to the
checkout's parent dir; passes on a properly-sited checkout).

## Operator landing step (NOT done here — no production writes)

1. Merge, sync the live checkout / pins per the deploy SOP.
2. Re-run the weekly refresh (or wait for Saturday's cron):
   `python -m renquant_base_data.sec_fundamentals --mode both --universe scripts/watchlist_universe.json --data-dir data --end-year 2026`
   (fetch ~952 requests ≈ 9-10 min wall; default budget suffices).
3. Verify: `python -m renquant_base_data.sec_fundamentals --verify --data-dir data`
   → expect exit 0; earnings_yield `n_have` should jump 67 → ≥119 immediately
   (≈776 once the OHLCV cache freshens beyond the ~131 currently-priced
   names), book_to_price 70 → ≥130, gross_profitability 317 → ≈511.

## Codex CHANGES_REQUESTED response (2026-07-11, commit `360b778`)

Codex (haorensjtu-dev) reviewed commit `7d15771` CHANGES_REQUESTED: a
coverage/freshness guard that can certify stale or unavailable ratio
inputs as healthy is worse than no guard, because it creates false
confidence. Two P1 findings, both fixed; one test-suite claim corrected.

**P1 provenance freshness.** `forward_fill_to_daily`'s `ffill()` carries a
missing raw concept across filings, but the emitted
`fiscal_period_end`/`available_at` was always the NEWEST filing row — a
daily ratio could use shares/revenue/equity from an older filing while
appearing to have the newest quarter's provenance, defeating
`P-FUND-FRESHNESS`. Fixed by tracking provenance PER CARRIED CONCEPT:
`forward_fill_to_daily(track_concept_provenance=True)` adds
`<concept>__source_available_at`/`__source_fiscal_period_end` companion
columns that travel WITH each carried value (ffilled alongside it, so they
describe the OLDER filing that actually supplied it, not the newest
one). `compute_derived_features` derives each feature's own
`<feature>_source_available_at`/`_source_fiscal_period_end`/
`_source_age_days` from the OLDEST of its actual contributing operands
(via `_coalesce_with_provenance` + `_oldest_operand_provenance`) — a ratio
is only as fresh as its stalest input, even when a DIFFERENT operand's
newer filing made the row look current. `compute_feature_freshness()`
exposes a contractable per-feature max-age verdict
(`DEFAULT_FEATURE_MAX_AGE_DAYS`, default 150d).

Regression `test_carried_concept_retains_older_source_age_and_fails_tight_freshness_bound`
reproduces the exact scenario Codex specified: a 10-K (`end=2020-03-31`,
`available=2020-05-01`) tags `CommonStockSharesOutstanding=10.0`; the next
10-Q (`end=2020-06-30`, `available=2020-08-01`) refreshes
`NetIncomeLoss`/`Assets`/`StockholdersEquity` but omits the shares tag.
Evaluated on 2020-08-10: (a) `earnings_yield`/`book_to_price` are finite
(coverage still increases via carry-forward); (b) row-level provenance is
unchanged (`fiscal_period_end=2020-06-30`, `available_at=2020-08-01` — the
insufficient signal); (c) but `earnings_yield_source_available_at` /
`book_to_price_source_available_at` correctly equal `2020-05-01` (the
10-K), age 101 days; (d) a tight 60-day "current quarter" freshness bound
FAILS for `earnings_yield`/`book_to_price` but PASSES for `roe` (NI+equity
both refreshed by the 10-Q, age 9 days) — freshness is genuinely
per-feature, driven by each ratio's own operands.

**P1 coverage denominator.** The manifest measured
`earnings_yield`/`book_to_price` coverage only over `n_priced` (131 of 831
served) — 0.91 coverage could be reported while ~700 served/scored names
silently received no price-dependent ratio at all; the OHLCV failure was
hidden by the denominator choice. Fixed: `compute_feature_coverage()` now
emits BOTH the legacy priced/served-relative `coverage`/`n_expected` AND
`universe_coverage`/`n_universe_expected` (same finite-cell count, against
the FULL declared/scored `universe` — not `n_priced`), plus an axis-level
`prerequisite_price_coverage` (priced tickers / declared universe) that
caps `earnings_yield`/`book_to_price` regardless of ratio-input coverage.
The combined `coverage_ok` verdict (`legacy_coverage_ok AND
universe_coverage_ok AND prerequisite_price_coverage_ok AND
freshness_ok`) is unhealthy when ANY required contract fails; alarm/block
POLICY is left to the pipeline/orchestrator consumer per the review's own
scoping instruction.

Regression `test_universe_denominator_exposes_degraded_price_coverage_hidden_by_priced_denominator`
reproduces a 1-of-10-priced universe: `earnings_yield` reports a
misleading perfect legacy `coverage=1.0` (1/1 priced) while
`universe_coverage=0.10` and `prerequisite_price_coverage=0.10` correctly
flag the axis unhealthy; a price-INdependent feature
(`gross_profitability`) stays healthy on both denominators, showing only
the price-dependent features are capped by the OHLCV outage.

**Test-suite claim, corrected.** The prior claim ("the only failing test
… is a worktree-siting artifact") was investigated, not re-asserted
unchanged: `test_fetchers_lift::test_byte_equivalent_to_umbrella`'s
`_UMBRELLA = parents[2]/"RenQuant"/…` heuristic can find a directory
merely NAMED `RenQuant` at that fixed relative depth in an ad hoc
scratch/worktree layout. Reproduced directly in this session: an earlier
session's worktree parent directory contained a stale, non-git `RenQuant`
directory copy (no working `.git`, no `subrepos.lock.json`) with a
divergent `kernel/` — the OLD heuristic would find it and report a
genuine byte-**mismatch FAILURE**, not a clean skip, purely from siting,
not from any actual code drift. Fixed: the resolver now only trusts a
candidate that also carries the umbrella's own `subrepos.lock.json`
marker file (unique to a genuine checkout — every subrepo's
`RENQUANT_REPOS.md` is auto-generated FROM it), plus an explicit
`RENQUANT_UMBRELLA_PATH` env override for CI/worktree configs. Verified
three ways: (1) the real sibling checkout at
`/Users/renhao/git/github/RenQuant` → resolved, real byte comparison,
PASSES; (2) this fix's own isolated worktree (no `RenQuant` sibling) →
resolves to `None`, clean SKIP; (3) the earlier session's worktree
directory (the stale same-named copy) → now ALSO resolves to `None`
(marker check fails), clean SKIP instead of a false FAILURE.

`make test` in the isolated fix worktree: **402 passed, 1 skipped** (the
skip is `test_byte_equivalent_to_umbrella`, correctly skipping here — no
genuine `RenQuant` sibling in this worktree).

Manifest schema bumped to `sec-fundamentals-manifest-v2` (additive fields
only; no external consumer wired to this manifest yet — confirmed by a
cross-repo grep across renquant-pipeline/renquant-orchestrator/RenQuant
for `ingestion_manifest_sec_fundamentals_daily`,
`write_daily_ingestion_manifest`, `verify_daily_feed`, and
`fundamentals_serving_axis`).

Evidence: [renquant-artifacts#20](https://github.com/hallovorld/renquant-artifacts/pull/20)
— sealed before/after reproduction of both fixes
(`store/experiments/sec-fund-provenance-coverage-fix-20260711/`).
