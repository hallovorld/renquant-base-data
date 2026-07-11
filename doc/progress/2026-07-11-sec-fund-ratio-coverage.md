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
