# 2026-07-30 — ratio denominator floor: NaN not 1e19 (PR #55)

STATUS:   SPLIT per codex review. This PR now contains ONLY the ratio-safety
          change. The 45->60 filing-lag fallback fix that was bundled here
          (via the #56 merge into this branch) is moved to base-data#57, on a
          clean branch from main with no #56 history — it changes
          point-in-time semantics for most future rebuilt rows and needs its
          own focused PIT review, validation and rollback path, which it
          cannot get while bundled with an unrelated numerical-safety fix.
          Verified after the split: FILING_LAG_FALLBACK_DAYS is back to main's
          45 on this branch and MEASURED_10K_P95_FILING_LAG_DAYS is absent, so
          the two PRs touch disjoint regions of sec_fundamentals.py.

STATUS:    delivered
WHAT:      `src/renquant_base_data/sec_fundamentals.py` replaces every
           `num / (denom + 1e-9)` derived-ratio expression (`earnings_yield`,
           `book_to_price`, `gross_profitability`, `roe`, `asset_turnover`,
           `profit_margin`, `return_on_assets`, `debt_to_assets`) with a new
           `_safe_ratio()` helper: NaN on a zero or non-finite denominator,
           NaN when `|ratio| > MAX_ABS_RATIO (1e6)`, sign preserved for a
           genuinely negative denominator. 11 new tests in
           `tests/test_ratio_denominator_floor.py`;
           `tests/test_sec_fund_ratio_coverage.py` updated because it had the
           `+1e-9` epsilon baked into its own expected values (was pinning
           the defect, not the fix).
WHY/DIR:   The `+1e-9` epsilon does not bound explosion (denom=1e-9 still
           multiplies the numerator by 1e9) and invents a sign on an exact-zero
           denominator. The 2026-06-24 `_safe_ratio(eps=1.0)` fix in the
           umbrella's `scripts/fetch_sec_fundamentals.py` never reached
           production because that script is dead code — the live producer is
           this package's `sec_fundamentals.py`, invoked by
           `weekly_fundamental_refresh.sh` — so the defect survived five weeks
           past its documented fix date. In the extended path the subsequent
           z-score + clip to `[-3, 3]` turned the explosion into a
           legitimate-looking `+/-3.0`, hiding rather than removing it.
EVIDENCE:
  artifact:      `src/renquant_base_data/sec_fundamentals.py::_safe_ratio`
                 (this PR); measured on the shipped 830-name panel
                 `[VERIFIED-now]`.
  prod or exp:   read/compute path only — no config, artifact, panel, or pin
                 touched by this PR. Existing panels on disk still contain
                 the pre-fix values; regenerating them is a separate,
                 explicitly out-of-scope action (changes ~1.6% of
                 `book_to_price` rows on an input the production scorer
                 trained on — needs its own A/B, not a quiet swap).
  existing data: pre-fix, `book_to_price` reached 1.6832e19 on 21,722 rows
                 (1.616% of non-null, 26 tickers); `earnings_yield` 7.82e17 on
                 19,736 rows (1.500%); `gross_profitability` 9.061e17 on 374
                 rows; `roe` 6.3311e16 on 688 rows `[VERIFIED-now]`.
                 `book_to_price` carries 2.0% of the production scorer's gain
                 `[VERIFIED-prior]`.
  best-known?:   yes for this defect class — bounds the OUTPUT magnitude
                 (unit-free) rather than the denominator (unit-coupled); an
                 absolute USD floor (first attempt, `1e6`) broke 10 of 465
                 previously-passing tests by voiding legitimately
                 small-denominator filers.
  scope:         "this is a compute-path bug fix in renquant-base-data,
                 read/compute only, vs the dead-code umbrella fix that never
                 shipped; baseline (`origin/main`, separate worktree) 465
                 passed / 0 failed, this branch 476 passed / 1 skipped / 0
                 failed"
NEXT:      regenerate the on-disk panel with the fixed ratios and A/B the
           production scorer against the corrected `book_to_price` /
           `earnings_yield` / `gross_profitability` / `roe` columns; repoint
           `RenQuant#545` (filed against the dead `fetch_sec_fundamentals.py`)
           at this file instead.
