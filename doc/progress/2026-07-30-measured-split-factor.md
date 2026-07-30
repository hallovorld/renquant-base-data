# Progress: split factors MEASURED against the stored series, not reconstructed

STATUS:   builder + fail-closed guard + fixture tests DELIVERED. The corrected
          panel and its continuity statistics are **PENDING REPRODUCIBLE
          EXECUTION** — not delivered. No production path written.

          Revised after codex P1/P1/P2. What changed and what it means:

          P1a FAIL-OPEN CLOSED (data-correctness bug). `reconstructed_factor`
          emitted `cum_factor = 1.0` whenever the union calendar had no row for
          a ticker. That conflates two different things: FMP answering "no split
          history" (authoritative) with the request having FAILED (nothing
          known). `harvest_splits_830.py` already recorded both and its own
          manifest note says "errors are NOT authoritative and must be treated
          as unknown" — the builder just never read it. So an errored ticker
          with no raw-price fallback silently published an UNADJUSTED
          market-cap basis as if verified. Now `load_split_retrieval_status()`
          reads the manifest and 1.0 is emitted ONLY against a recorded
          authoritative no-event response; anything else FAILS. The default
          argument is `None`, which DENIES, so a caller that forgets to pass
          the set cannot fail open either. A missing manifest fails every
          empty-calendar ticker — no manifest, no authority.

          Additional gap found while fixing it: the manifest recorded only
          `error_samples: errs[:20]`, so even a builder that DID read it could
          not know every errored ticker. It now also writes the full
          `error_tickers` list; the loader falls back to deriving from the
          truncated samples for older manifests, and says so.

          P1b REPRODUCIBILITY. All four tools hard-coded an agent-session path
          under `/private/tmp`, which made every number they produce
          unreproducible by anyone else. Now `_work_dir()` reads
          `RQ_SPLIT_FIX_DIR`, defaulting to the previous path so existing
          artifacts keep resolving. The panel/continuity claims are downgraded
          to PENDING above rather than restated: the run bundle with input
          checksums, request status, retrieval time and output fingerprints
          does not exist yet, and the model A/B stays BLOCKED on it.

          P2 CI COVERAGE. `tests/test_split_factor_fail_closed.py` (7 tests)
          exercises the builder against fixtures instead of only an untracked
          scratch run: the exact error-plus-missing-raw case, an authoritative
          no-event ticker still getting 1.0, a ticker absent from the manifest,
          a missing manifest, an omitted status argument, a real split event
          still reconstructing, and a legacy truncated-manifest fallback.
          Verified load-bearing rather than decorative: with the guard reverted,
          5 of the 7 FAIL.

WHAT:     `tools/build_split_factor.py` — per-ticker daily cumulative adjustment
          factor. `tools/build_pit_panel_v2.py` — the PIT panel rebuilt with
          split-adjusted share counts. `tools/harvest_splits_830.py`,
          `tools/harvest_raw_prices.py` — inputs.
          `doc/research/data/2026-07-30-split-factor-validation.txt` + the
          continuity CSV.

WHY/DIR:  `earnings_yield` and `book_to_price` were BLOCKED in the as-filed PIT
          panel: as-filed `dei:EntityCommonStockSharesOutstanding` is not
          retroactively split-adjusted while `data/ohlcv` closes are, so
          `market_cap = shares x price` was discontinuous by the split factor.
          `book_to_price` alone carries 2.0% of the production scorer's gain, so
          these are the two fundamentals the live model leans on most.

EVIDENCE: `[VERIFIED-now]`
  the defect      the step does NOT bite at the ex-date (shares have not updated
                  yet) but at the FIRST FILING AFTER it. NVDA 10:1 -> 9.33x
                  market-cap step; CMG 50:1 -> 48.89x; BKNG 25:1 -> 24.55x;
                  AMZN 20:1 -> 20.09x; AAPL 4:1 -> 3.97x; TSLA 5:1 -> 5.14x.
  after repair    those steps become 0.933 / 0.978 / 0.982 / 1.005 / 0.993 /
                  1.028. Median |log step| 1.1136 -> 0.0192, a 58x reduction;
                  0 of 70 corrected steps exceed 1.5x, against 63 of 70 before.
  magnitudes      earnings_yield max 3.134 (was 12.046), book_to_price max 9.528
                  (was 13.695); |x| > 1e3 is ZERO on both, so the 1e19 class that
                  motivated renquant-base-data#55 is absent by construction.
  universe        the two features go from 0 usable tickers to 685 at >=250
                  non-null days (book_to_price 711, earnings_yield 700). The
                  3-feature baseline reproduces at exactly 515, unchanged.
  negative ctrl   COHR, LITE, LOW have factor == 1.0: max|new - old| =
                  0.000e+00 across 3161 / 2724 / 2657 rows. The correction does
                  not move unsplit names.
  prod or exp:    Read-only. No production file written; mtime scan confirms zero
                  files changed under the umbrella tree.
  best-known?:    Yes, and it corrects a warning I issued myself. I told the
                  builder that `split_ratio != 1.0` yields ~2,404 false events on
                  AAPL. The real figure is 150,111 across the universe, because
                  the column's "no split" sentinel is 0.0 (91.0% of rows) or NaN
                  (8.95%) and ZERO rows carry a literal 1.0 -- and only 63 of 830
                  tickers have the column at all. It is a dead legacy field with
                  no live writer, unusable as a primary route.
  scope:          `renquant-base-data` tools + docs only. No pin, no config, no
                  artifact, no panel overwritten.

WHY MEASURED, NOT RECONSTRUCTED:
          `CumAdj(t) = raw_price(t) / close_stored(t)`, using FMP's
          non-split-adjusted endpoint for true raw prices. AAPL 2020-08-28:
          499.24 / 124.8075 = 4.000080, and 1.000000 after the split.
          Three cases a split CALENDAR would have got wrong, each verified:
            * MMM 2024-04-01 (1.196) is ABSENT from FMP's split endpoint yet
              demonstrably present in the stored prices;
            * HON 2026-06-29: FMP says 0.500, the ohlcv column says 0.9535, and
              the measured factor is 0.476771 = 0.5 x 0.9535 -- two co-dated
              events whose factors MULTIPLY, so either source alone is wrong;
            * CBSH: the stored close embeds only 3 of 4 FMP-listed 5% stock
              dividends, so applying the 4th would inject a 5% error.
          Cross-validation: 811/825 measured factors agree with the calendar
          product within 0.5%, and every disagreement is a case where the
          measurement is right about the stored series. 103 strongly
          discriminating events all test ADJUSTED, zero UNADJUSTED.

SCOPE/LIMITS:
          * The factor is evaluated at each share fact's OWN COVER DATE, not the
            trading date. Between an ex-date and the next filing the newest
            as-filed count is still pre-split, and using F(t) there understates
            market cap by the entire factor.
          * 5 tickers FAIL CLOSED -- APTV, DELL, FERG, RBC, SW: raw and stored
            price series disagree about the instrument (5.5-30% within-segment
            residual), so no factor is published and both features are NaN,
            never silently 1.0.
          * 1 unresolved: QXO (616x, an implausible 6.64e5 share `prev`).
          * 53 as-filed share facts across 36 tickers dropped as dei UNITS
            errors (PKG 8.99e10, AJG 1.9e14, LIN 2.5e4). Pre-existing EDGAR
            tagging defect, NOT the split mismatch; it accounts for 749 of the
            115,952 changed rows (0.65%), the other 99.35% falling inside the
            split-factor range at a median correction of 3.0x.
          * NOT fixed, flagged: the stored `close` series is itself internally
            inconsistent for a few names (CBSH's missing 4th stock dividend; nine
            tickers with an anomalous trailing bar dated 2026-05-12). The factor
            compensates correctly for market-cap purposes BECAUSE it was measured
            against that same stored close, but the panel's own `price` column
            still carries those discontinuities.
          * 2,657 spurious exact zeros surfaced and were closed: all GLD, a gold
            trust reporting StockholdersEquity = 0, now NaN via the same equity
            floor `roe` already used.
          * The builder's own first pass had a bug, found and fixed: nine tickers
            anchored on a 1-row segment at a file end-date, propagating a 3-5%
            error across their history. Steps must now persist or be corroborated
            by a real ex-date.

VERIFICATION:
          `doc/research/data/2026-07-30-split-factor-validation.txt` carries the
          full report; the continuity CSV is auditable by eye per the table
          above. Denominator safety matches #55's intent exactly: no epsilon,
          zero or non-finite denominator -> NaN, |ratio| > 1e6 -> NaN rather than
          clipped, and the guard sits strictly upstream of any z-scoring.

NEXT:     This unblocks the two features the production scorer leans on most, so
          the v1-vs-v2 A/B (renquant-model#107) can be re-registered to include
          them -- its Stage A deliberately excluded both, which is why its
          "no measurable source difference" verdict says nothing about them.
