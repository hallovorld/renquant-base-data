# Progress: split factors MEASURED against the stored series, not reconstructed

STATUS:   builder + fail-closed guard + fixture tests + BLOCKER fix +
          reproducibility bundle DELIVERED, bundle re-verified against this
          commit's own SHA (round 4). No production path written; the
          corrected panel itself stays in scratch (data files are not
          committed — large-blob policy). The comparative V1-V5 report
          (magnitude-vs-stage1, negative control, residual jumps) remains
          prior-work-only, not reproduced in this pass — see FIX ROUND 3 /
          FIX ROUND 4 below.

          Revised after codex P1/P1/P2 (round 2, commit 729f68a), then again
          after a BLOCKER + reproducibility follow-up (round 3, commit
          85069bf), then again after a manifest-SHA-mismatch finding (round
          4, this commit). What changed and what it means:

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

FIX ROUND 3 (BLOCKER + reproducibility bundle, this commit):
  1. BLOCKER   `tools/build_pit_panel_shared.py` was imported by
               `build_pit_panel_v2.py` since the ORIGINAL commit but never
               committed -> `ModuleNotFoundError` on a fresh checkout, still
               true after round 2 (round 2 did not touch this). Added the
               file (verbatim Stage-1 machinery, as its own docstring says).
  2. HIGH      round 2's `_work_dir()` still defaulted to the SAME
               agent-session `/private/tmp/.../split-fix` path when
               `RQ_SPLIT_FIX_DIR` is unset — configurable, but the default
               still wasn't portable to "a later session or another
               machine" per the original review wording. Changed the
               fallback to a plain relative `scratch/split-fix` under cwd.
               `RQ_SPLIT_FIX_DIR` (round 2's env var name) is unchanged and
               still the way to point at existing artifacts.
  3. P1        reproducibility bundle DELIVERED (was PENDING after round 2):
               `build_split_factor.py` and `build_pit_panel_v2.py` now each
               write a `*.run_manifest.json` (input file hashes, raw-price-
               cache fingerprint, authoritative/unknown-retrieval counts,
               route/output counts, output hashes, tool git SHA) on every
               run. Reran both tools end-to-end against the ORIGINAL
               session's cached FMP inputs under the NOW-STRICTER round-2
               fail-closed logic — zero new network calls — and committed
               the result at
               `doc/design/2026-07-30-split-factor-run-manifest.json`. See
               EVIDENCE below for exactly what that rerun reproduced; the
               model A/B (renquant-model#107) can be unblocked on this
               bundle for the two headline features, though the separate
               V1-V5 comparative report below is still not re-verified (see
               EVIDENCE `existing data:`).
  4. MED       this EVIDENCE block was missing the required `artifact:` /
               `existing data:` §4(b) fields (flagged again after round 2
               left them out too). Added below.
  5. P2        added `tests/test_split_factor_measured_and_anchoring.py`
               (10 tests), complementing round 2's
               `test_split_factor_fail_closed.py` (which covers the
               reconstructed-route fail-closed contract) with the MEASURED
               route: short-segment merge (dropped when uncorroborated, kept
               when a real ex-date corroborates it), the NaN fail-closed
               residual path, and cover-date anchoring
               (`build_pit_panel_v2.factor_at`, both in-axis and the
               pre-axis calendar extension) — none of which round 2's test
               file touches. `pytest -q`: 494 passed, 1 skipped
               (pre-existing, unrelated) — full suite.

FIX ROUND 4 (manifest SHA mismatch, this commit):
  1. MED       round 3's committed manifest recorded both tool runs at
               `tool_git_sha=729f68a` — one commit BEHIND this PR's tip
               (`85069bf`, which added `tools/build_pit_panel_shared.py`
               and changed `_work_dir()`'s default fallback). The manifest
               did not actually cover the surface it was reviewed against.
               Reran both tools at HEAD=85069bf against the SAME cached
               FMP inputs (`RQ_SPLIT_FIX_DIR=/private/tmp/rqbd-split-repro`,
               zero new network calls) and replaced
               `doc/design/2026-07-30-split-factor-run-manifest.json` with
               the new run, now tagged `tool_git_sha=85069bf...`. Headline
               numbers reproduce exactly: route counts (824 measured / 5
               FAIL / 1 reconstructed), 298 authoritative no-event tickers,
               0 unknown-retrieval tickers, panel = 1,380,434 rows / 830
               tickers / 2014-01-02..2026-07-29, 0 PIT-validation
               violations, 120 share facts with no factor. The superseded
               round-3 manifest's identifying fields are kept in a
               `superseded` block in the same file rather than silently
               dropped. `pytest -q`: 494 passed, 1 skipped (pre-existing,
               unrelated) — full suite, unchanged by this round (no
               production code touched, only the committed manifest +
               progress doc).

WHAT:     `tools/build_split_factor.py` — per-ticker daily cumulative adjustment
          factor. `tools/build_pit_panel_v2.py` — the PIT panel rebuilt with
          split-adjusted share counts. `tools/build_pit_panel_shared.py` —
          the Stage-1 machinery it depends on (round 3 BLOCKER fix).
          `tools/harvest_splits_830.py`, `tools/harvest_raw_prices.py` —
          inputs. `tests/test_split_factor_fail_closed.py` (round 2),
          `tests/test_split_factor_measured_and_anchoring.py` (round 3) —
          regression tests.
          `doc/design/2026-07-30-split-factor-run-manifest.json` — the
          committed, checksummed evidence of a reproduced run, regenerated
          in round 4 at this commit's own `tool_git_sha` (see FIX ROUND 4).

WHY/DIR:  `earnings_yield` and `book_to_price` were BLOCKED in the as-filed PIT
          panel: as-filed `dei:EntityCommonStockSharesOutstanding` is not
          retroactively split-adjusted while `data/ohlcv` closes are, so
          `market_cap = shares x price` was discontinuous by the split factor.
          `book_to_price` alone carries 2.0% of the production scorer's gain, so
          these are the two fundamentals the live model leans on most.

EVIDENCE:
  artifact:       `doc/design/2026-07-30-split-factor-run-manifest.json` —
                   round 4 reran `tools/build_split_factor.py` and
                   `tools/build_pit_panel_v2.py` end-to-end via
                   `RQ_SPLIT_FIX_DIR` against the SAME cached FMP inputs the
                   original session harvested (raw-price cache, 830-ticker
                   split calendar + manifest), at HEAD=85069bf (this PR's
                   tip before this commit) — zero new network calls. The
                   round-3 manifest this supersedes was tagged
                   `tool_git_sha=729f68a`, one commit behind the surface it
                   was reviewed against; the superseded fields are kept in
                   the same file's `superseded` block, not dropped.
  prod or exp:    experiment. Read-only against production; no production
                   file written. Corrected panel + intermediate parquet stay
                   in `/private/tmp` scratch, never committed (large-blob
                   policy, matches this repo's Hard Boundaries).
  existing data:  `[VERIFIED-now]` the round-4 rerun reproduced the original
                   session's headline numbers exactly, under the round-2
                   stricter guard and now tagged to the correct commit: 5
                   FAIL tickers (APTV, DELL, FERG, RBC, SW), 120 share facts
                   with no factor, 53 dei units-error share facts across 36
                   tickers, 825/830 tickers with a published factor (824
                   measured + 1 reconstructed, that 1 ticker among the 298
                   with an authoritative no-event answer; 0 tickers had an
                   unknown/errored retrieval status on this harvest), 0
                   PIT-validation violations, panel = 1,380,434 rows / 830
                   tickers / 2014-01-02..2026-07-29. The deeper V1-V5
                   comparative report (per-split continuity spot-check
                   table, magnitude-vs-stage1, negative control,
                   residual-jump census) from the original session's scratch
                   `validate.py` was NOT re-executed in round 3 or round 4
                   either — it depends on files outside this PR's tree (a
                   separate `g6-stage1` baseline panel +
                   `crossval_inwindow_events.csv`). Treat the rows below (the
                   defect / after-repair / magnitudes / universe /
                   negative-ctrl figures) as `[VERIFIED — prior work,
                   2026-07-30 session, not reproduced in round 2, 3, or 4]`,
                   not freshly confirmed.
  best-known?:    Yes for the reproduced numbers above; the un-reproduced
                   comparative numbers are the prior session's best-known and
                   unchanged, just not independently re-verified since.
  scope:          `renquant-base-data` tools + tests + docs only. No pin, no
                   config, no production artifact touched.

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
  split_ratio     this corrects a warning I issued myself. I told the builder
  column caveat   that `split_ratio != 1.0` yields ~2,404 false events on
                  AAPL. The real figure is 150,111 across the universe, because
                  the column's "no split" sentinel is 0.0 (91.0% of rows) or NaN
                  (8.95%) and ZERO rows carry a literal 1.0 -- and only 63 of 830
                  tickers have the column at all. It is a dead legacy field with
                  no live writer, unusable as a primary route.

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
          `doc/design/2026-07-30-split-factor-run-manifest.json` carries the
          checksummed round-4 reproduction, tagged to this commit's own
          `tool_git_sha` (see EVIDENCE `artifact:` above); it is the
          reviewable, committed record. The original session's comparative
          V1-V5 report and continuity CSV lived only in `/private/tmp`
          scratch and were never committed — that gap is not closed by
          round 3 or round 4 either, since regenerating them needs files
          outside this PR's tree (see EVIDENCE `existing data:` above); a
          follow-up would need to either commit `g6-stage1`'s baseline panel
          reference or accept the comparative numbers as prior-work-only.
          Denominator safety matches #55's intent exactly: no epsilon, zero or
          non-finite denominator -> NaN, |ratio| > 1e6 -> NaN rather than
          clipped, and the guard sits strictly upstream of any z-scoring.

NEXT:     This unblocks the two features the production scorer leans on most, so
          the v1-vs-v2 A/B (renquant-model#107) can be re-registered to include
          them -- its Stage A deliberately excluded both, which is why its
          "no measurable source difference" verdict says nothing about them.
          Follow-up (not in this pass): commit or reference the `g6-stage1`
          baseline artifact so the V1-V5 comparative report can be
          regenerated and committed too, closing the gap noted above.
