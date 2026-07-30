# 2026-07-30 — filing-lag fallback: 45d was LOOK-AHEAD on 10-Ks (PR #57)

STATUS:    delivered
WHAT:      `src/renquant_base_data/sec_fundamentals.py::FILING_LAG_FALLBACK_DAYS`
           raised `45 -> 60`. This is the fallback lag applied when the SEC
           frames API returns no `filed` date (90.37% of rows in production —
           the `sec_filed` provenance tier never fires because frames carry no
           filing timestamp). New `MEASURED_10K_P95_FILING_LAG_DAYS = 60`
           constant pins the measured bound. 3 new tests in
           `tests/test_filing_lag_fallback.py` pin the direction of the trade
           (fallback must not sit below the measured 10-K p95; must actually
           reject the old `45`; must stay under 120d).

           STACKING RESOLVED (was stale in an earlier revision of this doc,
           which said "Stacked on #55... retargets to `main` once #55
           merges"). #55 MERGED to `main` as `abefef7`. Verified this session:
           this branch is 0 commits behind `origin/main` and 2 ahead; its diff
           against `main` touches only the filing-lag constant, this doc, and
           `tests/test_filing_lag_fallback.py`; and it does NOT touch the
           ratio-denominator guard #55 landed (`git diff origin/main --
           sec_fundamentals.py` matches 0 lines mentioning
           computable/NaN/denom). So the two changes are cleanly separated in
           `main` + this branch, with nothing duplicated or reverted.
WHY/DIR:   Measured against 36,564 real SEC filing dates: 10-K median lag 53d,
           p95 60d, with 77.6% of 10-K filings exceeding the old 45d assumption
           (median +10d look-ahead, p95 +16d). 10-Ks are 24.6% of filings, so
           roughly 19% of filing events claimed a value was knowable before it
           was filed — concentrated in the annual-report window, which moves a
           fundamental feature the most. A single constant cannot be exact for
           both forms (frames payload carries no form field at the stamp
           point). **2026-07-30 review correction**: raising the fallback to
           the measured p95 is a RISK REDUCTION (cuts 10-K look-ahead
           incidence 77.6% -> ~5% of the same sample), not a correctness fix —
           by definition of "p95" the residual ~5% of 10-K filings (and a
           smaller 10-Q tail) still file after the new stamp, so this does NOT
           stop the fallback tier from being look-ahead, only reduces its
           incidence. No max-lag bound verified against the full filing corpus
           exists; that is unstarted follow-up. This does NOT make the panel
           point-in-time — real `filed` timestamps for 830 tickers exist on
           disk but under an unversioned, gitignored path with no refresh job
           or review (base-data #51/#53, both open).
EVIDENCE:
  artifact:      `src/renquant_base_data/sec_fundamentals.py::FILING_LAG_FALLBACK_DAYS`
                 (this PR); measured `[VERIFIED-now]` against 36,564 real SEC
                 filing dates.
  prod or exp:   read/compute path only — no config, artifact, panel, or pin
                 touched. Shifts availability dates on 90.37% of rows for any
                 panel regenerated after this lands; nothing is regenerated
                 here — that is a separate action requiring an A/B, since it
                 changes a production model input.
  existing data: 10-Q: n=27,554, median lag 33d, p95 40d, 0.5% exceed 45d
                 (conservative on 99.3%). 10-K: n=9,010, median lag 53d, p95
                 60d, 77.6% exceed 45d (median +10d look-ahead, p95 +16d)
                 `[VERIFIED-now]`. Tier mix: `expected_filing_lag` 90.37%,
                 `fmp_accepted` 2.88%, NULL 6.75%, `sec_filed` 0.00%.
  best-known?:   yes for this fallback-only fix — 60d is the measured 10-K
                 p95, a RISK REDUCTION (not a conservative bound — ~5% of
                 measured 10-K filings still exceed it) at the cost of ~15-27
                 extra staleness days on quarterly rows. A form-aware fallback
                 (infer fiscal-year-end per ticker, apply 60d only to annual
                 rows) would recover the lost quarterly freshness but adds
                 code/failure surface — named as follow-up, not built here.
                 A genuinely conservative (max-lag) bound verified against the
                 full filing corpus is a separate, unstarted follow-up; the
                 real PIT fix (base-data #51/#53) remains the only way to
                 eliminate the residual, not just reduce it.
  scope:         "this is a fallback-constant bug fix in renquant-base-data,
                 read/compute only, vs the pre-fix 45d that was look-ahead on
                 77.6% of 10-Ks; full suite 480 passed / 0 failed here
                 (author reported 479 passed / 1 skipped on the pre-rebase
                 tip), against a 465-passed / 0-failed baseline on `origin/main`"
NEXT:      wire `available_at_v2` (EDGAR true-PIT stamps, `doc/progress/2026-07-25-edgar-filing-dates.md`)
           into the fundamentals serving/panel build once its coverage claim
           is re-verified, which removes the need for this fallback entirely
           on the 830-ticker universe it covers; consider the form-aware
           fallback as an interim improvement if that wiring slips.
