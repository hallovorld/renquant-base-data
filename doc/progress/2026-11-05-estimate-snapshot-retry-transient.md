# 2026-11-05 — PIT estimate snapshot: retry transient fetch failures (3 consecutive partial days)

STATUS:   delivered, awaiting review — zero reviews at head.
WHAT:     `fmp_estimate_revisions.fetch_endpoint` now retries transport errors and HTTP 429/500/502/503/504 up to `FETCH_ATTEMPTS=4` with backoff 1 s / 3 s / 8 s before returning an error; the last error is returned if all attempts fail. 401/403/404 (plan-locked / bad key), bad JSON and unexpected shape are NOT retried. The partial / do-not-publish contract is unchanged: any (endpoint, symbol) that still fails after retries counts as a fetch error.
WHY/DIR:  `PIT estimate snapshot FAILED rc=1` on 2026-11-02, 11-03, 11-04. Each day 0-13 symbols per endpoint ended as `fetch_err` (a dropped connection — the same RemoteDisconnected flakiness that hung the rq105 collector) with NO retry; `had_fetch_failure` then forced `status=partial` and the day was not published. PIT history cannot be backfilled ("every missed day is UNRECOVERABLE"), so a handful of one-shot network drops was costing whole days.
EVIDENCE: see §4(b) below.
  artifact:      src/renquant_base_data/fmp_estimate_revisions.py, tests/test_fmp_estimate_revisions.py (31 passed).
  prod or exp:   exp — effective only after merge AND `renquant-base-data-run` sync (merged ≠ deployed); next scheduled snapshot picks it up.
  existing data: logs/pit_snapshots/estimate_snapshot_2026-11-02/03/04.log [VERIFIED 2026-11-05 11:3x PST]: http_err=0 on every endpoint, fetch_err 0-13 per endpoint (11-04: 7/3/11/12; 11-03: 4/8/13/10; 11-02: 2/4/5/0), status=partial, published=false. no_data 4-8 per endpoint is the known plan-locked set and is not affected.
  best-known?:   yes — anti-vacuity: with the source change stashed 6 of the 7 new tests fail; with it 31/31 pass.
  scope:         base-data src + tests only; no data written. Worst-case added wall time if FMP is fully down: 12 s x 4 endpoints x 142 symbols ≈ 2 h (bounded; run is a nightly job). Does not lower the coverage floor.
NEXT:     merge, sync -run (ask-first). The failed 11-02..11-04 days remain unrecoverable (FMP serves only the current estimate state); only forward days are saved.
