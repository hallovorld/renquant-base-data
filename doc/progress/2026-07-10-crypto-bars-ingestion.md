# Crypto bars ingestion + UTC session watermarks (D-C2)

**Date:** 2026-07-10
**PR:** base-data#41
**Status:** DRAFT — blocked on D-C1 (renquant-common), per Codex review round 1
**Spec:** merged crypto RFC — orchestrator `doc/design/2026-07-10-crypto-trading-rfc.md`
§2.3 (B1/B2/B3/B5/B6), §3.3, §3.5 (sealed data / watermark contract). Deliverable D-C2.

## Codex review round 1 (2026-07-10) — data-provenance and leakage fixes

Codex found the equity boundary sound but flagged four data-contract gaps.
All four addressed:

1. **D-C1 dependency (sequencing, not yet resolvable)**: `pair_slug`/
   `slug_pair` duplicate the canonical `renquant-common` helper the RFC
   assigns to D-C1, which doesn't exist as a merged PR yet — this creates
   exactly the duplicated-contract class `orchestrator#454` (architecture
   compliance audit) was merged to eliminate. **This PR stays in DRAFT**
   until D-C1 lands. Not silently deferred: the module docstring now
   carries an explicit, trackable TODO block naming the required follow-up
   (delete the local implementations, import the common helper, add a
   cross-repo parity test) as a condition for coming out of draft — not
   optional cleanup.
2. **Watermark completeness**: `ingest_crypto_bars` previously computed
   `watermark_utc` over only `status=="ok"` symbols, so a partial fetch
   (one pair failing) could silently advance a "signal-eligible-looking"
   watermark while another requested pair was missing. Fixed: the manifest
   now persists an immutable `expected_universe` (the full requested pair
   list) + `expected_universe_hash`, plus a `universe_complete` boolean
   (true only if EVERY expected pair sealed an `ok` bar) and a
   `signal_eligible` boolean (`universe_complete AND watermark_utc is not
   None`) that a consumer must check before trusting the manifest at all.
   `watermark_utc` itself stays populated from the ok-subset for ops
   diagnostics — it is explicitly documented as non-authoritative on its
   own. New test: 2 requested pairs, 1 fails — `universe_complete=False`,
   `signal_eligible=False`, watermark still recorded but the new
   eligibility gate rejects it outright.
3. **Manifest-bound eligibility**: `bars_eligible_for_session` only checked
   bar-close timestamps on an arbitrary dataframe — a late fetch could be
   labelled eligible purely because its bars happened to close at the right
   boundary. New `manifest_eligible_for_session(manifest, session_date, df)`
   is the mandatory gate: verifies fingerprint integrity, universe
   completeness, a non-null watermark, that the manifest's watermark
   actually matches the session's frozen watermark, AND that
   `generated_at_utc` falls inside the RFC §3.5 frozen Class-A signal
   cutoff window `[D 00:00, D 00:15)` UTC — rejecting (fail closed, raises
   `ManifestNotSignalEligibleError`) any manifest generated outside that
   window, any fingerprint mismatch, any incomplete universe, or any
   watermark that doesn't match the requested session. 7 new tests cover
   the happy path and every rejection mode (tamper, incomplete, watermark
   mismatch, before/at/inside/after the cutoff window).
4. **Hourly-fallback completeness**: `resample_hourly_to_utc_daily` used to
   emit a resampled daily candle from however many hourly bars existed,
   merely stamping `n_source_bars` as metadata — a day with 10 of 24 hours
   (provider outage, thin/late-listed pair) silently became a "valid" daily
   bar. Fixed: a day is now emitted ONLY if it has all 24 DISTINCT UTC
   hourly slots (`00:00`..`23:00`) present — checked via the actual hour-of-
   day SET, not just the row count, so a duplicate-hour row can't mask a
   real gap in the middle of the day. 3 new tests: 23 contiguous hours
   (fails), 24 rows but with a duplicate masking a mid-day gap (fails), and
   exactly 24 contiguous hours (succeeds).

Tests: `tests/test_crypto_bars.py` grew from 44 to 59 (15 new). Full suite:
353 passed, 1 skipped, zero regressions.

## What

`crypto_bars.py` — Alpaca spot-crypto bar ingestion (daily + intraday,
`CryptoHistoricalDataClient`, api_version v1beta3), slug store, UTC-session
watermark manifest, yfinance two-source parity check, and the alpha158
price/volume feature groundwork. Additive: the equity yfinance daily path is
behavior-unchanged (pinned by test).

## Decisions (with the RFC clause they implement)

1. **Slug encoding = `BTC/USD` → `BTC-USD`** (RFC §3.0, fixes B5): pair form
   for configs/API calls, slash→dash slug for every path/cache key. The slug
   coincides with yfinance's crypto ticker, so the cross-check needs no third
   form. `pair_slug`/`slug_pair` live here as a documented LOCAL STAND-IN —
   the RFC homes them in renquant-common (D-C1, not yet merged); when D-C1
   lands this module repoints and the round-trip tests freeze the semantics.
   Store layout: `data/crypto_ohlcv/{SLUG}/{tf}.parquet` (`1d`, `1h`, ...).
   Malformed symbols fail fast at validation (found by CLI probing:
   `BTC/USD/X` previously passed through as "pair form").

2. **UTC-day keying is verified, not assumed** (RFC §3.5): Alpaca's crypto
   1Day boundary is not contractually documented as UTC midnight, so
   `normalize_daily_bars_utc` REQUIRES exact UTC-midnight bar opens and
   fails closed (`VendorDailyNotUtcAlignedError`) into a deterministic
   1Hour→UTC-day resample fallback. Live check 2026-07-10 [VERIFIED]:
   v1beta3 1Day bars for BTC/ETH ARE UTC-midnight aligned (opens
   `T00:00:00+00:00`), so production takes the direct path; the fallback
   guards against vendor drift.

3. **Watermark contract** (RFC §3.5): every stored row carries
   `bar_close_utc`; only bars **closed AND fetched** (close ≤ fetch time) are
   ever written — today's in-progress bar is dropped, so a late vendor bar
   cannot backfill into a frozen signal. `ingest_crypto_bars` writes an
   atomic, self-fingerprinted manifest (`ingestion_manifest_{tf}.json`):
   per-symbol last-bar-close stamps + content sha256 (canonical bar bytes,
   parquet-encoding-independent), global `watermark_utc` = min over symbols,
   manifest fingerprint = sha256 over sorted-keys JSON.
   `load_crypto_ingestion_manifest` fails closed on tamper.
   `bars_eligible_for_session(df, D)` implements "session D consumes only
   bars closing ≤ D 00:00:00 UTC" (day D-1's bar, closing exactly at the
   watermark, IS eligible per §3.5). Determinism pinned: same bars + same
   clock ⇒ identical content sha and manifest fingerprint.

4. **Provider seam, equity path untouched** (B1/B2):
   `fetch_ohlcv(provider="alpaca_crypto")` delegates immediately to
   `crypto_bars.fetch_crypto_daily_cached` (own store, own UTC freshness
   clock — `loaders.data`'s NYSE logic never runs for crypto). The yfinance
   branch is byte-identical; `tests/test_crypto_bars.py::
   test_equity_daily_path_byte_identity` pins cache-serve frame equality and
   store layout. Crypto freshness = last completed UTC day (ALWAYS_OPEN
   stand-in until D-C1/M2 ships the canonical calendar). No feed argument on
   the crypto client (single US feed); same optional `ALPACA_API_KEY`/
   `ALPACA_SECRET_KEY` env creds; `call_with_timeout` on every network call.

5. **Feature groundwork, reuse not fork** (B7):
   `build_crypto_features_for_pair` delegates to the existing
   `alpha158_qlib_panel.build_features_for_ticker`, which is verified
   asset-agnostic — it consumes only the shared `alpha158_ops`
   kbar/price/rolling operators over OHLCV (fundamentals live in the
   separate `alpha158_fund_panel`, never touched). Test pins the output to
   exactly the 158 price/volume features and identity with the shared
   builder. **No label changes**: the SPY-excess sidecar stays equity-only;
   crypto labels are D-C3/D-C4 scope.

6. **Manifest registry (B6)**: `manifests/crypto-ohlcv-{1d,1h}.json` register
   the datasets as `asset_class:"crypto"` under the existing schema (no
   schema change); the authoritative per-run fingerprints live in the
   ingestion manifests.

7. **Two-source parity (§3.3)**: `crosscheck_daily_close` (pure) +
   `run_yfinance_crosscheck` (injectable secondary fetcher; slug = yfinance
   ticker). Live check 2026-07-10 [VERIFIED]: 15 overlapping days BTC,
   max relative close delta 9.4 bps, 0 breaches at 1% tolerance.

## Verification (beyond unit tests)

Drove the CLI end-to-end against the real (free, unauthenticated, read-only)
Alpaca v1beta3 API into an isolated tmp store: daily ingestion (15 bars,
in-progress 07-10 bar correctly excluded, watermark `2026-07-10T00:00Z`),
hourly ingestion (in-progress hour excluded), manifest fingerprint reload,
session-eligibility filter, provider seam via `fetch_ohlcv`, malformed-pair
and unknown-provider error paths. No production path written.

## Tests

`tests/test_crypto_bars.py` — 44 tests: symbol round-trip + rejection, slug
store paths (B5 nested-dir break impossible), UTC freshness, watermark
boundary semantics, vendor-alignment fail-closed + hourly fallback, sealed
manifest (validation, tamper-evidence, determinism), cache-first reads,
provider seam, equity byte-identity pin, alpha158 reuse identity, parity
check, fake-client fetch (skips without alpaca-py). No live API dependency:
CI (no alpaca-py/openbb/pyarrow) skips only the SDK-shaped test and
parquet-dependent tests via importorskip. Full suite: 338 passed, 1
pre-existing environment-only failure (`test_fetchers_lift` compares against
a sibling `../RenQuant` checkout; passes in the primary checkout, unrelated
to this change).

## Codex review round 2 (2026-07-10) — content-binding + duplicate-hour gaps

Codex confirmed round 1's universe-completeness, cutoff-window, and
24-distinct-hour fixes, and found two further provenance defects (the raw
review text had some function/variable names dropped by what looks like a
markdown code-span stripping glitch on Codex's end; both points below were
reconstructed by reading the actual code directly, not guessed from the
degraded text):

1. **Manifest content-binding**: `manifest_eligible_for_session` verified
   the manifest's OWN fingerprint (`verify_crypto_manifest` — internal
   self-consistency) but never checked that the `df` a caller hands in
   alongside the manifest is the data the manifest actually sealed. An
   intact, untampered, complete-universe manifest could therefore be paired
   with modified rows, or even another symbol's frame, and the function
   would still wave it through. Fixed: `manifest_eligible_for_session` now
   takes a required `symbol` argument whenever `df` is given, recomputes
   `_content_sha256(df)`, and requires it to equal
   `manifest["symbols"][symbol]["content_sha256"]` exactly — fails closed
   (`ManifestNotSignalEligibleError`) on a content mismatch, an unsealed
   symbol, or (via a plain `ValueError`) a missing `symbol` argument when
   `df` is supplied. Four new tests: happy path re-verified with a real
   content hash, tampered-row rejection, wrong-symbol rejection, unsealed-
   symbol rejection.
2. **Duplicate-hour double-counting**: `resample_hourly_to_utc_daily`'s
   round-1 fix (hour-of-day SET equals all 24 slots) correctly catches a
   missing hour, but not a 25-row day where one hour appears twice and the
   SET of hours still covers all 24 — that duplicate would silently double-
   count its hour's volume in the aggregation, inflating an otherwise
   apparently-valid day's reported volume. Fixed: completeness now requires
   BOTH the hour-set-equals-24 check AND an exact `len(group) == 24` row
   count; either condition alone is insufficient (a set-only check misses
   the duplicate-inflated case; a count-only check misses a genuinely
   missing hour padded by an unrelated duplicate). New regression test: 25
   rows, all 24 distinct hours present via a duplicated hour — day is
   dropped, not emitted with inflated volume.

Verified meaningful: reverted only the source fix (kept the new tests) and
confirmed all 6 new/changed tests fail without it — 3 with `TypeError:
unexpected keyword argument 'symbol'` (proving the parameter genuinely
didn't exist before), 3 with assertion/logic failures (proving the
duplicate-hour and content-mismatch bugs are real, not hypothetical).

Tests: `tests/test_crypto_bars.py` 59 → 64 (5 new). Full repo suite: 358
passed, 1 skipped, zero regressions. PR remains DRAFT (point 1 from round 1
— the D-C1 `renquant-common` dependency — is unresolved and out of scope
for this round).

## Round 2 follow-up (same review): store-artifact binding path

Codex r2's ask was "manifest-and-**store/content-hash** bound: require the
symbol/**store artifact** or an explicit df". The first r2 fix delivered the
explicit-`df` half; this follow-up completes the store half so the gate can
do the load-and-verify itself (a consumer that hand-loads a frame and then
"forgets" the binding step is exactly the foot-gun the review names):

- `manifest_eligible_for_session(..., symbol=, store=)` now loads the
  sealed artifact `{SLUG}/{tf}.parquet` from the given `CryptoLocalStore`
  (timeframe taken from the manifest itself), recomputes the canonical
  content hash, and requires equality with the sealed per-symbol
  `content_sha256`. Missing/empty artifact fails closed
  (`ManifestNotSignalEligibleError`), as does an unknown manifest
  timeframe. `symbol` accepts pair or slug form (normalized via the strict
  helpers, so a malformed symbol is rejected rather than mis-looked-up).
- 3 new tests: end-to-end ingest→store-bound consumption (including
  post-seal parquet tamper → rejection), missing-artifact rejection,
  store-without-symbol rejection.

Tests: `tests/test_crypto_bars.py` 64 → 67. Full repo suite: 361 passed
(the 1 `test_fetchers_lift` failure is the known worktree-environment
sibling-path artifact; passes in the primary checkout). PR remains DRAFT on
the D-C1 renquant-common prerequisite (canonical always-open calendar +
`pair_slug` helper); the local stand-ins repoint when D-C1 lands.
