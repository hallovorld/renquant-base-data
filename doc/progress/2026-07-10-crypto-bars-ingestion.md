# Crypto bars ingestion + UTC session watermarks (D-C2)

**Date:** 2026-07-10
**PR:** TBD (base-data)
**Status:** ready for review
**Spec:** merged crypto RFC — orchestrator `doc/design/2026-07-10-crypto-trading-rfc.md`
§2.3 (B1/B2/B3/B5/B6), §3.3, §3.5 (sealed data / watermark contract). Deliverable D-C2.

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
