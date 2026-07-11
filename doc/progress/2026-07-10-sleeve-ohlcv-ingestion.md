# sleeve-ohlcv-1d — SGOV/SPY daily-bar ingestion + pinned serving contract (pipeline #185 / RS-1)

**Date:** 2026-07-10 · **Branch:** `feat/sleeve-ohlcv-ingestion` · **Repo:** renquant-base-data
**Supersedes:** umbrella RenQuant#459 (closed per Codex repo-ownership verdict)

## Bottom line

renquant-pipeline#185 (parking-sleeve `mode="live"`, RS-1 SGOV floor) fail-closes
on a missing SGOV price; SGOV daily bars exist in no production store
`[VERIFIED 2026-07-10]`. Codex ruled the umbrella must not regain runtime data
ownership — ingestion, storage, freshness/manifest fingerprinting, and the
serving contract belong HERE, consumed by pipeline #185 as a pinned artifact
through the multi-repo run manifest. This PR ships that base-data slice as
code only; the actual backfill run stays a separate operator-granted landing
step.

## What shipped

* `src/renquant_base_data/sleeve_bars.py` — dataset `sleeve-ohlcv-1d`
  (SPY + SGOV legs), patterned on the crypto RFC D-C2 ingestion
  (`crypto_bars.py`), with NYSE-session freshness instead of UTC watermarks:
  * `ingest_sleeve_bars` — idempotent ingestion via the existing
    `loaders.data.fetch_ohlcv_incremental` (cache-first, ~10y cold start,
    timeout-protected) into the equity store layout
    `<store>/{SYMBOL}/1d.parquet`; stamps the sealed per-run manifest
    `<store>/ingestion_manifest_sleeve_1d.json` with per-symbol content
    sha256, universe-completeness hash, and a single `serving_eligible`
    verdict (complete AND every leg session-fresh).
  * Fingerprint mechanism REUSED from `crypto_bars.manifest_fingerprint`
    (sha256 over sorted-keys JSON minus `fingerprint`) — one impl on
    purpose; the calibrator triple-impl bug is the counterexample.
  * `resolve_sleeve_leg_bars` — the serving contract: consumer passes the
    run-manifest's pinned fingerprint; fail-closed on manifest tamper, pin
    mismatch, non-`ok` symbol, and parquet-vs-manifest content drift.
  * CLI `python -m renquant_base_data.sleeve_bars` — DRY-RUN by default;
    `--write` ingests + stamps; `--verify` audits seal/content/freshness
    (the manifest's `validation_command`); exit 2 refusal if a provided
    strategy config has the SGOV leg in its watchlist.
* `manifests/sleeve-ohlcv-1d.json` — registry entry (dataset_id, schema,
  fingerprint pointer, `store://ohlcv/1d` URI, provider, owner, freshness
  rule, retention class, validation command) resolvable via
  `registry.resolve_data_manifest`; passes `validate_data_manifest`.
* `tests/test_sleeve_bars.py` — 23 tests, no network, no production paths:
  normalization mirror-pin ×6, ingestion/manifest (seal, tamper, partial
  universe, stale leg, determinism, normalization) ×6, CLI ×6, and the
  **cross-repo pinned-artifact integration class** ×5: registry resolution
  of the real checked-in manifest → ingestion → run-manifest fingerprint
  pin → `resolve_sleeve_leg_bars("SGOV", pinned_fingerprint=...)` returns
  content-verified bars; fail-closed variants for pin mismatch, post-seal
  store drift, non-eligible symbol; explicit assertion that no resolved
  path references the umbrella.

## Findings carried over from RenQuant#459 (still true, cited)

1. **Symbol source of truth** — the umbrella's
   `adapters/sleeve_prices.parking_sleeve_leg_tickers` normalization
   (st104#39 follow-up) is authoritative. base-data is UPSTREAM of every
   consumer, so importing it here would invert the dependency graph;
   `sleeve_leg_tickers` mirrors it exactly and is mirror-pinned by test
   (same convention `crypto_bars` used pre-canonicalization). Consumers
   holding a strategy config keep resolving legs with the umbrella helper
   and pass them in. If Codex prefers a hard single impl, the lift target
   is `renquant-common` (follow-up, same as pair_slug → common#29).
2. **Watchlist / P-CONFIG-FP non-coupling `[VERIFIED]`** — the panel config
   fingerprint (`renquant_common.config_consistency._model_relevant_fields`)
   hashes watchlist / `panel_ltr` flags / sector maps only; neither the
   `sleeve` section nor any bar store enters the hash. SGOV joins price
   coverage only, never panel scoring/admission. The CLI additionally
   refuses (exit 2) on an sgov-in-watchlist config.

## Not in scope (explicit)

Serving-to-runner wiring: the daily runner today reads its own store; the
consuming repos (pipeline/orchestrator run manifest) adopt
`resolve_sleeve_leg_bars` + the pinned fingerprint as a follow-up. Nothing
here imports, references, or wires the umbrella.

## Evidence

* `tests/test_sleeve_bars.py` → **23 passed**.
* Full suite → **388 passed, 1 failed** — the failure is
  `test_fetchers_lift.py::test_byte_equivalent_to_umbrella`, pre-existing
  on clean `main` in the same environment (compares against a sibling
  umbrella checkout; unrelated to this diff).

## Operator landing step (after merge — separate grant)

```bash
cd <renquant-base-data checkout>
python -m renquant_base_data.sleeve_bars                       # inspect (dry-run)
python -m renquant_base_data.sleeve_bars --write               # backfill + stamp manifest
python -m renquant_base_data.sleeve_bars --verify              # audit the sealed artifact
```

Store target is the deployment's OHLCV store root via `--data-dir` (the
repo-root `data/ohlcv` by default). Expected: `SGOV/1d.parquet` (~10y bars),
refreshed `SPY/1d.parquet`, `ingestion_manifest_sleeve_1d.json` with
`serving_eligible=true`, exit 0.

## Rollback

Revert the commit — additive module + manifest + tests; no existing module
was modified.
