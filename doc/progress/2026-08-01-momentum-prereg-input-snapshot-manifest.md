# Frozen-input snapshot manifest for the GOAL-7 momentum prereg (rev 2: content-addressed identity)

## Why this lives here

Review of renquant-model#171 (codex): a prereg amendment cannot point at an untracked
791 MB directory; large inputs need a durable manifest with complete fingerprints, and
**base-data owns the model-factory input**. This manifest is that publication.

## Review round 1, addressed

The first cut made a filesystem path the manifest's `location`. The review is right
that a mutable, provisional path cannot anchor an immutable dataset id. Rev 2:

* **Identity is the digest set, full stop** — `identity` block = dataset_id + panel /
  sector / combined-OHLCV sha256. No path, host, or root is part of it, so relocation
  NEVER touches identity.
* **Resolver is content-addressed** (`content-addressed-v1`): a VALID ROOT is any
  directory where every one of the 294 files verifies against its listed sha256.
  `candidate_roots` is a non-normative cache-hint list; digest failure is a finding,
  never grounds to silently try another root.
* **Validation tests added** (5, pure manifest-content, CI-safe): schema/version; the
  294-entry set with exactly 292 unique OHLCV names incl. SPY; the combined digest
  RECOMPUTED from the listed per-file entries (equals the stated value AND the frozen
  prereg pin, byte-for-byte); identity == files == prereg pins; no path inside
  identity. `[本次实测 2026-08-01]` 5 passed; full base-data suite 500 passed.

## One factual correction to the review

The review states `renquant-orchestrator#742` "does not exist". It does:
<https://github.com/hallovorld/renquant-orchestrator/issues/742> (the self-reported
umbrella-write incident whose operator disposition — KEEP / RELOCATE / REMOVE —
governs the current cache root's fate). The manifest now carries the full URL.

## What it pins `[本次实测 2026-08-01]`

294 file entries, each sha256 + byte size: the panel (`55811f63…`, 797,218,434 bytes —
read back from the manifest, not asserted), the sector snapshot (`ec26bb1e…`), and 292
per-ticker OHLCV files whose combined digest `4d4638a9…` reproduces the prereg §2
arithmetic exactly.

## Durable publication (the remaining half, operator-gated)

The BYTES currently sit in the provisional cache root recorded in `candidate_roots`,
pending orch#742. The recommended disposition is RELOCATE into a content-addressed
store directory outside every protected tree; because identity is digest-only, that is
an append to `candidate_roots` — no identity edit. The byte-move itself is a
machine-landing action and waits for operator authorization; nothing in this PR's
correctness depends on which root ends up hosting the bytes.

## Ordering (per the #171 review)

1. **This PR** (durable fingerprint record + self-validation);
2. renquant-model#171 references this manifest (identity by digest);
3. the #169 runner resolves THROUGH the manifest (verify-then-read, already
   implemented on its branch against the rev-2 shape).
