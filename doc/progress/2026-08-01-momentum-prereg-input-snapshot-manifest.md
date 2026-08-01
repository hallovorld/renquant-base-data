# Frozen-input snapshot manifest for the GOAL-7 momentum prereg

## Why this lives here

Review of renquant-model#171 (codex): a prereg amendment cannot point at an untracked
791 MB directory; large inputs need a durable manifest with complete fingerprints, and
**base-data owns the model-factory input**. This manifest is that publication.

## What it pins `[本次实测 2026-08-01]`

`manifests/momentum-prereg-inputs-20260801.json` — 294 file entries, each with sha256 +
byte size: the panel (`55811f63…`, 797,218,434 bytes — read back from the manifest, not asserted), the sector snapshot
(`ec26bb1e…`), and 292 per-ticker OHLCV files whose combined digest `4d4638a9…`
reproduces the prereg's §2 arithmetic exactly (verified at manifest build; the three
headline digests are byte-identical to the frozen prereg pins).

## Contract

Digests NORMATIVE, location ADVISORY. The location is PROVISIONAL pending
renquant-orchestrator#742 (the snapshot was written to the umbrella tree in breach of a
session rule — self-reported; the operator picks KEEP / RELOCATE / REMOVE). RELOCATE =
a reviewed revision of the `location` field only. Consumers must verify every read
against these digests and refuse on mismatch (UNRESOLVED-DATA), never fall back to the
live paths.

## Ordering (per the #171 review)

1. **This PR** (the durable fingerprint record);
2. renquant-model#171 revised to reference this manifest by path + content sha;
3. the #169 runner revised to resolve inputs THROUGH the manifest (verify-then-read).
