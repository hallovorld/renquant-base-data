# Progress: sidecar single-writer unification amendment

Date: 2026-07-18

## What

Amendment to the merged sentiment-reconciliation RFC after AC-1
falsified its premise: the served sidecar has TWO active weekly writers
with contradictory recipes (base-data builder 176+extension, rejected
weekly; orchestrator σ-head refresh 179 no-extension, succeeds weekly) —
a writer war, not a legacy holdout. The (b) migration recommendation is
REVERSED by evidence: 99 active sanity contracts (incl. the prod XGB
scorer) name the sentiment columns.

Amended resolution: one file one writer — base-data builder becomes the
sole producer with a 179-col sentiment-carrying contract; σ-head refresh
stops writing (consumes or derives its view, AC-B'); extension-row
disposition frozen by consumer evidence; guard passes by construction.
ACs: writer cessation, σ-head fit equivalence, Saturday-chain dry run,
migration integrity inherited verbatim, signature closure.

## Status

RFC amendment only — no implementation, no migration.
