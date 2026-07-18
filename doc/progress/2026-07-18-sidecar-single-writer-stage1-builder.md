# Progress: sidecar single-writer — Stage 1 (canonical 179-col builder)

Date: 2026-07-18

## What

Stage 1 of the merged single-writer amendment
([`doc/design/2026-07-18-sidecar-single-writer-amendment.md`](../design/2026-07-18-sidecar-single-writer-amendment.md),
base-data#48). Makes `renquant_base_data.rawlabel_sidecar` the SOLE
writer's builder for the served
`alpha158_291_fundamental_dataset_rawlabel.parquet` by unifying its recipe
with the σ-head writer's served schema. **Code + tests only — NO migration,
NO served-file mutation, NO landing** (state migration is AC-D, an ask-first
umbrella runbook, not in this PR).

## Changes (`src/renquant_base_data/rawlabel_sidecar.py`)

- **Un-froze `SENTIMENT_COLS` (§2.2).** `RAWLABEL_SIDECAR_COLUMNS` is now the
  full 178-column fund-panel schema (sentiment INCLUDED) + `fwd_60d_excess_raw`
  = **179 columns**, matching the served file the 99 active/candidate sanity
  contracts require (AC-1). The three sentiment columns sit at the panel tail,
  just before the appended raw label — exactly the σ-head served order.
  Sentiment is now a REQUIRED contract column (a sentiment-less panel fails
  closed as schema drift, replacing the old "tolerated legacy shape").
- **Flipped `extend_to_bar_frontier` default OFF (§2.3, AC-B').** The canonical
  served-file recipe now carries ZERO bar-frontier extension rows — the last
  recipe divergence from the σ-head writer, which rejected them. The opt-in
  `extend_to_bar_frontier=True` path is retained (reserved for a SEPARATE
  artifact, never a recipe fork of the served file); CLI flag changed from
  `--no-extend-to-bar-frontier` to opt-in `--extend-to-bar-frontier`.
- **NaN-never-ffill for missing sentiment (§2.4).** Sentiment values flow
  through from the panel unchanged; a missing value stays NaN (the builder
  never forward-fills — a ffill would fabricate staleness as signal). Pinned
  by test.
- **Deleted the obsolete "the served sidecar predates them" docstring** (both
  the module recipe step and the `SENTIMENT_COLS` doc comment) — falsified by
  AC-1. Module/param docstrings updated to the new contract; the top WHY
  `max(date)` claim adjusted to the panel feature frontier (no extension).

## Tests

- `test_rawlabel_sidecar_columns_179.json` (renamed from `_176.json`) +
  `test_rawlabel_sidecar_schema_export.py`: drift guard now pins the 179-col
  contract with sentiment carried at the panel tail.
- `test_rawlabel_sidecar.py`:
  - 179-col contract shape (sentiment present, ordered) [pinned].
  - **AC-B' zero-extension-row pin** on the plain default build.
  - sentiment carried from the panel; missing sentiment stays NaN (never
    ffilled) [§2.4].
  - missing sentiment columns now **fail closed** (was: tolerated).
  - **canonical-contract parity (§2.5):** base-data tests only its OWN
    schema + row-domain contract — the default (canonical) build emits EXACTLY
    the 179-column schema, in order, sentiment INCLUDED, with ZERO bar-frontier
    extension rows (`n_extension_rows == 0`, `n_rows == n_panel_rows`). The
    earlier draft ported one branch of the RenQuant umbrella refresh guard
    (`scripts/refresh_transformer_corpus.py::_sanity_reasons`) into a base-data
    test; that was **removed** — a partial port of a foreign guard is a
    cross-repo boundary violation (inconsistent with AC-C umbrella ownership)
    that would silently diverge as the umbrella evolves. Exact guard execution
    (the new 179 output passes the guard by construction; the old 176-col
    recipe reproduces the "dropped columns (recipe/schema drift)" Saturday
    rejection naming the 3 sentiment cols) is deferred to the **AC-C umbrella
    integration/runbook stage against the pinned base-data revision** — see
    "Not in this PR" — and is NOT implemented in this repo.
  - Opt-in extension / PIT-label / fail-closed coverage preserved by passing
    `extend_to_bar_frontier=True` where the axis matters.

Full suite: **458 passed** (baseline 453 → +4 net new tests; a review revision
removed the two ported umbrella-guard tests and added one base-data-local
canonical-contract test in their place, net −1 vs the first draft).

## Not in this PR

- σ-head writer cessation + canonical-file consumption = Stage 2 (orchestrator
  PR, amendment §2.1 / AC-A / AC-B).
- The one supervised 179 regeneration + AC-2 digest integrity + Saturday
  dry-run (AC-C/AC-D/AC-E) = ask-first UMBRELLA runbook, never the scheduled
  job.
- **Exact umbrella refresh-guard execution (AC-C).** Proving the new 179-col
  output passes `scripts/refresh_transformer_corpus.py`'s full sanity surface
  by construction, and that the old 176-col recipe reproduces the guard's
  "dropped columns (recipe/schema drift)" Saturday rejection, belongs in the
  umbrella integration/runbook stage against the PINNED base-data revision —
  where the real guard runs. It is deliberately NOT ported into a base-data
  test (repo-boundary + single-branch-divergence risk).
- `sidecar_sanity_contract_scan.py` (base-data#47) repurposing to the (y)
  migration-day regression guard is AC-C/AC-D runbook work; left untouched here
  (its docstring still narrates the now-reversed 179→176 direction).
