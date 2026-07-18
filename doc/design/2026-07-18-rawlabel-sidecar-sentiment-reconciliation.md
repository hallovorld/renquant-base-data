# Design: rawlabel-sidecar sentiment-column contract reconciliation

Date: 2026-07-18
Status: RFC — design review required before any implementation
Owner: drafted personally per design-review policy

## 1. Problem — a contract deadlock failing the PatchTST weekly retrain

Every Saturday since at least 2026-07-11, `weekly_retrain_patchtst` fails
at corpus refresh with:

```
rebuilt rawlabel sidecar rejected (kept prior sidecar): staged corpus
dropped columns (recipe/schema drift):
['mean_sentiment', 'n_articles_log', 'sentiment_pos_share']
```

Root cause is NOT data drift — it is a frozen contradiction between two
reviewed surfaces [all VERIFIED on disk 2026-07-18]:

- the fund panel (`alpha158_291_fundamental_dataset.parquet`, 178 cols)
  and the transformer training corpus (178 cols) both CARRY the three
  sentiment columns; the corpus swaps cleanly every week;
- the sidecar BUILDER (`renquant_base_data.rawlabel_sidecar`) EXCLUDES
  them by design — `SENTIMENT_COLS` is an explicit drop list, docstring:
  "the served sidecar predates them"; its contract = panel minus
  sentiment = 176 cols;
- the SERVED sidecar on disk
  (`alpha158_291_fundamental_dataset_rawlabel.parquet`) has **179**
  cols INCLUDING sentiment — produced under an older/other recipe,
  contradicting the builder's own freeze;
- the refresh guard (`scripts/refresh_transformer_corpus.py`) compares
  staged (176) against served (179), reads the difference as "dropped
  columns", rejects fail-closed, and the retrain aborts.

The builder will never emit the columns; the guard will never accept
their absence: a deterministic weekly deadlock. The guard's fail-closed
DIRECTION is correct and is not changed by this RFC.

## 2. Impact (bounded)

- PatchTST shadow model stays at its current vintage (G4 evidence
  accrues under an aging model); no live-trading surface reads the
  rejected staging output; the transformer corpus itself updates
  normally. No capital impact; chronic alarm noise + blocked retrain.

## 3. Options

**(a) Builder carries sentiment (contract → 179).** Un-freeze
`SENTIMENT_COLS`; the sidecar mirrors the panel. Cost: extends the
serving-axis extension rows' semantics — sentiment for extension rows
(dates beyond the fundamentals frontier) must be defined (ffill? NaN?);
the sidecar contract churns for a feature set whose global value was
measured NEGATIVE-to-mixed (analyst/sentiment features: adds BULL_CALM
only, hurts BULL_VOLATILE — 2026-06 finding; no global retrain adopted
them).

**(b) Served sidecar migrates to the builder's 176-col contract —
RECOMMENDED (pending AC-1).** Declare the served file's sentiment
columns vestigial; one-time supervised regeneration of the served
sidecar via the CURRENT builder; the guard's baseline then matches the
recipe forever after. Preliminary evidence: no WF-gate code reads the
three columns from the SIDECAR (grep of renquant-backtesting wf_gate:
zero hits — the training corpus, which keeps them, is a different
file). Cheapest, aligns disk with the reviewed recipe, changes no
recipe semantics.

**(c) Guard allowlist ("tolerate these 3 known-dropped columns").**
Rejected: encodes the contradiction permanently instead of resolving
it, and per the check-existing-contract rule a guard should not carry a
side-channel exception to its own reference.

## 4. Acceptance criteria

- AC-1 (gates the (b) recommendation): a FULL consumer sweep across all
  repos proving nothing reads `mean_sentiment` / `n_articles_log` /
  `sentiment_pos_share` FROM THE SIDECAR file (readers of the panel or
  corpus are out of scope — those keep the columns). Any consumer found
  → option (a) or a consumer migration, re-review required.
- AC-2: after migration, a dry-run rebuild produces a staged sidecar
  whose schema matches the served file exactly (guard passes; retrain
  proceeds past corpus refresh).
- AC-3: the migration step is a supervised, backed-up, one-time
  regeneration executed as an ask-first landing action (production
  data-path mutation under the live-tree preflight rule) — never by the
  scheduled job itself.
- AC-4: the 07-11/07-18 failure signature is documented in the progress
  doc and the sentinel ack for weekly-retrain-patchtst updated to
  reference this RFC until the migration lands.

## 5. Rollout

1. This RFC (base-data owns the recipe) → adversarial review.
2. AC-1 consumer sweep (read-only, committed as evidence appendix).
3. Migration runbook PR (umbrella): backup + regenerate + verify script
   with the exact revert step (restore .bak).
4. Ask-first operator landing; next Saturday's retrain is the live
   verification.
