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
RECOMMENDED (pending the r2 AC-1).** Declare the served file's
sentiment columns vestigial; one-time supervised regeneration of the
served sidecar via the CURRENT builder; the guard's baseline then
matches the recipe forever after.

Evidence (r2 — review-verified):
- The horizon-variant sidecars ALREADY follow the current recipe:
  `_rawlabel30d.parquet` = 176 cols, `_rawlabel_20d.parquet` = 177
  cols, both ZERO sentiment columns [verified on disk]. The served 60d
  file is the lone legacy holdout — supporting "vestigial", not
  "re-freeze the builder around them".
- The r1 claim "no WF-gate consumer reads the columns" rested on a
  NAME GREP, which the round-2 review proved insufficient: the
  sidecar-path consumer inventory is **5 readers across 4 repos**
  (wf_gate/runner.py; model patchtst + gbdt fit_calibrator;
  orchestrator build_patchtst_wf_manifest + retrain_alpha158_fund),
  and wf_gate reads the file with DYNAMIC column resolution from the
  model's sanity contract (`missing = [c for c in feat_cols ...]`) —
  a corpus-trained model whose sanity `feat_cols` include sentiment
  (the corpus keeps them) would silently flip that sanity run from the
  direct path into the supplement/merge path post-migration. No
  textual sweep can see this; AC-1 is therefore behavioral, per §4 r2.

**(c) Guard allowlist ("tolerate these 3 known-dropped columns").**
Rejected: encodes the contradiction permanently instead of resolving
it, and per the check-existing-contract rule a guard should not carry a
side-channel exception to its own reference.

## 4. Acceptance criteria (r2 — per the round-1/round-2 review demands)

- **AC-1 — BEHAVIORAL consumer disposition (gates recommendation (b);
  a name grep is only an initial lead, never promotion evidence):**
  a committed exact-pins consumer inventory of the SIDECAR PATH,
  covering at minimum the five known readers
  (backtesting `wf_gate/runner.py`; model `patchtst/fit_calibrator.py`
  + `gbdt/fit_calibrator_alpha158_fund.py`; orchestrator
  `build_patchtst_wf_manifest.py` + `retrain_alpha158_fund.py`) plus
  any found by the sweep, EACH with an evidence-backed disposition and
  an EXECUTABLE consumer test against a 176-column fixture.
  The sweep spec closes three aliasing holes: (i) key on the exact
  sidecar filename, never the shared `alpha158_291_fundamental_dataset`
  prefix; (ii) chase path indirection — constants
  (`DEFAULT_RAW_LABEL_PANEL*`) and CLI/config-supplied paths, not just
  literals; (iii) exempt-and-state the frozen pinned copies under
  `artifacts/diagnostics/modal_sweep_*/bundle/subrepos/` (archives).
  **The wf_gate dynamic-consumption case is disposed explicitly:**
  `runner.py` resolves columns from the model's sanity contract, so a
  corpus-trained model whose `feat_cols` include sentiment would flip
  from the direct path into the supplement/merge path post-migration —
  AC-1 must either (x) verify no ACTIVE/candidate model sanity contract
  names the three columns (and add a migration precondition check for
  it), or (y) explicitly accept-and-document the merge-path semantics
  for such models, with a test pinning whichever is chosen.
- **AC-2 — migration data integrity (destructive one-time step):**
  recorded BEFORE and AFTER: builder revision, input fingerprints,
  sidecar SHA-256 + schema digest, row count, primary-key/date
  coverage, and a checksum over all RETAINED columns. The diff may
  contain ONLY the three declared column removals. Rollback restores
  the exact backed-up digest (verified by hash), not merely a `.bak`
  filename.
- **AC-3 — prove the actual failure path clears:** run the REAL
  refresh/guard against the migrated candidate (guard passes), then
  execute a NON-PROMOTING PatchTST retrain preparation through the
  former failure boundary. A schema comparison alone is insufficient
  (the wf_gate merge-path flip is a concrete example of what it cannot
  prove).
- **AC-4 — containment shape:** the migration is a supervised,
  backed-up, one-time regeneration executed as an ask-first landing
  action (live-tree mutation preflight) — never by the scheduled job.
  Runbook implementation lives in the UMBRELLA (it mutates the live
  served file); this RFC stays base-data-owned.
- **AC-5:** the 07-11/07-18 failure signature is documented in the
  progress doc and the sentinel ack for weekly-retrain-patchtst updated
  to reference this RFC until the migration lands.

## 5. Rollout (r2)

1. This RFC (base-data owns the recipe) → adversarial review.
2. AC-1 behavioral consumer disposition (evidence appendix: inventory +
   executable 176-col fixture tests + the wf_gate sanity-contract
   precondition check) — committed and independently reviewed.
3. Migration runbook PR (UMBRELLA — it mutates the live served file):
   backup + regenerate + AC-2 digest recording + hash-verified rollback
   + the AC-3 refresh/guard + non-promoting retrain-prep proof, all
   scripted.
4. Ask-first operator landing; next Saturday's retrain is the live
   verification.
