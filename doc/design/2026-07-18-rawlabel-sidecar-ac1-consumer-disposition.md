# AC-1 evidence appendix: rawlabel-sidecar consumer disposition (176-column migration)

Date: 2026-07-18
Status: evidence appendix for AC-1 of
[`2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md`](2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md)
(merged RFC). Evidence + executable tests only — NO migration, NO
served-file mutation, NO umbrella runbook here (that is rollout step 3).

**Bottom line first — two decisive findings, both requiring a design call
before recommendation (b) can execute as written:**

1. **The AC-1 (x) precondition FAILS.** The committed scan
   (`renquant_base_data.sidecar_sanity_contract_scan`, run read-only against
   the live tree 2026-07-18) found **99 active/candidate sanity contracts
   naming the three sentiment columns** — including the ACTIVE prod XGB
   scorer (`artifacts/prod/panel-ltr.alpha158_fund.json`, 172 feature_cols),
   its shadow twin, TODAY's weekly staging candidate
   (`panel-ltr.alpha158_fund.weekly_20260718T110005Z.staging.json`), every
   weekly rollback restore-candidate, and both GBDT WF corpora
   (`walkforward_v2_20260602` 39 cuts + `walkforward_gbdt_prod_recipe_v2`
   43 cuts). **All 99 record NO `training_contract.dataset`**, i.e. every
   one of them resolves sanity through the SIDECAR path — at 176 columns
   every one of them flips from the direct path to the supplement/merge
   path. Per the RFC's own AC-1 wording, the (x) route is therefore
   **STOPPED**: the disposition flips to (y) (accept-and-document the merge
   semantics, now executably pinned by the companion tests) or the
   migration is blocked/redesigned. That call is the operator's, not this
   appendix's.
2. **The served sidecar has a second, ACTIVE, contradicting WRITER.** The
   179-column served file is not a frozen legacy one-off: it was
   regenerated TODAY (2026-07-18T11:02:35Z, provenance receipt
   `data/alpha158_291_fundamental_dataset_rawlabel.parquet.provenance.json`)
   by the orchestrator σ-head refresh
   (`renquant_orchestrator.retrain_alpha158_fund`, scheduled weekly via
   `weekly_wf_promote.sh` → `daily_retrain_alpha158_fund.sh`), whose
   builder emits the FULL panel schema + raw label — sentiment INCLUDED —
   and whose pre-swap validator never checks the column contract. A
   one-time 179→176 regeneration is therefore insufficient by itself: the
   next Saturday σ-head refresh re-emits 179 and re-arms the deadlock. The
   migration design must couple the two writer recipes (executably pinned
   in the orchestrator companion PR).

## 1. Sweep method and totals

Per the RFC's AC-1 sweep spec: keyed on the EXACT sidecar filename
`alpha158_291_fundamental_dataset_rawlabel.parquet` (never the shared
`alpha158_291_fundamental_dataset` prefix; the `_rawlabel30d` /
`_rawlabel_20d` horizon variants alias under loose greps and are excluded
by exact-name classification), PLUS constant/indirection chase
(`DEFAULT_RAW_LABEL_PANEL`, `DEFAULT_RAW_LABEL_PANEL_FILENAME`,
`DEFAULT_RAWLABEL_FILENAME`, `DEFAULT_RAWLABEL_RELPATH`,
`raw_label_panel` / `--raw-label-panel` / `--rawlabel-path` CLI+config
bindings).

Repos swept, each at origin/main (scratch clones, read-only):

| repo | pin | filename hits | constant/CLI hits |
|---|---|---|---|
| RenQuant (umbrella) | `70393a2d` | 121 | 35 |
| renquant-orchestrator | `9ddb723` | 12 | 20 |
| renquant-model | `7703a81` | 3 | 28 |
| renquant-backtesting | `2b87dfd` | 6 | 5 |
| renquant-base-data | `b72dd92` | 4 | 1 |
| renquant-artifacts | `7859f8d` | 1 (provenance record) | 1 |
| renquant-pipeline `a871166`, renquant-common `fe832fa`, renquant-execution `69f01b1`, renquant-strategy-104 `082dccd`, renquant-state-backup `f60a8f4`, renquant-model-gbdt `39f9c2e` (archived), renquant-model-patchtst `9448b83` (archived) | — | 0 | 0 |

Raw sweep totals: 172 filename-stem lines + 115 constant/CLI lines across
all repos, classified below into code consumers/writers, path plumbers,
tests, docs, and non-reader provenance records.

**Exempt-and-stated (AC-1 (iii)):**
- `backtesting/renquant_104/artifacts/diagnostics/modal_sweep_*/` frozen
  bundle archives: **81 directories on the live tree only** — zero hits at
  umbrella origin/main (untracked archives). Not live consumers.
- Artifact provenance RECORDS (not readers): `panel-rank-calibration.json`
  and calibration artifacts in prod/shadow/sim/WF-corpus dirs (and the
  renquant-artifacts store copy) embed the sidecar path string as
  fit-provenance metadata (`raw_label_panel` / `expected_return_label_source`).
  They are written once at fit time and never read the parquet.
- Docs/progress/research markdown references (both repos): historical
  context, not consumers.

## 2. Consumer inventory (exact pins, columns consumed, disposition)

### 2.1 Committed sub-repo surfaces (the RFC's five known readers first)

| # | reader (pin) | path binding | columns consumed | disposition |
|---|---|---|---|---|
| 1 | backtesting `wf_gate/runner.py` `_load_sanity_panel` — read `runner.py:2044-2153`, contract resolution `:2470-2491`, label `:511-524` | hardcoded `REPO/"data/alpha158_291_fundamental_dataset_rawlabel.parquet"` (`:2044`); reached ONLY when the artifact records no usable `training_contract.dataset` (`:2026-2043`) | bare `pd.read_parquet` (ALL columns, `:2047`); then dynamically: `label` (artifact `label_col`, prod XGB = `fwd_60d_excess`) + every `feat_cols` present; missing `feat_cols` → supplement/merge from the training panel (`:2064-2128`) | **merge-path-flip-exposed.** At 176, every sentiment-naming contract (the live population, §4) flips direct→merge. Executably pinned before/after in companion tests (backtesting PR): direct at 179, merge at 176 with `feature_cols_supplied_by_feature_panel == the 3 sentiment cols` |
| 2 | model `patchtst/fit_calibrator.py` — `DEFAULT_RAW_LABEL_PANEL:29`, read `:166`, CLI `--raw-label-panel:470`, er-label inference `:81-89` | constant default `data/alpha158_291_fundamental_dataset_rawlabel.parquet`, CLI-overridable; orchestrator passes it explicitly | `pd.read_parquet(path, columns=["ticker","date", er_label_col])` — er label = `fwd_60d_excess_raw` (inferred from `fwd_60d_excess`) | **safe-at-176** (column-pruned read; all three consumed columns survive). Executable test in model companion PR |
| 3 | model `gbdt/fit_calibrator_alpha158_fund.py` — `DEFAULT_RAW_LABEL_PANEL_FILENAME:31`, read `:154`, CLI `:339` | constant default resolved under `--data-dir`, CLI-overridable | `columns=["ticker","date", chosen]`, chosen = `fwd_60d_excess_raw` | **safe-at-176.** Executable test in model companion PR |
| 4 | orchestrator `build_patchtst_wf_manifest.py` — `DEFAULT_RAW_LABEL_PANEL_REL:65`, plumbing `:226-227, :275-279, :379, :444, :501` | constant, resolved under the data root; handed to `renquant_model_patchtst.fit_calibrator` as `--raw-label-panel`; also stamped into the manifest JSON | **none** — pure path plumber, never opens the parquet (`retrain_patchtst.py:112,248,307` same) | **safe-at-176 transitively** (via #2's pruned read). Executable pass-through test in orchestrator companion PR |
| 5 | orchestrator `retrain_alpha158_fund.py` — `DEFAULT_RAWLABEL_FILENAME:109`, `RAWLABEL_COLUMN:113`, task config `:206-226`, builder `:578-627`, validator `:693-788` | `repo_dir/data/<served filename>` — **this is the served production path**, written via staging + atomic swap | **WRITER, not a reader.** Builder = port of the ORIGINAL Track-A recipe: full panel schema + `fwd_60d_excess_raw` appended → **sentiment INCLUDED (179)** from the 178-col panel. Validator checks keys/label/coverage ONLY — column-contract-blind — and REJECTS bar-frontier extension rows (base-data recipe rows) | **recipe-conflict / needs-migration-coupling** (decisive finding 2). Scheduled weekly: `ops/launchd_manifest.json` `com.renquant.weekly-wf-promote` → `weekly_wf_promote.sh:246` → `daily_retrain_alpha158_fund.sh:70`. All three facts executably pinned in orchestrator companion PR |

### 2.2 Additional sub-repo surfaces found by the sweep

| reader (pin) | binding / columns | disposition |
|---|---|---|
| backtesting `wf_gate/fit_walkforward_calibrators.py:66,101-102,133` | `--raw-label-panel` forwarded verbatim to the model-repo fitter subprocesses; never opens the parquet | safe-at-176 transitively (plumber) |
| model `alpha158_linear/calibrator.py:18,38-58` | its `raw_label_panel` DEFAULT is `transformer_dataset_engineered.parquet` — **not this sidecar**; no caller binds it to the 60d sidecar (`retrain_alpha158_linear.sh`: zero rawlabel refs) | not a sidecar consumer (named for disambiguation — the shared constant name aliases under constant-greps) |
| orchestrator `scripts/s9_track_a_conditional.py:379,390,412`; `scripts/s9_independent_verification.py:91-93,392,398`; `scripts/d3_core_shrink_check.py:56-59,549` | pruned reads: keys + `fwd_60d_excess`(+`_raw`) / `date` only | safe-at-176; read-only research/evidence scripts, no scheduled invoker (0 hits in `ops/launchd_manifest.json`) |

### 2.3 Umbrella surfaces (RenQuant @ `70393a2d`)

| surface (pin) | role / columns | disposition |
|---|---|---|
| `scripts/refresh_transformer_corpus.py` — `DEFAULT_RAWLABEL_RELPATH:149`, builds via `renquant_base_data.rawlabel_sidecar.build_rawlabel_sidecar` (`:714-730`), guard `_sanity_reasons:1080-1123`, reject `:1231-1245` | **the 176-col WRITER + the fail-closed guard.** Guard reads schema names + `ticker`/`date` counts only; comparison is one-directional (prior−staged) → the 179→176 shrink always trips `dropped columns`; `swap_fail_on_regression=True` default raises `CorpusRefreshError` → aborts the weekly job (`weekly_retrain_patchtst.sh:15 set -euo pipefail`, invocation `:77-79`) | the migration mechanism itself; scheduled (`com.renquant.weekly-retrain-patchtst`). The RFC's failure signature (07-11/07-18 logs) is this guard doing its job |
| `scripts/weekly_retrain_patchtst.sh:59-82,168-171` / `scripts/weekly_wf_promote.sh:74-77,141-157,246` | schedulers; promote path note: default WF gate runner is `multirepo` (backtesting #1); `run_wf_gate.py` only under explicit `RQ_WF_GATE_RUNNER=umbrella` rollback | run-surface wiring, no direct column reads |
| `scripts/run_wf_gate.py:2132-2243,2466-2476` | **own COPY of `_load_sanity_panel`** (does not import backtesting): bare full-column read `:2137`, dynamic `feat_cols` `:2141`, same supplement/merge fallback; the sentiment case is already unit-pinned (`tests/test_wf_gate_recipe_scope.py:614-648` supplements `mean_sentiment` from the training panel, asserts `feature_panel_merge is True`) | merge-path-flip-exposed, rollback surface only. Same disposition as #1; behavior pinned by the existing umbrella test — flip semantics land in the (y) decision |
| `scripts/promote_shadow_patchtst.py:144-145,489,542-564` | rawlabel source = SLA freshness probe: reads `columns=["date"]` only | safe-at-176; scheduled |
| `scripts/fit_calibrator_alpha158_fund.py:208,274-278,324-327` and `scripts/fit_hf_patchtst_calibrator.py:53-72,192,207-210` | pruned reads `columns=["ticker","date", er_label]` | safe-at-176; research-adjacent predecessors of the model-repo fitters, no scheduled invoker |
| `scripts/fit_walkforward_calibrators.py:30-31,261-262,304` / `scripts/train_walkforward_patchtst.py:33,127-141,273-275` | pass-through plumbers to the model-repo fitters | safe-at-176 transitively; not scheduled |
| `scripts/train_ngboost_proper.py:249,289-296,336-357,397` (via manual `run_a3_ngboost_retrain.sh`) | full-column read; features = `--panel-artifact` `feature_cols`, **error-on-missing by default** → conditional exposure | safe-at-176 TODAY: prod+shadow NGBoost heads = 169 feature_cols, sentiment-free [verified 2026-07-18]. Manually-triggered production-artifact producer; covered by the committed scan (prod/shadow surfaces) |
| `scripts/refit_gate_b_offline.py:49,73,82` | full-column read; features from the NGB head artifact → same conditional class | safe-at-176 today (169-col head); research one-off, no invoker |
| `scripts/build_raw_fwd60d_label.py:38-40,43,86-96` | the ORIGINAL one-off 179-writer (full panel copy + raw label; confirmed sentiment INCLUDED; keeps `fwd_60d_excess` too) | dormant (no invoker, no CLI); the migration runbook should retire/mark it — its recipe now lives on ONLY through `retrain_alpha158_fund`'s port (§2.1 #5) |
| research one-offs `train_quantile_head_rawlabel.py:43,54` · `train_qhead_neural.py:154,164` · `train_qhead_catboost_multiquantile.py:35,46` · `train_ngb_vol_adjusted_label.py:46,50` · `test_insider_features.py:47,109` · `qhead_purged_baseline.py:66,72` · `qhead_phaseA_experiments.py:103,110` · `qhead_neural_sanity.py:34,42` · `ngb_proper_placebo.py:53,58` · `long_short_prereq_gate.py:35,42-48` | full-column reads; features dynamically from artifact JSONs (none names sentiment literally; `long_short_prereq_gate` drops missing gracefully) | research one-offs, invoker: none found. Disposition-only (no executable tests): frozen research surfaces, not on any scheduled path; exposure class is covered by the committed contract scan |
| umbrella tests `test_rawlabel_refresh.py:164-183` (pins the guard's `dropped columns` rejection — the exact class the migration swap must pass through), `test_wf_gate_recipe_scope.py:614-648` (pins the sentiment supplement), `test_train_walkforward_patchtst.py:28,88`, `test_fit_walkforward_calibrators.py:123,132`, `test_manifest_sanity_placebo_analysis.py:154-313`, `test_train_ngboost_proper_rawlabel_admission.py`, `test_fit_calibrator_raw_label_contract.py:94-113` | existing pins of the above surfaces | evidence corroboration |

### 2.4 base-data (this repo)

`rawlabel_sidecar.py:74-85` — the 176-column recipe (SENTIMENT_COLS drop
list, `RAWLABEL_SIDECAR_COLUMNS` = panel − sentiment + raw label);
`tests/test_rawlabel_sidecar.py` — recipe tests; consumed by the umbrella
weekly refresh (§2.3). This PR adds the schema export + drift guard
(`tests/rawlabel_sidecar_columns_176.json`,
`tests/test_rawlabel_sidecar_schema_export.py`) that every companion-repo
fixture embeds.

## 3. Executable consumer tests (176-column fixture)

Each live consumer surface has an executable test against a fixture carrying
the builder's EXACT 176-column contract (embedded exports of
`RAWLABEL_SIDECAR_COLUMNS` @ `b72dd92`; drift guard in this repo):

| repo / PR | tests |
|---|---|
| renquant-backtesting (companion PR, test-only) | `tests/wf_gate/test_sidecar_176_contract.py` — 4 tests: direct path at 176 for sentiment-free contracts; prod-shape (172) contract direct at 179; **the same contract flipping to merge at 176** (the AC-1 flip, pinned); label survival |
| renquant-model (companion PR, test-only) | `tests/test_sidecar_176_consumer_evidence.py` — 2 tests: patchtst + gbdt loaders merge `fwd_60d_excess_raw` from the 176-col fixture via their column-pruned reads |
| renquant-orchestrator (companion PR, test-only) | `tests/test_sidecar_176_consumer_evidence.py` — 6 tests: exact-filename binding + pass-through plumbing (no read); **σ-head builder re-emits sentiment from a sentiment-carrying panel**; **σ-head validator is column-contract-blind (admits 176 and 179 alike)**; **σ-head validator rejects bar-frontier extension rows** (recipe incompatibility both directions); embedded schema sanity |
| renquant-base-data (this PR) | `tests/test_sidecar_sanity_contract_scan.py` — 8 tests for the committed (x) checker; `tests/test_rawlabel_sidecar_schema_export.py` — 2 drift-guard tests |

Umbrella surfaces: executable proof belongs to the AC-3 runbook PR (the RFC
assigns the real refresh/guard + non-promoting retrain-prep run to the
UMBRELLA rollout step 3); their behavior today is already pinned by the
umbrella tests cited in §2.3.

## 4. The wf_gate sanity-contract precondition — AC-1 (x) RESULT

Committed checker: `renquant_base_data.sidecar_sanity_contract_scan`
(strict per the RFC: ANY active/candidate contract naming the three columns
fails; unparseable payloads and unresolvable manifest entries fail closed;
surfaces = prod, shadow, `walkforward_*` corpora, sim, top-level artifacts,
walkforward manifests with `.pt → .pt.metadata.json` chasing; diagnostics /
modal_sweep archives exempt).

Live run 2026-07-18 (read-only, `--umbrella-root /Users/renhao/git/github/RenQuant`):

```
n_scanned=292 payloads, n_contracts=149, n_violations=99  →  PRECONDITION FAILS (exit 1)
  prod:   16  (ACTIVE panel-ltr.alpha158_fund.json; pre-restamp/previous;
               4 weekly staging incl TODAY 20260718T110005Z; 9 weekly rollbacks)
  shadow:  1  (panel-ltr.alpha158_fund.shadow.json)
  walkforward_gbdt_prod_recipe_v2: 43 cuts   walkforward_v2_20260602: 39 cuts
  every violation: n_feature_cols=172, names all 3 sentiment cols,
                   dataset_recorded=False  →  100% sidecar-path exposed
  unresolved (fail-closed): 43 entries of ONE stale manifest
  (sim/walkforward_manifest_gbdt_prod_recipe_calibrated.json → deleted
   walkforward_172_sentiment/ corpus — the dir name itself records the
   172-with-sentiment recipe lineage)
```

Clean surfaces: every PatchTST artifact records
`training_contract.dataset = data/transformer_v4_wl200_clean.parquet`
(sanity never touches the sidecar); NGBoost heads 169 sentiment-free;
alpha158_linear 158; sim corpus 169-feature cuts sentiment-free.

Why the population is 172-with-sentiment: the prod XGB recipe deliberately
keeps the 3 sentiment features with a runtime-zeroing serving gate
(orchestrator `retrain_alpha158_fund.py:151-157` — config-fingerprint
parity with the WF v2 manifest cuts). This is a reviewed recipe decision,
not drift — which is precisely why (x) cannot be satisfied by waiting.

**Consequence (per the RFC's AC-1 x/y clause): the (x) path is STOPPED.**
The operator design call is now between:
- **(y)** accept-and-document the merge-path semantics for
  sentiment-naming contracts. The flip is now executably pinned
  (backtesting companion test; umbrella `test_wf_gate_recipe_scope.py`
  already pinned the mechanism); factual impact note: post-migration the
  supplement source (`alpha158_291_fundamental_dataset.parquet`) is the
  SAME panel the σ-head build copies sentiment from today, so values
  match up to refresh timing — the change is provenance/plumbing
  (`feature_panel_merge: True`, third supplement file in the sanity
  provenance, 1% tail-gap drop tolerance), not a different feature source;
- **block/redesign** the migration.
Either way, decisive finding 2 (the σ-head writer conflict) must be
resolved in the migration design regardless of the (x)/(y) choice — a
one-time regeneration alone is re-broken by the next weekly σ-head refresh.

The checker doubles as the migration-day precondition/regression guard once
the design call is made: under (y) its scope assertion becomes "no NEW
sidecar-exposed contract class appears"; under a recipe change it verifies
the population drained.

## 5. What this appendix explicitly does NOT do

No migration, no served-file mutation, no umbrella runbook, no builder
change, no guard change. AC-2 (integrity digests), AC-3 (real refresh/guard
+ non-promoting retrain-prep), AC-4 (containment shape) and AC-5 (sentinel
ack) remain with the umbrella runbook PR per the RFC rollout.

## 6. Companion PRs

- renquant-backtesting: test-only, `test/sidecar-176-consumer-evidence`
- renquant-model: test-only, `test/sidecar-176-consumer-evidence`
- renquant-orchestrator: test-only, `test/sidecar-176-consumer-evidence`

(URLs recorded in the PR description; all authored under the same identity
discipline as this PR, reviewer haorensjtu-dev, none self-merged.)
