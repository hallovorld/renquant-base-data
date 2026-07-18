# Progress: AC-1 consumer disposition evidence (rawlabel sidecar RFC)

Date: 2026-07-18
Scope: rollout step 2 of the merged RFC
`doc/design/2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md` —
the AC-1 behavioral consumer disposition. Evidence + executable tests only;
no migration, no served-file mutation, no umbrella runbook.

## Delivered

- `doc/design/2026-07-18-rawlabel-sidecar-ac1-consumer-disposition.md` —
  exact-pins consumer inventory (12 repos swept at origin/main pins; the
  RFC's 5 known readers + every additional surface found; per-reader
  columns-consumed evidence and disposition; modal_sweep_* archives and
  provenance-record JSONs exempt-and-stated).
- `src/renquant_base_data/sidecar_sanity_contract_scan.py` +
  `tests/test_sidecar_sanity_contract_scan.py` — the committed AC-1 (x)
  precondition checker (strict, fail-closed; walkforward-manifest chasing).
- `tests/rawlabel_sidecar_columns_176.json` +
  `tests/test_rawlabel_sidecar_schema_export.py` — canonical 176-column
  schema export + drift guard for the companion-repo fixture embeds.
- `doc/design/2026-07-18-sanity-contract-scan-live.json` — the live-tree
  scan report (read-only run, 2026-07-18).
- Companion test-only PRs (backtesting / model / orchestrator) with
  executable 176-column fixture tests per reader — linked in the PR body.

## The two decisive results

1. **AC-1 (x) precondition FAILS**: 99 active/candidate sanity contracts
   name the three sentiment columns (ACTIVE prod XGB + shadow + today's
   weekly staging + rollbacks + both GBDT WF corpora; all 172-feature,
   none records `training_contract.dataset` → 100% sidecar-path exposed).
   Per the RFC, the (x) route is STOPPED; the (y)-vs-block call is the
   operator's design decision.
2. **Second active writer**: the σ-head refresh
   (`renquant_orchestrator.retrain_alpha158_fund`, weekly via
   `weekly_wf_promote.sh`) regenerated the served sidecar at 179 columns
   TODAY (provenance receipt 2026-07-18T11:02:35Z); its builder keeps
   sentiment and its validator is column-contract-blind. A one-time
   179→176 migration is re-broken by the next weekly run unless the two
   writer recipes are unified in the migration design.

## Failure signature (AC-5 forward-reference)

`logs/weekly_retrain_patchtst/2026-07-11.log:20` and `2026-07-18.log:20`:
`CorpusRefreshError: rebuilt rawlabel sidecar rejected (kept prior
sidecar): staged corpus dropped columns (recipe/schema drift):
['mean_sentiment', 'n_articles_log', 'sentiment_pos_share']` — emitted by
`refresh_transformer_corpus.py:1231-1245` (guard working as designed
against the 179-column serving maintained by writer 2). The sentinel-ack
update itself stays with the umbrella runbook PR per AC-5.
