# Amendment: single-writer unification for the rawlabel sidecar

Date: 2026-07-18
Status: RFC amendment to the merged
`2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md` — design
review required before implementation. Drafted personally.

## 1. AC-1 falsified the base RFC's premise; the recommendation reverses

The merged RFC recommended option (b) — migrate the served sidecar to
the builder's 176-col contract — "pending AC-1". AC-1 ran (evidence PRs
base-data#47, backtesting#73, model#62, orch#552) and produced two
decisive facts:

1. **The sentiment columns are NOT vestigial.** 99 active/candidate
   sanity contracts name them — including the ACTIVE prod XGB scorer
   (172-feature recipe keeps sentiment with runtime zeroing), its
   shadow, today's weekly staging candidate, 9 weekly rollbacks, and
   both GBDT WF corpora. At 176 cols every one of those sanity loads
   flips from the direct path into the supplement/merge path in
   `_load_sanity_panel` (both the backtesting copy and the umbrella
   rollback copy). The (x) precondition of AC-1 is unsatisfiable.
2. **The served file has a SECOND active weekly writer.** The
   orchestrator σ-head refresh (`retrain_alpha158_fund` via
   `weekly_wf_promote.sh`) regenerated the served sidecar at 179 cols
   TODAY (provenance receipt 2026-07-18T11:02:35Z). Its recipe is
   panel-schema+raw-label (sentiment included, NO bar-frontier
   extension rows) and its validator is column-contract-blind. Any
   one-time 179→176 migration is re-broken the next Saturday.

The base RFC's "lone legacy holdout" framing was therefore FALSE: the
weekly failure is not a stale file vs a frozen builder — it is a
**writer war**: two active weekly writers with contradictory recipes on
one served artifact, where the base-data builder's rebuild is rejected
by the guard (served ≠ its contract) while the σ-head writer succeeds
and re-imposes its own schema. This is precisely the multi-writer
pathology the AC4 bundle-transactionality program exists to eliminate;
resolving it by data migration alone is treating the symptom.

## 2. Amended resolution — one file, one writer

1. **Single canonical writer.** `renquant_base_data.rawlabel_sidecar`
   becomes the SOLE producer of the served sidecar. The orchestrator
   σ-head refresh STOPS writing the file and CONSUMES the canonical
   file directly (r2 — viable now that §2.3 drops extension rows, the
   sole feature its validator rejected). Its column-contract-blind
   validator is retired in favor of the canonical guard.
2. **Canonical contract carries sentiment (option (a)-variant,
   evidence-forced).** `SENTIMENT_COLS` is un-frozen: the contract =
   panel schema INCLUDING the three sentiment columns + raw fwd60d
   label = **179 cols**, matching what the active consumer population
   (the 99 contracts) requires; the wf_gate direct path is preserved
   and no merge-path flip ever occurs. The builder docstring's "the
   served sidecar predates them" is deleted as factually obsolete.
3. **Extension-row disposition — FROZEN: the canonical contract
   DROPS bar-frontier extension rows (r2, review-adjudicated).** The
   AC-1 inventory shows NO consumer of THIS file requires them
   (calibrator fitters read labeled rows; wf_gate reads model
   feat_cols on eval dates), the σ-head validator rejects them, and
   today's served file carries none — dropping them is ZERO behavior
   change for every inventoried consumer and removes the last recipe
   divergence between the two former writers. The builder's
   `extend_to_bar_frontier` default flips off for THIS artifact; any
   future consumer needing a bar-frontier view gets its OWN artifact,
   never a recipe fork of this one.
4. **Sentiment for unlabeled tail rows (dates whose fwd60d label is
   not yet realized) = whatever the panel carries; any MISSING
   sentiment value = NaN, never ffill** (event-driven features; ffill
   would fabricate staleness as signal; the XGB runtime zeroing path
   handles NaN by design — compatibility test required).
5. **Guard passes by construction** thereafter: builder contract ==
   served file, single writer, no drift source. The guard itself stays
   unchanged (fail-closed direction preserved).

## 3. Acceptance criteria

- AC-A (writer cessation): after implementation, NO code path in the
  weekly promote chain writes the served sidecar except the canonical
  builder — proven by the AC-1 sweep re-run showing exactly one writer,
  plus a σ-head-path test asserting it no longer opens the file for
  write.
- AC-B (σ-head fit equivalence, r2 — population and tolerance frozen
  now that the row domain is closed): over the IDENTICAL labeled-row
  population (canonical file rows == the σ-head's former self-built
  rows, proven by row-set digest equality), the fit output artifacts
  are BYTE-IDENTICAL under fixed seeds; if any nondeterminism source
  is documented (library/BLAS), the fallback tolerance is per-parameter
  relative diff ≤ 1e-9 with the source named. Anything looser fails
  AC-B.
- AC-B' (extension rows): FROZEN as DROPPED per §2.3; a contract test
  pins that the canonical output contains zero bar-frontier extension
  rows.
- AC-C (deadlock closure, r2 strengthened): a full dry-run of the
  Saturday chain (refresh → guard → non-promoting retrain preparation,
  per the base RFC's AC-3) passes end-to-end against the unified
  contract, WITH a served-file digest watch across the whole chain —
  the digest may change only at the canonical builder's swap step;
  any other mutation (e.g. the dormant `build_raw_fwd60d_label.py`
  or an unswept writer) fails the dry run. This covers writers
  invisible to AC-A's runtime assertions.
- AC-D (migration integrity): the one supervised regeneration to the
  canonical contract inherits the base RFC's AC-2 verbatim (before/
  after digests, retained-column checksum, hash-verified rollback) —
  ask-first operator landing, never the scheduled job.
- AC-E: the 07-11/07-18 failure signature closes; the sentinel ack for
  weekly-retrain-patchtst is retired after the first green Saturday.

## 4. Ownership

Recipe + contract: base-data (this repo). σ-head writer cessation:
orchestrator PR. Guard baseline: unchanged (umbrella script reads the
contract). Migration runbook: umbrella. Mutual review throughout; no
implementation before this amendment is approved.
