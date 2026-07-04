# alpha158 train/serve operator unification — shared module home (campaign B8)

Date: 2026-07-04. Campaign: compliance fix campaign Group B item B8
(orchestrator `doc/design/2026-07-04-compliance-fix-campaign.md`), fixing
pipeline#168 audit finding §6.3 (top P1): the pipeline's serve-side
`alpha158_features.py` claimed its operators were shared imports with the
training builder but hand-mirrored them — the anti-skew invariant on the live
XGB primary feature path was stated, not enforced.

## What changed (this repo = the module home)

- NEW `src/renquant_base_data/alpha158_ops.py` — the ONE shared train/serve
  alpha158 operator module. Both grains moved VERBATIM:
  - train grain (`kbar_features` / `price_features` / `rolling_features` +
    `slope`/`rsquare`/`resi`/`idx_max_n`/`idx_min_n`/`greater`/`less`) from
    `alpha158_qlib_panel.py`;
  - serve grain (`compute_alpha158_at` / `compute_alpha158_frame` /
    `alpha158_feature_names` + at-bar kernels) from renquant-pipeline
    `kernel/panel_pipeline/alpha158_features.py`.
  - `KNOWN_TRAIN_SERVE_DIVERGENCES` registry documenting the measured,
    pre-existing grain divergences (below).
- `alpha158_qlib_panel.py` now imports every operator from `alpha158_ops`
  (mirrors deleted; `_load_ohlcv` kept — IO, not an operator). Public API
  unchanged.
- NEW `tests/test_alpha158_ops.py`: builder-uses-shared-ops identity,
  cross-grain lockstep (exact for order-identical families, 1e-8 for
  fp-accumulation families), RANK tie divergence pinned as documented,
  registry shape.

Module home rationale (import matrix): the training panel builder lives HERE
(`renquant_orchestrator.retrain_alpha158_fund` → `alpha158_qlib_panel` is the
production retrain path; prod model trained 2026-06-21 via it), and
renquant-pipeline already declares `renquant-base-data>=0.1.0`. base-data owns
the training-data contract, and feature definitions ARE that contract.
renquant-common would have worked but adds a third repo for no consumer.

## Protection-contract proof (read-only, real prod OHLCV)

Old-vs-unified byte equivalence, 40 tickers x 40 recent dates = 1600 rows
sampled from the prod panel universe, PLUS full builder frames for all 40
tickers (~97k rows): **max|delta| = 0.0 exactly** for old-train vs
unified-train and old-serve vs unified-serve (both `compute_alpha158_at` and
`compute_alpha158_frame`). Suites A/B: base-data 238→244 passed (0 regressions),
pipeline 1291→1297 passed (0 regressions).

## Findings (measured live train/serve skew — reported, NOT changed)

1. **RANK5-60 (material)**: train = `rolling(n).rank(pct=True)` (average rank
   on ties) vs serve = `(window <= today).sum()/n` (max rank). On real rows:
   1-2.8% of rows per window diverge; max|delta| = 0.2 (RANK5). Serve reads
   HIGHER whenever today's close ties an earlier close in the window. This has
   been live the whole time the XGB primary served. Convergence = a
   model-lifecycle decision (pick a convention, retrain, gate) — proposed as a
   campaign follow-up, deliberately NOT bundled here.
2. **CORD (fp)**: train correlates `c/c_lag1`, serve `c/c_lag1 - 1` — corr is
   shift-invariant; measured <= 1.6e-11.
3. **scalar-vs-vector accumulation (fp)**: serve-at recomputes windows with
   numpy vs pandas rolling on the train grain / serve-frame grain; measured
   <= 7.1e-10 (CORR10). Same profile as the pre-existing serve-internal
   frame-vs-at (cache hit vs miss) difference.

## Deploy path

Merge order: this PR first, then the renquant-pipeline shim PR (its CI checks
out base-data@main). Live behavior changes ZERO bytes by proof; the live
umbrella kernel mirror sync + pin bump follows the campaign Group C
governance.
