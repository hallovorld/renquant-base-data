# renquant-base-data

Data-manifest repository for RenQuant.

Operating model: https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Repository map: [RENQUANT_REPOS.md](RENQUANT_REPOS.md)

Local automation:

```bash
make test
make doctor
```

This repo tracks data contracts, fingerprints, schemas, and object locations.
It does not store large parquet/zip/database files in normal Git.

SEC EDGAR refresh jobs must set `SEC_USER_AGENT` to an operator contact string,
for example `RenQuant ops@example.com`. If unset, the package uses a generic
library identifier instead of a personal email.

## Pipeline Rule

Data validation and materialization workflows are `renquant-common`
Task/Job/Pipeline chains.

## Track B Feature Readiness

The BULL_CALM Track B feature-readiness manifest lives at
`manifests/track-b-bull-calm-feature-readiness.json`. It checks only the
required feature surface for full WF retrain readiness; it does not run
training.

```bash
PYTHONPATH=../renquant-common/src:src python -m renquant_base_data.track_b_readiness \
  --columns mom_carry_12_1 beta_dm rvar_total idio_vol_market
```

## Initial Split Source

`hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

## Local Test

```bash
PYTHONPATH=../renquant-common/src:src python -m pytest -q
```
