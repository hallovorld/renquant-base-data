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

## Initial Split Source

`hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

## Local Test

```bash
PYTHONPATH=../renquant-common/src:src python -m pytest -q
```
