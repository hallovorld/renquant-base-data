# renquant-base-data

Data-manifest repository for RenQuant.

This repo tracks data contracts, fingerprints, schemas, and object locations.
It does not store large parquet/zip/database files in normal Git.

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
