# CLAUDE.md

Canonical operating model:
https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Local repo map: `RENQUANT_REPOS.md`.

Branch policy: `main` is the stable interface consumed by other repos and
automation. Experiments, optimizations, and large upgrades happen on feature
branches, then merge back only after tests and integration checks pass.

## Repo Role

`renquant-base-data` owns data manifests, schemas, freshness contracts,
fingerprints, materialization rules, and backup pointers.

## Hard Boundaries

- Git stores manifests and schemas, not large parquet/zip/database blobs.
- Large data belongs in object storage, DVC remote, Git LFS where deliberately
  configured, or controlled local stores referenced by manifest.
- Consumers must resolve data through manifests and fingerprints.
- Data API fallback and cache materialization belong here, not in model or
  execution repos.
- Large source-provider or schema changes use a feature branch.
- Do not delete or empty the source umbrella repo at
  `/Users/renhao/git/github/RenQuant`.

## Required Evidence

Every dataset manifest must declare schema, URI, fingerprint, source,
freshness rule, owner, retention class, and validation command.

## Workflow

```bash
make test
make doctor
```
