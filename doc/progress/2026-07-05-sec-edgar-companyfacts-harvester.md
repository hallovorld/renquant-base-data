# SEC EDGAR companyfacts harvester

**Date:** 2026-07-05
**PR:** TBD (base-data)
**Status:** ready for review

## What

`sec_edgar_companyfacts_harvester.py` — fetches full per-issuer XBRL fact
history from SEC EDGAR's `companyfacts` API (revenue, net income, diluted
EPS, total assets), preserving each fact's real `filed` date as the PIT
anchor. Free, no API key.

Relocated here from `renquant-orchestrator#350` per Codex review: raw
data-vendor extraction/harvesting is base-data's ownership, not
orchestrator's (orchestrator orchestrates pinned subrepos and schedules
data workflows — it should carry a thin wrapper here, not the extraction
logic itself). Matches the established `renquant_base_data.fmp_estimate_revisions`
pattern already used by orchestrator's `pit_revision_collector.py`, and the
`AlpacaBrokerPort` (orchestrator → renquant-execution, PR #291) /
`SoftwareStopRegistry` (umbrella → renquant-pipeline, PR #167) relocation
precedent from earlier this session.

Named `sec_edgar_companyfacts_harvester.py` (not `sec_edgar_harvester.py`) to
stay distinct from the existing `sec_fundamentals.py`, which uses the
`frames` API (one cross-sectional value per concept/period across ALL
issuers) — a different access pattern for a different use case. This module
uses `companyfacts` (full per-issuer history in one call), the right shape
for a per-ticker PIT harvest.

## Fixes from the original PR #350 review

1. **Repo boundary** — moved the harvester here (see above).
2. **Output-contract bug**: `extract_facts()` previously mapped
   `RevenueFromContractWithCustomerExcludingAssessedTax` to a DIFFERENT
   output field (`revenue_alt`) than the plain `Revenues` tag (`revenue`) —
   the same economic concept landed under different field names depending on
   issuer/ASC-606 adoption era, silently breaking the stated four-field
   contract for any downstream consumer keying on `revenue`. Fixed via
   `CANONICAL_CONCEPTS`: every known revenue-tag variant (`Revenues`,
   `RevenueFromContractWithCustomerExcludingAssessedTax`,
   `RevenueFromContractWithCustomerIncludingAssessedTax`, `SalesRevenueNet`)
   normalizes to the single canonical `revenue` field; the actual tag
   matched is preserved per-record as `xbrl_tag` provenance, never folded
   into the field name.
3. **Resumability**: the original `load_completed_tickers()` treated ANY
   record's presence for a ticker as proof that ticker was fully harvested —
   so a crash/kill mid-ticker (some fact records written, then interrupted)
   would cause a rerun to silently skip re-harvesting it, permanently losing
   the remaining facts. Fixed: each ticker's batch of records is now written
   as one atomic append immediately followed by an explicit
   `{"ticker": ..., "_harvest_complete": true}` marker record;
   `load_completed_tickers()` only counts a ticker as done if that marker is
   present. `test_harvest_rerun_after_partial_crash_reharvests_ticker`
   reproduces the crash scenario and confirms the fixed code re-harvests
   rather than skips.

## Tests

30 tests in `tests/test_sec_edgar_companyfacts_harvester.py` — CIK mapping,
fact extraction, the ASC-606 canonical-field-normalization fix (2 dedicated
tests), the marker-based resumability fix (5 dedicated tests including the
partial-crash-rerun scenario), harvest integration (mocked HTTP), output
format. Full base-data suite: 293 passed, 1 skipped, no regressions.

## Orchestrator side

`renquant-orchestrator#350` is being reduced to a thin scheduling wrapper
around this module (subprocess-invoking
`renquant_base_data.sec_edgar_companyfacts_harvester`), matching
`pit_revision_collector.py`'s existing pattern — see that PR's own updated
progress doc for the wrapper-side changes.
