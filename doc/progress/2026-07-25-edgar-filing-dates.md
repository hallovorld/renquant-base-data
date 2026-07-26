# 2026-07-25 — EDGAR filing-date backfill: authoritative PIT stamps (issue #51)

STATUS:    module + tests + one-off artifact delivered; panel-rebuild wiring = next
WHAT:      `src/renquant_base_data/edgar_filing_dates.py` — ticker→CIK mapping
           (share-class variants), EDGAR submissions fetch (paginated, rate-limited,
           original 10-K/10-Q only), `available_at_v2` = acceptance + 1 BD,
           `restamp_fundamentals()` join helper. 3 network-free unit tests.
WHY/DIR:   96.9% of `sec_fundamentals_daily` stamps are an assumed 45d lag (issue
           #51). Ground truth now measured: 18.0% of periods violate the assumption;
           13.2% of assumed rows were served EARLIER than public (median 10d
           look-ahead) — on the model's dominant signal source.
EVIDENCE:
  artifact:      umbrella `data/edgar_pit/{filing_dates,available_at_v2}.parquet`
                 (12,522 filings, 275/292 panel tickers, 2014-2026; 8 of 9 misses
                 are ETFs with no 10-K/10-Q — correct absences). ADDITIVE only.
  prod or exp:   data-infrastructure; nothing in the production serving path changes
                 until the panel/serving builders consume `available_at_v2` via a
                 reviewed PR (explicitly NOT done here).
  existing data: validation battery in issue #51 comments, incl. the DOWNWARD
                 correction of the original 58.2% claim (small-subset bias) and the
                 70.3%-within-±2d fmp cross-check (demoted to diagnostic — EDGAR is
                 authoritative; the tail indicts fmp's acceptedDate).
  best-known?:   SEC acceptanceDateTime is the definitive public-availability record
  scope:         "true-PIT stamps exist for the panel universe 2014-2026; first
                 consumer result: asset_growth clean IC rises to +0.0162 @60d under
                 v2 stamps (house 0.015 bar: PASS) — screen-grade, single feature"
NEXT:      (1) wire `available_at_v2` into the fundamentals serving/panel build
           behind a flag, with a paired v1-vs-v2 model-level A/B; (2) extend the
           fetch to the full 824-ticker universe; (3) re-test ey_within_volcohort
           and the specialist-blend famA line on the honest stamps.
