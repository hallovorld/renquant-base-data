# 2026-07-29 — survivorship: what the existing subscriptions can and cannot buy

STATUS:    findings delivered, no spend made, decision requested. Relocated
           from renquant-orchestrator#596 per codex BLOCKER (repo-ownership
           violation — this is a base-data panel/vendor question, not
           orchestration).

WHAT:      Read-only live API probes into whether a delisting-inclusive,
           survivorship-safe PIT universe is buildable on the FMP + Alpaca
           subscriptions already owned, given GOAL-6 Stage 1's breadth panel
           (830 names) failed its survivorship criterion outright: 0 of 23
           probed known-delisted large caps present, zero exits across 10.3
           years, a today-alive screen whose size variation is OHLCV harvest
           vintage, not survivorship. Every Stage-2 result trained on that
           panel inherits the bias.

WHY/DIR:   Before spending on a new vendor product or scoping the experiment
           down to an explicitly-biased universe, measure precisely what the
           gap is and what it costs to close, rather than assuming either
           "we already have this" or "we need to buy something new."

EVIDENCE:
  artifact:      this progress doc (no code/artifact changed — probes only,
                 read-only against live vendor APIs) `[VERIFIED — this PR's
                 diff]`.
  prod or exp:   experiment — live read-only API probes, no serving/panel
                 code touched, no purchase made.
  existing data: live probes `[VERIFIED — 2026-07-29]`:
                 (1) bars for a KNOWN delisted symbol — FMP
                 `stable/historical-price-eod/full` — WORKS: TWTR returns 459
                 bars ending 2022-10-27, its real final trading days;
                 (2) the delisting REGISTRY (enumerate who left, and when) —
                 FMP `stable/delisted-companies` — page 0 only: 100 rows
                 spanning 2026-07-02..07-27; page 1 returns HTTP 402 Payment
                 Required on the Starter plan;
                 (3) an inactive-symbol universe — Alpaca
                 `/v2/assets?status=inactive` — 19,209 symbols but
                 OTC-dominated (16,355 OTC vs 1,310 NASDAQ / 845 NYSE / 97
                 AMEX), and hits only 2 of 9 probed known-delisted majors
                 (CERN, XLNX; misses TWTR, ATVI, VMW, SIVB, FRC, PXD, SPLK).
  best-known?:   the constraint is precise and it is NOT the price data — we
                 can already fetch history for a delisted name once we know
                 its ticker; we cannot enumerate the delisted universe. FMP
                 gives ~4 weeks of registry depth on this plan; Alpaca's
                 inactive list is not a delisting registry, it is whatever
                 Alpaca stopped carrying, missing 7 of 9 large-cap
                 delistings checked and drowned in OTC. Survivorship is
                 blocked on a SYMBOL LIST, not on prices — and that
                 enumeration is exactly the expensive part of any vendor
                 product.
  scope:         "this is a data-availability probe only; no IC/Sharpe claim
                 is made, no panel/serving code is touched, no purchase is
                 made. The §4(b) sanity triad does not apply — this is a
                 vendor-capability measurement, not a model result."

## Options, costed

1. **FMP tier upgrade** — unblocks `delisted-companies` paging directly, and
   the price endpoint already works. Cost: the delta above Starter ($29).
   Cheapest path IF the registry's history reaches back to 2016; the probe
   cannot confirm depth because page 1 is gated. **Verify depth before
   paying** — a plan that pages further but still only covers recent years
   buys nothing.
2. **Historical index constituents** (e.g. S&P 500 membership by date) — a
   different product, gives PIT membership rather than a delisting registry.
   Solves survivorship *for an index-defined universe*, arguably the
   better-defined experiment anyway.
3. **Reconstruct from what we hold** — the panel's own history plus EDGAR
   filings can identify names that stopped filing, but conflates delisting
   with acquisition, going private, and filer-status changes. Cheap, noisy,
   hard to defend in a prereg.
4. **Scope down** — run Stage 2 on an explicitly today-alive universe and
   report the survivorship bias as a stated limitation rather than pretending
   it is absent. Costs nothing and is honest, but every number it produces
   carries an asterisk that cannot be removed later.

## Recommendation

**Option 2, then 1.** An index-constituent-by-date list defines the universe
the strategy actually competes against, and makes "who was investable on
date d" answerable without a delisting registry at all. Option 1 is the
fallback if constituent history proves harder to source than the tier
upgrade.

**Not option 4 silently.** If Stage 2 runs before this is fixed, its
limitation section must state the bias in the same place its results are
quoted, not in a footnote.

## What this probe does NOT establish

Whether FMP's paid registry reaches back to 2016 (page 1 is gated, so depth
is unmeasured); what an index-constituent product costs; whether Alpaca's
inactive list would be adequate after filtering to names that ever appeared
in our panel — that last one is cheap to check and is the obvious next probe.

NEXT:      Operator decision between an index-constituent-by-date product
           (recommended — defines the universe the strategy competes
           against, sidesteps the registry entirely), an FMP tier upgrade
           (verify registry DEPTH before paying — page 1 is gated so depth
           is unmeasured), or running Stage 2 with the bias stated in the
           results rather than a footnote. Cheap next probe: whether
           Alpaca's inactive list is adequate once filtered to names that
           ever appeared in our panel. Once a universe manifest/fingerprint
           is chosen and built here, the orchestrator-facing contract is a
           separate, narrow PR in renquant-orchestrator that consumes the
           published fingerprint and enforces a run gate — not this probe's
           prose.
