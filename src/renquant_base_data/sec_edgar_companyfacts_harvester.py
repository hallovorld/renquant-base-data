"""SEC EDGAR XBRL company-facts harvester — free PIT financial data (N2/RS-3).

Fetches full per-company XBRL fact history from SEC EDGAR's ``companyfacts``
API, preserving the ``filed`` date as the point-in-time (PIT) timestamp. This
is the ground-truth filing date — when the SEC received the document — and
is the correct ``available_at`` anchor for leak-free backtesting.

This is distinct from ``sec_fundamentals.py`` (the ``frames`` API, one
cross-sectional value per concept/period across ALL issuers) — this module
uses the ``companyfacts`` API (full per-issuer history across all periods
and forms in one call), which is the right shape for a per-ticker PIT
harvest.

Output: JSONL with one record per (ticker, field, period, form), each
carrying the ``filed`` date. Never writes to canonical ``data/`` paths.

SEC API rules: User-Agent header required, <=10 req/sec.

Usage:
    sec_edgar_companyfacts_harvester.py --tickers AAPL,GRMN,MU --output /tmp/edgar.jsonl
    sec_edgar_companyfacts_harvester.py --watchlist watchlist.txt --output /tmp/edgar.jsonl
    sec_edgar_companyfacts_harvester.py --tickers AAPL          # stdout
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

log = logging.getLogger("renquant_base_data.sec_edgar_companyfacts_harvester")

USER_AGENT = "RenQuant research renhao.overflow@gmail.com"
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
REQUEST_DELAY = 0.15  # <=10 req/sec

# Multiple XBRL concept tags can carry the SAME economic concept, varying by
# issuer and accounting era (e.g. ASC 606 revenue-recognition adoption moved
# many issuers off the plain ``Revenues`` tag onto a
# ``RevenueFromContractWithCustomer*`` variant). All tags in one concept's
# tuple normalize to that concept's single canonical field name; the specific
# tag actually matched is preserved per-record as provenance
# (``xbrl_tag``), never folded into the field name itself.
CANONICAL_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "net_income": (
        "NetIncomeLoss",
    ),
    "eps_diluted": (
        "EarningsPerShareDiluted",
    ),
    "total_assets": (
        "Assets",
    ),
}

# Reverse lookup: concept name -> canonical field name.
_CONCEPT_TO_FIELD: dict[str, str] = {
    concept: field_name
    for field_name, concepts in CANONICAL_CONCEPTS.items()
    for concept in concepts
}

_HARVEST_COMPLETE_MARKER = "_harvest_complete"


def _session() -> "requests.Session":
    if requests is None:
        raise ImportError("requests is required: pip install requests")
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    s.headers["Accept"] = "application/json"
    return s


def fetch_ticker_cik_map(session: "requests.Session") -> dict[str, int]:
    """Download SEC's ticker->CIK mapping. Returns {TICKER: cik_int}."""
    time.sleep(REQUEST_DELAY)
    resp = session.get(TICKER_CIK_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return parse_ticker_cik_map(data)


def parse_ticker_cik_map(data: dict[str, Any]) -> dict[str, int]:
    """Parse the SEC ticker-CIK JSON (for testing without HTTP)."""
    return {
        entry["ticker"].upper(): int(entry["cik_str"])
        for entry in data.values()
        if "ticker" in entry and "cik_str" in entry
    }


def fetch_company_facts(
    session: "requests.Session", cik: int
) -> dict[str, Any] | None:
    """Fetch XBRL company facts for a CIK. Returns None on error."""
    url = COMPANY_FACTS_URL.format(cik=str(cik).zfill(10))
    time.sleep(REQUEST_DELAY)
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            log.warning("CIK %s: 404 (no XBRL filings)", cik)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        log.exception("CIK %s: fetch failed", cik)
        return None


def extract_facts(
    ticker: str, facts_json: dict[str, Any]
) -> list[dict[str, Any]]:
    """Extract target fields from XBRL company facts JSON.

    Every XBRL concept tag mapped in ``CANONICAL_CONCEPTS`` normalizes to its
    concept's single canonical ``field`` name (e.g. ``revenue``), regardless
    of which specific tag variant the issuer/era used. The originating tag
    is preserved per-record as ``xbrl_tag`` for provenance/debugging — never
    folded into the field name.
    """
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    records: list[dict[str, Any]] = []

    for concept_name, field_name in _CONCEPT_TO_FIELD.items():
        concept = us_gaap.get(concept_name)
        if concept is None:
            continue

        units = concept.get("units", {})
        unit_key = "USD/shares" if "EarningsPerShare" in concept_name else "USD"
        entries = units.get(unit_key, [])

        for entry in entries:
            form = entry.get("form", "")
            if form not in ("10-K", "10-Q"):
                continue

            records.append({
                "ticker": ticker,
                "field": field_name,
                "xbrl_tag": f"us-gaap:{concept_name}",
                "value": entry.get("val"),
                "filed_date": entry.get("filed"),
                "period_end": entry.get("end"),
                "period_start": entry.get("start"),
                "fiscal_year": entry.get("fy"),
                "fiscal_period": entry.get("fp"),
                "form": form,
                "accession_number": entry.get("accn"),
                "source": "sec_edgar_xbrl",
            })

    return records


def load_completed_tickers(output_path: Path) -> set[str]:
    """Read fully-harvested tickers from an existing JSONL file.

    A ticker only counts as complete if its explicit completion marker
    record is present — not merely because SOME record for it exists. This
    makes resumability correct across a rerun that follows a crash/kill
    mid-ticker: a partially-written ticker (fact records written, but the
    process died before the marker) is correctly re-harvested, not silently
    treated as done.
    """
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get(_HARVEST_COMPLETE_MARKER):
                seen.add(rec.get("ticker", ""))
    return seen


def harvest(
    tickers: list[str],
    output: Path | None = None,
    *,
    session: "requests.Session | None" = None,
    ticker_cik_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Harvest EDGAR XBRL facts for a list of tickers.

    Returns all extracted records. If *output* is given, appends JSONL there
    (resumable: skips tickers whose completion marker is already present).

    Each ticker's records are written and flushed as a single atomic batch,
    immediately followed by its own completion-marker record — so a
    kill/crash between two tickers, or mid-write of one ticker's records
    (before the marker line lands), never leaves a ticker looking "done"
    when it is not.
    """
    if session is None:
        session = _session()

    if ticker_cik_map is None:
        log.info("Fetching SEC ticker->CIK mapping...")
        ticker_cik_map = fetch_ticker_cik_map(session)
        log.info("Loaded %d ticker->CIK mappings", len(ticker_cik_map))

    skip = load_completed_tickers(output) if output else set()
    all_records: list[dict[str, Any]] = []
    out_fh = open(output, "a") if output else None

    try:
        for i, ticker in enumerate(tickers):
            ticker = ticker.upper().strip()
            if ticker in skip:
                log.info("[%d/%d] %s: skipped (already harvested)", i + 1, len(tickers), ticker)
                continue

            cik = ticker_cik_map.get(ticker)
            if cik is None:
                log.warning("[%d/%d] %s: no CIK found", i + 1, len(tickers), ticker)
                continue

            facts = fetch_company_facts(session, cik)
            if facts is None:
                continue

            records = extract_facts(ticker, facts)
            all_records.extend(records)
            log.info(
                "[%d/%d] %s (CIK %d): %d records",
                i + 1, len(tickers), ticker, cik, len(records),
            )

            if out_fh:
                batch = "".join(
                    json.dumps(rec, sort_keys=True) + "\n" for rec in records
                )
                batch += json.dumps(
                    {"ticker": ticker, _HARVEST_COMPLETE_MARKER: True,
                     "record_count": len(records)},
                    sort_keys=True,
                ) + "\n"
                out_fh.write(batch)
                out_fh.flush()
    finally:
        if out_fh:
            out_fh.close()

    return all_records


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tickers", help="Comma-separated ticker list (e.g. AAPL,GRMN,MU)"
    )
    group.add_argument(
        "--watchlist", help="Path to a file with one ticker per line"
    )
    ap.add_argument(
        "--output",
        help="Output JSONL path (default: stdout). NEVER use data/ paths.",
    )
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = Path(args.watchlist).read_text().strip().splitlines()
        tickers = [t.strip() for t in tickers if t.strip() and not t.startswith("#")]

    output = Path(args.output) if args.output else None
    records = harvest(tickers, output)

    if output is None:
        for rec in records:
            print(json.dumps(rec, sort_keys=True))

    total = len(records)
    tickers_ok = len({r["ticker"] for r in records})
    log.info("Done: %d records from %d tickers", total, tickers_ok)


if __name__ == "__main__":
    main()
