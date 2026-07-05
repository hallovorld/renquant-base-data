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
import os
import tempfile
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


def _replace_ticker_batch(output: Path, ticker: str, batch: str) -> None:
    """Atomically replace *ticker*'s lines in *output* with *batch*.

    Two hazards, both closed here:

    1. **Crash mid-write.** A plain ``open(output, "a").write(...)`` is not
       crash-safe — a hard kill or power loss mid-syscall can leave a
       partial batch on disk (some fact lines persisted, the trailing
       completion-marker line truncated or missing).
    2. **Stale partial-write duplication.** Because a ticker with no marker
       is (correctly) re-harvested on the next run, any fact lines already
       on disk for that ticker are leftovers from an interrupted, never-
       completed attempt — not data to preserve. Naively *appending* the
       fresh batch on top of them (even if that append itself were atomic)
       would duplicate every fact the interrupted run had already written.

    This function reads the existing file, drops every line belonging to
    *ticker* (stale partial data, since we are here specifically because
    that ticker had no completion marker), appends the freshly-harvested
    *batch*, and commits the result via write-temp + fsync + ``os.replace``
    (an atomic rename on POSIX) — so *output* is always either its previous
    complete state or the new state, never a mix, and never contains more
    than one copy of any fact record for a given rerun. Output files here
    are one harvest run's worth of a bounded watchlist (at most low
    hundreds of tickers), so rewriting the whole file per ticker is cheap.
    """
    kept_lines: list[str] = []
    if output.exists():
        for line in output.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ticker") == ticker:
                continue  # stale partial data for the ticker being redone
            kept_lines.append(line)

    new_content = "".join(l + "\n" for l in kept_lines) + batch
    fd, tmp_name = tempfile.mkstemp(
        dir=str(output.parent), prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(new_content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, output)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


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

    Each ticker's records plus its completion-marker record are committed to
    *output* via an atomic whole-file replace that also drops any stale,
    marker-less lines already on disk for that ticker (see
    :func:`_replace_ticker_batch`) — so a kill/crash while writing one
    ticker's batch, or a rerun after one, can never leave duplicate fact
    rows for that ticker on disk.
    """
    if session is None:
        session = _session()

    if ticker_cik_map is None:
        log.info("Fetching SEC ticker->CIK mapping...")
        ticker_cik_map = fetch_ticker_cik_map(session)
        log.info("Loaded %d ticker->CIK mappings", len(ticker_cik_map))

    skip = load_completed_tickers(output) if output else set()
    all_records: list[dict[str, Any]] = []

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

        if output:
            batch = "".join(
                json.dumps(rec, sort_keys=True) + "\n" for rec in records
            )
            batch += json.dumps(
                {"ticker": ticker, _HARVEST_COMPLETE_MARKER: True,
                 "record_count": len(records)},
                sort_keys=True,
            ) + "\n"
            _replace_ticker_batch(output, ticker, batch)

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
