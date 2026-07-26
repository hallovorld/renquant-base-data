"""EDGAR filing-date backfill — authoritative PIT stamps for fundamentals.

Fetches SEC `acceptanceDateTime` for 10-K/10-Q filings from the EDGAR
submissions API and emits `available_at_v2` = acceptance + 1 business day.
This replaces the assumed 45-day `expected_filing_lag` stamps that cover
96.9% of `sec_fundamentals_daily.parquet` (issue #51): measured against
this ground truth, 13.2% of assumed-lag rows were served EARLIER than the
filing was public (median 10 days of look-ahead) and 18.0% of periods
violate the 45d assumption (median true lag 36d, p90 54d).

Sources (free, no vendor):
  https://www.sec.gov/files/company_tickers.json          ticker -> CIK
  https://data.sec.gov/submissions/CIK##########.json     filings + acceptance

Usage::

    python -m renquant_base_data.edgar_filing_dates \
        --panel  <path to a parquet with a `ticker` column> \
        --out-dir <dir>   # writes filing_dates.parquet + available_at_v2 needs
                          # a fundamentals file to join; see --fundamentals

Notes
-----
* SEC fair-access: custom User-Agent + <=8 req/s. This module sleeps 0.12 s
  between requests.
* ETFs (no 10-K/10-Q) legitimately produce no rows.
* Ticker->CIK tries the raw symbol, then '.'<->'-' share-class variants.
* Amended forms (10-K/A, 10-Q/A) are EXCLUDED from v2 stamps: the original
  filing is the first public availability of the period's data.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

USER_AGENT = "RenQuant research renhao.overflow@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FORMS = ("10-K", "10-Q")
SLEEP_S = 0.12


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def cik_variants(ticker: str) -> list[str]:
    """Symbol spellings to try against company_tickers.json."""
    t = ticker.upper()
    out = [t]
    if "." in t:
        out.append(t.replace(".", "-"))
    if "-" in t:
        out.append(t.replace("-", "."))
    return out


def load_cik_map() -> dict[str, int]:
    raw = _get(TICKERS_URL)
    return {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}


def fetch_filing_dates_for_cik(cik: int, min_period: str = "2014-01-01") -> list[dict]:
    """All 10-K/10-Q (original forms) for one CIK, incl. paginated history.

    Fails closed: a fetch failure on any paginated history file propagates
    (instead of being swallowed) so `build_filing_dates`'s per-ticker
    try/except marks the whole CIK missing, rather than silently returning
    a partial history mislabeled as complete coverage.
    """
    root = _get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"))
    frames = [root["filings"]["recent"]]
    for extra in root["filings"].get("files", []):
        frames.append(_get(SUBMISSIONS_URL.format(name=extra["name"])))
        time.sleep(SLEEP_S)
    rows = []
    for r in frames:
        for form, acc, rep, fdate in zip(r["form"], r["acceptanceDateTime"],
                                         r["reportDate"], r["filingDate"]):
            if form in FORMS and rep and rep >= min_period:
                rows.append({"cik": cik, "form": form, "period_end": rep,
                             "accepted_at": acc, "filing_date": fdate})
    return rows


def build_filing_dates(tickers: list[str], min_period: str = "2014-01-01",
                       log=print) -> tuple[pd.DataFrame, list[str]]:
    cikmap = load_cik_map()
    rows, missing = [], []
    for i, t in enumerate(sorted(set(tickers))):
        cik = next((cikmap[v] for v in cik_variants(t) if v in cikmap), None)
        if cik is None:
            missing.append(t)
            continue
        try:
            for r in fetch_filing_dates_for_cik(cik, min_period):
                rows.append({"ticker": t, **r})
            time.sleep(SLEEP_S)
        except Exception:
            missing.append(t)
        if i % 50 == 0:
            log(f"  [{i}] filings={len(rows)}")
    df = pd.DataFrame(rows).drop_duplicates(["ticker", "form", "period_end"])
    return df, missing


def stamp_available_v2(filing_dates: pd.DataFrame) -> pd.DataFrame:
    """available_at_v2 = acceptance date + 1 business day (normalized)."""
    out = filing_dates.copy()
    acc = pd.to_datetime(out["accepted_at"]).dt.tz_localize(None)
    out["available_at_v2"] = (acc + pd.offsets.BDay(1)).dt.normalize()
    out["period_end"] = pd.to_datetime(out["period_end"])
    return out[["ticker", "period_end", "form", "available_at_v2"]]


def restamp_fundamentals(fundamentals: pd.DataFrame,
                         stamps: pd.DataFrame) -> pd.DataFrame:
    """Join v2 stamps onto a fundamentals record set by (ticker, fiscal period).

    Rows with no EDGAR match keep their original `available_at` (conservative:
    the measured direction of the assumed-lag error is mostly late-serving).
    Null availability rows are dropped — they cannot be served point-in-time.
    """
    rec = fundamentals.copy()
    rec["fiscal_period_end"] = pd.to_datetime(rec["fiscal_period_end"])
    j = rec.merge(stamps.rename(columns={"period_end": "fiscal_period_end"}),
                  on=["ticker", "fiscal_period_end"], how="left")
    j["available_source_v2"] = j["available_at_v2"].notna().map(
        {True: "edgar_accepted", False: "carried_v1"})
    j["available_at_v2"] = j["available_at_v2"].fillna(pd.to_datetime(j["available_at"]))
    j = j.dropna(subset=["available_at_v2"])
    return j


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True,
                    help="parquet with a `ticker` column (defines the universe)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-period", default="2014-01-01")
    args = ap.parse_args()
    tickers = sorted(pd.read_parquet(args.panel, columns=["ticker"])
                     ["ticker"].unique())
    df, missing = build_filing_dates(tickers, args.min_period)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "filing_dates.parquet")
    stamp_available_v2(df).to_parquet(out / "available_at_v2.parquet")
    print(f"filings={len(df)} tickers={df['ticker'].nunique()}/{len(tickers)} "
          f"missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
