"""Harvest the full FMP split calendar for the 830-ticker panel universe.

Why: the in-repo sources are both partial --
  * data/ohlcv/<T>/1d.parquet::split_ratio exists for only 63 / 830 tickers
  * data/fmp_harvest/splits_291.parquet requested only 291 tickers (211 with data)
A cumulative adjustment factor needs an authoritative "no split" answer for EVERY
ticker, so absence-of-event must itself be sourced. This fetches all 830.

Uses the same endpoint already recorded in
data/fmp_harvest/splits_291.manifest.json: stable/splits?symbol={sym}
Existing FMP Starter subscription; no new spend.

Writes ONLY into this scratch directory.
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

RQ = Path("/Users/renhao/git/github/RenQuant")
def _work_dir() -> Path:
    """Scratch work dir for this split-fix lane, overridable.

    Was a hard-coded agent-session path under /private/tmp, which made every
    number these tools produce unreproducible by anyone else (codex review on
    base-data#58, round 2). Set RQ_SPLIT_FIX_DIR to relocate. The default is
    now a plain relative dir under cwd -- NOT the prior session's ephemeral
    /private/tmp path, which does not exist on another machine/session and
    defeated the point of making this overridable.
    """
    import os
    env = os.environ.get("RQ_SPLIT_FIX_DIR")
    return Path(env).expanduser() if env else Path("scratch/split-fix")


OUT = _work_dir()
assert not str(OUT).startswith(str(RQ)), "output must be outside the production tree"

KEY = None
for line in (RQ / ".env").read_text().splitlines():
    if line.startswith("FMP_API_KEY"):
        KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
assert KEY, "FMP_API_KEY not found"

TICKERS = sorted((OUT / "tickers.txt").read_text().split())
rows, no_data, errs = [], [], []
t0 = time.time()
for i, sym in enumerate(TICKERS):
    url = f"https://financialmodelingprep.com/stable/splits?symbol={sym}&apikey={KEY}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                js = json.load(r)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                errs.append({"ticker": sym, "err": f"{type(e).__name__}: {e}"})
                js = None
            else:
                time.sleep(2.0 * (attempt + 1))
    if js is None:
        continue
    if not js:
        no_data.append(sym)
        continue
    for d in js:
        rows.append({"ticker": sym, "date": d.get("date"),
                     "numerator": d.get("numerator"), "denominator": d.get("denominator"),
                     "splitType": d.get("splitType")})
    time.sleep(0.20)
    if i % 100 == 0:
        print("  %4d/%d  elapsed=%.0fs rows=%d" % (i, len(TICKERS), time.time() - t0, len(rows)))

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df["ratio"] = df.numerator / df.denominator
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
df.to_parquet(OUT / "fmp_splits_830.parquet", index=False)

man = {
    "endpoint": "stable/splits?symbol={sym}",
    "requested": len(TICKERS), "with_data": int(df.ticker.nunique()),
    "no_data": len(no_data), "errors": len(errs),
    "rows": len(df),
    # FULL list, not a sample: build_split_factor.py treats a ticker with an
    # empty calendar as "authoritative never split" ONLY if it appears in
    # no_data_tickers. A truncated error list would let an errored ticker look
    # like a clean no-event answer, which is the fail-open this manifest exists
    # to prevent. error_samples stays for human reading; error_tickers is the
    # machine-readable contract.
    "no_data_tickers": no_data,
    "error_tickers": sorted({e["ticker"] for e in errs}),
    "error_samples": errs[:20],
    "fetched_at": pd.Timestamp.utcnow().isoformat(),
    "note": ("no_data == FMP has no split history for the symbol == authoritative "
             "'never split'. errors are NOT authoritative and must be treated as unknown."),
}
(OUT / "fmp_splits_830.manifest.json").write_text(json.dumps(man, indent=2))
print("\nrequested=%d with_data=%d no_data=%d errors=%d rows=%d  (%.0fs)"
      % (len(TICKERS), df.ticker.nunique(), len(no_data), len(errs), len(df), time.time() - t0))
if errs:
    print("ERRORS (treated as unknown, NOT as no-split):")
    print(pd.DataFrame(errs).to_string(index=False))
