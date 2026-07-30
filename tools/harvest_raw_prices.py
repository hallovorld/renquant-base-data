"""Harvest TRUE RAW (unadjusted) closes so the cumulative adjustment factor can be
MEASURED rather than reconstructed from a split calendar.

Route rationale
---------------
data/ohlcv/<T>/1d.parquet::close is back-adjusted, so
    close_stored(t) = price_raw(t) / CumAdj(t)
and therefore
    CumAdj(t) = price_raw(t) / close_stored(t)                     <-- measured, per date

FMP stable/historical-price-eod/non-split-adjusted returns price_raw (verified: AAPL
2020-08-28 adjClose=499.24 against a stored close of 124.8075, ratio 4.000080).

Measuring beats reconstructing because it needs no split/spinoff classification, no
same-date conflict resolution, and it captures EXACTLY the factors the stored series
embeds -- including ones no calendar we have lists (MMM's 2024-04-01 Solventum factor is
absent from FMP's split endpoint yet is present in the stored prices), and including
co-dated events whose factors multiply (HON 2026-06-29: 0.5 reverse split x 0.9535
spinoff = 0.47675 measured).

Existing FMP Starter subscription; no new spend. Writes ONLY into this scratch dir.
"""
from __future__ import annotations
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

RQ = Path("/Users/renhao/git/github/RenQuant")
OHLCV = RQ / "data/ohlcv"
def _work_dir() -> Path:
    """Scratch work dir for this split-fix lane, overridable.

    Was a hard-coded agent-session path under /private/tmp, which made every
    number these tools produce unreproducible by anyone else (codex review on
    base-data#58). Set RQ_SPLIT_FIX_DIR to relocate; the previous path stays as
    the default so existing artifacts keep resolving.
    """
    import os
    env = os.environ.get("RQ_SPLIT_FIX_DIR")
    if env:
        return Path(env).expanduser()
    return Path("/private/tmp/claude-502/"
                "-Users-renhao-git-github-renquant-orchestrator/"
                "428feb92-8ee7-4b4f-afed-1e4fa82ef367/scratchpad/split-fix")


OUT = _work_dir()
CACHE = OUT / "raw_price_cache"
CACHE.mkdir(parents=True, exist_ok=True)
assert not str(OUT).startswith(str(RQ)), "output must be outside the production tree"

KEY = [l.split("=", 1)[1].strip().strip('"').strip("'")
       for l in (RQ / ".env").read_text().splitlines() if l.startswith("FMP_API_KEY")][0]
TICKERS = sorted((OUT / "tickers.txt").read_text().split())


def span(t: str):
    p = OHLCV / t / "1d.parquet"
    if not p.exists():
        return None
    idx = pd.to_datetime(pd.read_parquet(p, columns=[]).index).normalize()
    return idx.min(), idx.max()


def fetch(t: str, a, b):
    url = ("https://financialmodelingprep.com/stable/historical-price-eod/"
           f"non-split-adjusted?symbol={t}&from={a:%Y-%m-%d}&to={b:%Y-%m-%d}&apikey={KEY}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def main() -> None:
    done = skipped = failed = 0
    errs = []
    t0 = time.time()
    for i, t in enumerate(TICKERS):
        cp = CACHE / f"{t}.parquet"
        if cp.exists():
            skipped += 1
            continue
        s = span(t)
        if s is None:
            continue
        try:
            js = fetch(t, s[0], s[1])
        except Exception as e:  # noqa: BLE001
            errs.append({"ticker": t, "err": f"{type(e).__name__}: {e}"})
            failed += 1
            continue
        if not js:
            pd.DataFrame(columns=["date", "raw_close"]).to_parquet(cp, index=False)
            done += 1
            continue
        d = pd.DataFrame(js)
        col = "adjClose" if "adjClose" in d.columns else "close"
        d = d[["date", col]].rename(columns={col: "raw_close"})
        d["date"] = pd.to_datetime(d["date"])
        d = d.dropna().drop_duplicates("date").sort_values("date")
        d.to_parquet(cp, index=False)
        done += 1
        time.sleep(0.18)
        if done % 100 == 0:
            print("  fetched=%d skipped=%d failed=%d  elapsed=%.0fs"
                  % (done, skipped, failed, time.time() - t0), flush=True)
    print("DONE fetched=%d cached_skipped=%d failed=%d elapsed=%.0fs"
          % (done, skipped, failed, time.time() - t0))
    if errs:
        (OUT / "raw_price_errors.json").write_text(json.dumps(errs, indent=2))
        print("errors written; %d tickers unresolved" % len(errs))


if __name__ == "__main__":
    main()
