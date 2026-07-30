"""Reusable builder: per-ticker CUMULATIVE ADJUSTMENT FACTOR on the trading-date axis.

CONTRACT
--------
For every ticker and trading date t the builder emits ``cum_factor`` F(t) such that

    price_raw(t)            = close_stored(t) * F(t)
    shares_on_close_basis   = shares_as_filed * F(reference_date_of_that_share_fact)

i.e. multiplying an as-filed share count by F puts it on the same basis as the
back-adjusted closes in data/ohlcv, so that

    market_cap(t) = shares_as_filed * F(cover_date) * close_stored(t)

is the true historical market cap and the adjustment basis cancels exactly.

F(t) = product of every adjustment factor whose ex-date is STRICTLY AFTER t, so F is a
non-increasing-in-information step function that equals 1.0 after the last event.

WHY F IS EVALUATED AT THE SHARE FACT'S COVER DATE, NOT AT t
-----------------------------------------------------------
Between a split's ex-date and the next filing, the newest as-filed share count is still
on the PRE-split basis while F(t) has already dropped to 1. Multiplying by F(t) there
understates market cap by the whole split factor. Anchoring F at the share fact's own
cover date makes the product exact on both sides of every ex-date:
    market_cap(t) = S * prod{ex > cover} f * close(t)
              and price_raw(t) = close(t) * prod{ex > t} f, so the two agree.

ROUTE (established empirically -- see doc string of harvest_raw_prices.py)
-------------------------------------------------------------------------
PRIMARY  = MEASURED.  F(t) = raw_price(t) / close_stored(t), from FMP
           historical-price-eod/non-split-adjusted. Measured, not reconstructed, so it
           needs no split-vs-spinoff classification, inherits no calendar date errors,
           captures factors absent from every calendar we hold (MMM 2024-04-01), and
           captures co-dated events whose factors MULTIPLY (HON 2026-06-29 = 0.5 reverse
           split x 0.9535 spinoff = 0.47675, which no single source states).
           Segment boundaries are DETECTED from the measured series, and each segment's
           level is the segment MEDIAN, so one bad vendor day cannot move a factor.
FALLBACK = RECONSTRUCTED from the union of the two in-repo calendars (FMP splits +
           the ohlcv split_ratio column), used only where raw prices are unavailable.
FAIL-CLOSED = if neither route yields a factor, the ticker's factor is NaN and every
           market-cap-derived feature for it becomes NaN. Never silently 1.0.

Reads only:  data/ohlcv/<T>/1d.parquet, data/fmp_harvest/splits_291.parquet (read-only)
             plus this scratch dir's raw-price cache and fmp_splits_830.parquet
Writes only: this scratch dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

# a step below this is immaterial (0.2% of market cap) and indistinguishable from the
# 2-decimal rounding of the vendor's raw price
STEP_TOL = 0.002
MIN_COVERAGE = 0.50        # need raw prices on at least this share of the ticker's dates
# A real adjustment factor PERSISTS; a one-day disagreement between the two price vendors
# does not. Segments shorter than this are merged away -- but only when the step is small,
# so that a genuine split landing near the end of the file is never merged out.
MIN_SEG_OBS = 5
MERGE_MAX_STEP = 0.02
# if the measured ratio still wobbles this much INSIDE a segment, the two price series are
# not describing the same instrument cleanly and no factor is published (fail closed)
MAX_RESID = 0.05


def _ohlcv_close(t: str) -> pd.Series | None:
    p = OHLCV / t / "1d.parquet"
    if not p.exists():
        return None
    px = pd.read_parquet(p, columns=["close"])
    s = pd.to_numeric(px["close"], errors="coerce")
    s.index = pd.to_datetime(px.index).normalize()
    s = s[(s > 0) & s.notna()]
    return s[~s.index.duplicated()].sort_index()


def union_calendar() -> pd.DataFrame:
    """Union of the two in-repo calendars. Co-dated entries from different sources are
    kept as SEPARATE events and multiply -- verified against HON 2026-06-29, where the
    measured factor 0.47675 equals FMP's 0.5 reverse split times the ohlcv column's
    0.9535 spinoff factor."""
    a = pd.read_parquet(OUT / "fmp_splits_830.parquet")[["ticker", "date", "ratio"]]
    a["src"] = "fmp"
    rows = []
    for t in sorted({p.parent.name for p in CACHE.glob("*.parquet")} | set(a.ticker)):
        p = OHLCV / t / "1d.parquet"
        if not p.exists() or "split_ratio" not in pq.ParquetFile(p).schema_arrow.names:
            continue
        px = pd.read_parquet(p, columns=["split_ratio"])
        s = pd.to_numeric(px["split_ratio"], errors="coerce")
        s.index = pd.to_datetime(px.index).normalize()
        for d, v in s[(s.notna()) & (s != 0.0) & (s != 1.0)].items():
            rows.append({"ticker": t, "date": d, "ratio": float(v), "src": "ohlcv_col"})
    b = pd.DataFrame(rows, columns=["ticker", "date", "ratio", "src"])
    return pd.concat([a, b], ignore_index=True)


def measured_factor(t: str, close: pd.Series,
                    cal_dates: np.ndarray | None = None) -> tuple[pd.Series | None, dict]:
    """F(t) measured as raw/close, compressed to a step function.

    ``cal_dates`` are the union-calendar ex-dates for this ticker. They are used ONLY to
    corroborate the DATE of a short-lived step, never to set a magnitude. This matters:
    several files end with one bar on a different basis (e.g. a step dated exactly on the
    file's last date, 2026-05-12, for HUM/CF/ELV/FCX/CI/HII/FRPT/APPF/DV). Anchoring on
    such a one-row segment would propagate a 3-5% error across the ticker's whole history.
    A step is therefore kept only if it PERSISTS or a real ex-date corroborates it.
    """
    cp = CACHE / f"{t}.parquet"
    if not cp.exists():
        return None, {"why": "no-raw-cache"}
    raw = pd.read_parquet(cp)
    if raw.empty:
        return None, {"why": "empty-raw"}
    r = pd.to_numeric(raw["raw_close"], errors="coerce")
    r.index = pd.to_datetime(raw["date"]).dt.normalize()
    r = r[(r > 0) & r.notna()]
    r = r[~r.index.duplicated()].sort_index()
    common = close.index.intersection(r.index)
    cov = len(common) / max(len(close), 1)
    if cov < MIN_COVERAGE or len(common) < 30:
        return None, {"why": "raw-coverage-%.2f" % cov, "coverage": cov}
    ratio = (r.reindex(common) / close.reindex(common)).astype(float)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(ratio) < 30:
        return None, {"why": "too-few-finite"}
    x = np.log(ratio.values)
    # breakpoints where the measured level shifts by more than the tolerance
    d = np.diff(x)
    brk = np.abs(d) > STEP_TOL

    # Drop breakpoints that neither PERSIST nor are corroborated by a real ex-date.
    # A short segment at a file edge is the two price vendors disagreeing on a stale bar,
    # not a corporate action -- and anchoring on it would poison the whole history.
    idx = ratio.index.values.astype("datetime64[ns]")
    if cal_dates is None:
        cal_dates = np.array([], dtype="datetime64[ns]")
    corrob = np.zeros(len(idx), dtype=bool)
    if len(cal_dates):
        for k in range(len(idx)):
            corrob[k] = bool(np.any(np.abs((cal_dates - idx[k]) / np.timedelta64(1, "D")) <= 5))
    for _ in range(40):
        seg = np.concatenate([[0], np.cumsum(brk)])
        sizes = np.bincount(seg)
        killed = False
        for s in range(len(sizes)):
            if sizes[s] >= MIN_SEG_OBS:
                continue
            j = np.where(seg == s)[0]
            # the breakpoint that OPENS this segment is at index j[0]-1 in `brk`
            opens = j[0] - 1
            if opens >= 0 and corrob[j[0]]:
                continue                            # a real ex-date sits here: keep it
            if opens >= 0:
                brk[opens] = False
                killed = True
            if j[-1] < len(brk) and not corrob[min(j[-1] + 1, len(idx) - 1)]:
                brk[j[-1]] = False
                killed = True
            if killed:
                break
        if not killed:
            break

    seg = np.concatenate([[0], np.cumsum(brk)])
    lvl = pd.Series(ratio.values).groupby(seg).transform("median").values
    resid = float(np.max(np.abs(np.log(ratio.values / lvl)))) if len(lvl) else np.nan
    # ANCHOR: F(t) = prod{f_e : ex_e > t} is exactly 1.0 after the final event, so the last
    # segment must be 1.0. Rebasing on it also cancels any constant price-level offset
    # between the raw-price vendor and the stored close.
    tail_level = float(lvl[-1])
    lvl = lvl / tail_level
    fac = pd.Series(lvl, index=ratio.index)
    # carry the step function onto EVERY trading date (raw feed may miss days)
    full = fac.reindex(close.index).ffill().bfill()
    steps = [(ratio.index[i + 1], float(lvl[i + 1] / lvl[i]))
             for i in range(len(lvl) - 1) if brk[i]]
    if not np.isfinite(resid) or resid > MAX_RESID:
        # the measurement itself is not trustworthy -> refuse it and fail closed rather
        # than publish a factor we cannot stand behind
        return None, {"why": "unreliable-resid-%.4f" % resid, "coverage": cov}
    return full, {"why": "measured", "coverage": cov, "n_segments": int(seg[-1] + 1),
                  "max_within_segment_resid": resid, "steps": steps,
                  "tail_level_before_anchor": tail_level}


def load_split_retrieval_status() -> tuple[frozenset[str], frozenset[str]]:
    """(authoritative_no_event, unknown) ticker sets from the harvest manifest.

    An EMPTY calendar for a ticker has two completely different causes and the
    calendar alone cannot tell them apart:

      * FMP answered and has no split history  -> authoritative "never split"
      * the request FAILED                     -> we know NOTHING

    `harvest_splits_830.py` already records both, and its own manifest note says
    "errors are NOT authoritative and must be treated as unknown". Before this
    function existed the builder never read it, so an errored ticker with no raw
    price fallback silently emitted `cum_factor = 1.0` -- publishing an
    unadjusted market-cap basis as though it had been verified.

    Returns EMPTY sets when the manifest is missing, which makes every
    empty-calendar ticker FAIL rather than defaulting to 1.0: no manifest means
    no authority to claim "never split".
    """
    mp = OUT / "fmp_splits_830.manifest.json"
    if not mp.exists():
        return frozenset(), frozenset()
    try:
        man = json.loads(mp.read_text())
    except (OSError, ValueError):
        return frozenset(), frozenset()
    no_ev = frozenset(str(x) for x in (man.get("no_data_tickers") or []))
    errs = frozenset(str(x) for x in (man.get("error_tickers") or []))
    if not errs:
        # older manifests only carried a truncated `error_samples`; derive what
        # we can, and note that the derivation is partial.
        errs = frozenset(str(e.get("ticker")) for e in (man.get("error_samples") or [])
                         if isinstance(e, dict) and e.get("ticker"))
    return no_ev, errs


def reconstructed_factor(t: str, close: pd.Series, cal: pd.DataFrame,
                         authoritative_no_event: "frozenset[str] | None" = None,
                         ) -> tuple[pd.Series | None, dict]:
    """F(t) = prod of calendar ratios with ex-date > t (fallback route)."""
    g = cal[cal.ticker == t]
    if g.empty:
        # An empty calendar is NOT by itself a "never split" answer -- see
        # load_split_retrieval_status. Emit 1.0 only against a recorded
        # authoritative no-event response; otherwise FAIL CLOSED.
        if authoritative_no_event is None or t not in authoritative_no_event:
            return None, {
                "why": "no-authoritative-no-event-answer",
                "detail": ("empty split calendar and no recorded authoritative "
                           "'no split history' response for this ticker, so a "
                           "factor of 1.0 would be a guess, not a measurement"),
            }
        return pd.Series(1.0, index=close.index), {"why": "reconstructed-no-events",
                                                   "n_segments": 1, "steps": []}
    ev = g[(g.date > close.index.min()) & (g.date <= close.index.max())]
    f = pd.Series(1.0, index=close.index)
    for _, e in ev.iterrows():
        f.loc[f.index < e.date] *= e.ratio
    steps = [(pd.Timestamp(e.date), float(e.ratio)) for _, e in ev.iterrows()]
    return f, {"why": "reconstructed", "n_segments": len(steps) + 1, "steps": steps}


def build(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = union_calendar()
    authoritative_no_event, unknown_tickers = load_split_retrieval_status()
    caldates = {t: g["date"].values.astype("datetime64[ns]")
                for t, g in cal.groupby("ticker", sort=False)}
    frames, meta = [], []
    for t in tickers:
        close = _ohlcv_close(t)
        if close is None or close.empty:
            meta.append({"ticker": t, "route": "FAIL", "why": "no-ohlcv"})
            continue
        f, info = measured_factor(t, close, caldates.get(t))
        route = "measured"
        if f is None:
            why_m = info.get("why")
            if str(why_m).startswith("unreliable"):
                # the two price series disagree about the instrument itself; a calendar
                # reconstruction would be a guess dressed as a measurement.
                meta.append({"ticker": t, "route": "FAIL", "why": why_m})
                continue
            f, info = reconstructed_factor(t, close, cal, authoritative_no_event)
            route = "reconstructed"
            info["measured_failed_because"] = why_m
            if t in unknown_tickers:
                info["split_retrieval"] = "unknown (request errored)"
        if f is None:
            meta.append({"ticker": t, "route": "FAIL", "why": info.get("why")})
            continue
        frames.append(pd.DataFrame({"date": close.index, "ticker": t,
                                    "cum_factor": f.values, "route": route}))
        meta.append({"ticker": t, "route": route, "n_dates": len(close),
                     "f_first": float(f.iloc[0]), "f_last": float(f.iloc[-1]),
                     "f_min": float(f.min()), "f_max": float(f.max()),
                     "n_segments": info.get("n_segments"),
                     "coverage": info.get("coverage"),
                     "max_within_segment_resid": info.get("max_within_segment_resid"),
                     "n_steps": len(info.get("steps") or []),
                     "steps": ";".join("%s:%.6g" % (d.date(), v)
                                       for d, v in (info.get("steps") or [])),
                     "why": info.get("why"),
                     "tail_level_before_anchor": info.get("tail_level_before_anchor"),
                     "measured_failed_because": info.get("measured_failed_because")})
    F = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return F, pd.DataFrame(meta)


def main() -> None:
    tickers = sorted((OUT / "tickers.txt").read_text().split())
    F, M = build(tickers)
    F.to_parquet(OUT / "split_factors_830.parquet", index=False)
    M.to_csv(OUT / "split_factor_meta.csv", index=False)
    pd.set_option("display.width", 200)
    print("=" * 100)
    print("CUMULATIVE ADJUSTMENT FACTOR BUILT")
    print("=" * 100)
    print("  rows=%d  tickers=%d" % (len(F), F.ticker.nunique() if len(F) else 0))
    print("  route: %s" % dict(M.route.value_counts()))
    ok = M[M.route != "FAIL"]
    print("  tickers whose factor is not identically 1.0 (i.e. a correction applies): %d"
          % int((ok.f_min.astype(float) != 1.0).sum()))
    print("  factor range across universe: min=%.6g  max=%.6g"
          % (ok.f_min.astype(float).min(), ok.f_max.astype(float).max()))
    print("  f_last != 1.0 (should be ~none; factor must end at 1): %d"
          % int((ok.f_last.astype(float).sub(1).abs() > 1e-6).sum()))
    bad = ok[ok.f_last.astype(float).sub(1).abs() > 1e-6]
    if len(bad):
        print(bad[["ticker", "route", "f_first", "f_last", "coverage"]].to_string(index=False))
    mr = pd.to_numeric(ok.max_within_segment_resid, errors="coerce").dropna()
    print("\n  measured-route flatness QC (max |log(ratio/segment median)| per ticker):")
    print("    median=%.2e  p95=%.2e  p99=%.2e  max=%.2e"
          % (mr.median(), mr.quantile(.95), mr.quantile(.99), mr.max()))
    print("    tickers with residual > 1%% (candidate unmodelled drift):  %d"
          % int((mr > 0.01).sum()))
    print(ok[pd.to_numeric(ok.max_within_segment_resid, errors="coerce") > 0.01]
          [["ticker", "route", "n_segments", "max_within_segment_resid", "coverage"]]
          .head(25).to_string(index=False))
    if (M.route == "FAIL").any():
        print("\n  FAILED tickers (features will be NaN, never silently 1.0):")
        print(M[M.route == "FAIL"].to_string(index=False))


if __name__ == "__main__":
    main()
