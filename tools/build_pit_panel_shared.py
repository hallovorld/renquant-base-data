"""Stage-1 PIT machinery, lifted VERBATIM from g6-stage1/build_pit_panel.py.

Kept byte-identical in behaviour so that the split correction is the ONLY difference
between the two panels -- that is what makes the unsplit-ticker negative control and the
three-unchanged-features check meaningful.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RQ = Path("/Users/renhao/git/github/RenQuant")
RAW = RQ / "data/edgar_pit/companyfacts_asfiled_raw.parquet"

TTM_SPAN = (330, 400)
AVAIL_BUFFER_DAYS = 1


def load_asfiled() -> pd.DataFrame:
    raw = pd.read_parquet(RAW)
    for c in ("filed", "end", "start"):
        raw[c] = pd.to_datetime(raw[c], errors="coerce")
    raw["val"] = pd.to_numeric(raw["val"], errors="coerce")
    raw = raw.dropna(subset=["ticker", "concept", "end", "val", "filed"])
    raw = raw[raw["end"] <= raw["filed"]]
    raw = raw.sort_values("filed").drop_duplicates(["ticker", "concept", "start", "end"], keep="first")
    raw["span"] = (raw["end"] - raw["start"]).dt.days
    return raw


def _dur(raw, concepts, span):
    sub = raw[raw.concept.isin(concepts) & raw.span.between(*span)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ticker", "end", "val", "filed"])
    sub["prio"] = sub.concept.map({c: i for i, c in enumerate(concepts)})
    return (sub.sort_values(["ticker", "end", "prio", "filed"])
               .drop_duplicates(["ticker", "end"], keep="first")[["ticker", "end", "val", "filed"]]
               .reset_index(drop=True))


def instant(raw, concept):
    sub = raw[(raw.concept == concept) & raw.start.isna()]
    if sub.empty:
        sub = raw[raw.concept == concept]
    return (sub.sort_values(["ticker", "end", "filed"])
               .drop_duplicates(["ticker", "end"], keep="first")[["ticker", "end", "val", "filed"]]
               .reset_index(drop=True))


def derive_q4(q: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    out = []
    qi = {t: g.sort_values("end") for t, g in q.groupby("ticker", sort=False)}
    for t, ga in ann.groupby("ticker", sort=False):
        gq = qi.get(t)
        if gq is None:
            continue
        for _, a in ga.iterrows():
            lo, hi = a["end"] - pd.Timedelta(days=366), a["end"] - pd.Timedelta(days=20)
            inside = gq[(gq.end > lo) & (gq.end <= hi)]
            if len(inside) != 3:
                continue
            if ((gq.end == a["end"]).any()):
                continue
            out.append({"ticker": t, "end": a["end"], "val": a["val"] - inside.val.sum(),
                        "filed": max(a["filed"], inside.filed.max())})
    return pd.DataFrame(out) if out else pd.DataFrame(columns=["ticker", "end", "val", "filed"])


def ttm(q: pd.DataFrame, ann: pd.DataFrame, name: str) -> pd.DataFrame:
    q4 = derive_q4(q, ann)
    qq = pd.concat([q, q4], ignore_index=True).sort_values(["ticker", "end", "filed"])
    qq = qq.drop_duplicates(["ticker", "end"], keep="first")
    out = []
    for t, g in qq.groupby("ticker", sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        if len(g) >= 4:
            val = g.val.rolling(4).sum()
            av = pd.to_datetime(g.filed.astype("int64").rolling(4).max(), unit="ns")
            av = av.where(g.filed.notna().rolling(4).sum() == 4)
            span = (g.end - g.end.shift(3)).dt.days
            ok = val.notna() & av.notna() & span.between(*TTM_SPAN)
            if ok.any():
                out.append(pd.DataFrame({"ticker": t, "end": g.end[ok],
                                         "val": val[ok], "filed": av[ok]}))
    roll = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["ticker", "end", "val", "filed"])
    both = pd.concat([ann.assign(_p=0), roll.assign(_p=1)], ignore_index=True)
    both = (both.sort_values(["ticker", "end", "_p"])
                .drop_duplicates(["ticker", "end"], keep="first")
                .drop(columns=["_p"]).reset_index(drop=True))
    return both.rename(columns={"val": name, "filed": f"{name}_filed"})


def asof(daily: pd.DataFrame, series: pd.DataFrame, val_col: str, filed_col: str, tag: str):
    """Place one raw quantity on the daily grid by its OWN first-publication date."""
    s = series.dropna(subset=[filed_col, val_col]).copy()
    if s.empty:
        daily[tag] = np.nan
        daily[f"{tag}__av"] = pd.NaT
        daily[f"{tag}__fpe"] = pd.NaT
        return daily
    s["date"] = pd.to_datetime(s[filed_col]) + pd.Timedelta(days=AVAIL_BUFFER_DAYS)
    s["__fpe"] = s["end"]
    s = (s[["date", val_col, "__fpe"]].sort_values("date")
           .drop_duplicates("date", keep="last"))
    s["__av"] = s["date"]
    j = pd.merge_asof(daily[["date"]], s, on="date", direction="backward")
    daily[tag] = j[val_col].values
    daily[f"{tag}__av"] = j["__av"].values
    daily[f"{tag}__fpe"] = j["__fpe"].values
    return daily
