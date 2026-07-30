"""G6 Stage-1b: rebuild the as-filed PIT panel with SPLIT-CORRECTED market-cap ratios.

This is the Stage-1 builder with exactly three changes; every PIT rule is preserved
verbatim (as-filed earliest-filing values, per-operand availability, per-operand backward
as-of, max-over-operands availability, +1 calendar day buffer, NO imputation -- warmup
stays NaN and is never zero-filled).

CHANGE 1 -- SPLIT BASIS.  As-filed dei:EntityCommonStockSharesOutstanding is not
  retroactively split-adjusted; data/ohlcv closes are. Each share FACT is multiplied by
  the cumulative adjustment factor evaluated at ITS OWN cover date (`end`), which makes
  shares and price share one basis and makes the basis cancel out of the ratio:
      market_cap(t) = S * F(cover_date) * close(t) = S_raw(t) * price_raw(t)
  Anchoring F at the cover date rather than at the trading date is essential: between an
  ex-date and the next filing the newest as-filed count is still pre-split, and using
  F(t) there would understate market cap by the entire split factor.

CHANGE 2 -- DENOMINATOR / OUTPUT SAFETY, matching renquant-base-data#55's intent.
  No epsilon anywhere. A denominator must be finite and non-zero or the result is NaN;
  additionally any |ratio| > MAX_ABS_RATIO is NaN, because that magnitude is the
  signature of a divide-by-near-zero rather than a real value. NaN, never a clip -- a
  clip is what let a 1.68e19 masquerade as a legitimate-looking z-scored +-3.0.
  The universe-appropriate positive floors from Stage-1 are retained (they also screen
  the known dei placeholder counts of exactly 100 / 1000 shares).

CHANGE 3 -- FAIL-CLOSED on an unresolved split basis. A ticker whose cumulative factor
  could not be established gets NaN for both market-cap ratios; it is never silently
  treated as factor 1.0. Same for as-filed share SPIKES (a value that is >1.5x both its
  neighbours in the same direction), which are dei units errors, not share counts.

Reads only:  data/edgar_pit/companyfacts_asfiled_raw.parquet, data/ohlcv/<T>/1d.parquet
Writes only: the scratch --out path. NEVER a production path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RQ = Path("/Users/renhao/git/github/RenQuant")
RAW = RQ / "data/edgar_pit/companyfacts_asfiled_raw.parquet"
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


SCRATCH = _work_dir()

Q_SPAN, ANNUAL_SPAN, TTM_SPAN = (80, 100), (350, 380), (330, 400)
AVAIL_BUFFER_DAYS = 1
MKTCAP_FLOOR, EQUITY_FLOOR, ASSETS_FLOOR, SHARES_FLOOR = 1e6, 1e5, 1e5, 1e3
MAX_ABS_RATIO = 1.0e6            # renquant-base-data#55
SPIKE = 1.5
GROSS = 20.0        # a share count this far from its local level is a units error

REVENUE_CHAIN = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
COST_CHAIN = ["CostOfRevenue", "CostOfGoodsAndServicesSold"]
CORRECTED = ["earnings_yield", "book_to_price"]
UNCHANGED = ["roe", "gross_profitability", "asset_growth"]
ALL_FEATS = CORRECTED + UNCHANGED

# ------------------------------------------------------------------ Stage-1 machinery
from build_pit_panel_shared import derive_q4, ttm, _dur, instant, load_asfiled, asof  # noqa: E402


def _safe_ratio(numerator, denominator, *, max_abs_ratio: float = MAX_ABS_RATIO) -> pd.Series:
    """``numerator / denominator``, NaN where the result is not a real ratio.

    Mirrors renquant-base-data#55 `_safe_ratio`: no epsilon; a zero or non-finite
    denominator yields NaN; a magnitude above ``max_abs_ratio`` is the signature of a
    divide-by-near-zero and is DISCARDED (NaN), not clipped. A legitimately negative
    denominator divides normally and keeps its sign.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    usable = np.isfinite(den) & (den != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num.where(usable) / den.where(usable)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.where(out.abs() <= float(max_abs_ratio))


def guard(s, floor, positive_only=False):
    """Stage-1 floor guard: below the floor is NOT COMPUTABLE -> NaN (never an epsilon)."""
    s = pd.to_numeric(s, errors="coerce")
    m = np.isfinite(s) & (s.abs() > floor)
    if positive_only:
        m = m & (s > 0)
    return s.where(m)


# ------------------------------------------------------------------ split basis
def load_factor_lookup():
    F = pd.read_parquet(SCRATCH / "split_factors_830.parquet")
    meta = pd.read_csv(SCRATCH / "split_factor_meta.csv")
    failed = set(meta.loc[meta.route == "FAIL", "ticker"])
    per = {}
    for t, g in F.groupby("ticker", sort=False):
        g = g.sort_values("date")
        per[t] = (g["date"].values.astype("datetime64[ns]"), g["cum_factor"].values)
    cal = pd.read_parquet(SCRATCH / "fmp_splits_830.parquet")[["ticker", "date", "ratio"]]
    calper = {t: g.sort_values("date") for t, g in cal.groupby("ticker", sort=False)}
    return per, calper, failed


def factor_at(t, dates, per, calper):
    """F evaluated at arbitrary (possibly non-trading, possibly pre-axis) reference dates.

    Pre-axis cover dates are extended with the calendar: any event between the cover date
    and the first priced date still has to be applied, or an early share count would be
    left on the wrong basis.
    """
    ax = per.get(t)
    if ax is None:
        return np.full(len(dates), np.nan)
    d = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[ns]")
    idx = np.searchsorted(ax[0], d, side="right") - 1
    out = np.where(idx >= 0, ax[1][np.clip(idx, 0, len(ax[1]) - 1)], np.nan)
    pre = idx < 0
    if pre.any():
        f0 = ax[1][0]
        t0 = ax[0][0]
        g = calper.get(t)
        ext = np.full(pre.sum(), f0)
        if g is not None and len(g):
            gd = g["date"].values.astype("datetime64[ns]")
            gr = g["ratio"].values
            for k, dd in enumerate(d[pre]):
                m = (gd > dd) & (gd <= t0)
                if m.any():
                    ext[k] = f0 * float(np.prod(gr[m]))
        out[pre] = ext
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_path = Path(args.out).expanduser().resolve()
    assert not str(out_path).startswith(str(RQ)), "refusing to write inside the production tree"

    per, calper, failed = load_factor_lookup()
    print("split basis: %d tickers with a factor, %d FAILED (features forced NaN)"
          % (len(per), len(failed)))

    raw = load_asfiled()
    ni_q, ni_a = _dur(raw, ["NetIncomeLoss"], Q_SPAN), _dur(raw, ["NetIncomeLoss"], ANNUAL_SPAN)
    gp_q, gp_a = _dur(raw, ["GrossProfit"], Q_SPAN), _dur(raw, ["GrossProfit"], ANNUAL_SPAN)
    rv_q, rv_a = _dur(raw, REVENUE_CHAIN, Q_SPAN), _dur(raw, REVENUE_CHAIN, ANNUAL_SPAN)
    cs_q, cs_a = _dur(raw, COST_CHAIN, Q_SPAN), _dur(raw, COST_CHAIN, ANNUAL_SPAN)
    t_ni = ttm(ni_q, ni_a, "ttm_ni")
    t_gp = ttm(gp_q, gp_a, "ttm_gp_d")
    t_rv = ttm(rv_q, rv_a, "ttm_rev")
    t_cs = ttm(cs_q, cs_a, "ttm_cost")
    fb = t_rv.merge(t_cs, on=["ticker", "end"], how="inner")
    fb["ttm_gp"] = fb.ttm_rev - fb.ttm_cost
    fb["ttm_gp_filed"] = fb[["ttm_rev_filed", "ttm_cost_filed"]].max(axis=1)
    gp_all = pd.concat([
        t_gp.rename(columns={"ttm_gp_d": "ttm_gp", "ttm_gp_d_filed": "ttm_gp_filed"}).assign(_p=0),
        fb[["ticker", "end", "ttm_gp", "ttm_gp_filed"]].assign(_p=1)], ignore_index=True)
    gp_all = (gp_all.sort_values(["ticker", "end", "_p"])
                    .drop_duplicates(["ticker", "end"], keep="first").drop(columns=["_p"]))

    assets = instant(raw, "Assets").rename(columns={"val": "assets", "filed": "assets_filed"})
    equity = instant(raw, "StockholdersEquity").rename(columns={"val": "equity", "filed": "equity_filed"})
    shares = instant(raw, "EntityCommonStockSharesOutstanding").rename(
        columns={"val": "shares", "filed": "shares_filed"})

    # ---- CHANGE 1: put every share fact on the back-adjusted-close basis
    shares = shares.sort_values(["ticker", "end"]).reset_index(drop=True)
    shares["fac"] = np.nan
    for t, g in shares.groupby("ticker", sort=False):
        shares.loc[g.index, "fac"] = factor_at(t, g["end"], per, calper)
    n_nofac = int(shares.fac.isna().sum())
    shares["shares_adj"] = shares["shares"] * shares["fac"]

    # ---- CHANGE 3b: drop dei UNITS ERRORS.
    # Two independent signatures, because one alone leaves residue:
    #  (i) an isolated spike: >1.5x BOTH neighbours in the same direction;
    # (ii) a gross level error: >GROSS x away from the local level. Filers that tag the
    #      cover-page count in thousands produce exact 1e3/1e6 offsets (AJG 1.9e14,
    #      LIN 2.5e4, GRMN 1.98e11), and those survive (i) whenever two land in a row or
    #      one sits at a series edge. No real corporate action changes a share count by
    #      20x -- the largest genuine step in this universe is KDP's 7.7x merger.
    g = shares.groupby("ticker", sort=False)["shares_adj"]
    prev, nxt = g.shift(1), g.shift(-1)
    up = (shares.shares_adj / prev > SPIKE) & (shares.shares_adj / nxt > SPIKE)
    dn = (shares.shares_adj / prev < 1 / SPIKE) & (shares.shares_adj / nxt < 1 / SPIKE)
    spike = (up | dn).fillna(False)
    loc = g.transform(lambda s: s.rolling(7, center=True, min_periods=3).median())
    gross = ((shares.shares_adj / loc > GROSS) | (shares.shares_adj / loc < 1 / GROSS)).fillna(False)
    print("  units-error detail: isolated spikes=%d, gross level errors=%d, union=%d"
          % (int(spike.sum()), int(gross.sum()), int((spike | gross).sum())))
    spike = spike | gross
    print("as-filed share facts: %d | no factor: %d | units SPIKES dropped: %d (%d tickers)"
          % (len(shares), n_nofac, int(spike.sum()), shares.loc[spike, "ticker"].nunique()))
    shares.loc[spike, "shares_adj"] = np.nan
    shares[["ticker", "end", "shares_filed", "shares", "fac", "shares_adj"]] \
        .assign(spike=spike.values).to_parquet(SCRATCH / "shares_adjusted.parquet", index=False)

    ag = assets.sort_values(["ticker", "end"]).copy()
    ag["assets_lag"] = ag.groupby("ticker").assets.shift(4)
    span = (ag.end - ag.groupby("ticker").end.shift(4)).dt.days
    ag["assets_lag"] = ag.assets_lag.where(span.between(*TTM_SPAN))
    ag = ag.dropna(subset=["assets_lag"])[["ticker", "end", "assets", "assets_lag", "assets_filed"]]

    frames = []
    tickers = sorted(set(raw.ticker))
    for i, t in enumerate(tickers):
        if i % 200 == 0:
            print("  %d/%d" % (i, len(tickers)))
        p = OHLCV / t / "1d.parquet"
        if not p.exists():
            continue
        px = pd.read_parquet(p)
        dcol = px["date"] if "date" in px.columns else px.index
        day = pd.DataFrame({"date": pd.to_datetime(dcol).normalize(),
                            "price": pd.to_numeric(px["close"], errors="coerce").values})
        day = day.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        if day.empty:
            continue
        day = asof(day, t_ni[t_ni.ticker == t], "ttm_ni", "ttm_ni_filed", "ni")
        day = asof(day, gp_all[gp_all.ticker == t], "ttm_gp", "ttm_gp_filed", "gp")
        day = asof(day, assets[assets.ticker == t], "assets", "assets_filed", "as")
        day = asof(day, equity[equity.ticker == t], "equity", "equity_filed", "eq")
        sht = shares[shares.ticker == t].dropna(subset=["shares_adj"])
        day = asof(day, sht, "shares_adj", "shares_filed", "sh")
        # the UNCORRECTED as-filed count, on the Stage-1 basis, carried so that the
        # before/after of the correction is auditable by eye and the unsplit-ticker
        # negative control can be checked numerically rather than argued.
        day = asof(day, shares[shares.ticker == t], "shares", "shares_filed", "shraw")
        gg = ag[ag.ticker == t]
        day = asof(day, gg, "assets", "assets_filed", "agn")
        day = asof(day, gg.rename(columns={"assets_lag": "aglag"}), "aglag", "assets_filed", "agl")

        sh_g = guard(day["sh"], SHARES_FLOOR, True)
        if t in failed:
            sh_g = pd.Series(np.nan, index=day.index)          # CHANGE 3: fail closed
        mc = guard(sh_g * day["price"], MKTCAP_FLOOR, True)

        sh_raw = guard(day["shraw"], SHARES_FLOOR, True)
        res = pd.DataFrame({"date": day["date"], "ticker": t, "price": day["price"],
                            "shares_asfiled": sh_raw, "shares_adj": sh_g,
                            "market_cap": mc,
                            "market_cap_stage1": guard(sh_raw * day["price"], MKTCAP_FLOOR, True)})
        res["earnings_yield"] = _safe_ratio(day["ni"], mc)
        # an as-filed equity of exactly 0 is not a book value: GLD (a gold trust) reports
        # StockholdersEquity=0 for its whole history and produced 2657 spurious 0.0 rows.
        # Same floor the panel already applies to roe, so |negative equity| survives.
        res["book_to_price"] = _safe_ratio(guard(day["eq"], EQUITY_FLOOR), mc)
        res["gross_profitability"] = _safe_ratio(day["gp"], guard(day["as"], ASSETS_FLOOR, True))
        res["roe"] = _safe_ratio(day["ni"], guard(day["eq"], EQUITY_FLOOR, True))
        res["asset_growth"] = (_safe_ratio(day["agn"], guard(day["agl"], ASSETS_FLOOR, True))
                               - 1.0).clip(-0.99, 5.0)

        avail_ops = {"earnings_yield": ["ni", "sh"], "book_to_price": ["eq", "sh"],
                     "gross_profitability": ["gp", "as"], "roe": ["ni", "eq"],
                     "asset_growth": ["agn"]}
        fpe_ops = {"earnings_yield": ["ni"], "book_to_price": ["eq"],
                   "gross_profitability": ["gp", "as"], "roe": ["ni", "eq"],
                   "asset_growth": ["agn"]}
        for f, ops in avail_ops.items():
            res[f"{f}_available_at"] = day[[f"{o}__av" for o in ops]].max(axis=1)
            res[f"{f}_fiscal_period_end"] = day[[f"{o}__fpe" for o in fpe_ops[f]]].max(axis=1)
        frames.append(res)

    panel = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for f in ALL_FEATS:
        m = panel[f].notna()
        panel.loc[~m, [f"{f}_available_at", f"{f}_fiscal_period_end"]] = pd.NaT
        panel[f"{f}_age_days"] = (panel["date"] - panel[f"{f}_available_at"]).dt.days

    cols = (["date", "ticker", "price", "shares_asfiled", "shares_adj",
             "market_cap", "market_cap_stage1"] + ALL_FEATS +
            [c for f in ALL_FEATS for c in (f"{f}_available_at", f"{f}_fiscal_period_end",
                                            f"{f}_age_days")])
    panel = panel[cols].sort_values(["ticker", "date"]).reset_index(drop=True)

    print("\n=== PIT VALIDATION (fail-closed) ===")
    bad = 0
    for f in ALL_FEATS:
        m = panel[f].notna()
        la = int((m & (panel[f"{f}_available_at"] > panel["date"])).sum())
        imp = int((m & (panel[f"{f}_available_at"] < panel[f"{f}_fiscal_period_end"])).sum())
        npv = int((m & panel[f"{f}_available_at"].isna()).sum())
        bad += la + imp + npv
        print("  %-22s nonnull=%8d  look_ahead=%d  avail<fiscal_end=%d  no_provenance=%d"
              % (f, int(m.sum()), la, imp, npv))
    if bad:
        raise SystemExit("PIT VALIDATION FAILED (%d violations) - refusing to write" % bad)
    print("  ALL CLEAR: 0 violations")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print("\nwrote %s  rows=%d tickers=%d dates=%s..%s"
          % (out_path, len(panel), panel.ticker.nunique(),
             panel.date.min().date(), panel.date.max().date()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
