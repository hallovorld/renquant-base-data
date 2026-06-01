"""Build the alpha158 + fundamentals production training panel.

This module is the subrepo-owned lift of RenQuant's
``scripts/build_alpha158_fund_panel.py``. It keeps the same data contract while
removing the machine-specific umbrella repo root dependency: callers pass a
``data_dir`` containing the alpha158, SEC fundamentals, earnings surprise, and
optional sentiment parquet caches.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd


log = logging.getLogger("renquant_base_data.alpha158_fund_panel")

FUND_COLS = [
    "earnings_yield",
    "book_to_price",
    "gross_profitability",
    "roe",
    "asset_growth",
]
PEAD_COLS = ["days_since_earnings", "pead_signal", "pead_quintile_rank"]
SUE_COLS = ["sue_signal", "surprise_momentum", "surprise_streak"]
SENT_COLS = ["sentiment_pos_share", "mean_sentiment", "n_articles_log"]

PEAD_DECAY_DAYS = 60
SUE_WINDOW = 4

DEFAULT_ALPHA_FILENAME = "alpha158_qlib_dataset.parquet"
DEFAULT_FUND_FILENAME = "sec_fundamentals_daily.parquet"
DEFAULT_OUTPUT_FILENAME = "alpha158_291_fundamental_dataset.parquet"


def build_alpha158_fund_panel(
    data_dir: str | Path,
    *,
    truncate_to_sec_max: bool = False,
    output_path: str | Path | None = None,
) -> Path:
    """Merge alpha158, fundamental, PEAD, SUE, and sentiment features.

    Parameters
    ----------
    data_dir:
        Directory containing ``alpha158_qlib_dataset.parquet`` and
        ``sec_fundamentals_daily.parquet`` plus optional feature cache folders.
    truncate_to_sec_max:
        If alpha158 extends past SEC fundamentals max date, drop those trailing
        alpha rows instead of failing. Production retrain uses this because the
        training cutoffs are historic and recent SEC lag is label-irrelevant.
    output_path:
        Optional output parquet path. Defaults to
        ``data_dir/alpha158_291_fundamental_dataset.parquet``.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    alpha_path = data_dir / DEFAULT_ALPHA_FILENAME
    fund_path = data_dir / DEFAULT_FUND_FILENAME
    out_path = Path(output_path).expanduser().resolve() if output_path else data_dir / DEFAULT_OUTPUT_FILENAME

    log.info("Loading alpha158 panel: %s", alpha_path)
    alpha = pd.read_parquet(alpha_path)
    alpha["date"] = pd.to_datetime(alpha["date"])
    log.info(
        "alpha rows=%d tickers=%d cols=%d dates=%s..%s",
        len(alpha),
        alpha["ticker"].nunique(),
        len(alpha.columns),
        alpha["date"].min().date(),
        alpha["date"].max().date(),
    )

    log.info("Loading fundamentals panel: %s", fund_path)
    fund = pd.read_parquet(fund_path)
    fund["date"] = pd.to_datetime(fund["date"])
    keep = ["ticker", "date"] + [col for col in FUND_COLS if col in fund.columns]
    fund = fund[keep]

    alpha = _enforce_sec_coverage(alpha, fund, truncate_to_sec_max=truncate_to_sec_max)
    merged = _merge_fundamentals(alpha, fund)

    t0 = time.time()
    merged = _add_pead_features(merged, data_dir=data_dir)
    log.info("PEAD features added in %.1fs", time.time() - t0)

    t0 = time.time()
    merged = _add_sue_features(merged, data_dir=data_dir)
    log.info("SUE features added in %.1fs", time.time() - t0)

    t0 = time.time()
    merged = _add_sentiment_features(merged, data_dir=data_dir)
    log.info("sentiment features added in %.1fs", time.time() - t0)

    expected_cols = len(alpha.columns) + len(FUND_COLS) + len(PEAD_COLS) + len(SUE_COLS) + len(SENT_COLS)
    if len(merged.columns) != expected_cols:
        extra = set(merged.columns) - set(alpha.columns) - set(FUND_COLS) - set(PEAD_COLS) - set(SUE_COLS) - set(SENT_COLS)
        log.warning("column count %d expected %d; extra=%s", len(merged.columns), expected_cols, sorted(extra))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    log.info(
        "wrote %s rows=%d cols=%d tickers=%d size=%.1fMB",
        out_path,
        len(merged),
        len(merged.columns),
        merged["ticker"].nunique(),
        out_path.stat().st_size / 1e6,
    )
    return out_path


def _enforce_sec_coverage(
    alpha: pd.DataFrame,
    fund: pd.DataFrame,
    *,
    truncate_to_sec_max: bool,
) -> pd.DataFrame:
    panel_max = alpha["date"].max()
    sec_max = fund["date"].max()
    if panel_max <= sec_max:
        return alpha
    if truncate_to_sec_max:
        n_drop = int((alpha["date"] > sec_max).sum())
        log.warning(
            "SEC coverage guard bypassed by truncate: dropped %d alpha rows after %s",
            n_drop,
            sec_max.date(),
        )
        return alpha[alpha["date"] <= sec_max].reset_index(drop=True)
    raise RuntimeError(
        f"SEC coverage guard: alpha158 panel max date {panel_max.date()} > "
        f"sec_fundamentals_daily max {sec_max.date()}. Refresh fundamentals or pass "
        "--truncate-to-sec-max for historic-data rebuilds."
    )


def _merge_fundamentals(alpha: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    merged = alpha.merge(fund, on=["ticker", "date"], how="left")
    log.info("fund merge rows=%d elapsed=%.1fs", len(merged), time.time() - t0)
    if len(merged) != len(alpha):
        raise RuntimeError(
            f"fund merge changed row count: {len(alpha)} -> {len(merged)}; "
            "check duplicate (ticker,date) pairs in fundamentals"
        )

    for col in FUND_COLS:
        if col not in merged.columns:
            log.warning("fundamental column missing: %s; filling with 0", col)
            merged[col] = 0.0
            continue
        med = merged.groupby("date")[col].transform("median")
        merged[col] = merged[col].fillna(med).fillna(0.0)
    return merged


def _load_earnings(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    earn = pd.read_parquet(path).reset_index()
    earn = earn.rename(columns={earn.columns[0]: "earnings_date"})
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
    return earn.sort_values("earnings_date").reset_index(drop=True)


def _add_pead_features(panel: pd.DataFrame, *, data_dir: Path) -> pd.DataFrame:
    earn_dir = data_dir / "earnings_surprise"
    n_with_earn = 0
    out_blocks: list[pd.DataFrame] = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True).copy()
        earn = _load_earnings(earn_dir / f"{ticker}.parquet")
        if earn is None or earn.empty:
            for col in PEAD_COLS + ["pead_surprise"]:
                group[col] = np.nan
            out_blocks.append(group)
            continue

        n_with_earn += 1
        group_dates = group["date"].values
        earn_dates = earn["earnings_date"].values
        surprises = earn["surprise_pct"].astype(float).values
        idxs = np.searchsorted(earn_dates, group_dates, side="right") - 1

        days_since = np.full(len(group), np.nan)
        surprise = np.full(len(group), np.nan)
        valid = idxs >= 0
        diff = (group_dates[valid] - earn_dates[idxs[valid]]).astype("timedelta64[D]").astype(int)
        days_since[valid] = diff
        surprise[valid] = surprises[idxs[valid]]

        days_since = np.where(days_since > PEAD_DECAY_DAYS, np.nan, days_since)
        surprise = np.where(np.isnan(days_since), np.nan, surprise)
        decay = np.where(np.isnan(days_since), 0.0, np.maximum(0.0, 1.0 - days_since / PEAD_DECAY_DAYS))

        group["days_since_earnings"] = days_since
        group["pead_signal"] = surprise * decay
        group["pead_surprise"] = surprise
        out_blocks.append(group)

    log.info("PEAD coverage: %d/%d tickers", n_with_earn, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)
    out["pead_quintile_rank"] = out.groupby("date")["pead_surprise"].rank(pct=True, na_option="keep")
    for col in PEAD_COLS + ["pead_surprise"]:
        med = out.groupby("date")[col].transform("median")
        out[col] = out[col].fillna(med).fillna(0.0)
    return out.drop(columns=["pead_surprise"])


def _add_sue_features(panel: pd.DataFrame, *, data_dir: Path) -> pd.DataFrame:
    earn_dir = data_dir / "earnings_surprise"
    n_with_data = 0
    out_blocks: list[pd.DataFrame] = []
    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True).copy()
        earn = _load_earnings(earn_dir / f"{ticker}.parquet")
        if earn is None or earn.empty:
            for col in SUE_COLS:
                group[col] = np.nan
            out_blocks.append(group)
            continue

        n_with_data += 1
        s = earn["surprise_pct"].astype(float)
        rolling_std = s.shift(1).rolling(SUE_WINDOW, min_periods=2).std()
        sue_per_event = (s / (rolling_std + 1e-6)).clip(-5, 5)
        momentum_per_event = s.diff()
        sign = np.sign(s).fillna(0).astype(int)
        streak = np.zeros(len(s), dtype=int)
        for idx in range(len(s)):
            if idx == 0 or sign.iloc[idx] == 0 or sign.iloc[idx] != sign.iloc[idx - 1]:
                streak[idx] = sign.iloc[idx]
            else:
                streak[idx] = streak[idx - 1] + sign.iloc[idx]

        group_dates = group["date"].values
        earn_dates = earn["earnings_date"].values
        idxs = np.searchsorted(earn_dates, group_dates, side="right") - 1
        days_since = np.full(len(group), np.nan)
        sue = np.full(len(group), np.nan)
        momentum = np.full(len(group), np.nan)
        streak_daily = np.full(len(group), np.nan)
        valid = idxs >= 0
        diff = (group_dates[valid] - earn_dates[idxs[valid]]).astype("timedelta64[D]").astype(int)
        days_since[valid] = diff
        sue[valid] = sue_per_event.iloc[idxs[valid]].values
        momentum[valid] = momentum_per_event.iloc[idxs[valid]].values
        streak_daily[valid] = streak[idxs[valid]]

        out_of_window = (days_since > PEAD_DECAY_DAYS) | np.isnan(days_since)
        decay = np.where(out_of_window, 0.0, np.maximum(0.0, 1.0 - days_since / PEAD_DECAY_DAYS))
        group["sue_signal"] = np.where(out_of_window, 0.0, sue * decay)
        group["surprise_momentum"] = np.where(out_of_window, 0.0, momentum * decay)
        group["surprise_streak"] = np.where(out_of_window, 0.0, streak_daily * decay)
        out_blocks.append(group)

    log.info("SUE coverage: %d/%d tickers", n_with_data, panel["ticker"].nunique())
    out = pd.concat(out_blocks, ignore_index=True)
    for col in SUE_COLS:
        med = out.groupby("date")[col].transform("median")
        out[col] = out[col].fillna(med).fillna(0.0)
    return out


def _add_sentiment_features(panel: pd.DataFrame, *, data_dir: Path) -> pd.DataFrame:
    sent_dir = data_dir / "news_sentiment_alpaca"
    files = sorted(sent_dir.glob("*.parquet")) if sent_dir.exists() else []
    if not files:
        for col in SENT_COLS:
            panel[col] = 0.0
        return panel

    parts: list[pd.DataFrame] = []
    dropped_pre2020 = 0
    for path in files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.rename(columns={"symbol": "ticker"})
        df["date"] = pd.to_datetime(df["date"])
        pre = int((df["date"] < pd.Timestamp("2020-01-01")).sum())
        if pre:
            dropped_pre2020 += pre
            df = df[df["date"] >= pd.Timestamp("2020-01-01")]
        if df.empty:
            continue
        df["n_articles_log"] = np.log1p(df["n_articles"].astype(float))
        parts.append(df[["ticker", "date"] + SENT_COLS])

    if dropped_pre2020:
        log.info("dropped %d pre-2020 sentiment rows", dropped_pre2020)
    if not parts:
        for col in SENT_COLS:
            panel[col] = 0.0
        return panel

    sent = pd.concat(parts, ignore_index=True)
    merged = panel.merge(sent, on=["ticker", "date"], how="left")
    if len(merged) != len(panel):
        raise RuntimeError(
            f"sentiment merge changed row count: {len(panel)} -> {len(merged)}; "
            "check duplicate (ticker,date) pairs in sentiment"
        )
    for col in SENT_COLS:
        med = merged.groupby("date")[col].transform("median")
        merged[col] = merged[col].fillna(med).fillna(0.0)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--truncate-to-sec-max", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    build_alpha158_fund_panel(
        args.data_dir,
        truncate_to_sec_max=args.truncate_to_sec_max,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
