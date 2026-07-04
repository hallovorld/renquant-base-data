"""Point-in-time revision-drift feature builder for the 106 signal pipeline.

Reads dated snapshot directories produced by ``fmp_estimate_revisions.py`` and
computes four trailing-revision features per ticker per snapshot date:

  * ``eps_revision_3m``      -- % change in consensus EPS estimate over ~3 months
  * ``revenue_revision_3m``  -- % change in consensus revenue estimate over ~3 months
  * ``target_revision_3m``   -- % change in analyst target price over ~3 months
  * ``revision_breadth``     -- (frac analysts UP) - (frac analysts DOWN)

PIT PROVENANCE: the ``available_at`` column is set to the **snapshot directory
date** (the date the data was actually fetched), NOT today, NOT the estimate
period. This is the only honest as-of for downstream joins -- any row with
``available_at = D`` was computed from snapshots that existed on date D, so
using it keyed on a feature date >= D introduces zero look-ahead.

Usage::

    python -m renquant_base_data.pit_feature_builder \\
        --snapshots data/estimate_snapshots \\
        --out data/pit_features.parquet

    # incremental: only compute dates not already in the output
    python -m renquant_base_data.pit_feature_builder \\
        --snapshots data/estimate_snapshots \\
        --out data/pit_features.parquet \\
        --incremental
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

log = logging.getLogger("renquant_base_data.pit_feature_builder")

# --- snapshot reading ---------------------------------------------------------

# The FMP ``analyst-estimates`` endpoint returns per-fiscal-period rows with
# at least: symbol, date (fiscal period end), estimatedEpsAvg,
# estimatedRevenueAvg, numberAnalystEstimatedEps, numberAnalystsEstimatedRevenue.
# We take the FIRST row (nearest fiscal period end) as the current consensus.
#
# The ``price-target-consensus`` endpoint returns a single row per symbol with:
# targetConsensus, targetHigh, targetLow, targetMedian.
#
# The ``price-target-summary`` endpoint returns a single row per symbol with:
# lastMonthAvgPriceTarget, lastQuarterAvgPriceTarget.
#
# These are the column names from the FMP stable API (verified against the
# harvest manifests and the live collector test fixtures).

_EPS_COL = "estimatedEpsAvg"
_REV_COL = "estimatedRevenueAvg"
_TARGET_COL = "targetConsensus"
# Breadth: numberAnalystEstimatedEps is the total analyst count; we approximate
# UP/DOWN from the delta in the consensus (if the consensus increased, we treat
# all analysts as net-UP, etc.) -- a true UP/DOWN decomposition requires the
# individual-analyst-level estimate history which FMP free/Starter does not
# provide. For a richer breadth measure we can later splice in the
# ``grades_consensus`` buy/hold/sell counts.
_GRADES_BUY_COL = "buy"
_GRADES_SELL_COL = "sell"
_GRADES_HOLD_COL = "hold"
_GRADES_SB_COL = "strongBuy"
_GRADES_SS_COL = "strongSell"


def _read_snapshot_parquet(snapshot_dir: Path, filename: str) -> pd.DataFrame | None:
    """Read a single parquet file from a snapshot date dir; None if missing."""
    path = snapshot_dir / filename
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        log.warning("failed to read %s", path)
        return None


def _available_snapshot_dates(snapshots_root: Path) -> list[str]:
    """Return sorted YYYY-MM-DD date strings for published snapshot dirs."""
    dates = []
    if not snapshots_root.is_dir():
        return dates
    for d in sorted(snapshots_root.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            # validate it looks like a date
            try:
                datetime.strptime(d.name, "%Y-%m-%d")
                dates.append(d.name)
            except ValueError:
                continue
    return dates


def _find_lookback_date(
    available_dates: list[str], current_date: str, lookback_days: int
) -> str | None:
    """Find the snapshot date closest to ``current_date - lookback_days``.

    Returns the available date whose distance to the ideal target is smallest,
    with a tolerance of +/- 30 days. Returns None if nothing is close enough.
    """
    current = datetime.strptime(current_date, "%Y-%m-%d").date()
    target = current - timedelta(days=lookback_days)
    best: str | None = None
    best_dist = float("inf")
    for d_str in available_dates:
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        if d >= current:
            continue
        dist = abs((d - target).days)
        if dist < best_dist:
            best_dist = dist
            best = d_str
    # Only accept if within a reasonable tolerance
    if best is not None and best_dist <= 30:
        return best
    return None


def _extract_eps_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """From ``analyst_estimates.parquet``, extract per-ticker consensus EPS/rev.

    Takes the first row per symbol (nearest fiscal period) as the current
    consensus estimate.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", _EPS_COL, _REV_COL])
    # Ensure we have the needed columns (graceful on missing)
    cols_present = set(df.columns)
    eps_col = _EPS_COL if _EPS_COL in cols_present else None
    rev_col = _REV_COL if _REV_COL in cols_present else None
    if eps_col is None and rev_col is None:
        # Try fallback column names
        if "epsAvg" in cols_present:
            eps_col = "epsAvg"
        if "revenueAvg" in cols_present:
            rev_col = "revenueAvg"
    if "symbol" not in cols_present:
        return pd.DataFrame(columns=["symbol", _EPS_COL, _REV_COL])

    # First row per symbol (assumes ordered by fiscal period, nearest first)
    first = df.groupby("symbol", sort=False).first().reset_index()
    out_cols = ["symbol"]
    renames = {}
    if eps_col and eps_col in first.columns:
        out_cols.append(eps_col)
        if eps_col != _EPS_COL:
            renames[eps_col] = _EPS_COL
    if rev_col and rev_col in first.columns:
        out_cols.append(rev_col)
        if rev_col != _REV_COL:
            renames[rev_col] = _REV_COL
    result = first[out_cols].copy()
    if renames:
        result = result.rename(columns=renames)
    # Ensure both columns exist even if one wasn't found
    for c in [_EPS_COL, _REV_COL]:
        if c not in result.columns:
            result[c] = float("nan")
    return result[["symbol", _EPS_COL, _REV_COL]]


def _extract_target(df: pd.DataFrame) -> pd.DataFrame:
    """From ``price_target_consensus.parquet``, extract per-ticker target."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", _TARGET_COL])
    if "symbol" not in df.columns or _TARGET_COL not in df.columns:
        return pd.DataFrame(columns=["symbol", _TARGET_COL])
    first = df.groupby("symbol", sort=False).first().reset_index()
    return first[["symbol", _TARGET_COL]]


def _extract_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """From ``grades_consensus.parquet``, compute net-UP fraction.

    ``revision_breadth`` = (strongBuy + buy) / total - (strongSell + sell) / total
    where total = strongBuy + buy + hold + sell + strongSell.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "revision_breadth"])
    required = {"symbol", _GRADES_BUY_COL, _GRADES_SELL_COL, _GRADES_HOLD_COL}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["symbol", "revision_breadth"])
    first = df.groupby("symbol", sort=False).first().reset_index()
    buy = first.get(_GRADES_SB_COL, 0)
    if isinstance(buy, int):
        buy = pd.Series(0, index=first.index)
    buy = buy.fillna(0) + first[_GRADES_BUY_COL].fillna(0)

    sell = first.get(_GRADES_SS_COL, 0)
    if isinstance(sell, int):
        sell = pd.Series(0, index=first.index)
    sell = sell.fillna(0) + first[_GRADES_SELL_COL].fillna(0)

    hold = first[_GRADES_HOLD_COL].fillna(0)
    total = buy + hold + sell
    breadth = ((buy - sell) / total).where(total > 0, other=float("nan"))
    return pd.DataFrame({"symbol": first["symbol"], "revision_breadth": breadth})


# --- feature computation -----------------------------------------------------


def _pct_change(new_val: pd.Series, old_val: pd.Series) -> pd.Series:
    """Percentage change, guarded against zero/NaN denominators."""
    result = (new_val - old_val) / old_val.abs()
    # Replace inf with NaN (division by zero)
    result = result.replace([float("inf"), float("-inf")], float("nan"))
    return result


def build_revision_drift_one_date(
    snapshots_root: Path,
    current_date: str,
    lookback_days: int = 90,
    available_dates: list[str] | None = None,
) -> pd.DataFrame | None:
    """Compute revision-drift features for a single snapshot date.

    Returns a DataFrame with columns:
        ticker, available_at, eps_revision_3m, revenue_revision_3m,
        target_revision_3m, revision_breadth

    Returns None if no lookback snapshot is available (too few snapshots yet).
    """
    if available_dates is None:
        available_dates = _available_snapshot_dates(snapshots_root)

    old_date = _find_lookback_date(available_dates, current_date, lookback_days)
    if old_date is None:
        log.info(
            "no lookback snapshot for %s (need ~%dd ago); skipping",
            current_date,
            lookback_days,
        )
        return None

    cur_dir = snapshots_root / current_date
    old_dir = snapshots_root / old_date

    # --- read snapshots ---
    cur_estimates = _read_snapshot_parquet(cur_dir, "analyst_estimates.parquet")
    old_estimates = _read_snapshot_parquet(old_dir, "analyst_estimates.parquet")
    cur_targets = _read_snapshot_parquet(cur_dir, "price_target_consensus.parquet")
    old_targets = _read_snapshot_parquet(old_dir, "price_target_consensus.parquet")
    cur_grades = _read_snapshot_parquet(cur_dir, "grades_consensus.parquet")

    # --- extract per-ticker values ---
    cur_er = _extract_eps_revenue(cur_estimates)
    old_er = _extract_eps_revenue(old_estimates)
    cur_tgt = _extract_target(cur_targets)
    old_tgt = _extract_target(old_targets)
    breadth = _extract_breadth(cur_grades)

    # --- merge current + old ---
    # EPS / revenue revision
    merged_er = cur_er.merge(
        old_er, on="symbol", how="left", suffixes=("_cur", "_old")
    )
    merged_er["eps_revision_3m"] = _pct_change(
        merged_er[f"{_EPS_COL}_cur"], merged_er[f"{_EPS_COL}_old"]
    )
    merged_er["revenue_revision_3m"] = _pct_change(
        merged_er[f"{_REV_COL}_cur"], merged_er[f"{_REV_COL}_old"]
    )

    # Target revision
    merged_tgt = cur_tgt.merge(
        old_tgt, on="symbol", how="left", suffixes=("_cur", "_old")
    )
    merged_tgt["target_revision_3m"] = _pct_change(
        merged_tgt[f"{_TARGET_COL}_cur"], merged_tgt[f"{_TARGET_COL}_old"]
    )

    # --- assemble ---
    result = merged_er[["symbol", "eps_revision_3m", "revenue_revision_3m"]].copy()
    result = result.merge(
        merged_tgt[["symbol", "target_revision_3m"]], on="symbol", how="left"
    )
    result = result.merge(breadth, on="symbol", how="left")

    # PIT-correct: available_at = the snapshot date, not today
    result["available_at"] = current_date
    result = result.rename(columns={"symbol": "ticker"})
    return result[
        [
            "ticker",
            "available_at",
            "eps_revision_3m",
            "revenue_revision_3m",
            "target_revision_3m",
            "revision_breadth",
        ]
    ]


def build_revision_drift(
    snapshots_root: Path, lookback_days: int = 90
) -> pd.DataFrame:
    """Build revision-drift features for ALL available snapshot dates.

    Reads all snapshot date directories under ``snapshots_root``, and for each
    date that has a matching lookback snapshot, computes the 4 features.

    Returns a DataFrame with columns:
        ticker, available_at, eps_revision_3m, revenue_revision_3m,
        target_revision_3m, revision_breadth
    """
    available_dates = _available_snapshot_dates(snapshots_root)
    if not available_dates:
        return pd.DataFrame(
            columns=[
                "ticker",
                "available_at",
                "eps_revision_3m",
                "revenue_revision_3m",
                "target_revision_3m",
                "revision_breadth",
            ]
        )

    frames = []
    for d in available_dates:
        one = build_revision_drift_one_date(
            snapshots_root, d, lookback_days, available_dates
        )
        if one is not None:
            frames.append(one)

    if not frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "available_at",
                "eps_revision_3m",
                "revenue_revision_3m",
                "target_revision_3m",
                "revision_breadth",
            ]
        )
    return pd.concat(frames, ignore_index=True)


# --- incremental update ------------------------------------------------------


def incremental_update(
    snapshots_root: Path,
    existing: pd.DataFrame | None = None,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Compute revision features for snapshot dates not already in ``existing``.

    If ``existing`` is None or empty, computes for all dates (equivalent to
    ``build_revision_drift``). Otherwise, only processes dates whose
    ``available_at`` is not already present.

    Returns the union of ``existing`` and the newly computed rows.
    """
    available_dates = _available_snapshot_dates(snapshots_root)

    if existing is not None and not existing.empty:
        done_dates = set(existing["available_at"].unique())
    else:
        done_dates = set()

    new_dates = [d for d in available_dates if d not in done_dates]
    if not new_dates:
        if existing is not None:
            return existing
        return pd.DataFrame(
            columns=[
                "ticker",
                "available_at",
                "eps_revision_3m",
                "revenue_revision_3m",
                "target_revision_3m",
                "revision_breadth",
            ]
        )

    frames = []
    for d in new_dates:
        one = build_revision_drift_one_date(
            snapshots_root, d, lookback_days, available_dates
        )
        if one is not None:
            frames.append(one)

    if not frames:
        if existing is not None:
            return existing
        return pd.DataFrame(
            columns=[
                "ticker",
                "available_at",
                "eps_revision_3m",
                "revenue_revision_3m",
                "target_revision_3m",
                "revision_breadth",
            ]
        )

    new_df = pd.concat(frames, ignore_index=True)
    if existing is not None and not existing.empty:
        return pd.concat([existing, new_df], ignore_index=True)
    return new_df


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build PIT revision-drift features from estimate snapshots."
    )
    ap.add_argument(
        "--snapshots",
        required=True,
        help="root directory of estimate snapshots (contains YYYY-MM-DD subdirs)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output parquet path for the feature table",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="lookback window in days for revision calculation (default 90)",
    )
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="only compute dates not already in --out (appends to existing output)",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshots_root = Path(args.snapshots)
    out_path = Path(args.out)

    if not snapshots_root.is_dir():
        print(f"error: snapshots directory not found: {snapshots_root}", file=sys.stderr)
        return 2

    existing = None
    if args.incremental and out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            log.info("loaded %d existing rows from %s", len(existing), out_path)
        except Exception as exc:
            print(
                f"warning: could not read existing output {out_path}: {exc}",
                file=sys.stderr,
            )
            existing = None

    result = incremental_update(
        snapshots_root, existing=existing, lookback_days=args.lookback_days
    )

    if result.empty:
        print(
            "warning: no revision features computed (need at least 2 snapshots "
            f"~{args.lookback_days} days apart)",
            file=sys.stderr,
        )
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    n_dates = result["available_at"].nunique()
    n_tickers = result["ticker"].nunique()
    print(
        f"pit_feature_builder: wrote {len(result)} rows "
        f"({n_tickers} tickers x {n_dates} dates) to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
