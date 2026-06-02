"""Build the RenQuant alpha158 Qlib feature panel.

This module is the subrepo-owned lift of RenQuant's
``scripts/build_alpha158_qlib.py``. It preserves the production data contract
while removing the umbrella-repo path dependency: callers pass a ``data_dir``
or explicit input paths, and the ticker feature phase runs through
``renquant_common.run_parallel``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from renquant_common import Job, Pipeline, Task, run_parallel

from .track_b_features import (
    TRACK_B_FEATURES,
    add_track_b_features,
)


for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")


log = logging.getLogger("renquant_base_data.alpha158_qlib_panel")

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12
MAX_SPY_LABEL_FFILL_DAYS = 5
STD_DDOF = 1
MIN_OHLCV_ROWS = 70
EXPECTED_ALPHA158_FEATURES = 158

DEFAULT_INVENTORY_FILENAME = "transformer_universe_inventory.json"
DEFAULT_INTEGRITY_REPORT_FILENAME = "transformer_data_integrity_report.json"
DEFAULT_EXISTING_ENGINEERED_FILENAME = "transformer_dataset_engineered.parquet"
DEFAULT_OHLCV_DIRNAME = "ohlcv"
DEFAULT_OUTPUT_FILENAME = "alpha158_qlib_dataset.parquet"

# Minimum number of finite observations per Track B feature on the train
# split before the panel build is allowed to proceed. Matches one full
# 252-day warmup window so the per-feature train mean / std come from at
# least one full canonical lookback. Anything less leaves the
# NormalizeAndAnnotateJob computing NaN means/stds that get masked by the
# downstream fillna(0.0) — silently emitting zero columns advertised as
# live features (PR #16 codex HIGH finding, 2026-06-02).
MIN_TRACK_B_TRAIN_OBS = 252


class InsufficientTrainHistoryError(RuntimeError):
    """Raised when ``include_track_b=True`` but the train split lacks enough
    finite observations for one or more Track B features. Names the offending
    feature(s) + observed counts so operators see the missing dependency
    directly. Operators may either extend the input history OR drop
    ``--include-track-b``; silently emitting all-zero columns is forbidden.
    """


@dataclass
class Alpha158QlibConfig:
    data_dir: Path = Path("data")
    inventory_path: Path | None = None
    integrity_report_path: Path | None = None
    existing_engineered_path: Path | None = None
    ohlcv_dir: Path | None = None
    output_path: Path | None = None
    tickers: int = 0
    max_workers: int | None = None
    timeout_seconds: float | None = None
    progress_log_seconds: float = 30.0
    # Track B (BULL_CALM signal recovery, 2026-06-02): when True, append
    # the 4 canonical low-vol/momentum features to the panel. Off by default
    # so the existing 158-feature artifact contract is byte-stable.
    include_track_b: bool = False

    def resolved(self) -> "Alpha158QlibConfig":
        data_dir = Path(self.data_dir).expanduser().resolve()
        return Alpha158QlibConfig(
            data_dir=data_dir,
            inventory_path=_resolve_path(self.inventory_path, data_dir / DEFAULT_INVENTORY_FILENAME),
            integrity_report_path=_resolve_path(
                self.integrity_report_path,
                data_dir / DEFAULT_INTEGRITY_REPORT_FILENAME,
            ),
            existing_engineered_path=_resolve_path(
                self.existing_engineered_path,
                data_dir / DEFAULT_EXISTING_ENGINEERED_FILENAME,
            ),
            ohlcv_dir=_resolve_path(self.ohlcv_dir, data_dir / DEFAULT_OHLCV_DIRNAME),
            output_path=_resolve_path(self.output_path, data_dir / DEFAULT_OUTPUT_FILENAME),
            tickers=int(self.tickers or 0),
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
            progress_log_seconds=float(self.progress_log_seconds),
            include_track_b=bool(self.include_track_b),
        )


@dataclass
class Alpha158QlibContext:
    config: Alpha158QlibConfig
    universe: list[str] = field(default_factory=list)
    panel: pd.DataFrame | None = None
    feature_cols: list[str] = field(default_factory=list)
    output_path: Path | None = None


@dataclass
class TickerFeatureContext:
    ticker: str
    ohlcv_dir: Path
    features: pd.DataFrame | None = None


def _resolve_path(path: Path | str | None, default: Path) -> Path:
    return Path(path).expanduser().resolve() if path is not None else default.expanduser().resolve()


def _slope(s: pd.Series, n: int) -> pd.Series:
    x_mean = (n - 1) / 2.0
    var_x = sum((i - x_mean) ** 2 for i in range(n))

    def slope_fn(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        return sum((i - x_mean) * (arr[i] - y_mean) for i in range(n)) / var_x

    return s.rolling(n).apply(slope_fn, raw=True)


def _rsquare(s: pd.Series, n: int) -> pd.Series:
    x_mean = (n - 1) / 2.0
    var_x = sum((i - x_mean) ** 2 for i in range(n))

    def rsq_fn(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        ss_tot = ((arr - y_mean) ** 2).sum()
        if ss_tot < EPS:
            return np.nan
        slope = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n)) / var_x
        intercept = y_mean - slope * x_mean
        ss_res = sum((arr[i] - intercept - slope * i) ** 2 for i in range(n))
        return 1.0 - ss_res / ss_tot

    return s.rolling(n).apply(rsq_fn, raw=True)


def _resi(s: pd.Series, n: int) -> pd.Series:
    x_mean = (n - 1) / 2.0
    var_x = sum((i - x_mean) ** 2 for i in range(n))

    def resi_fn(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y_mean = arr.mean()
        slope = sum((i - x_mean) * (arr[i] - y_mean) for i in range(n)) / var_x
        intercept = y_mean - slope * x_mean
        return arr[-1] - intercept - slope * (n - 1)

    return s.rolling(n).apply(resi_fn, raw=True)


def _idx_max_n(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).apply(lambda x: float(np.argmax(x)), raw=True)


def _idx_min_n(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).apply(lambda x: float(np.argmin(x)), raw=True)


def _greater(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).max(axis=1)


def _less(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.concat([a, b], axis=1).min(axis=1)


def _load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return df

    date_col = next((col for col in ("date", "Date", "timestamp", "Timestamp") if col in df.columns), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing OHLCV columns: {missing}")
    return df[required]


def kbar_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    span = (h - l) + EPS
    g_oc = _greater(o, c)
    l_oc = _less(o, c)
    return {
        "KMID": (c - o) / o,
        "KLEN": (h - l) / o,
        "KMID2": (c - o) / span,
        "KUP": (h - g_oc) / o,
        "KUP2": (h - g_oc) / span,
        "KLOW": (l_oc - l) / o,
        "KLOW2": (l_oc - l) / span,
        "KSFT": (2 * c - h - l) / o,
        "KSFT2": (2 * c - h - l) / span,
    }


def price_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["close"]
    vwap = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    return {
        "OPEN0": df["open"] / c,
        "HIGH0": df["high"] / c,
        "LOW0": df["low"] / c,
        "VWAP0": vwap / c,
    }


def rolling_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)

    c_lag1 = c.shift(1)
    c_diff = c - c_lag1
    abs_c_diff = c_diff.abs()
    log_v = np.log(v + 1)
    c_ret = c / c_lag1 - 1
    abs_c_ret = c_ret.abs()
    c_ret_norm = c / c_lag1
    v_ret_norm = v / v.shift(1)
    log_v_ret = np.log(v_ret_norm + 1)
    v_diff = v - v.shift(1)
    abs_v_diff = v_diff.abs()

    out: dict[str, pd.Series] = {}
    for n in WINDOWS:
        out[f"ROC{n}"] = c.shift(n) / c
        out[f"MA{n}"] = c.rolling(n).mean() / c
        out[f"STD{n}"] = c.rolling(n).std(ddof=STD_DDOF) / c
        out[f"BETA{n}"] = _slope(c, n) / c
        out[f"RSQR{n}"] = _rsquare(c, n)
        out[f"RESI{n}"] = _resi(c, n) / c
        out[f"MAX{n}"] = h.rolling(n).max() / c
        out[f"MIN{n}"] = l.rolling(n).min() / c
        out[f"QTLU{n}"] = c.rolling(n).quantile(0.8) / c
        out[f"QTLD{n}"] = c.rolling(n).quantile(0.2) / c
        out[f"RANK{n}"] = c.rolling(n).rank(pct=True)
        out[f"RSV{n}"] = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min() + EPS)
        out[f"IMAX{n}"] = _idx_max_n(h, n) / n
        out[f"IMIN{n}"] = _idx_min_n(l, n) / n
        out[f"IMXD{n}"] = (_idx_max_n(h, n) - _idx_min_n(l, n)) / n
        out[f"CORR{n}"] = c.rolling(n).corr(log_v)
        out[f"CORD{n}"] = c_ret_norm.rolling(n).corr(log_v_ret)
        out[f"CNTP{n}"] = (c > c_lag1).astype(float).rolling(n).mean()
        out[f"CNTN{n}"] = (c < c_lag1).astype(float).rolling(n).mean()
        out[f"CNTD{n}"] = out[f"CNTP{n}"] - out[f"CNTN{n}"]

        pos_ret = c_diff.clip(lower=0)
        neg_ret = (-c_diff).clip(lower=0)
        sum_abs = abs_c_diff.rolling(n).sum() + EPS
        out[f"SUMP{n}"] = pos_ret.rolling(n).sum() / sum_abs
        out[f"SUMN{n}"] = neg_ret.rolling(n).sum() / sum_abs
        out[f"SUMD{n}"] = out[f"SUMP{n}"] - out[f"SUMN{n}"]

        v_safe = v.where(np.isfinite(v) & (v > 0), v.rolling(20, min_periods=1).mean())
        v_safe = v_safe.where(np.isfinite(v_safe) & (v_safe > 0), 1.0)
        out[f"VMA{n}"] = v.rolling(n).mean() / v_safe
        out[f"VSTD{n}"] = v.rolling(n).std(ddof=STD_DDOF) / v_safe

        wv = abs_c_ret * v
        out[f"WVMA{n}"] = wv.rolling(n).std(ddof=STD_DDOF) / (wv.rolling(n).mean() + EPS)

        pos_v = v_diff.clip(lower=0)
        neg_v = (-v_diff).clip(lower=0)
        sum_abs_v = abs_v_diff.rolling(n).sum() + EPS
        out[f"VSUMP{n}"] = pos_v.rolling(n).sum() / sum_abs_v
        out[f"VSUMN{n}"] = neg_v.rolling(n).sum() / sum_abs_v
        out[f"VSUMD{n}"] = out[f"VSUMP{n}"] - out[f"VSUMN{n}"]
    return out


def build_features_for_ticker(ticker: str, ohlcv_dir: str | Path) -> pd.DataFrame | None:
    path = Path(ohlcv_dir) / ticker / "1d.parquet"
    if not path.exists():
        return None
    try:
        df = _load_ohlcv(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: OHLCV read failed: %s", ticker, exc)
        return None
    if df.empty or len(df) < MIN_OHLCV_ROWS:
        return None

    feats: dict[str, pd.Series] = {}
    feats.update(kbar_features(df))
    feats.update(price_features(df))
    feats.update(rolling_features(df))
    feat_df = pd.DataFrame(feats, index=df.index)
    feat_df.index.name = "date"
    feat_df = feat_df.reset_index()
    feat_df["date"] = pd.to_datetime(feat_df["date"])
    feat_df.insert(0, "ticker", ticker)
    return feat_df


def fit_raw_clip_bounds(
    panel: pd.DataFrame,
    feat_cols: list[str],
    train_mask: pd.Series,
    *,
    low_q: float = 0.001,
    high_q: float = 0.999,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    lows: dict[str, float | None] = {}
    highs: dict[str, float | None] = {}
    for col in feat_cols:
        train_col = panel.loc[train_mask, col]
        q_lo, q_hi = train_col.quantile([low_q, high_q])
        if np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo:
            lows[col] = float(q_lo)
            highs[col] = float(q_hi)
        else:
            lows[col] = None
            highs[col] = None
    return lows, highs


def _compute_excess_label_frame(
    ticker: str,
    close: pd.Series,
    spy_close: pd.Series,
    *,
    horizons: tuple[int, ...] = (5, 20, 60),
    max_spy_ffill_days: int = MAX_SPY_LABEL_FFILL_DAYS,
) -> pd.DataFrame:
    c = close.sort_index()
    spy = spy_close.sort_index()
    spy_aligned = spy.reindex(
        c.index,
        method="ffill",
        tolerance=pd.Timedelta(days=max_spy_ffill_days),
    )
    rec: dict[str, pd.Series] = {
        "ticker": pd.Series(ticker, index=c.index),
        "date": pd.Series(c.index, index=c.index),
    }
    for n in horizons:
        fwd_ticker = c.shift(-n) / c - 1
        fwd_spy = spy_aligned.shift(-n) / spy_aligned - 1
        rec[f"fwd_{n}d_excess"] = fwd_ticker - fwd_spy
    return pd.DataFrame(rec)


class LoadUniverseJob(Job):
    def run(self, ctx: Alpha158QlibContext) -> None:
        cfg = ctx.config
        inv = json.loads(cfg.inventory_path.read_text())
        integ = json.loads(cfg.integrity_report_path.read_text())
        universe = set(inv.get("tier_A_tickers", [])) | set(inv.get("tier_B_tickers", []))

        failed: set[str] = set()
        for tier in ("A", "B"):
            for row in integ.get("per_ticker", {}).get(tier, []):
                if not row.get("ok"):
                    failed.add(row["ticker"])
        ctx.universe = sorted(universe - failed)
        if cfg.tickers > 0:
            ctx.universe = ctx.universe[: cfg.tickers]
        if not ctx.universe:
            raise RuntimeError("No tickers remain after inventory/integrity filtering")
        log.info("Building alpha158 Qlib panel for %d tickers", len(ctx.universe))


class BuildTickerFeatureTask(Task):
    def run(self, ctx: TickerFeatureContext) -> bool | None:
        ctx.features = build_features_for_ticker(ctx.ticker, ctx.ohlcv_dir)
        return True


class BuildTickerFeatureJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [BuildTickerFeatureTask()]


class BuildFeaturePanelJob(Job):
    def run(self, ctx: Alpha158QlibContext) -> None:
        cfg = ctx.config
        ticker_contexts = [TickerFeatureContext(ticker=ticker, ohlcv_dir=cfg.ohlcv_dir) for ticker in ctx.universe]
        run_parallel(
            ticker_contexts,
            BuildTickerFeatureJob(),
            max_workers=cfg.max_workers,
            timeout_seconds=cfg.timeout_seconds,
            progress_log_seconds=cfg.progress_log_seconds,
        )
        rows = [item.features for item in ticker_contexts if item.features is not None and not item.features.empty]
        if not rows:
            raise RuntimeError("No tickers produced alpha158 features")
        panel = pd.concat(rows, ignore_index=True)
        ctx.feature_cols = [col for col in panel.columns if col not in ("ticker", "date")]
        if len(ctx.feature_cols) != EXPECTED_ALPHA158_FEATURES:
            raise RuntimeError(
                f"alpha158 feature count changed: {len(ctx.feature_cols)} != "
                f"{EXPECTED_ALPHA158_FEATURES} (baseline alpha158 stage; Track B "
                f"features are appended later by AddTrackBFeaturesJob)"
            )
        panel[ctx.feature_cols] = panel[ctx.feature_cols].replace([np.inf, -np.inf], np.nan)
        ctx.panel = panel
        log.info("Raw alpha158 panel rows=%d feature_cols=%d", len(panel), len(ctx.feature_cols))


class AddTrackBFeaturesJob(Job):
    """Track B (BULL_CALM signal recovery, 2026-06-02): append the 4 canonical
    low-vol/momentum features (mom_carry_12_1, beta_dm, rvar_total, idio_vol_market)
    to the alpha158 panel. Reads each ticker's raw close+volume from the OHLCV
    parquet (already validated upstream) and uses the SPY close as the market leg.
    Skipped when ``config.include_track_b`` is False; the existing 158-feature
    artifact is byte-stable in that branch.
    """

    def should_skip(self, ctx: Alpha158QlibContext) -> bool:
        return not bool(ctx.config.include_track_b)

    def run(self, ctx: Alpha158QlibContext) -> None:
        cfg = ctx.config
        panel = _require_panel(ctx)
        spy_path = cfg.ohlcv_dir / "SPY" / "1d.parquet"
        if not spy_path.exists():
            raise FileNotFoundError(
                f"SPY OHLCV not found at {spy_path}; cannot compute Track B features"
            )
        spy_close = _load_ohlcv(spy_path)["close"].sort_index().rename("spy_close")
        # Re-attach raw close+volume per ticker so the feature builder can
        # operate; the alpha158 build dropped them. Per-ticker join keeps the
        # alpha158 panel rows immutable.
        ohlcv_rows: list[pd.DataFrame] = []
        for ticker in panel["ticker"].unique():
            path = cfg.ohlcv_dir / ticker / "1d.parquet"
            if not path.exists():
                continue
            try:
                tk = _load_ohlcv(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: track-b OHLCV read failed: %s", ticker, exc)
                continue
            tk = tk[["close", "volume"]].copy()
            tk["ticker"] = ticker
            tk["date"] = pd.to_datetime(tk.index)
            ohlcv_rows.append(tk.reset_index(drop=True))
        if not ohlcv_rows:
            raise RuntimeError("Track B: no OHLCV available for any ticker in panel")
        ohlcv_panel = pd.concat(ohlcv_rows, ignore_index=True)
        merged = panel.merge(ohlcv_panel, on=["ticker", "date"], how="left")
        if len(merged) != len(panel):
            raise RuntimeError(
                f"Track B OHLCV merge changed row count: {len(panel)} -> {len(merged)}"
            )
        with_tb = add_track_b_features(merged, spy_close=spy_close)
        # Drop the temporary raw OHLCV columns; only the 4 new features stay.
        with_tb = with_tb.drop(columns=["close", "volume"])
        with_tb[list(TRACK_B_FEATURES)] = (
            with_tb[list(TRACK_B_FEATURES)].replace([np.inf, -np.inf], np.nan)
        )
        # HIGH-finding guard (codex PR #16 review, 2026-06-02): fail-loud
        # when train-split history is too short for the 252-day windows.
        # Fired AT THE SOURCE so downstream stats fitting never sees a
        # silently-zeroed column.
        _validate_track_b_train_history(with_tb, cfg)
        ctx.feature_cols = list(ctx.feature_cols) + list(TRACK_B_FEATURES)
        with_tb[ctx.feature_cols] = with_tb[ctx.feature_cols].replace([np.inf, -np.inf], np.nan)
        ctx.panel = with_tb
        log.info(
            "Track B features added: %s (panel rows=%d feature_cols=%d)",
            ", ".join(TRACK_B_FEATURES), len(with_tb), len(ctx.feature_cols),
        )


def _validate_track_b_train_history(
    panel: pd.DataFrame, cfg: Alpha158QlibConfig
) -> None:
    """Per-feature finite-observation gate on the train split.

    Reads the same ``existing_engineered_path`` split label that
    ``NormalizeAndAnnotateJob`` uses, slices the train rows, and counts
    finite (non-NaN, non-Inf) observations per Track B feature. Raises
    ``InsufficientTrainHistoryError`` listing every offending feature when
    any count is below ``MIN_TRACK_B_TRAIN_OBS``.
    """
    existing = pd.read_parquet(cfg.existing_engineered_path)
    if "split_label" not in existing.columns:
        raise ValueError(
            f"{cfg.existing_engineered_path} missing split_label "
            "(needed to gate Track B train-history sufficiency)"
        )
    existing["date"] = pd.to_datetime(existing["date"])
    date_split = (
        existing[["date", "split_label"]]
        .drop_duplicates("date")
        .set_index("date")["split_label"]
    )
    panel_dates = pd.to_datetime(panel["date"])
    split_labels = panel_dates.map(date_split).fillna("test")
    train_mask = (split_labels == "train").to_numpy()
    deficiencies: dict[str, int] = {}
    for col in TRACK_B_FEATURES:
        col_train = panel.loc[train_mask, col]
        finite_count = int(np.isfinite(col_train.to_numpy(dtype=float)).sum())
        if finite_count < MIN_TRACK_B_TRAIN_OBS:
            deficiencies[col] = finite_count
    if deficiencies:
        details = ", ".join(
            f"{col}={count}<{MIN_TRACK_B_TRAIN_OBS}"
            for col, count in sorted(deficiencies.items())
        )
        raise InsufficientTrainHistoryError(
            f"Track B train-history gate failed: {details}. Each feature "
            f"requires ≥{MIN_TRACK_B_TRAIN_OBS} finite observations in the "
            "train split (one full 252-day warmup window). Either extend "
            "the input OHLCV history covering the train window, or drop "
            "--include-track-b for this build."
        )


class BuildLabelsJob(Job):
    def run(self, ctx: Alpha158QlibContext) -> None:
        cfg = ctx.config
        panel = _require_panel(ctx)
        spy_path = cfg.ohlcv_dir / "SPY" / "1d.parquet"
        if not spy_path.exists():
            raise FileNotFoundError(f"SPY OHLCV not found at {spy_path}; cannot compute excess labels")
        spy_close = _load_ohlcv(spy_path)["close"].sort_index().rename("spy_close")

        label_rows: list[pd.DataFrame] = []
        for ticker in panel["ticker"].unique():
            path = cfg.ohlcv_dir / ticker / "1d.parquet"
            if not path.exists():
                continue
            try:
                ticker_df = _load_ohlcv(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: label OHLCV read failed: %s", ticker, exc)
                continue
            label_rows.append(_compute_excess_label_frame(ticker, ticker_df["close"], spy_close))
        if not label_rows:
            raise RuntimeError("No labels produced from OHLCV")

        labels_panel = pd.concat(label_rows, ignore_index=True)
        labels_panel["date"] = pd.to_datetime(labels_panel["date"])
        ctx.panel = panel.merge(labels_panel, on=["ticker", "date"], how="inner")
        log.info(
            "After label merge rows=%d tickers=%d",
            len(ctx.panel),
            ctx.panel["ticker"].nunique(),
        )


class NormalizeAndAnnotateJob(Job):
    def run(self, ctx: Alpha158QlibContext) -> None:
        cfg = ctx.config
        panel = _require_panel(ctx)
        feat_cols = ctx.feature_cols

        existing = pd.read_parquet(cfg.existing_engineered_path)
        existing["date"] = pd.to_datetime(existing["date"])
        if "split_label" not in existing.columns:
            raise ValueError(f"{cfg.existing_engineered_path} missing split_label")
        date_split = existing[["date", "split_label"]].drop_duplicates("date").set_index("date")["split_label"]
        panel["split_label"] = panel["date"].map(date_split).fillna("test")
        panel = panel.dropna(subset=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"]).copy()
        train_mask = panel["split_label"] == "train"
        if int(train_mask.sum()) == 0:
            raise RuntimeError("No train rows available for alpha158 feature normalization")

        panel.loc[:, feat_cols] = panel[feat_cols].replace([np.inf, -np.inf], np.nan)
        raw_clip_low, raw_clip_high = fit_raw_clip_bounds(panel, feat_cols, train_mask)
        for col in feat_cols:
            q_lo = raw_clip_low[col]
            q_hi = raw_clip_high[col]
            if q_lo is not None and q_hi is not None:
                panel.loc[:, col] = panel[col].clip(q_lo, q_hi)

        feature_stats: dict[str, dict[str, float]] = {}
        for col in feat_cols:
            col_train = panel.loc[train_mask, col]
            mean = float(col_train.mean())
            std = float(col_train.std())
            feature_stats[col] = {"mean": mean, "std": std}
            if std > 1e-9:
                panel.loc[:, col] = (panel[col] - mean) / std
            else:
                panel.loc[:, col] = panel[col] - mean

        panel.loc[:, feat_cols] = panel[feat_cols].fillna(0.0)
        for col in feat_cols:
            panel.loc[:, col] = panel[col].clip(-5.0, 5.0)

        for label in ("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"):
            date_mean = panel.groupby("date")[label].transform("mean")
            date_std = panel.groupby("date")[label].transform("std")
            panel.loc[:, label] = (panel[label] - date_mean) / (date_std + EPS)

        stats_path = cfg.output_path.with_suffix(".stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                {
                    "feature_cols": feat_cols,
                    "feature_means": [feature_stats[col]["mean"] for col in feat_cols],
                    "feature_stds": [feature_stats[col]["std"] for col in feat_cols],
                    "feature_raw_clip_low": [raw_clip_low[col] for col in feat_cols],
                    "feature_raw_clip_high": [raw_clip_high[col] for col in feat_cols],
                    "feature_raw_clip_quantiles": [0.001, 0.999],
                    "feature_raw_clip_fit_split": "train",
                    "feature_preprocess_version": 2,
                    "n_train_rows": int(train_mask.sum()),
                    "clip_sigma": 5.0,
                },
                indent=2,
                default=str,
            )
        )
        ctx.panel = panel
        log.info("Saved train-only feature stats: %s", stats_path)


class PersistResultsJob(Job):
    def run(self, ctx: Alpha158QlibContext) -> None:
        cfg = ctx.config
        panel = _require_panel(ctx)
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cfg.output_path, index=False)
        ctx.output_path = cfg.output_path
        log.info(
            "Wrote %s rows=%d cols=%d tickers=%d",
            cfg.output_path,
            len(panel),
            len(panel.columns),
            panel["ticker"].nunique(),
        )


def _require_panel(ctx: Alpha158QlibContext) -> pd.DataFrame:
    if ctx.panel is None:
        raise RuntimeError("alpha158 panel is not initialized")
    return ctx.panel


def build_alpha158_qlib_pipeline() -> Pipeline:
    return Pipeline(
        [
            LoadUniverseJob(),
            BuildFeaturePanelJob(),
            AddTrackBFeaturesJob(),
            BuildLabelsJob(),
            NormalizeAndAnnotateJob(),
            PersistResultsJob(),
        ],
        name="alpha158-qlib-panel-build",
    )


def build_alpha158_qlib_panel(
    data_dir: str | Path = Path("data"),
    *,
    inventory_path: str | Path | None = None,
    integrity_report_path: str | Path | None = None,
    existing_engineered_path: str | Path | None = None,
    ohlcv_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    tickers: int = 0,
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
    progress_log_seconds: float = 30.0,
    include_track_b: bool = False,
) -> Path:
    config = Alpha158QlibConfig(
        data_dir=Path(data_dir),
        inventory_path=Path(inventory_path) if inventory_path is not None else None,
        integrity_report_path=Path(integrity_report_path) if integrity_report_path is not None else None,
        existing_engineered_path=Path(existing_engineered_path) if existing_engineered_path is not None else None,
        ohlcv_dir=Path(ohlcv_dir) if ohlcv_dir is not None else None,
        output_path=Path(output_path) if output_path is not None else None,
        tickers=tickers,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        progress_log_seconds=progress_log_seconds,
        include_track_b=include_track_b,
    ).resolved()
    ctx = Alpha158QlibContext(config=config)
    build_alpha158_qlib_pipeline().run(ctx)
    if ctx.output_path is None:
        raise RuntimeError("alpha158 panel build completed without an output path")
    return ctx.output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--integrity-report", type=Path, default=None)
    parser.add_argument("--existing-engineered", type=Path, default=None)
    parser.add_argument("--ohlcv-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tickers", type=int, default=0, help="Limit to first N tickers for smoke runs.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--progress-log-seconds", type=float, default=30.0)
    parser.add_argument(
        "--include-track-b",
        action="store_true",
        help=(
            "Append the 4 Track B BULL_CALM-regime features (mom_carry_12_1, "
            "beta_dm, rvar_total, idio_vol_market). Default: off — preserves the "
            "158-feature baseline contract."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    build_alpha158_qlib_panel(
        args.data_dir,
        inventory_path=args.inventory,
        integrity_report_path=args.integrity_report,
        existing_engineered_path=args.existing_engineered,
        ohlcv_dir=args.ohlcv_dir,
        output_path=args.output,
        tickers=args.tickers,
        max_workers=args.max_workers,
        timeout_seconds=args.timeout_seconds,
        progress_log_seconds=args.progress_log_seconds,
        include_track_b=args.include_track_b,
    )
    return 0


__all__ = [
    "Alpha158QlibConfig",
    "EXPECTED_ALPHA158_FEATURES",
    "InsufficientTrainHistoryError",
    "MIN_TRACK_B_TRAIN_OBS",
    "build_alpha158_qlib_panel",
    "build_alpha158_qlib_pipeline",
    "build_features_for_ticker",
    "fit_raw_clip_bounds",
]


if __name__ == "__main__":
    raise SystemExit(main())
