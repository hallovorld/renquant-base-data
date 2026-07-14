"""Build the crypto alpha158 feature panel for XGB model training (D-C3).

Crypto counterpart of :mod:`alpha158_qlib_panel` (equity). Reuses the
canonical alpha158 operators from :mod:`alpha158_ops` — those are
OHLCV-agnostic — but differs in three ways:

1. **Labels** — raw forward returns (``fwd_Nd``) and BTC-excess
   (``fwd_Nd_btc_excess``). SPY-excess makes no sense for crypto; BTC
   is the cross-sectional "market" leg.
2. **Calendar** — UTC calendar days (365/year), not NYSE (252). Horizons
   are in calendar days to match the equity convention.
3. **No fundamentals** — no SEC, PEAD, SUE, or sentiment overlay.

Input: crypto bars from :class:`crypto_bars.CryptoLocalStore` (layout
``crypto_ohlcv/{slug}/1d.parquet``).
Output: stacked panel parquet ``(date, pair, alpha158_features..., labels...)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .alpha158_ops import compute_alpha158_frame, alpha158_feature_names
from .crypto_bars import CryptoLocalStore, pair_slug, CRYPTO_OHLCV_DIRNAME

log = logging.getLogger("renquant_base_data.crypto_alpha158_panel")

BTC_SLUG = "BTC-USD"
DEFAULT_LABEL_HORIZONS = (5, 20, 60)
MIN_BARS_FEATURES = 70
MIN_BARS_PANEL = 90
DEFAULT_OUTPUT_FILENAME = "crypto_alpha158_panel.parquet"
EXPECTED_FEATURE_COUNT = 158


@dataclass
class CryptoPanelConfig:
    crypto_ohlcv_dir: Path | None = None
    output_path: Path | None = None
    label_horizons: tuple[int, ...] = DEFAULT_LABEL_HORIZONS
    btc_excess: bool = True
    min_bars: int = MIN_BARS_FEATURES
    min_panel_dates: int = MIN_BARS_PANEL
    min_pairs: int = 2
    pairs: list[str] | None = None


def discover_pairs(store_dir: Path) -> list[str]:
    """List available crypto pair slugs from the store directory."""
    if not store_dir.is_dir():
        return []
    slugs = sorted(
        d.name
        for d in store_dir.iterdir()
        if d.is_dir() and (d / "1d.parquet").exists()
    )
    return slugs


def _load_ohlcv(store: CryptoLocalStore, slug: str) -> pd.DataFrame | None:
    df = store.load(slug, "1d")
    if df is None or df.empty:
        return None
    required = ["open", "high", "low", "close", "volume", "bar_close_utc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        if "bar_close_utc" in missing:
            raise ValueError(
                f"{slug}: bar_close_utc column is required but missing. "
                f"Input bars must carry a verified bar_close_utc from D-C2 ingestion."
            )
        log.warning("%s: missing OHLCV columns %s, skipping", slug, missing)
        return None
    df = df[required].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    expected_close = df.index + pd.Timedelta(days=1)
    bar_close_ts = pd.to_datetime(df["bar_close_utc"])
    misaligned = bar_close_ts != expected_close
    if misaligned.any():
        first_bad = df.index[misaligned][0]
        raise ValueError(
            f"{slug}: bar_close_utc does not match UTC daily convention "
            f"(index + 1 day) at {first_bad}. Got {bar_close_ts[misaligned].iloc[0]}, "
            f"expected {expected_close[misaligned][0]}."
        )
    return df


def build_features_for_pair(
    slug: str, store: CryptoLocalStore, *, min_bars: int = MIN_BARS_FEATURES,
) -> pd.DataFrame | None:
    """Compute alpha158 features for a single crypto pair."""
    ohlcv = _load_ohlcv(store, slug)
    if ohlcv is None or len(ohlcv) < min_bars:
        log.info("%s: insufficient bars (%s < %d), skipping features",
                 slug, len(ohlcv) if ohlcv is not None else 0, min_bars)
        return None
    feats = compute_alpha158_frame(ohlcv, min_bars=min_bars)
    if feats.empty:
        return None
    feats = feats.copy()
    feats["pair"] = slug
    feats["date"] = feats.index
    feats = feats.reset_index(drop=True)
    return feats


FFILL_LIMIT_DAYS = 3


def compute_forward_returns(
    ohlcv: pd.DataFrame,
    slug: str,
    *,
    horizons: tuple[int, ...] = DEFAULT_LABEL_HORIZONS,
    btc_close: pd.Series | None = None,
    btc_bar_close_utc: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute raw and optionally BTC-excess forward returns.

    Labels are in UTC calendar days: both the pair and BTC close series
    are reindexed to a complete daily calendar before shifting, so
    ``shift(-n)`` always means exactly *n* calendar days forward
    regardless of missing observations.  Gaps beyond ``FFILL_LIMIT_DAYS``
    produce NaN (no silent extrapolation).

    Availability timestamps are derived from the verified ``bar_close_utc``
    column on ``ohlcv`` (and ``btc_bar_close_utc`` for BTC-excess),
    not from arithmetic offsets.
    """
    c = ohlcv["close"].sort_index()
    pair_close_ts = ohlcv["bar_close_utc"].sort_index() if "bar_close_utc" in ohlcv.columns else None
    max_horizon = max(horizons)
    cal = pd.date_range(
        c.index.min(),
        c.index.max() + pd.Timedelta(days=max_horizon),
        freq="D",
    )
    c_cal = c.reindex(cal, method="ffill", limit=FFILL_LIMIT_DAYS)

    real_obs = c.reindex(cal).notna()

    if pair_close_ts is not None:
        close_ts_cal = pair_close_ts.reindex(cal)
    else:
        close_ts_cal = None

    output_dates = c.index
    rec: dict[str, pd.Series] = {
        "pair": pd.Series(slug, index=output_dates),
        "date": pd.Series(output_dates, index=output_dates),
    }
    if btc_close is not None:
        btc_sorted = btc_close.sort_index()
        btc_cal = btc_sorted.reindex(cal, method="ffill", limit=FFILL_LIMIT_DAYS)
        btc_real = btc_sorted.reindex(cal).notna()
        btc_start_has_real = btc_real.reindex(output_dates).fillna(False).infer_objects(copy=False)
        if btc_bar_close_utc is not None:
            btc_close_ts_cal = btc_bar_close_utc.sort_index().reindex(cal)
        else:
            btc_close_ts_cal = None
    else:
        btc_cal = btc_real = btc_start_has_real = None
        btc_close_ts_cal = None

    for n in horizons:
        fwd = c_cal.shift(-n) / c_cal - 1
        terminal_has_real = real_obs.shift(-n).reindex(output_dates).fillna(False).infer_objects(copy=False)
        fwd_out = fwd.reindex(output_dates)
        fwd_out = fwd_out.where(terminal_has_real, other=np.nan)

        if close_ts_cal is not None:
            terminal_close_ts = close_ts_cal.shift(-n).reindex(output_dates)
        else:
            terminal_close_ts = pd.Series(
                output_dates + pd.Timedelta(days=n + 1), index=output_dates,
            )

        rec[f"fwd_{n}d"] = fwd_out
        rec[f"fwd_{n}d_available_after"] = terminal_close_ts.where(terminal_has_real, other=pd.NaT)

        if btc_cal is not None:
            fwd_btc = btc_cal.shift(-n) / btc_cal - 1
            btc_terminal_has_real = btc_real.shift(-n).reindex(output_dates).fillna(False).infer_objects(copy=False)
            excess_valid = terminal_has_real & btc_terminal_has_real & btc_start_has_real
            excess = fwd_out - fwd_btc.reindex(output_dates)
            excess = excess.where(excess_valid, other=np.nan)
            rec[f"fwd_{n}d_btc_excess"] = excess

            if btc_close_ts_cal is not None:
                btc_terminal_close_ts = btc_close_ts_cal.shift(-n).reindex(output_dates)
                excess_avail = pd.concat(
                    [terminal_close_ts, btc_terminal_close_ts], axis=1,
                ).max(axis=1)
            else:
                excess_avail = terminal_close_ts

            rec[f"fwd_{n}d_btc_excess_available_after"] = excess_avail.where(excess_valid, other=pd.NaT)
    return pd.DataFrame(rec)


def build_crypto_panel(cfg: CryptoPanelConfig) -> Path:
    """Build the full crypto alpha158 panel and write to parquet.

    Returns the output path.
    """
    store_dir = cfg.crypto_ohlcv_dir or (
        Path(__file__).resolve().parents[3] / "data" / CRYPTO_OHLCV_DIRNAME
    )
    store = CryptoLocalStore(store_dir)
    out_path = Path(cfg.output_path) if cfg.output_path else store_dir.parent / DEFAULT_OUTPUT_FILENAME

    if cfg.pairs:
        slugs = [pair_slug(p) for p in cfg.pairs]
    else:
        slugs = discover_pairs(store_dir)
    if not slugs:
        raise RuntimeError(f"No crypto pairs found in {store_dir}")
    log.info("Building crypto alpha158 panel for %d pairs: %s", len(slugs), slugs)

    btc_close: pd.Series | None = None
    btc_bar_close: pd.Series | None = None
    if cfg.btc_excess:
        btc_ohlcv = _load_ohlcv(store, BTC_SLUG)
        if btc_ohlcv is None:
            log.warning("BTC-USD bars not found; BTC-excess labels will be skipped")
        else:
            btc_close = btc_ohlcv["close"].sort_index().rename("btc_close")
            btc_bar_close = btc_ohlcv["bar_close_utc"].sort_index()

    feature_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    for slug in slugs:
        feats = build_features_for_pair(slug, store, min_bars=cfg.min_bars)
        if feats is None:
            continue
        feature_rows.append(feats)
        ohlcv = _load_ohlcv(store, slug)
        if ohlcv is not None:
            ohlcv_cache[slug] = ohlcv
            labels = compute_forward_returns(
                ohlcv, slug,
                horizons=cfg.label_horizons,
                btc_close=btc_close if slug != BTC_SLUG else None,
                btc_bar_close_utc=btc_bar_close if slug != BTC_SLUG else None,
            )
            label_rows.append(labels)

    if not feature_rows:
        raise RuntimeError("No pairs produced alpha158 features")

    features_panel = pd.concat(feature_rows, ignore_index=True)
    feature_cols = [c for c in features_panel.columns if c not in ("pair", "date", "feature_available_after")]
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Crypto alpha158 feature count mismatch: {len(feature_cols)} != "
            f"{EXPECTED_FEATURE_COUNT}. Columns: {feature_cols[:5]}..."
        )
    features_panel[feature_cols] = features_panel[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    log.info(
        "Features: %d rows, %d pairs, %d features, dates %s..%s",
        len(features_panel),
        features_panel["pair"].nunique(),
        len(feature_cols),
        features_panel["date"].min(),
        features_panel["date"].max(),
    )

    if label_rows:
        labels_panel = pd.concat(label_rows, ignore_index=True)
        labels_panel["date"] = pd.to_datetime(labels_panel["date"])
        features_panel["date"] = pd.to_datetime(features_panel["date"])
        panel = features_panel.merge(labels_panel, on=["pair", "date"], how="inner")
    else:
        panel = features_panel

    bar_close_rows = []
    for slug, ohlcv in ohlcv_cache.items():
        bcu = pd.to_datetime(ohlcv["bar_close_utc"]).sort_index()
        bar_close_rows.append(pd.DataFrame({
            "pair": slug, "date": bcu.index, "feature_available_after": bcu.values,
        }))
    if bar_close_rows:
        bcu_lookup = pd.concat(bar_close_rows, ignore_index=True)
        bcu_lookup["date"] = pd.to_datetime(bcu_lookup["date"])
        panel = panel.merge(bcu_lookup[["pair", "date", "feature_available_after"]],
                            on=["pair", "date"], how="left")
        missing = panel["feature_available_after"].isna()
        if missing.any():
            panel.loc[missing, "feature_available_after"] = (
                pd.to_datetime(panel.loc[missing, "date"]) + pd.Timedelta(days=1)
            )
    else:
        panel["feature_available_after"] = pd.to_datetime(panel["date"]) + pd.Timedelta(days=1)

    n_pairs = panel["pair"].nunique()
    n_dates = panel["date"].nunique()
    if n_pairs < cfg.min_pairs:
        raise RuntimeError(
            f"Panel has only {n_pairs} pairs (minimum {cfg.min_pairs})"
        )
    if n_dates < cfg.min_panel_dates:
        raise RuntimeError(
            f"Panel has only {n_dates} dates (minimum {cfg.min_panel_dates})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    parquet_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    input_bar_digests: dict[str, str] = {}
    for slug in sorted(panel["pair"].unique().tolist()):
        bar_path = store_dir / slug / "1d.parquet"
        if bar_path.exists():
            input_bar_digests[slug] = hashlib.sha256(bar_path.read_bytes()).hexdigest()

    feature_config_digest = hashlib.sha256(
        json.dumps(sorted(feature_cols), sort_keys=True).encode()
    ).hexdigest()

    has_btc_excess = cfg.btc_excess and btc_close is not None

    observation_end: dict[str, str] = {}
    label_available_at: dict[str, dict[str, str | None]] = {}
    for slug in sorted(panel["pair"].unique().tolist()):
        pair_rows = panel[panel["pair"] == slug]
        ohlcv = _load_ohlcv(store, slug)
        if ohlcv is not None:
            observation_end[slug] = str(ohlcv.index.max().date())
        label_avail: dict[str, str | None] = {}
        for n in cfg.label_horizons:
            col = f"fwd_{n}d"
            if col in pair_rows.columns:
                valid = pair_rows[pair_rows[col].notna()]
                label_avail[col] = str(valid["date"].max().date()) if not valid.empty else None
            else:
                label_avail[col] = None
        label_available_at[slug] = label_avail

    input_bar_watermarks: dict[str, str] = {}
    for slug in sorted(panel["pair"].unique().tolist()):
        ohlcv = _load_ohlcv(store, slug)
        if ohlcv is not None:
            input_bar_watermarks[slug] = str(ohlcv.index.max().date())

    cal_start = str(panel["date"].min().date())
    cal_end = str(panel["date"].max().date())
    calendar_identity = hashlib.sha256(
        json.dumps({"freq": "D", "tz": "UTC", "start": cal_start, "end": cal_end},
                   sort_keys=True).encode()
    ).hexdigest()

    manifest = {
        "schema_version": "crypto-alpha158-panel-v1",
        "output_path": str(out_path),
        "n_pairs": n_pairs,
        "n_dates": n_dates,
        "n_rows": len(panel),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "label_cols": [c for c in panel.columns if c.startswith("fwd_") and not c.endswith("_available_after")],
        "pairs": sorted(panel["pair"].unique().tolist()),
        "date_range": [cal_start, cal_end],
        "parquet_sha256": parquet_sha256,
        "input_bar_digests": input_bar_digests,
        "input_bar_watermarks": input_bar_watermarks,
        "feature_config_digest": feature_config_digest,
        "calendar_identity_digest": calendar_identity,
        "observation_end": observation_end,
        "label_available_at": label_available_at,
        "label_contract": {
            "type": "calendar_day_forward_return",
            "horizons_calendar_days": list(cfg.label_horizons),
            "ffill_limit_days": FFILL_LIMIT_DAYS,
            "btc_excess": has_btc_excess,
            "terminal_obs_required": True,
            "btc_start_obs_required": has_btc_excess,
            "row_level_pit_fields": True,
            "bar_timestamp_convention": "UTC_daily_open",
            "bar_close_offset_days": 1,
            "bar_close_convention_validated": True,
            "availability_derived_from": "bar_close_utc",
            "availability_rule": (
                "features at date D use close[D], available after "
                "bar_close_utc[D]; labels fwd_Nd use close[D+N], "
                "available after bar_close_utc[D+N]; BTC-excess "
                "available after max(pair, BTC) terminal bar_close_utc"
            ),
        },
        "btc_excess": has_btc_excess,
    }

    if has_btc_excess:
        btc_bar_path = store_dir / BTC_SLUG / "1d.parquet"
        if btc_bar_path.exists() and BTC_SLUG not in input_bar_digests:
            input_bar_digests[BTC_SLUG] = hashlib.sha256(btc_bar_path.read_bytes()).hexdigest()
        if btc_close is not None and BTC_SLUG not in input_bar_watermarks:
            input_bar_watermarks[BTC_SLUG] = str(btc_close.index.max().date())
        btc_ohlcv_for_manifest = _load_ohlcv(store, BTC_SLUG)
        if btc_ohlcv_for_manifest is not None and BTC_SLUG not in observation_end:
            observation_end[BTC_SLUG] = str(btc_ohlcv_for_manifest.index.max().date())
        manifest["benchmark_inputs"] = {BTC_SLUG: {"role": "excess_return_denominator"}}
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    log.info(
        "Crypto panel written: %s (%d rows, %d pairs, %d dates, sha256=%s)",
        out_path, len(panel), n_pairs, n_dates, parquet_sha256[:16],
    )
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build crypto alpha158 panel (D-C3)")
    parser.add_argument(
        "--crypto-ohlcv-dir", type=Path, default=None,
        help="Crypto OHLCV store directory (default: data/crypto_ohlcv)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path")
    parser.add_argument("--pairs", nargs="*", help="Specific pairs to include (default: all)")
    parser.add_argument(
        "--horizons", type=int, nargs="*", default=list(DEFAULT_LABEL_HORIZONS),
        help="Forward return label horizons in calendar days",
    )
    parser.add_argument("--no-btc-excess", action="store_true", help="Skip BTC-excess labels")
    parser.add_argument("--min-bars", type=int, default=MIN_BARS_FEATURES)
    parser.add_argument("--min-dates", type=int, default=MIN_BARS_PANEL)
    parser.add_argument("--min-pairs", type=int, default=2)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = CryptoPanelConfig(
        crypto_ohlcv_dir=args.crypto_ohlcv_dir,
        output_path=args.output,
        label_horizons=tuple(args.horizons),
        btc_excess=not args.no_btc_excess,
        min_bars=args.min_bars,
        min_panel_dates=args.min_dates,
        min_pairs=args.min_pairs,
        pairs=args.pairs,
    )
    build_crypto_panel(cfg)


if __name__ == "__main__":
    main()
