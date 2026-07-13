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
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning("%s: missing OHLCV columns %s, skipping", slug, missing)
        return None
    df = df[required].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


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
) -> pd.DataFrame:
    """Compute raw and optionally BTC-excess forward returns.

    Labels are in UTC calendar days: both the pair and BTC close series
    are reindexed to a complete daily calendar before shifting, so
    ``shift(-n)`` always means exactly *n* calendar days forward
    regardless of missing observations.  Gaps beyond ``FFILL_LIMIT_DAYS``
    produce NaN (no silent extrapolation).
    """
    c = ohlcv["close"].sort_index()
    max_horizon = max(horizons)
    cal = pd.date_range(
        c.index.min(),
        c.index.max() + pd.Timedelta(days=max_horizon),
        freq="D",
    )
    c_cal = c.reindex(cal, method="ffill", limit=FFILL_LIMIT_DAYS)

    real_obs = c.reindex(cal).notna()

    output_dates = c.index
    rec: dict[str, pd.Series] = {
        "pair": pd.Series(slug, index=output_dates),
        "date": pd.Series(output_dates, index=output_dates),
    }
    for n in horizons:
        fwd = c_cal.shift(-n) / c_cal - 1
        terminal_has_real = real_obs.shift(-n).reindex(output_dates).fillna(False).infer_objects(copy=False)
        fwd_out = fwd.reindex(output_dates)
        fwd_out = fwd_out.where(terminal_has_real, other=np.nan)
        rec[f"fwd_{n}d"] = fwd_out
        if btc_close is not None:
            btc_cal = btc_close.sort_index().reindex(
                cal, method="ffill", limit=FFILL_LIMIT_DAYS,
            )
            btc_real = btc_close.sort_index().reindex(cal).notna()
            fwd_btc = btc_cal.shift(-n) / btc_cal - 1
            btc_terminal_has_real = btc_real.shift(-n).reindex(output_dates).fillna(False).infer_objects(copy=False)
            excess = fwd_out - fwd_btc.reindex(output_dates)
            excess = excess.where(terminal_has_real & btc_terminal_has_real, other=np.nan)
            rec[f"fwd_{n}d_btc_excess"] = excess
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
    if cfg.btc_excess:
        btc_ohlcv = _load_ohlcv(store, BTC_SLUG)
        if btc_ohlcv is None:
            log.warning("BTC-USD bars not found; BTC-excess labels will be skipped")
        else:
            btc_close = btc_ohlcv["close"].sort_index().rename("btc_close")

    feature_rows: list[pd.DataFrame] = []
    label_rows: list[pd.DataFrame] = []
    for slug in slugs:
        feats = build_features_for_pair(slug, store, min_bars=cfg.min_bars)
        if feats is None:
            continue
        feature_rows.append(feats)
        ohlcv = _load_ohlcv(store, slug)
        if ohlcv is not None:
            labels = compute_forward_returns(
                ohlcv, slug,
                horizons=cfg.label_horizons,
                btc_close=btc_close if slug != BTC_SLUG else None,
            )
            label_rows.append(labels)

    if not feature_rows:
        raise RuntimeError("No pairs produced alpha158 features")

    features_panel = pd.concat(feature_rows, ignore_index=True)
    feature_cols = [c for c in features_panel.columns if c not in ("pair", "date")]
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
        "label_cols": [c for c in panel.columns if c.startswith("fwd_")],
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
        },
        "btc_excess": has_btc_excess,
    }
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
