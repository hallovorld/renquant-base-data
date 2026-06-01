"""CLI for refreshing Alpaca options-IV parquet caches from base-data."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from renquant_base_data.alpaca_common import TokenBucket, load_strategy_watchlist


log = logging.getLogger("renquant_base_data.options_iv_refresh")

OPTIONS_IV_DIRNAME = "options_iv_alpaca"
_OCC_RE = re.compile(r"^([A-Z]{1,6})_?(\d{6})([CP])(\d{8})$")


def parse_occ(occ: str) -> dict[str, object] | None:
    match = _OCC_RE.match(str(occ).upper())
    if not match:
        return None
    underlying, expiry_raw, option_type, strike_raw = match.groups()
    try:
        expiry = date(2000 + int(expiry_raw[:2]), int(expiry_raw[2:4]), int(expiry_raw[4:6]))
        strike = int(strike_raw) / 1000.0
    except ValueError:
        return None
    return {
        "underlying": underlying,
        "expiry": expiry,
        "option_type": option_type,
        "strike": strike,
    }


def nearest_atm_iv(
    contracts: list[dict[str, object]],
    target_dte: int,
    option_type: str,
    spot: float,
    *,
    today: date | None = None,
    dte_tolerance: int = 10,
) -> tuple[float, int, float] | None:
    today = today or date.today()
    candidates = [
        contract
        for contract in contracts
        if contract.get("option_type") == option_type
        and abs((contract["expiry"] - today).days - target_dte) <= dte_tolerance
        and contract.get("iv") is not None
        and float(contract["iv"]) > 0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda contract: abs((contract["expiry"] - today).days - target_dte))
    nearest_dte = (candidates[0]["expiry"] - today).days
    same_expiry = [contract for contract in candidates if (contract["expiry"] - today).days == nearest_dte]
    atm = min(same_expiry, key=lambda contract: abs(float(contract["strike"]) - spot))
    return float(atm["iv"]), int(nearest_dte), float(atm["strike"])


def fetch_spot(symbol: str) -> float | None:
    try:
        import yfinance as yf

        history = yf.Ticker(symbol).history(period="1d", auto_adjust=True)
        if history.empty:
            return None
        return float(history["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return None


def fetch_iv_features(
    client,
    symbol: str,
    spot: float,
    bucket: TokenBucket,
    *,
    today: date | None = None,
) -> dict[str, object] | None:
    from alpaca.data.requests import OptionChainRequest

    today = today or date.today()
    bucket.acquire()
    backoff = 1.0
    for attempt in range(5):
        try:
            chain = client.get_option_chain(OptionChainRequest(underlying_symbol=symbol))
            break
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "rate" in message.lower() or "429" in message:
                log.warning("%s: rate-limited on attempt %d; backoff %.1fs", symbol, attempt + 1, backoff)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            log.warning("%s: option chain fetch failed: %s", symbol, exc)
            return None
    else:
        return None

    contracts: list[dict[str, object]] = []
    for occ, snapshot in (chain or {}).items():
        parsed = parse_occ(occ)
        if parsed is None:
            continue
        iv = getattr(snapshot, "implied_volatility", None)
        if iv is None or float(iv) <= 0:
            continue
        parsed["iv"] = float(iv)
        contracts.append(parsed)
    if not contracts:
        return None

    c30 = nearest_atm_iv(contracts, 30, "C", spot, today=today)
    p30 = nearest_atm_iv(contracts, 30, "P", spot, today=today)
    c60 = nearest_atm_iv(contracts, 60, "C", spot, today=today)
    p60 = nearest_atm_iv(contracts, 60, "P", spot, today=today)

    iv_30d_call = c30[0] if c30 else np.nan
    iv_30d_put = p30[0] if p30 else np.nan
    iv_60d_call = c60[0] if c60 else np.nan
    iv_60d_put = p60[0] if p60 else np.nan
    iv_skew_30d = iv_30d_put - iv_30d_call if not np.isnan(iv_30d_put) and not np.isnan(iv_30d_call) else np.nan

    iv_term_struct = np.nan
    if not np.isnan(iv_60d_call) and not np.isnan(iv_30d_call):
        iv_30d_atm = (iv_30d_call + iv_30d_put) / 2 if not np.isnan(iv_30d_put) else iv_30d_call
        iv_60d_atm = (iv_60d_call + iv_60d_put) / 2 if not np.isnan(iv_60d_put) else iv_60d_call
        iv_term_struct = iv_60d_atm - iv_30d_atm

    return {
        "symbol": symbol,
        "as_of": today.isoformat(),
        "spot": float(spot),
        "iv_30d_call_atm": iv_30d_call,
        "iv_30d_put_atm": iv_30d_put,
        "iv_60d_call_atm": iv_60d_call,
        "iv_60d_put_atm": iv_60d_put,
        "iv_skew_30d": iv_skew_30d,
        "iv_term_struct": iv_term_struct,
        "dte_30": c30[1] if c30 else None,
        "dte_60": c60[1] if c60 else None,
        "n_valid_iv_contracts": len(contracts),
    }


def merge_iv_snapshot(prior: pd.DataFrame | None, new_row: dict[str, object]) -> pd.DataFrame:
    new = pd.DataFrame([new_row])
    if prior is not None and not prior.empty:
        new = pd.concat([prior, new], ignore_index=True)
    return new.drop_duplicates(subset=["symbol", "as_of"], keep="last").sort_values("as_of").reset_index(drop=True)


def refresh_options_iv(
    *,
    symbols: list[str],
    data_dir: str | Path,
    client=None,
    spot_fn: Callable[[str], float | None] = fetch_spot,
    feature_fn: Callable[..., dict[str, object] | None] = fetch_iv_features,
    rate_limit: int = 180,
) -> dict[str, object]:
    out_dir = Path(data_dir).expanduser().resolve() / OPTIONS_IV_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    bucket = TokenBucket(max_calls=rate_limit, window_seconds=60.0)
    per_symbol: dict[str, dict[str, object]] = {}

    for symbol in [s.upper() for s in symbols]:
        out_path = out_dir / f"{symbol}.parquet"
        spot = spot_fn(symbol)
        if spot is None:
            per_symbol[symbol] = {"status": "missing_spot", "rows": 0, "path": str(out_path)}
            continue
        features = feature_fn(client, symbol, float(spot), bucket)
        if features is None:
            per_symbol[symbol] = {"status": "empty_chain", "rows": 0, "path": str(out_path)}
            continue
        prior = pd.read_parquet(out_path) if out_path.exists() else None
        merged = merge_iv_snapshot(prior, features)
        merged.to_parquet(out_path, index=False)
        per_symbol[symbol] = {"status": "ok", "rows": int(len(merged)), "path": str(out_path)}

    non_empty = sum(1 for item in per_symbol.values() if int(item["rows"]) > 0)
    return {
        "ok": True,
        "n_symbols": int(len(symbols)),
        "non_empty": int(non_empty),
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "per_symbol": per_symbol,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy-config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--rate-limit", type=int, default=180)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    if args.symbols is None and args.strategy_config is None:
        raise SystemExit("--symbols or --strategy-config is required")
    symbols = (
        [symbol.upper() for symbol in args.symbols]
        if args.symbols is not None
        else load_strategy_watchlist(args.strategy_config)
    )
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not in env")
        return 3
    from alpaca.data.historical.option import OptionHistoricalDataClient

    summary = refresh_options_iv(
        symbols=symbols,
        data_dir=args.data_dir,
        client=OptionHistoricalDataClient(api_key=key, secret_key=secret),
        rate_limit=args.rate_limit,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
