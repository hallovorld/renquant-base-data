"""Tests for crypto bars ingestion + UTC-session watermarks (crypto RFC D-C2).

No live API dependency anywhere: Alpaca payloads are faked/injected, the
yfinance cross-check uses an injected secondary fetcher, and the only
alpaca-py-dependent test skips when the SDK is absent (CI installs neither
alpaca-py nor openbb).
"""
from __future__ import annotations

import hashlib
import importlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data.crypto_bars import (
    BAR_CLOSE_COL,
    CryptoLocalStore,
    ManifestNotSignalEligibleError,
    VendorDailyNotUtcAlignedError,
    _content_sha256,
    bars_eligible_for_session,
    build_crypto_features_for_pair,
    crosscheck_daily_close,
    crypto_manifest_path,
    fetch_crypto_daily_cached,
    ingest_crypto_bars,
    last_completed_utc_session,
    load_crypto_ingestion_manifest,
    manifest_eligible_for_session,
    manifest_fingerprint,
    normalize_daily_bars_utc,
    pair_slug,
    resample_hourly_to_utc_daily,
    run_yfinance_crosscheck,
    session_watermark_utc,
    slug_pair,
    verify_crypto_manifest,
)


# ---------------------------------------------------------------------------
# Symbol policy (RFC §3.0): slash pair <-> dash slug, round-trip pinned
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("pair", "slug"),
    [
        ("BTC/USD", "BTC-USD"),
        ("ETH/USD", "ETH-USD"),
        ("USDT/USD", "USDT-USD"),
        ("DOGE/USD", "DOGE-USD"),
    ],
)
def test_pair_slug_round_trip(pair: str, slug: str) -> None:
    assert pair_slug(pair) == slug
    assert slug_pair(slug) == pair
    assert slug_pair(pair_slug(pair)) == pair
    assert pair_slug(slug_pair(slug)) == slug


def test_pair_slug_normalizes_case_and_whitespace() -> None:
    assert pair_slug(" btc/usd ") == "BTC-USD"
    assert slug_pair(" eth-usd ") == "ETH/USD"


@pytest.mark.parametrize("bad", ["BTCUSD", "BTC/USD/X", "BTC-USD", "", "/USD", "BTC/"])
def test_pair_slug_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        pair_slug(bad)


@pytest.mark.parametrize("bad", ["BTC/USD", "BTCUSD", "BTC-USD-X", "", "-USD", "BTC-"])
def test_slug_pair_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        slug_pair(bad)


def test_ingest_rejects_malformed_pair_fast(tmp_path: Path) -> None:
    # Found by CLI probing: "BTC/USD/X" must fail at symbol validation,
    # not silently produce a no_data manifest entry.
    with pytest.raises(ValueError, match="BTC/USD/X"):
        ingest_crypto_bars(
            ["BTC/USD/X"],
            store=CryptoLocalStore(tmp_path),
            fetch_fn=lambda *a, **k: {},
            now_fn=lambda: _NOW,
        )


def test_store_path_is_slug_never_nested(tmp_path: Path) -> None:
    store = CryptoLocalStore(tmp_path)
    for symbol in ("BTC/USD", "btc/usd", "BTC-USD"):
        path = store._path(symbol)  # noqa: SLF001
        assert path == tmp_path / "BTC-USD" / "1d.parquet"
        # The B5 break: a slash symbol must never become a nested directory.
        assert path.parent.parent == tmp_path
    assert store._path("ETH/USD", "1h") == tmp_path / "ETH-USD" / "1h.parquet"  # noqa: SLF001


def test_module_imports_without_alpaca() -> None:
    # CI installs neither alpaca-py nor openbb; the module must import clean.
    assert importlib.import_module("renquant_base_data.crypto_bars") is not None


# ---------------------------------------------------------------------------
# 24/7 session semantics (RFC §3.5): UTC-day watermark
# ---------------------------------------------------------------------------

def test_session_watermark_is_utc_midnight() -> None:
    wm = session_watermark_utc(date(2026, 7, 10))
    assert wm == pd.Timestamp("2026-07-10T00:00:00Z")
    assert session_watermark_utc("2026-07-10") == wm
    with pytest.raises(ValueError):
        session_watermark_utc("2026-07-10T05:00:00")


def test_last_completed_utc_session() -> None:
    ref = pd.Timestamp("2026-07-10T00:00:01Z")
    assert last_completed_utc_session(ref) == date(2026, 7, 9)
    # One second before midnight, day 07-09 is not complete yet.
    assert last_completed_utc_session(pd.Timestamp("2026-07-09T23:59:59Z")) == date(2026, 7, 8)


def _daily_frame(days: list[str], *, close_offset_days: int = 1) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in days], name="timestamp")
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(len(days))],
            "high": [101.0 + i for i in range(len(days))],
            "low": [99.0 + i for i in range(len(days))],
            "close": [100.5 + i for i in range(len(days))],
            "volume": [10.0 + i for i in range(len(days))],
        },
        index=idx,
    )
    df[BAR_CLOSE_COL] = df.index + pd.Timedelta(days=close_offset_days)
    return df


def test_bars_eligible_for_session_boundary_semantics() -> None:
    # Bars for UTC days 07-07..07-09; closes 07-08..07-10 00:00.
    df = _daily_frame(["2026-07-07", "2026-07-08", "2026-07-09"])
    eligible = bars_eligible_for_session(df, date(2026, 7, 10))
    # Day 07-09's bar closes exactly AT the session-D watermark -> eligible.
    assert list(eligible.index) == list(df.index)
    eligible_d9 = bars_eligible_for_session(df, date(2026, 7, 9))
    # Session 07-09 must not see the bar closing 07-10 00:00.
    assert list(eligible_d9.index) == list(df.index[:2])


def test_bars_eligible_requires_ingestion_stamp() -> None:
    df = _daily_frame(["2026-07-07"]).drop(columns=[BAR_CLOSE_COL])
    with pytest.raises(ValueError, match=BAR_CLOSE_COL):
        bars_eligible_for_session(df, date(2026, 7, 10))


# ---------------------------------------------------------------------------
# UTC-day keying: vendor-aligned path + fail-closed hourly fallback
# ---------------------------------------------------------------------------

def _vendor_daily(days: list[str], *, hour: int = 0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=hour) for d in days],
        name="timestamp",
    )
    n = len(days)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [110.0 + i for i in range(n)],
            "low": [90.0 + i for i in range(n)],
            "close": [105.0 + i for i in range(n)],
            "volume": [1000.0 + i for i in range(n)],
        },
        index=idx,
    )


def test_normalize_daily_bars_utc_stamps_and_seals() -> None:
    df = _vendor_daily(["2026-07-07", "2026-07-08", "2026-07-09"])
    # As of 07-09 12:00 UTC the 07-09 bar (close 07-10 00:00) is in progress.
    out = normalize_daily_bars_utc(
        df, fetched_through_utc=pd.Timestamp("2026-07-09T12:00:00Z")
    )
    assert list(out.index) == [
        pd.Timestamp("2026-07-07", tz="UTC"),
        pd.Timestamp("2026-07-08", tz="UTC"),
    ]
    assert list(out[BAR_CLOSE_COL]) == [
        pd.Timestamp("2026-07-08", tz="UTC"),
        pd.Timestamp("2026-07-09", tz="UTC"),
    ]


def test_normalize_daily_bars_utc_fails_closed_on_misalignment() -> None:
    df = _vendor_daily(["2026-07-07", "2026-07-08"], hour=5)  # 05:00Z ≈ NY-midnight keying
    with pytest.raises(VendorDailyNotUtcAlignedError):
        normalize_daily_bars_utc(df, fetched_through_utc=pd.Timestamp("2026-07-09T12:00:00Z"))


def test_resample_hourly_to_utc_daily_aggregation_and_sealing() -> None:
    hours = pd.date_range("2026-07-07T00:00Z", "2026-07-09T06:00Z", freq="1h")
    df = pd.DataFrame(
        {
            "open": [float(i) for i in range(len(hours))],
            "high": [float(i) + 0.5 for i in range(len(hours))],
            "low": [float(i) - 0.5 for i in range(len(hours))],
            "close": [float(i) + 0.25 for i in range(len(hours))],
            "volume": [1.0] * len(hours),
        },
        index=pd.DatetimeIndex(hours, name="timestamp"),
    )
    out = resample_hourly_to_utc_daily(
        df, fetched_through_utc=pd.Timestamp("2026-07-09T06:30:00Z")
    )
    # Days 07-07 and 07-08 complete; 07-09 partial (7 bars) -> excluded.
    assert list(out.index) == [
        pd.Timestamp("2026-07-07", tz="UTC"),
        pd.Timestamp("2026-07-08", tz="UTC"),
    ]
    d1 = out.loc[pd.Timestamp("2026-07-07", tz="UTC")]
    assert d1["open"] == 0.0  # first hourly open of the day
    assert d1["close"] == 23.25  # last hourly close of the day
    assert d1["high"] == 23.5
    assert d1["low"] == -0.5
    assert d1["volume"] == 24.0
    assert d1["n_source_bars"] == 24
    assert d1[BAR_CLOSE_COL] == pd.Timestamp("2026-07-08", tz="UTC")


def _hourly_frame(hours: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(hours)
    return pd.DataFrame(
        {
            "open": [1.0] * n,
            "high": [1.5] * n,
            "low": [0.5] * n,
            "close": [1.2] * n,
            "volume": [1.0] * n,
        },
        index=pd.DatetimeIndex(hours, name="timestamp"),
    )


def test_resample_hourly_drops_a_day_with_fewer_than_24_hours() -> None:
    # 07-07 has all 24 hours; 07-08 is missing hour 12 (23 hours) — both
    # days are within the fetched-through window, but 07-08 must still be
    # dropped: a <24-hour day is incomplete market data, not a valid sealed
    # day, regardless of the fetch cutoff (Codex review round 1 on #41).
    hours = pd.date_range("2026-07-07T00:00Z", "2026-07-08T23:00Z", freq="1h")
    hours = hours[hours != pd.Timestamp("2026-07-08T12:00Z")]
    df = _hourly_frame(hours)
    out = resample_hourly_to_utc_daily(
        df, fetched_through_utc=pd.Timestamp("2026-07-09T00:00:00Z")
    )
    assert list(out.index) == [pd.Timestamp("2026-07-07", tz="UTC")]


def test_resample_hourly_drops_a_day_with_a_mid_day_gap_even_at_24_rows() -> None:
    # 24 rows present, but hour 5 is duplicated and hour 17 is missing — the
    # naive "count == 24" check would pass this; the hour-SET check must not.
    hours = list(pd.date_range("2026-07-07T00:00Z", "2026-07-07T23:00Z", freq="1h"))
    hours.remove(pd.Timestamp("2026-07-07T17:00Z"))
    hours.append(pd.Timestamp("2026-07-07T05:00Z"))  # duplicate of an existing hour
    df = _hourly_frame(pd.DatetimeIndex(sorted(hours)))
    assert len(df) == 24  # same row count as a genuinely complete day
    out = resample_hourly_to_utc_daily(
        df, fetched_through_utc=pd.Timestamp("2026-07-08T00:00:00Z")
    )
    assert list(out.index) == []  # gap must still be caught


def test_resample_hourly_exactly_24_contiguous_hours_succeeds() -> None:
    hours = pd.date_range("2026-07-07T00:00Z", "2026-07-07T23:00Z", freq="1h")
    assert len(hours) == 24
    df = _hourly_frame(hours)
    out = resample_hourly_to_utc_daily(
        df, fetched_through_utc=pd.Timestamp("2026-07-08T00:00:00Z")
    )
    assert list(out.index) == [pd.Timestamp("2026-07-07", tz="UTC")]


def test_resample_hourly_drops_a_25_row_day_with_a_duplicated_hour(
) -> None:
    # Codex review round 2: all 24 distinct hours ARE present (the hour-SET
    # check alone would pass this), but hour 5 additionally appears TWICE,
    # so the row count is 25. A set-only completeness check would silently
    # emit this day with hour 5's volume double-counted. Both the row-count
    # check AND the hour-set check must be required together.
    hours = list(pd.date_range("2026-07-07T00:00Z", "2026-07-07T23:00Z", freq="1h"))
    hours.append(pd.Timestamp("2026-07-07T05:00Z"))  # duplicate of an existing hour
    df = _hourly_frame(pd.DatetimeIndex(sorted(hours)))
    assert len(df) == 25
    assert frozenset(df.index.hour) == frozenset(range(24))  # set check alone would pass
    out = resample_hourly_to_utc_daily(
        df, fetched_through_utc=pd.Timestamp("2026-07-08T00:00:00Z")
    )
    assert list(out.index) == []  # duplicate-inflated day must still be dropped


# ---------------------------------------------------------------------------
# Store freshness on the UTC-day clock (B3)
# ---------------------------------------------------------------------------

def test_store_utc_freshness(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    now = pd.Timestamp(datetime.now(timezone.utc))
    yesterday = (now.normalize() - pd.Timedelta(days=1)).tz_localize(None)

    fresh = _daily_frame([str((yesterday - pd.Timedelta(days=i)).date()) for i in (2, 1, 0)])
    store.save(fresh, "BTC/USD", "1d")
    assert store.has_range("BTC/USD") is True
    assert store.has_range("BTC-USD") is True  # both canonical forms accepted

    stale = _daily_frame([str((yesterday - pd.Timedelta(days=i)).date()) for i in (5, 4, 3)])
    store.save(stale, "ETH/USD", "1d")
    assert store.has_range("ETH/USD") is False
    assert store.has_range("ETH/USD", tolerance_days=10) is True
    # Coverage check with a tz-naive start string must not raise.
    assert store.has_range("BTC/USD", start=str(yesterday.date())) is True


# ---------------------------------------------------------------------------
# Ingestion: store writes + sealed manifest (watermark contract)
# ---------------------------------------------------------------------------

_NOW = pd.Timestamp("2026-07-09T12:00:00Z")


def _fake_fetch_aligned(pairs, *, timeframe="1Day", start=None, end=None, **_kw):
    assert timeframe == "1Day"
    out = {}
    if "BTC/USD" in pairs:
        out["BTC/USD"] = _vendor_daily(["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"])
    if "ETH/USD" in pairs:
        out["ETH/USD"] = _vendor_daily(["2026-07-06", "2026-07-07"])
    return out


def test_ingest_daily_writes_slug_store_and_watermark_manifest(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    summary = ingest_crypto_bars(
        ["BTC/USD", "ETH/USD"],
        timeframe="1Day",
        store=store,
        fetch_fn=_fake_fetch_aligned,
        now_fn=lambda: _NOW,
    )

    # Store layout: slug path, UTC-midnight keyed, bar_close stamped.
    btc_path = tmp_path / "BTC-USD" / "1d.parquet"
    assert btc_path.exists()
    btc = pd.read_parquet(btc_path)
    assert list(btc.index) == [
        pd.Timestamp(d, tz="UTC") for d in ("2026-07-06", "2026-07-07", "2026-07-08")
    ]  # 07-09 bar closes 07-10 00:00 > fetch time -> not sealed, not stored
    assert BAR_CLOSE_COL in btc.columns

    # Manifest: per-symbol close stamps + global watermark = min over symbols.
    manifest_file = crypto_manifest_path(store, "1Day")
    assert manifest_file == tmp_path / "ingestion_manifest_1d.json"
    payload = load_crypto_ingestion_manifest(manifest_file)
    assert payload["asset_class"] == "crypto"
    assert payload["dataset_id"] == "crypto-ohlcv-1d"
    assert payload["provider"] == "alpaca:v1beta3"
    btc_info = payload["symbols"]["BTC/USD"]
    eth_info = payload["symbols"]["ETH/USD"]
    assert btc_info["slug"] == "BTC-USD"
    assert btc_info["path"] == "BTC-USD/1d.parquet"
    assert pd.Timestamp(btc_info["last_bar_close_utc"]) == pd.Timestamp("2026-07-09", tz="UTC")
    assert pd.Timestamp(eth_info["last_bar_close_utc"]) == pd.Timestamp("2026-07-08", tz="UTC")
    assert pd.Timestamp(payload["watermark_utc"]) == pd.Timestamp("2026-07-08", tz="UTC")

    # The manifest is the summary (single artifact, no drift).
    assert summary["fingerprint"] == payload["fingerprint"]

    # It satisfies the repo's dataset-manifest schema (B6).
    from renquant_base_data import validate_data_manifest

    report = validate_data_manifest(payload)
    assert report["ok"] is True

    # Universe completeness (Codex review round 1 on #41): both requested
    # pairs sealed an 'ok' bar, so this manifest IS signal-eligible.
    assert payload["expected_universe"] == ["BTC/USD", "ETH/USD"]
    assert payload["universe_complete"] is True
    assert payload["signal_eligible"] is True


def test_ingest_partial_universe_fetch_is_not_signal_eligible(tmp_path: Path) -> None:
    """Codex review round 1 on #41: a manifest must not falsely claim an
    actionable watermark for an incomplete universe. Request two pairs, make
    one fail entirely (no data) — the resulting manifest must record the
    successful pair's watermark for OPS VISIBILITY only, but the manifest as
    a whole must be marked incomplete/not-signal-eligible, and no session
    manifest built from it may be consumed."""
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)

    def fake_fetch(pairs, *, timeframe="1Day", start=None, end=None, **_kw):
        assert timeframe == "1Day"
        out = {}
        if "BTC/USD" in pairs:
            out["BTC/USD"] = _vendor_daily(["2026-07-06", "2026-07-07", "2026-07-08"])
        # ETH/USD deliberately absent -> "no_data" status.
        return out

    payload = ingest_crypto_bars(
        ["BTC/USD", "ETH/USD"], store=store, fetch_fn=fake_fetch, now_fn=lambda: _NOW
    )
    assert payload["symbols"]["BTC/USD"]["status"] == "ok"
    assert payload["symbols"]["ETH/USD"]["status"] == "no_data"
    assert payload["expected_universe"] == ["BTC/USD", "ETH/USD"]
    assert payload["universe_complete"] is False
    assert payload["signal_eligible"] is False
    # The BTC watermark is still recorded for ops diagnostics (last BTC bar
    # 07-08 closes 07-09 00:00Z) ...
    assert pd.Timestamp(payload["watermark_utc"]) == pd.Timestamp("2026-07-09", tz="UTC")
    # ... but must not be usable as a session manifest: the eligibility gate
    # rejects it outright, regardless of the (diagnostic-only) watermark.
    with pytest.raises(ManifestNotSignalEligibleError, match="incomplete"):
        manifest_eligible_for_session(payload, date(2026, 7, 9))


def test_ingest_manifest_fingerprint_tamper_evident(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    ingest_crypto_bars(
        ["BTC/USD"], store=store, fetch_fn=_fake_fetch_aligned, now_fn=lambda: _NOW
    )
    manifest_file = crypto_manifest_path(store, "1Day")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert verify_crypto_manifest(payload) is True

    tampered = dict(payload)
    tampered["watermark_utc"] = "2026-07-10T00:00:00+00:00"  # stale-signal laundering attempt
    assert verify_crypto_manifest(tampered) is False
    manifest_file.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_crypto_ingestion_manifest(manifest_file)


def test_ingest_is_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    p1 = ingest_crypto_bars(
        ["BTC/USD"], store=store, fetch_fn=_fake_fetch_aligned, now_fn=lambda: _NOW
    )
    p2 = ingest_crypto_bars(
        ["BTC/USD"], store=store, fetch_fn=_fake_fetch_aligned, now_fn=lambda: _NOW
    )
    # Same bars + same clock -> identical content sha AND identical manifest
    # fingerprint (a parquet rewrite must not change the sealed identity).
    assert p1["symbols"]["BTC/USD"]["content_sha256"] == p2["symbols"]["BTC/USD"]["content_sha256"]
    assert p1["fingerprint"] == p2["fingerprint"]


def test_ingest_daily_misaligned_vendor_falls_back_to_hourly(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    calls: list[str] = []

    def fake_fetch(pairs, *, timeframe="1Day", start=None, end=None, **_kw):
        calls.append(timeframe)
        if timeframe == "1Day":
            # NY-midnight-style keying: NOT UTC-aligned -> must not be trusted.
            return {"BTC/USD": _vendor_daily(["2026-07-07", "2026-07-08"], hour=5)}
        assert timeframe == "1Hour"
        hours = pd.date_range("2026-07-07T00:00Z", "2026-07-08T23:00Z", freq="1h")
        return {
            "BTC/USD": pd.DataFrame(
                {
                    "open": [1.0] * len(hours),
                    "high": [2.0] * len(hours),
                    "low": [0.5] * len(hours),
                    "close": [1.5] * len(hours),
                    "volume": [3.0] * len(hours),
                },
                index=pd.DatetimeIndex(hours, name="timestamp"),
            )
        }

    payload = ingest_crypto_bars(
        ["BTC/USD"], store=store, fetch_fn=fake_fetch, now_fn=lambda: _NOW
    )
    assert calls == ["1Day", "1Hour"]
    info = payload["symbols"]["BTC/USD"]
    assert info["status"] == "ok"
    assert info["source"] == "alpaca-1hour-resampled-utc"
    stored = pd.read_parquet(tmp_path / "BTC-USD" / "1d.parquet")
    assert list(stored.index) == [
        pd.Timestamp("2026-07-07", tz="UTC"),
        pd.Timestamp("2026-07-08", tz="UTC"),
    ]
    assert stored["n_source_bars"].tolist() == [24, 24]


def test_ingest_intraday_stamps_bar_close_and_drops_in_progress(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    hours = pd.date_range("2026-07-09T09:00Z", "2026-07-09T12:00Z", freq="1h")

    def fake_fetch(pairs, *, timeframe="1Hour", start=None, end=None, **_kw):
        assert timeframe == "1Hour"
        return {
            "BTC/USD": pd.DataFrame(
                {
                    "open": [1.0] * len(hours),
                    "high": [1.0] * len(hours),
                    "low": [1.0] * len(hours),
                    "close": [1.0] * len(hours),
                    "volume": [1.0] * len(hours),
                },
                index=pd.DatetimeIndex(hours, name="timestamp"),
            )
        }

    payload = ingest_crypto_bars(
        ["BTC/USD"],
        timeframe="1Hour",
        store=store,
        fetch_fn=fake_fetch,
        now_fn=lambda: _NOW,  # 12:00Z -> the 12:00 bar (close 13:00) is in progress
    )
    stored = pd.read_parquet(tmp_path / "BTC-USD" / "1h.parquet")
    assert list(stored.index) == list(hours[:3])
    assert list(pd.to_datetime(stored[BAR_CLOSE_COL])) == [
        h + pd.Timedelta(hours=1) for h in hours[:3]
    ]
    assert pd.Timestamp(payload["watermark_utc"]) == pd.Timestamp("2026-07-09T12:00:00Z")
    assert crypto_manifest_path(store, "1Hour").exists()
    assert payload["dataset_id"] == "crypto-ohlcv-1h"


def test_manifest_fingerprint_helper_round_trip() -> None:
    payload = {"a": 1, "b": "x"}
    payload["fingerprint"] = manifest_fingerprint(payload)
    assert verify_crypto_manifest(payload) is True
    expected = "sha256:" + hashlib.sha256(
        json.dumps({"a": 1, "b": "x"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert payload["fingerprint"] == expected


# ---------------------------------------------------------------------------
# Manifest-BOUND eligibility (RFC §3.5, Codex review round 1 on #41): bar
# timestamps alone are not a leakage proof — the manifest's own identity,
# completeness, and generation time must all check out.
# ---------------------------------------------------------------------------

def _sealed_manifest_for(
    session_date: date,
    *,
    generated_at: "pd.Timestamp | None" = None,
    sealed_df: "pd.DataFrame | None" = None,
    symbol: str = "BTC/USD",
) -> dict:
    """A minimal, correctly-fingerprinted, complete, on-time manifest for
    the given session date — the happy path other tests mutate away from.

    ``sealed_df``, if given, has its real content_sha256 computed and stored
    under ``symbol`` (Codex review round 2: content-binding tests need a
    manifest whose sealed hash genuinely matches a real frame, not a
    placeholder string)."""
    watermark = session_watermark_utc(session_date)
    generated = generated_at if generated_at is not None else watermark + pd.Timedelta(minutes=5)
    symbol_entry: dict = {"status": "ok"}
    if sealed_df is not None:
        symbol_entry["content_sha256"] = _content_sha256(sealed_df)
    payload = {
        "dataset_id": "crypto-ohlcv-1d",
        "schema_version": "crypto-ohlcv-manifest-v1",
        "asset_class": "crypto",
        "provider": "alpaca:v1beta3",
        "timeframe": "1Day",
        "uri": "store://crypto_ohlcv/1d",
        "generated_at_utc": generated.isoformat(),
        "fetched_through_utc": generated.isoformat(),
        "expected_universe": [symbol],
        "expected_universe_hash": "sha256:x",
        "universe_complete": True,
        "watermark_utc": watermark.isoformat(),
        "signal_eligible": True,
        "symbols": {symbol: symbol_entry},
    }
    payload["fingerprint"] = manifest_fingerprint(payload)
    return payload


def test_manifest_eligible_for_session_happy_path() -> None:
    m = _sealed_manifest_for(date(2026, 7, 10))
    assert manifest_eligible_for_session(m, date(2026, 7, 10)) is None
    df = _daily_frame(["2026-07-09"])
    m_bound = _sealed_manifest_for(date(2026, 7, 10), sealed_df=df)
    out = manifest_eligible_for_session(m_bound, date(2026, 7, 10), df, symbol="BTC/USD")
    assert list(out.index) == list(df.index)


def test_manifest_eligible_for_session_requires_symbol_when_df_given() -> None:
    # Codex review round 2: without a declared symbol, content-binding
    # cannot be checked at all -- this must fail loud, not silently skip
    # the check.
    m = _sealed_manifest_for(date(2026, 7, 10))
    df = _daily_frame(["2026-07-09"])
    with pytest.raises(ValueError, match="symbol"):
        manifest_eligible_for_session(m, date(2026, 7, 10), df)


def test_manifest_eligible_for_session_rejects_tampered_rows() -> None:
    # Codex review round 2: an intact, correctly-fingerprinted, complete-
    # universe manifest -- but the df handed alongside it has been modified
    # after sealing (e.g. a mutated close price). Content-binding must
    # catch this even though every manifest-level check (1-5) passes.
    sealed_df = _daily_frame(["2026-07-09"])
    m = _sealed_manifest_for(date(2026, 7, 10), sealed_df=sealed_df)
    tampered_df = sealed_df.copy()
    tampered_df.loc[tampered_df.index[0], "close"] *= 1.01
    with pytest.raises(ManifestNotSignalEligibleError, match="does not match"):
        manifest_eligible_for_session(m, date(2026, 7, 10), tampered_df, symbol="BTC/USD")


def test_manifest_eligible_for_session_rejects_wrong_symbol_df() -> None:
    # Codex review round 2: manifest sealed for BTC/USD, but the df handed
    # in is actually ETH/USD's data (or any other symbol not covered by
    # this manifest's sealed entry) -- must be rejected even if internally
    # well-formed and even if it happens to have the right timestamps.
    btc_df = _daily_frame(["2026-07-09"])
    eth_df = _daily_frame(["2026-07-09"], close_offset_days=1)
    eth_df["close"] = eth_df["close"] * 7.3  # a different price series entirely
    m = _sealed_manifest_for(date(2026, 7, 10), sealed_df=btc_df)
    with pytest.raises(ManifestNotSignalEligibleError, match="does not match"):
        manifest_eligible_for_session(m, date(2026, 7, 10), eth_df, symbol="BTC/USD")


def test_manifest_eligible_for_session_rejects_unsealed_symbol() -> None:
    # symbol= names a pair the manifest never sealed at all.
    m = _sealed_manifest_for(date(2026, 7, 10))
    df = _daily_frame(["2026-07-09"])
    with pytest.raises(ManifestNotSignalEligibleError, match="no sealed"):
        manifest_eligible_for_session(m, date(2026, 7, 10), df, symbol="ETH/USD")


def test_manifest_eligible_for_session_rejects_tampered_fingerprint() -> None:
    m = _sealed_manifest_for(date(2026, 7, 10))
    # Mutate to a genuinely DIFFERENT value post-seal without recomputing the
    # fingerprint -- exactly what "tampered" means.
    m["watermark_utc"] = "2099-01-01T00:00:00+00:00"
    with pytest.raises(ManifestNotSignalEligibleError, match="fingerprint"):
        manifest_eligible_for_session(m, date(2026, 7, 10))


def test_manifest_eligible_for_session_rejects_incomplete_universe() -> None:
    m = _sealed_manifest_for(date(2026, 7, 10))
    m["universe_complete"] = False
    m["fingerprint"] = manifest_fingerprint(m)
    with pytest.raises(ManifestNotSignalEligibleError, match="incomplete"):
        manifest_eligible_for_session(m, date(2026, 7, 10))


def test_manifest_eligible_for_session_rejects_watermark_mismatch() -> None:
    # A perfectly valid manifest — but for the WRONG session.
    m = _sealed_manifest_for(date(2026, 7, 9))
    with pytest.raises(ManifestNotSignalEligibleError, match="does not match"):
        manifest_eligible_for_session(m, date(2026, 7, 10))


@pytest.mark.parametrize(
    "generated_offset_minutes",
    [-1, 15, 20, 60],  # before window open; at/after window close (exclusive upper bound)
)
def test_manifest_eligible_for_session_rejects_outside_signal_freeze_window(
    generated_offset_minutes: int,
) -> None:
    session = date(2026, 7, 10)
    watermark = session_watermark_utc(session)
    m = _sealed_manifest_for(
        session, generated_at=watermark + pd.Timedelta(minutes=generated_offset_minutes)
    )
    with pytest.raises(ManifestNotSignalEligibleError, match="cutoff window"):
        manifest_eligible_for_session(m, session)


@pytest.mark.parametrize("generated_offset_minutes", [0, 1, 14])
def test_manifest_eligible_for_session_accepts_inside_signal_freeze_window(
    generated_offset_minutes: int,
) -> None:
    session = date(2026, 7, 10)
    watermark = session_watermark_utc(session)
    m = _sealed_manifest_for(
        session, generated_at=watermark + pd.Timedelta(minutes=generated_offset_minutes)
    )
    assert manifest_eligible_for_session(m, session) is None


# ---------------------------------------------------------------------------
# Cache-first daily read (the provider seam's workhorse)
# ---------------------------------------------------------------------------

def test_fetch_crypto_daily_cached_serves_fresh_cache_without_network(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    now = pd.Timestamp(datetime.now(timezone.utc))
    days = [str((now.normalize() - pd.Timedelta(days=i)).date()) for i in (3, 2, 1)]
    store.save(_daily_frame(days), "BTC/USD", "1d")

    def forbidden_fetch(*_a, **_kw):
        raise AssertionError("fresh cache must not trigger a network fetch")

    df = fetch_crypto_daily_cached("BTC/USD", store=store, fetch_fn=forbidden_fetch)
    assert len(df) == 3


def test_fetch_crypto_daily_cached_ingests_when_stale(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    called = {"n": 0}
    now = pd.Timestamp(datetime.now(timezone.utc))
    days = [str((now.normalize() - pd.Timedelta(days=i)).date()) for i in (3, 2, 1)]

    def fake_fetch(pairs, *, timeframe="1Day", start=None, end=None, **_kw):
        called["n"] += 1
        assert timeframe == "1Day"
        return {"BTC/USD": _vendor_daily(days)}

    df = fetch_crypto_daily_cached("BTC/USD", store=store, fetch_fn=fake_fetch)
    assert called["n"] == 1
    assert len(df) == 3
    assert BAR_CLOSE_COL in df.columns
    # Ingestion also refreshed the sealed manifest.
    assert crypto_manifest_path(store, "1Day").exists()


# ---------------------------------------------------------------------------
# Provider seam + equity-path byte-identity pin
# ---------------------------------------------------------------------------

def test_fetch_ohlcv_alpaca_crypto_provider_delegates(monkeypatch, tmp_path: Path) -> None:
    from renquant_base_data.loaders.data import fetch_ohlcv

    sentinel = pd.DataFrame({"close": [1.0]})
    seen = {}

    def fake_cached(symbol, *, start=None, end=None, cache=True, timeout_sec=30.0):
        seen.update(symbol=symbol, start=start, end=end, cache=cache)
        return sentinel

    monkeypatch.setattr(
        "renquant_base_data.crypto_bars.fetch_crypto_daily_cached", fake_cached
    )
    out = fetch_ohlcv("BTC/USD", start="2026-01-01", provider="alpaca_crypto")
    assert out is sentinel
    assert seen["symbol"] == "BTC/USD"
    assert seen["start"] == "2026-01-01"


def test_fetch_ohlcv_unknown_provider_still_rejected() -> None:
    from renquant_base_data.loaders.data import fetch_ohlcv

    with pytest.raises(ValueError, match="Unknown provider"):
        fetch_ohlcv("AAPL", provider="not-a-provider")


def test_equity_daily_path_byte_identity(monkeypatch, tmp_path: Path) -> None:
    """Pin: the crypto provider is ADDITIVE — the equity yfinance path's
    cache-serve behavior and store layout are unchanged (no crypto namespace,
    no UTC clock, byte-identical frame out)."""
    pytest.importorskip("pyarrow")
    import renquant_base_data.loaders.data as data_mod

    store = data_mod.LocalStore(tmp_path)
    monkeypatch.setattr(data_mod, "_default_store", store)

    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    idx = pd.DatetimeIndex([today - pd.Timedelta(days=i) for i in (2, 1, 0)])
    seeded = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [10.0, 20.0, 30.0],
        },
        index=idx.sort_values(),
    )
    store.save(seeded, "AAPL")
    # Equity layout unchanged: {dir}/{SYMBOL}/1d.parquet, no slug, no crypto dir.
    assert (tmp_path / "AAPL" / "1d.parquet").exists()
    assert not (tmp_path / "crypto_ohlcv").exists()

    served = data_mod.fetch_ohlcv("AAPL")  # fresh cache -> no network
    pd.testing.assert_frame_equal(served, seeded)
    assert (
        hashlib.sha256(served.to_csv().encode()).hexdigest()
        == hashlib.sha256(seeded.to_csv().encode()).hexdigest()
    )


# ---------------------------------------------------------------------------
# Feature groundwork (B7): alpha158 price/volume subset reused, not forked
# ---------------------------------------------------------------------------

def test_crypto_features_reuse_alpha158_price_volume_ops(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from renquant_base_data.alpha158_ops import alpha158_feature_names
    from renquant_base_data.alpha158_qlib_panel import build_features_for_ticker

    store = CryptoLocalStore(tmp_path)
    rng = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    base = pd.Series(range(len(rng)), index=rng, dtype="float64")
    bars = pd.DataFrame(
        {
            "open": 100.0 + base,
            "high": 101.0 + base,
            "low": 99.0 + base,
            "close": 100.5 + base * 1.001,
            "volume": 1000.0 + (base % 7) * 13.0,
        },
        index=rng,
    )
    bars[BAR_CLOSE_COL] = bars.index + pd.Timedelta(days=1)
    store.save(bars, "BTC/USD", "1d")

    feats = build_crypto_features_for_pair("BTC/USD", tmp_path)
    assert feats is not None
    # Exactly the 158 price/volume alpha158 features — nothing fundamental,
    # nothing forked. ticker carries the slug (store directory key).
    assert set(feats.columns) - {"ticker", "date"} == set(alpha158_feature_names())
    assert (feats["ticker"] == "BTC-USD").all()

    # Identity with the shared equity builder on the same store: reuse, no fork.
    ref = build_features_for_ticker("BTC-USD", tmp_path)
    pd.testing.assert_frame_equal(feats, ref)


# ---------------------------------------------------------------------------
# Two-source parity (yfinance cross-check), no network
# ---------------------------------------------------------------------------

def test_crosscheck_daily_close_pass_and_breach() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2026-07-01", periods=5, freq="D", tz="UTC"))
    primary = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
    secondary = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=pd.DatetimeIndex(pd.date_range("2026-07-01", periods=5, freq="D")),  # naive ok
    )
    ok_report = crosscheck_daily_close(primary, secondary, rel_tol=0.01)
    assert ok_report["ok"] is True
    assert ok_report["n_overlap"] == 5
    assert ok_report["n_breaches"] == 0

    secondary.loc[secondary.index[2], "close"] = 110.0  # 7.5% divergence
    breach_report = crosscheck_daily_close(primary, secondary, rel_tol=0.01)
    assert breach_report["ok"] is False
    assert breach_report["n_breaches"] == 1
    assert breach_report["breach_dates"] == ["2026-07-03"]

    empty_report = crosscheck_daily_close(
        primary, pd.DataFrame({"close": []}, index=pd.DatetimeIndex([]))
    )
    assert empty_report["ok"] is False
    assert empty_report["n_overlap"] == 0


def test_run_yfinance_crosscheck_with_injected_secondary(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    store = CryptoLocalStore(tmp_path)
    days = ["2026-07-06", "2026-07-07", "2026-07-08"]
    store.save(_daily_frame(days), "BTC/USD", "1d")

    def fake_secondary(slug: str) -> pd.DataFrame:
        assert slug == "BTC-USD"  # slug form IS the yfinance ticker
        return pd.DataFrame(
            {"close": [100.5, 101.5, 102.5]},
            index=pd.DatetimeIndex(days),
        )

    report = run_yfinance_crosscheck("BTC/USD", store=store, fetch_secondary=fake_secondary)
    assert report["ok"] is True
    assert report["pair"] == "BTC/USD"
    assert report["slug"] == "BTC-USD"


# ---------------------------------------------------------------------------
# Registry entries (B6): asset_class="crypto" resolves via the existing schema
# ---------------------------------------------------------------------------

def test_committed_crypto_registry_manifests_resolve() -> None:
    from renquant_base_data import resolve_data_manifest

    manifests_dir = Path(__file__).resolve().parents[1] / "manifests"
    daily = resolve_data_manifest(manifests_dir, dataset_id="crypto-ohlcv-1d")
    assert daily["asset_class"] == "crypto"
    assert daily["uri"] == "store://crypto_ohlcv/1d"
    hourly = resolve_data_manifest(manifests_dir, dataset_id="crypto-ohlcv-1h")
    assert hourly["asset_class"] == "crypto"


# ---------------------------------------------------------------------------
# Raw provider fetch with a fake client (skips without alpaca-py; never live)
# ---------------------------------------------------------------------------

def test_fetch_crypto_bars_fake_client_pair_keying() -> None:
    pytest.importorskip("alpaca")
    from renquant_base_data.crypto_bars import fetch_crypto_bars

    idx = pd.MultiIndex.from_product(
        [["BTC/USD"], pd.date_range("2026-07-07", periods=2, freq="D", tz="UTC")],
        names=["symbol", "timestamp"],
    )
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1.0, 1.0],
        },
        index=idx,
    )

    class FakeBarSet:
        df = frame

    class FakeClient:
        def __init__(self) -> None:
            self.requests = []

        def get_crypto_bars(self, req):
            self.requests.append(req)
            return FakeBarSet()

    client = FakeClient()
    # Slug input must be translated to pair form for the API call.
    out = fetch_crypto_bars(["BTC-USD"], timeframe="1Day", client=client)
    assert list(out.keys()) == ["BTC/USD"]
    assert client.requests[0].symbol_or_symbols == ["BTC/USD"]
    assert out["BTC/USD"].index.tz is not None
    with pytest.raises(ValueError, match="Unknown crypto timeframe"):
        fetch_crypto_bars(["BTC/USD"], timeframe="2Day", client=client)
