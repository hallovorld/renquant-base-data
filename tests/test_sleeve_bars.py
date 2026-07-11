"""sleeve_bars — parking-sleeve equity daily bars (SGOV + SPY) contract.

Pins the base-data slice of the renquant-pipeline#185 (RS-1 SGOV floor)
data dependency: symbol normalization mirror-pin, ingestion + fingerprinted
manifest (tamper-evident, completeness-aware), the dry-run-default CLI, and
the pinned-artifact serving contract a downstream consumer resolves bars
through. No test touches the network or any production store, and no test
references any umbrella path.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data import sleeve_bars as sb
from renquant_base_data.loaders.data import LocalStore
from renquant_base_data.registry import resolve_data_manifest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _fresh_frame(days: int = 30) -> pd.DataFrame:
    """Daily bars ending 'now' in NY — always passes NYSE-session freshness."""
    end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    idx = pd.bdate_range(end=end, periods=days)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 1000},
        index=idx,
    )


def _saving_fetch(store: LocalStore):
    def fetch(symbol: str):
        store.save(_fresh_frame(), symbol)
    return fetch


# ---------------------------------------------------------------------------
# Symbol resolution — mirror-pinned against the umbrella normalization
# ---------------------------------------------------------------------------

class TestSleeveLegTickers:
    """EXACT mirror of adapters/sleeve_prices.parking_sleeve_leg_tickers.

    base-data is upstream of every consumer, so the umbrella helper cannot
    be imported here (dependency direction); these cases replicate that
    helper's pinned behaviors byte-for-byte. Any change must land in
    lockstep on both sides.
    """

    def test_defaults_regardless_of_enabled(self):
        assert sb.sleeve_leg_tickers(None) == ["SPY", "SGOV"]
        assert sb.sleeve_leg_tickers({}) == ["SPY", "SGOV"]
        assert sb.sleeve_leg_tickers({"sleeve": {"enabled": False}}) == ["SPY", "SGOV"]
        assert sb.sleeve_leg_tickers({"sleeve": {"enabled": True}}) == ["SPY", "SGOV"]

    def test_normalization(self):
        cfg = {"sleeve": {"spy_symbol": " ivv ", "sgov_symbol": "bil"}}
        assert sb.sleeve_leg_tickers(cfg) == ["IVV", "BIL"]

    def test_blank_symbols_fall_back_to_defaults(self):
        cfg = {"sleeve": {"spy_symbol": "  ", "sgov_symbol": ""}}
        assert sb.sleeve_leg_tickers(cfg) == ["SPY", "SGOV"]

    def test_identical_legs_deduped(self):
        cfg = {"sleeve": {"spy_symbol": "SGOV", "sgov_symbol": "SGOV"}}
        assert sb.sleeve_leg_tickers(cfg) == ["SGOV"]

    def test_malformed_section_uses_defaults(self):
        assert sb.sleeve_leg_tickers({"sleeve": "yes"}) == ["SPY", "SGOV"]
        assert sb.sleeve_leg_tickers({"sleeve": None}) == ["SPY", "SGOV"]

    def test_sgov_ticker_accessor(self):
        assert sb.sleeve_sgov_ticker({}) == "SGOV"
        assert sb.sleeve_sgov_ticker({"sleeve": {"sgov_symbol": "bil"}}) == "BIL"


# ---------------------------------------------------------------------------
# Ingestion + fingerprinted manifest
# ---------------------------------------------------------------------------

class TestIngestSleeveBars:
    def test_ingest_writes_store_and_sealed_manifest(self, tmp_path):
        store = LocalStore(tmp_path)
        summary = sb.ingest_sleeve_bars(store=store, fetch_fn=_saving_fetch(store))

        assert (tmp_path / "SGOV" / "1d.parquet").exists()
        assert (tmp_path / "SPY" / "1d.parquet").exists()
        manifest_file = tmp_path / "ingestion_manifest_sleeve_1d.json"
        assert manifest_file.exists()

        payload = sb.load_sleeve_ingestion_manifest(manifest_file)  # verifies fp
        assert payload == summary
        assert payload["dataset_id"] == "sleeve-ohlcv-1d"
        assert payload["expected_universe"] == ["SGOV", "SPY"]
        assert payload["universe_complete"] is True
        assert payload["serving_eligible"] is True
        for symbol in ("SGOV", "SPY"):
            info = payload["symbols"][symbol]
            assert info["status"] == "ok"
            assert info["fresh"] is True
            assert info["content_sha256"]
            # content sha matches an independent recompute from the parquet
            assert info["content_sha256"] == sb._content_sha256(store.load(symbol))

    def test_manifest_fingerprint_tamper_evident(self, tmp_path):
        store = LocalStore(tmp_path)
        sb.ingest_sleeve_bars(store=store, fetch_fn=_saving_fetch(store))
        manifest_file = sb.sleeve_manifest_path(store)
        payload = json.loads(manifest_file.read_text())
        assert sb.verify_sleeve_manifest(payload) is True

        tampered = dict(payload)
        tampered["serving_eligible"] = True
        tampered["universe_complete"] = True
        tampered["symbols"] = dict(payload["symbols"])
        tampered["symbols"]["SGOV"] = {**payload["symbols"]["SGOV"], "rows_total": 9999}
        assert sb.verify_sleeve_manifest(tampered) is False

        manifest_file.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            sb.load_sleeve_ingestion_manifest(manifest_file)

    def test_partial_universe_is_not_serving_eligible(self, tmp_path):
        store = LocalStore(tmp_path)

        def spy_only(symbol: str):
            if symbol == "SPY":
                store.save(_fresh_frame(), symbol)
            else:
                raise RuntimeError("vendor down")

        summary = sb.ingest_sleeve_bars(store=store, fetch_fn=spy_only)
        assert summary["symbols"]["SGOV"]["status"] == "no_data"
        assert summary["universe_complete"] is False
        assert summary["serving_eligible"] is False

    def test_stale_leg_is_not_serving_eligible(self, tmp_path):
        store = LocalStore(tmp_path)

        def stale_sgov(symbol: str):
            if symbol == "SGOV":
                df = _fresh_frame()
                store.save(df.iloc[:-10], symbol)  # ends 10 sessions ago
            else:
                store.save(_fresh_frame(), symbol)

        summary = sb.ingest_sleeve_bars(store=store, fetch_fn=stale_sgov)
        assert summary["symbols"]["SGOV"]["status"] == "stale"
        assert summary["serving_eligible"] is False

    def test_ingest_is_deterministic_for_fixed_inputs(self, tmp_path):
        store = LocalStore(tmp_path)
        now = pd.Timestamp("2026-07-10T21:00:00Z")
        fixed = _fresh_frame()

        def fetch(symbol: str):
            store.save(fixed, symbol)

        a = sb.ingest_sleeve_bars(store=store, fetch_fn=fetch, now_fn=lambda: now)
        b = sb.ingest_sleeve_bars(store=store, fetch_fn=fetch, now_fn=lambda: now)
        assert a["fingerprint"] == b["fingerprint"]

    def test_custom_symbols_normalized(self, tmp_path):
        store = LocalStore(tmp_path)
        seen: list[str] = []

        def fetch(symbol: str):
            seen.append(symbol)
            store.save(_fresh_frame(), symbol)

        summary = sb.ingest_sleeve_bars([" spy ", "bil", "SPY"],
                                        store=store, fetch_fn=fetch)
        assert seen == ["SPY", "BIL"]
        assert summary["expected_universe"] == ["BIL", "SPY"]


# ---------------------------------------------------------------------------
# CLI — dry-run default, --write, --verify, watchlist refusal
# ---------------------------------------------------------------------------

class TestCli:
    def test_dry_run_default_no_fetch_no_write(self, tmp_path, capsys, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("dry run must not fetch")
        monkeypatch.setattr(sb._ohlcv, "fetch_ohlcv_incremental", boom)

        rc = sb.main(["--data-dir", str(tmp_path / "ohlcv")])
        assert rc == 0
        assert not (tmp_path / "ohlcv" / "SGOV").exists()
        out = capsys.readouterr().out
        assert "DRY-RUN" in out
        assert "[MISSING]" in out and "SGOV" in out

    def test_write_stamps_manifest(self, tmp_path, capsys, monkeypatch):
        store_dir = tmp_path / "ohlcv"

        def fake_fetch(symbol, *, store, timeout_sec):
            store.save(_fresh_frame(), symbol)
        monkeypatch.setattr(sb._ohlcv, "fetch_ohlcv_incremental", fake_fetch)

        rc = sb.main(["--write", "--data-dir", str(store_dir)])
        assert rc == 0
        payload = sb.load_sleeve_ingestion_manifest(
            store_dir / "ingestion_manifest_sleeve_1d.json")
        assert payload["serving_eligible"] is True
        assert "[VERIFIED]" in capsys.readouterr().out

    def test_write_fails_closed_when_leg_missing(self, tmp_path, monkeypatch):
        store_dir = tmp_path / "ohlcv"

        def spy_only(symbol, *, store, timeout_sec):
            if symbol == "SPY":
                store.save(_fresh_frame(), symbol)
            else:
                raise RuntimeError("vendor down")
        monkeypatch.setattr(sb._ohlcv, "fetch_ohlcv_incremental", spy_only)

        rc = sb.main(["--write", "--data-dir", str(store_dir)])
        assert rc == 1

    def test_refuses_sgov_in_watchlist(self, tmp_path, capsys, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must refuse before fetching")
        monkeypatch.setattr(sb._ohlcv, "fetch_ohlcv_incremental", boom)

        cfg = tmp_path / "strategy_config.json"
        cfg.write_text(json.dumps(
            {"watchlist": ["AAPL", "SGOV"], "sleeve": {"enabled": False}}))
        rc = sb.main(["--write", "--strategy-config", str(cfg),
                      "--data-dir", str(tmp_path / "ohlcv")])
        assert rc == 2
        assert "REFUSING" in capsys.readouterr().out

    def test_spy_in_watchlist_is_fine(self, tmp_path, monkeypatch):
        # SPY is the benchmark and legitimately everywhere; only the T-bill
        # leg is barred from the watchlist (st104#39).
        def fake_fetch(symbol, *, store, timeout_sec):
            store.save(_fresh_frame(), symbol)
        monkeypatch.setattr(sb._ohlcv, "fetch_ohlcv_incremental", fake_fetch)

        cfg = tmp_path / "strategy_config.json"
        cfg.write_text(json.dumps(
            {"watchlist": ["AAPL", "SPY"], "sleeve": {"enabled": False}}))
        rc = sb.main(["--write", "--strategy-config", str(cfg),
                      "--data-dir", str(tmp_path / "ohlcv")])
        assert rc == 0

    def test_verify_ok_and_verify_tamper(self, tmp_path, capsys):
        store = LocalStore(tmp_path / "ohlcv")
        sb.ingest_sleeve_bars(store=store, fetch_fn=_saving_fetch(store))
        assert sb.main(["--verify", "--data-dir", str(store.data_dir)]) == 0
        assert "VERIFY OK" in capsys.readouterr().out

        # Rewrite one leg's parquet after sealing → content drift → fail.
        store.save(_fresh_frame(days=40), "SGOV")
        assert sb.main(["--verify", "--data-dir", str(store.data_dir)]) == 1
        assert "VERIFY FAIL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Cross-repo pinned-artifact integration: a consumer resolves SGOV bars
# from base-data's pinned artifact — registry manifest → ingestion manifest
# fingerprint pin → content-verified bars. No umbrella path anywhere.
# ---------------------------------------------------------------------------

class TestPinnedArtifactConsumerResolution:
    """Simulates the multi-repo consumer path (pipeline #185 via the
    orchestrator run manifest):

      1. resolve the dataset from the checked-in registry entry
         (``renquant_base_data.registry.resolve_data_manifest`` — a
         renquant_common Pipeline, i.e. the cross-repo primitive chain);
      2. pin the live ingestion manifest's fingerprint (what the run
         manifest records);
      3. resolve SGOV bars through the serving contract with that pin —
         fail-closed on pin mismatch and on post-seal store drift.
    """

    def _ingested_store(self, tmp_path) -> tuple[LocalStore, dict]:
        store = LocalStore(tmp_path / "ohlcv")
        summary = sb.ingest_sleeve_bars(store=store,
                                        fetch_fn=_saving_fetch(store))
        return store, summary

    def test_consumer_resolves_sgov_bars_from_pinned_artifact(self, tmp_path):
        # 1. Registry resolution from the REAL checked-in manifest.
        registry_dir = tmp_path / "manifests"
        registry_dir.mkdir()
        shutil.copy(_REPO_ROOT / "manifests" / "sleeve-ohlcv-1d.json",
                    registry_dir / "sleeve-ohlcv-1d.json")
        manifest = resolve_data_manifest(registry_dir,
                                         dataset_id="sleeve-ohlcv-1d")
        assert manifest["asset_class"] == "us_equity"
        assert manifest["uri"] == "store://ohlcv/1d"

        # 2. Ingestion produces the live artifact; the run manifest pins
        #    its fingerprint.
        store, summary = self._ingested_store(tmp_path)
        assert summary["serving_eligible"] is True
        pinned_fp = summary["fingerprint"]

        # 3. Consumer resolves SGOV bars against the pin.
        bars = sb.resolve_sleeve_leg_bars("SGOV", store=store,
                                          pinned_fingerprint=pinned_fp)
        assert len(bars) > 0
        assert float(bars["close"].iloc[-1]) > 0

        # The whole resolution chain names no umbrella path.
        touched = [str(store.data_dir), str(registry_dir),
                   str(sb.sleeve_manifest_path(store)),
                   manifest["uri"], manifest["source"]]
        assert not any("RenQuant/" in t for t in touched)

    def test_consumer_fails_closed_on_pin_mismatch(self, tmp_path):
        store, _ = self._ingested_store(tmp_path)
        with pytest.raises(ValueError, match="pinned"):
            sb.resolve_sleeve_leg_bars(
                "SGOV", store=store,
                pinned_fingerprint="sha256:" + "0" * 64)

    def test_consumer_fails_closed_on_post_seal_store_drift(self, tmp_path):
        store, summary = self._ingested_store(tmp_path)
        store.save(_fresh_frame(days=45), "SGOV")  # rewrite after sealing
        with pytest.raises(ValueError, match="content sha256 drifted"):
            sb.resolve_sleeve_leg_bars("SGOV", store=store,
                                       pinned_fingerprint=summary["fingerprint"])

    def test_consumer_fails_closed_on_non_eligible_symbol(self, tmp_path):
        store = LocalStore(tmp_path / "ohlcv")

        def spy_only(symbol: str):
            if symbol == "SPY":
                store.save(_fresh_frame(), symbol)
            else:
                raise RuntimeError("vendor down")

        sb.ingest_sleeve_bars(store=store, fetch_fn=spy_only)
        with pytest.raises(ValueError, match="not serving-eligible"):
            sb.resolve_sleeve_leg_bars("SGOV", store=store)

    def test_registry_entry_is_valid_against_repo_contract(self):
        # The checked-in registry entry passes the repo's own validation
        # pipeline (required keys, no developer-local URI).
        from renquant_base_data.registry import load_data_manifest
        manifest = load_data_manifest(
            _REPO_ROOT / "manifests" / "sleeve-ohlcv-1d.json")
        assert manifest["dataset_id"] == "sleeve-ohlcv-1d"
