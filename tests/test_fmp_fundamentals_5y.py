"""Tests for the one-shot FMP 5-year fundamentals harvester (M-SIG C2).

All tests use a dependency-injected fake fetcher and tmp_path output --
NO live FMP calls, no writes outside pytest scratch. Contracts covered:

  * coverage gate: a target below --min-coverage marks the run partial,
    publishes NOTHING, and main() exits non-zero;
  * manifest fields: fetched_at (ISO UTC), row/symbol coverage, sha256
    matching the published parquet bytes;
  * parquet schema: every row carries symbol, fiscal_date, harvest_period,
    fetched_at (vendor-native fields preserved);
  * dry-run: ZERO fetch calls, ZERO writes, full request plan;
  * idempotent verify / --force re-harvest;
  * canonical-path guard rejects non-dedicated targets.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data import fmp_fundamentals_5y as h5y


# --- fake fetch -----------------------------------------------------------------
def _make_fetch(policy, calls=None):
    """fetch_endpoint stand-in: policy(endpoint_path, sym) -> (records, error)."""

    def _fetch(session, endpoint_path, sym, api_key):
        if calls is not None:
            calls.append((endpoint_path, sym))
        return policy(endpoint_path, sym)

    return _fetch


def _all_ok_policy(records_per_sym=2):
    def policy(endpoint_path, sym):
        quarterly = "period=quarter&" in endpoint_path
        return (
            [
                {
                    "symbol": sym,
                    "date": f"2026-0{i + 1}-31",
                    "period": f"Q{i + 1}" if quarterly else "FY",
                    "fiscalYear": "2026",
                    "revenue": 100.0 + i,
                }
                for i in range(records_per_sym)
            ],
            None,
        )

    return policy


@pytest.fixture
def tickers():
    return [f"TKR{i:03d}" for i in range(20)]


def _run(*, tickers, out_dir, policy, calls=None, periods=h5y.DEFAULT_PERIODS,
         force=False, min_coverage=h5y.DEFAULT_MIN_COVERAGE, dry_run=False):
    return h5y.harvest(
        session=None,
        tickers=tickers,
        api_key="FAKE",
        out_dir=Path(out_dir),
        periods=periods,
        dry_run=dry_run,
        force=force,
        min_coverage=min_coverage,
        throttle_s=0.0,  # no courteous sleep in tests
        fetch=_make_fetch(policy, calls),
    )


# --- 1. happy path: all 8 targets published atomically ---------------------------
def test_happy_path_publishes_all_targets(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    res = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())
    assert res["status"] == "ok"
    assert res["published"] is True
    names = {t["name"] for t in h5y.build_targets(list(h5y.DEFAULT_PERIODS))}
    assert len(names) == 8  # 4 endpoints x 2 periods
    for name in names:
        assert (out / f"{name}.parquet").exists()
        assert (out / f"{name}.manifest.json").exists()
    assert (out / h5y.HARVEST_MANIFEST).exists()
    # No staging or backup residue.
    assert not list(tmp_path.glob(".stage-*"))
    assert not list(tmp_path.glob(".replaced-*"))


# --- 2. parquet schema ------------------------------------------------------------
def test_parquet_rows_carry_required_columns(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    res = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(records_per_sym=3))
    fetched_at = res["summary"]["fetched_at"]
    for name, period in (("key_metrics_annual", "annual"),
                         ("income_statement_quarterly", "quarterly")):
        df = pd.read_parquet(out / f"{name}.parquet")
        assert len(df) == 3 * len(tickers)
        for col in ("symbol", "fiscal_date", "harvest_period", "fetched_at"):
            assert col in df.columns, f"{name} missing {col}"
        assert set(df["symbol"]) == set(tickers)
        assert (df["harvest_period"] == period).all()
        assert (df["fetched_at"] == fetched_at).all()
        assert (df["fiscal_date"] == df["date"]).all()  # stamped from vendor date
    # Vendor-native period field preserved (FY vs Q1..) alongside harvest_period.
    dfq = pd.read_parquet(out / "income_statement_quarterly.parquet")
    assert set(dfq["period"]) <= {"Q1", "Q2", "Q3", "Q4"}
    dfa = pd.read_parquet(out / "key_metrics_annual.parquet")
    assert set(dfa["period"]) == {"FY"}


# --- 3. manifest fields ------------------------------------------------------------
def test_manifest_fields_and_sha(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    res = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(records_per_sym=2))
    for m in res["manifests"]:
        man = json.loads((out / f"{m['name']}.manifest.json").read_text())
        # fetched_at is a real ISO-8601 UTC timestamp shared with the rows.
        assert datetime.fromisoformat(man["fetched_at"]).tzinfo is not None
        assert man["fetched_at"] == res["summary"]["fetched_at"]
        assert man["requested"] == len(tickers)
        assert man["with_data"] == len(tickers)
        assert man["coverage"] == 1.0
        assert man["rows"] == 2 * len(tickers)
        assert man["status"] == "ok"
        assert man["endpoint"] in h5y.ENDPOINTS
        assert man["period"] in h5y.PERIODS
        assert "{sym}" in man["path_template"]
        assert man["sha256"] == h5y._sha256_file(out / man["output"])
    top = json.loads((out / h5y.HARVEST_MANIFEST).read_text())
    assert top["status"] == "ok"
    assert top["universe"] == len(tickers)
    assert top["universe_fingerprint"] == h5y._universe_fingerprint(tickers)
    assert top["harvester_version"] == h5y.HARVESTER_VERSION
    assert top["pit_classification"] == "research_descriptive_only"
    assert set(top["targets"]) == {m["name"] for m in res["manifests"]}
    for name, entry in top["targets"].items():
        # sha256/schema_columns bind every child artifact's exact content
        # into the top-level manifest -- not just an aggregate status claim.
        assert set(entry) == {
            "rows", "with_data", "coverage", "status", "sha256", "schema_columns",
        }
        assert entry["sha256"] == h5y._sha256_file(out / f"{name}.parquet")


# --- 4. coverage gate ---------------------------------------------------------------
def test_coverage_gate_blocks_publish(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    ok = _all_ok_policy()

    # One target (income-statement quarterly) returns rows for only half the
    # universe -> below the 0.9 floor -> the WHOLE run is partial, nothing lands.
    def policy(endpoint_path, sym):
        if endpoint_path.startswith("income-statement") and "period=quarter&" in endpoint_path:
            if sym in tickers[: len(tickers) // 2]:
                return ok(endpoint_path, sym)
            return ([], None)
        return ok(endpoint_path, sym)

    res = _run(tickers=tickers, out_dir=out, policy=policy)
    assert res["status"] == "partial"
    assert res["published"] is False
    assert res["partial_targets"] == ["income_statement_quarterly"]
    assert not out.exists()
    assert not list(tmp_path.glob(".stage-*"))


def test_plan_locked_quarterly_blocks_publish_annual_only_passes(tmp_path, tickers):
    """The probed 2026-07-02 reality: every quarterly request 402s (plan gate).
    The full run must fail loudly; --periods annual must still harvest clean."""
    out = tmp_path / "fmp_harvest_5y"
    ok = _all_ok_policy()

    def policy(endpoint_path, sym):
        if "period=quarter&" in endpoint_path:
            return (None, "http_402")
        return ok(endpoint_path, sym)

    res = _run(tickers=tickers, out_dir=out, policy=policy)
    assert res["status"] == "partial"
    assert res["published"] is False
    assert sorted(res["partial_targets"]) == sorted(
        f"{e}_quarterly" for e in h5y.ENDPOINTS
    )
    by_name = {m["name"]: m for m in res["manifests"]}
    assert by_name["ratios_quarterly"]["http_error"] == len(tickers)
    assert by_name["ratios_quarterly"]["coverage"] == 0.0
    assert not out.exists()

    annual = _run(tickers=tickers, out_dir=out, policy=policy, periods=("annual",))
    assert annual["status"] == "ok"
    assert annual["published"] is True
    assert (out / "key_metrics_annual.parquet").exists()
    assert not (out / "key_metrics_quarterly.parquet").exists()


def test_min_coverage_threshold_is_respected(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    ok = _all_ok_policy()
    missing = set(tickers[:3])  # 17/20 = 0.85 coverage on every target

    def policy(endpoint_path, sym):
        if sym in missing:
            return ([], None)
        return ok(endpoint_path, sym)

    strict = _run(tickers=tickers, out_dir=out, policy=policy, min_coverage=0.9)
    assert strict["status"] == "partial"
    assert not out.exists()

    lax = _run(tickers=tickers, out_dir=out, policy=policy, min_coverage=0.8)
    assert lax["status"] == "ok"
    man = json.loads((out / "ratios_annual.manifest.json").read_text())
    assert man["coverage"] == 0.85
    assert man["no_data"] == 3


# --- 5. dry-run ---------------------------------------------------------------------
def test_dry_run_makes_zero_fetch_calls_and_no_writes(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    calls: list = []
    res = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), calls=calls,
               dry_run=True)
    assert res["status"] == "dry_run"
    assert res["published"] is False
    assert calls == []  # ZERO fetch calls
    assert not out.exists()  # ZERO writes
    assert not list(tmp_path.iterdir())
    # The plan lists every (endpoint, period) request group with the exact path.
    assert res["planned_requests"] == len(tickers) * 8
    planned = {t["name"]: t for t in res["targets"]}
    assert planned["key_metrics_annual"]["path_template"] == (
        "key-metrics?symbol={sym}&period=annual&limit=10"
    )
    assert planned["income_statement_quarterly"]["path_template"] == (
        "income-statement?symbol={sym}&period=quarter&limit=40"
    )


def test_main_dry_run_exits_zero_without_key_or_network(tmp_path, capsys):
    universe = tmp_path / "universe.txt"
    universe.write_text("AAA\nBBB\n")
    rc = h5y.main(
        ["--dry-run", "--universe", str(universe), "--out", str(tmp_path / "fmp_harvest_5y")]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "PLAN key_metrics_annual" in err
    assert "16 requests total" in err  # 2 symbols x 8 targets


# --- 6. idempotency / force ----------------------------------------------------------
def test_idempotent_rerun_is_noop(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    r1 = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())
    assert r1["published"] is True
    sha_before = json.loads((out / "ratios_annual.manifest.json").read_text())["sha256"]

    def boom(endpoint_path, sym):
        raise AssertionError("idempotent rerun must not refetch")

    r2 = _run(tickers=tickers, out_dir=out, policy=boom)
    assert r2["status"] == "skipped"
    assert r2["published"] is False
    sha_after = json.loads((out / "ratios_annual.manifest.json").read_text())["sha256"]
    assert sha_before == sha_after


def test_force_reharvests_and_partial_preserves_prior(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(records_per_sym=1))
    good_sha = json.loads((out / "ratios_annual.manifest.json").read_text())["sha256"]

    # force + all-good: re-published with the new (2-record) data.
    r2 = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(records_per_sym=2),
              force=True)
    assert r2["published"] is True
    df = pd.read_parquet(out / "ratios_annual.parquet")
    assert len(df) == 2 * len(tickers)

    # force + coverage failure: publishes nothing, the prior harvest survives.
    def bad(endpoint_path, sym):
        return (None, "http_500")

    r3 = _run(tickers=tickers, out_dir=out, policy=bad, force=True)
    assert r3["status"] == "partial"
    assert r3["published"] is False
    still = json.loads((out / "ratios_annual.manifest.json").read_text())["sha256"]
    assert still != good_sha  # it is the r2 harvest ...
    assert len(pd.read_parquet(out / "ratios_annual.parquet")) == 2 * len(tickers)


# --- corrupted / stale prior harvest must never be silently skipped -------------


def test_corrupted_parquet_forces_reharvest_not_silent_skip(tmp_path, tickers):
    """A prior harvest whose recorded sha256 no longer matches the on-disk
    parquet (truncation/corruption) must not be accepted as 'already
    published' -- the old skip check only looked at manifest presence and
    would have silently trusted this forever."""
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())
    # Corrupt one published parquet in place (truncate to zero bytes).
    (out / "ratios_annual.parquet").write_bytes(b"")

    calls: list = []
    r2 = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), calls=calls)
    assert r2["status"] == "ok"
    assert r2["published"] is True
    assert calls, "corrupted prior harvest must trigger a real re-fetch, not a skip"
    # The re-harvest actually repaired the corrupted file.
    df = pd.read_parquet(out / "ratios_annual.parquet")
    assert len(df) == 2 * len(tickers)


def test_incomplete_target_manifest_forces_reharvest(tmp_path, tickers):
    """A prior harvest missing one target's manifest (e.g. an interrupted
    publish) must not be accepted as 'already published'."""
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())
    (out / "ratios_annual.manifest.json").unlink()

    calls: list = []
    r2 = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), calls=calls)
    assert r2["status"] == "ok"
    assert r2["published"] is True
    assert calls, "incomplete prior harvest must trigger a real re-fetch, not a skip"


def test_changed_universe_forces_reharvest(tmp_path, tickers):
    """A prior harvest published against a DIFFERENT ticker universe than the
    one now configured must not be accepted as 'already published' -- its
    coverage claims don't apply to the new target set."""
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())

    new_universe = tickers + ["NEWTICKER"]
    calls: list = []
    r2 = _run(tickers=new_universe, out_dir=out, policy=_all_ok_policy(), calls=calls)
    assert r2["status"] == "ok"
    assert r2["published"] is True
    assert calls, "a changed universe must trigger a real re-fetch, not a skip"
    df = pd.read_parquet(out / "ratios_annual.parquet")
    assert set(df["symbol"]) == set(new_universe)


def test_stale_harvester_version_forces_reharvest(tmp_path, tickers, monkeypatch):
    """A prior harvest published under an older harvester_version must not be
    accepted as 'already published' even if its bytes are perfectly intact --
    a version bump means the CONTRACT's semantics changed."""
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())

    top = json.loads((out / h5y.HARVEST_MANIFEST).read_text())
    top["harvester_version"] = -1
    (out / h5y.HARVEST_MANIFEST).write_text(json.dumps(top, indent=2) + "\n")

    calls: list = []
    r2 = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), calls=calls)
    assert r2["status"] == "ok"
    assert r2["published"] is True
    assert calls, "a stale harvester_version must trigger a real re-fetch, not a skip"


def test_verify_published_harvest_reports_specific_reason(tmp_path, tickers):
    """_verify_published_harvest must name the exact mismatch, not just
    return a bare boolean -- this is what lets a caller log WHY a re-harvest
    is happening instead of silently skipping or silently overwriting."""
    out = tmp_path / "fmp_harvest_5y"
    targets = h5y.build_targets(list(h5y.DEFAULT_PERIODS))

    is_valid, reason = h5y._verify_published_harvest(out, targets, tickers)
    assert is_valid is False
    assert "does not exist" in reason

    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())
    is_valid, reason = h5y._verify_published_harvest(out, targets, tickers)
    assert is_valid is True
    assert reason == ""

    (out / "ratios_annual.parquet").write_bytes(b"corrupt")
    is_valid, reason = h5y._verify_published_harvest(out, targets, tickers)
    assert is_valid is False
    assert "ratios_annual" in reason
    assert "corrupt" in reason.lower() or "hash mismatch" in reason.lower()


def test_pit_classification_and_manifest_provenance_fields(tmp_path, tickers):
    """Every published manifest (top-level and per-target) must carry the
    research_descriptive_only PIT classification, harvester_version, and
    (top-level) universe_fingerprint -- the fields _verify_published_harvest
    and any downstream consumer rely on."""
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())

    top = json.loads((out / h5y.HARVEST_MANIFEST).read_text())
    assert top["pit_classification"] == "research_descriptive_only"
    assert top["harvester_version"] == h5y.HARVESTER_VERSION
    assert top["universe_fingerprint"] == h5y._universe_fingerprint(tickers)

    per_target = json.loads((out / "ratios_annual.manifest.json").read_text())
    assert per_target["pit_classification"] == "research_descriptive_only"
    assert per_target["harvester_version"] == h5y.HARVESTER_VERSION
    assert set(h5y._REQUIRED_HARVESTER_COLUMNS) <= set(per_target["schema_columns"])


# --- 7. canonical-path guard ----------------------------------------------------------
@pytest.mark.parametrize("leaf", sorted(h5y._FORBIDDEN))
def test_guard_rejects_forbidden_leaves(leaf):
    assert h5y.is_canonical_path(Path("data") / leaf) is True


def test_guard_requires_dedicated_leaf_or_scratch():
    assert h5y.is_canonical_path(Path("data/fmp_harvest_5y")) is False
    assert h5y.is_canonical_path(Path("/tmp/fmp5y_demo")) is False
    assert h5y.is_canonical_path(Path("data/whatever")) is True
    assert h5y.is_canonical_path(Path("data/fmp_harvest")) is True  # canonical harvest


def test_main_refuses_non_dedicated_out(tmp_path, capsys):
    universe = tmp_path / "universe.txt"
    universe.write_text("AAA\n")
    rc = h5y.main(["--out", "data/fmp_harvest", "--universe", str(universe)])
    assert rc == 2
    assert "refusing" in capsys.readouterr().err


# --- 8. target construction -----------------------------------------------------------
def test_build_targets_rejects_unknown_or_empty_periods():
    with pytest.raises(ValueError, match="unknown period"):
        h5y.build_targets(["monthly"])
    with pytest.raises(ValueError, match="at least one period"):
        h5y.build_targets([])
