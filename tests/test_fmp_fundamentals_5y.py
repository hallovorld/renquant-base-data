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
  * VERIFIED idempotency: the skip path deep-verifies the published bundle
    (child parquet + manifest hashes, target set, universe fingerprint,
    schema/harvester version, coverage floor, row columns); corrupted parquet,
    missing files, tampered manifests, stale/incomplete target sets, and a
    changed universe all fail loudly (verify_failed, exit 3) instead of being
    silently accepted, and --force recovers by atomic replacement;
  * top-level contract manifest binds universe fingerprint, endpoint x period
    config, code/schema version, PIT stamp, and every child hash;
  * PIT stamp: admissible_use=research_descriptive_only on every manifest
    (restated vendor-current history is NOT a C2 confirmatory input; see
    renquant-orchestrator PR #243 r4);
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
    assert set(top["targets"]) == {m["name"] for m in res["manifests"]}
    for entry in top["targets"].values():
        assert {"endpoint", "period", "path_template", "output", "rows",
                "with_data", "coverage", "status", "sha256",
                "manifest_sha256"} <= set(entry)


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
    assert r2["reason"] == "already_published_verified"
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


# --- 9. verified idempotency: the skip path must deep-verify the bundle ----------------
def _boom(endpoint_path, sym):
    raise AssertionError("verification path must not refetch")


def _publish_good(tmp_path, tickers, **kw):
    out = tmp_path / "fmp_harvest_5y"
    res = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), **kw)
    assert res["published"] is True
    return out


def test_verify_failed_on_corrupted_parquet_then_force_recovers(tmp_path, tickers):
    """A truncated/corrupt prior harvest must NOT be silently accepted, and
    --force must recover by re-harvesting + atomic replacement."""
    out = _publish_good(tmp_path, tickers)
    pq = out / "key_metrics_annual.parquet"
    pq.write_bytes(b"definitely not parquet")

    res = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert res["published"] is False
    assert any("parquet sha256 mismatch" in p for p in res["problems"])
    # Nothing skipped, nothing refetched (policy=_boom), nothing touched.
    assert pq.read_bytes() == b"definitely not parquet"

    # force-replacement recovery: the bad bundle is atomically replaced ...
    rec = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), force=True)
    assert rec["status"] == "ok" and rec["published"] is True
    assert len(pd.read_parquet(pq)) == 2 * len(tickers)
    assert not list(tmp_path.glob(".replaced-*"))
    # ... and the repaired bundle verifies clean again.
    assert _run(tickers=tickers, out_dir=out, policy=_boom)["status"] == "skipped"


def test_verify_failed_on_missing_parquet_or_manifest(tmp_path, tickers):
    out = _publish_good(tmp_path, tickers)
    (out / "ratios_quarterly.parquet").unlink()
    res = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert any("missing parquet" in p for p in res["problems"])

    (out / "ratios_annual.manifest.json").unlink()
    res2 = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res2["status"] == "verify_failed"
    assert any("missing child manifest" in p for p in res2["problems"])


def test_verify_failed_on_stale_or_incomplete_target_set(tmp_path, tickers):
    out = tmp_path / "fmp_harvest_5y"
    ann = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(),
               periods=("annual",))
    assert ann["published"] is True

    # The annual-only bundle is INCOMPLETE for the default annual+quarterly ask.
    res = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert any("lacks requested target" in p for p in res["problems"])

    # The reverse (bundle carries MORE than requested) is a mismatch too.
    full = _run(tickers=tickers, out_dir=out, policy=_all_ok_policy(), force=True)
    assert full["published"] is True
    res2 = _run(tickers=tickers, out_dir=out, policy=_boom, periods=("annual",))
    assert res2["status"] == "verify_failed"
    assert any("unrequested target" in p for p in res2["problems"])


def test_verify_failed_on_changed_universe(tmp_path, tickers):
    out = _publish_good(tmp_path, tickers)
    res = _run(tickers=list(tickers) + ["NEWCO"], out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert any("universe fingerprint mismatch" in p for p in res["problems"])
    # Same ticker SET in another order / with dupes is the same universe.
    shuffled = list(reversed(tickers)) + [tickers[0]]
    assert _run(tickers=shuffled, out_dir=out, policy=_boom)["status"] == "skipped"


def test_verify_failed_on_schema_or_harvester_version_drift(
    tmp_path, tickers, monkeypatch
):
    out = _publish_good(tmp_path, tickers)
    monkeypatch.setattr(h5y, "SCHEMA_VERSION", h5y.SCHEMA_VERSION + 1)
    res = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert any("schema_version mismatch" in p for p in res["problems"])

    monkeypatch.undo()
    monkeypatch.setattr(h5y, "HARVESTER_VERSION", "999.0.0")
    res2 = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res2["status"] == "verify_failed"
    assert any("harvester_version mismatch" in p for p in res2["problems"])


def test_verify_failed_on_tampered_child_manifest(tmp_path, tickers):
    out = _publish_good(tmp_path, tickers)
    man = out / "ratios_annual.manifest.json"
    doc = json.loads(man.read_text())
    doc["rows"] = 999_999  # cook the books; parquet bytes stay valid
    man.write_text(json.dumps(doc, indent=2) + "\n")
    res = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert any("child manifest sha256 mismatch" in p for p in res["problems"])


def test_verify_failed_when_requested_floor_exceeds_recorded_coverage(
    tmp_path, tickers
):
    out = tmp_path / "fmp_harvest_5y"
    ok = _all_ok_policy()
    missing = {tickers[0]}  # 19/20 = 0.95 coverage on every target

    def policy(endpoint_path, sym):
        return ([], None) if sym in missing else ok(endpoint_path, sym)

    assert _run(tickers=tickers, out_dir=out, policy=policy,
                min_coverage=0.9)["published"] is True
    res = _run(tickers=tickers, out_dir=out, policy=_boom, min_coverage=0.99)
    assert res["status"] == "verify_failed"
    assert any("below the requested floor" in p for p in res["problems"])
    # At the floor it was harvested under, it still verifies clean.
    assert _run(tickers=tickers, out_dir=out, policy=_boom,
                min_coverage=0.9)["status"] == "skipped"


def test_verify_failed_on_unmanifested_existing_dir(tmp_path, tickers):
    """A pre-existing out_dir that is not a published bundle must fail loudly,
    not be silently replaced (that now takes an explicit --force)."""
    out = tmp_path / "fmp_harvest_5y"
    out.mkdir()
    (out / "stray.txt").write_text("leftover\n")
    res = _run(tickers=tickers, out_dir=out, policy=_boom)
    assert res["status"] == "verify_failed"
    assert any("missing top-level manifest" in p for p in res["problems"])
    assert (out / "stray.txt").exists()  # untouched without --force


# --- 10. top-level contract manifest ----------------------------------------------------
def test_top_manifest_binds_full_contract(tmp_path, tickers):
    out = _publish_good(tmp_path, tickers)
    top = json.loads((out / h5y.HARVEST_MANIFEST).read_text())
    assert top["harvester"] == h5y.HARVESTER
    assert top["schema_version"] == h5y.SCHEMA_VERSION
    assert top["harvester_version"] == h5y.HARVESTER_VERSION
    assert top["universe_sha256"] == h5y.universe_fingerprint(tickers)
    assert top["universe"] == len(tickers)
    assert top["periods"] == list(h5y.DEFAULT_PERIODS)
    assert top["required_row_columns"] == list(h5y.REQUIRED_ROW_COLUMNS)
    expected = {t["name"]: t for t in h5y.build_targets(list(h5y.DEFAULT_PERIODS))}
    assert set(top["targets"]) == set(expected)
    for name, entry in top["targets"].items():
        # Every child hash is bound: parquet bytes AND the child manifest file.
        assert entry["sha256"] == h5y._sha256_file(out / entry["output"])
        assert entry["manifest_sha256"] == h5y._sha256_file(
            out / f"{name}.manifest.json"
        )
        assert entry["path_template"] == expected[name]["path_template"]


def test_universe_fingerprint_is_order_and_dupe_insensitive():
    a = h5y.universe_fingerprint(["AAPL", "MSFT", "NVDA"])
    assert a == h5y.universe_fingerprint(["NVDA", "AAPL", "MSFT", "AAPL", " MSFT "])
    assert a != h5y.universe_fingerprint(["AAPL", "MSFT"])


# --- 11. PIT stamp: research/descriptive only, never C2-confirmatory --------------------
def test_pit_stamp_on_every_manifest(tmp_path, tickers):
    out = _publish_good(tmp_path, tickers)
    top = json.loads((out / h5y.HARVEST_MANIFEST).read_text())
    assert top["admissible_use"] == "research_descriptive_only"
    assert top["pit_provenance"] == "vendor_current_values_no_revision_identity"
    # The note must carry the operative caveats: restatement risk and the
    # admissible confirmatory path (genuine filing timestamps / SEC EDGAR,
    # fail closed) from the merged M-SIG spec.
    note = top["pit_note"]
    for needle in ("RESTATED", "#243", "acceptedDate", "EDGAR", "INADMISSIBLE"):
        assert needle in note, f"pit_note missing {needle!r}"
    for name in top["targets"]:
        child = json.loads((out / f"{name}.manifest.json").read_text())
        assert child["admissible_use"] == top["admissible_use"]
        assert child["pit_provenance"] == top["pit_provenance"]
        assert child["schema_version"] == h5y.SCHEMA_VERSION
        assert child["harvester_version"] == h5y.HARVESTER_VERSION


# --- 12. CLI surfacing of verify_failed --------------------------------------------------
def test_main_verify_failed_exits_3_and_prints_problems(
    tmp_path, tickers, capsys, monkeypatch
):
    out = tmp_path / "fmp_harvest_5y"
    _run(tickers=tickers, out_dir=out, policy=_all_ok_policy())
    (out / "key_metrics_annual.parquet").write_bytes(b"garbage")

    universe = tmp_path / "universe.txt"
    universe.write_text("".join(f"{t}\n" for t in tickers))
    monkeypatch.setenv("FMP_API_KEY", "FAKE")

    # No live session and no network: verification runs before any fetch.
    class _FakeRequestsModule:
        class Session:  # noqa: D106 - test stub
            pass

    monkeypatch.setattr(h5y, "_require_requests", lambda: _FakeRequestsModule)

    rc = h5y.main(["--out", str(out), "--universe", str(universe)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "VERIFY FAILED" in err
    assert "parquet sha256 mismatch" in err
    assert "--force" in err
    # The corrupt bundle was neither skipped nor replaced.
    assert (out / "key_metrics_annual.parquet").read_bytes() == b"garbage"
