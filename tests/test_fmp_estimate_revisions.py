"""Focused tests for the FMP estimate-revision forward snapshotter.

These cover the contracts Codex flagged as blocking on renquant-orchestrator
PR #205, using a fake/mocked fetch and /tmp output -- NO live FMP calls:

  * a historical OR future ``--as-of`` is REJECTED (no backdated/future PIT
    history); only today's UTC date is accepted;
  * snapshot_as_of is derived from the actual UTC fetch date;
  * idempotency is a no-op verify, not a destructive refetch;
  * a partial endpoint failure marks ``status: partial`` and does NOT publish
    over an existing good snapshot;
  * publication is atomic (no half-written final dir is ever observable);
  * the canonical-path guard rejects forbidden leaves AND symlinked targets;
  * manifest hashes match the published parquet bytes.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from renquant_base_data import fmp_estimate_revisions as snap


# --- fake fetch ---------------------------------------------------------------
class _FakeSession:
    """Stands in for requests.Session; routes by symbol/endpoint via a policy."""

    def __init__(self, policy):
        self._policy = policy

    def get(self, *a, **k):  # never called: collect_snapshot calls fetch_endpoint
        raise AssertionError("network must not be hit in tests")


def _make_fetch(policy):
    """Return a fetch_endpoint replacement driven by ``policy``.

    policy(endpoint_path, sym) -> (records, error) tuple, matching the real
    fetch_endpoint signature contract.
    """

    def _fetch(session, endpoint_path, sym, api_key):
        return policy(endpoint_path, sym)

    return _fetch


def _all_ok_policy(records_per_sym=1):
    def policy(endpoint_path, sym):
        return ([{"symbol": sym, "epsAvg": 1.23} for _ in range(records_per_sym)], None)

    return policy


@pytest.fixture
def tickers():
    return [f"TKR{i:03d}" for i in range(20)]


def _run(monkeypatch, *, tickers, as_of, out_root, policy, force=False,
         min_coverage=snap.DEFAULT_MIN_COVERAGE):
    monkeypatch.setattr(snap, "fetch_endpoint", _make_fetch(policy))
    monkeypatch.setattr(snap.time, "sleep", lambda *_: None)  # no throttle in tests
    return snap.collect_snapshot(
        session=_FakeSession(policy),
        tickers=tickers,
        api_key="FAKE",
        as_of=as_of,
        out_root=Path(out_root),
        dry_run=False,
        force=force,
        min_coverage=min_coverage,
    )


# --- 1. as-of backdating is forbidden ----------------------------------------
def test_resolve_as_of_rejects_past_date():
    today = date(2026, 6, 27)
    past = (today - timedelta(days=1)).isoformat()
    with pytest.raises(snap.AsOfError):
        snap.resolve_as_of(past, today=today)


def test_resolve_as_of_rejects_future_date():
    # A future --as-of pre-labels today's live fetch as future data -> fake
    # provenance. snapshot_as_of must equal the actual UTC fetch date, so a
    # future date is rejected just like a past one.
    today = date(2026, 6, 27)
    future = (today + timedelta(days=3)).isoformat()
    with pytest.raises(snap.AsOfError):
        snap.resolve_as_of(future, today=today)


def test_resolve_as_of_accepts_only_today():
    # The only honest as-of for a live fetch is today's UTC date: None defaults
    # to it, and an explicit value is accepted ONLY when it equals today.
    today = date(2026, 6, 27)
    assert snap.resolve_as_of(None, today=today) == "2026-06-27"
    assert snap.resolve_as_of("2026-06-27", today=today) == "2026-06-27"


def test_resolve_as_of_rejects_malformed():
    with pytest.raises(snap.AsOfError):
        snap.resolve_as_of("2026/06/27", today=date(2026, 6, 27))


def test_main_rejects_historical_as_of(tmp_path, capsys):
    yesterday = (date.today() - timedelta(days=2)).isoformat()
    rc = snap.main(["--as-of", yesterday, "--out", str(tmp_path / "x")])
    assert rc == 2
    assert "past" in capsys.readouterr().err.lower()


def test_main_rejects_future_as_of(tmp_path, capsys):
    future = (date.today() + timedelta(days=2)).isoformat()
    rc = snap.main(["--as-of", future, "--out", str(tmp_path / "x")])
    assert rc == 2
    assert "future" in capsys.readouterr().err.lower()


def test_snapshot_as_of_stamped_from_resolved_date(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    out = tmp_path  # tmp_path is under /var/folders -> scratch-allowed
    res = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=out,
               policy=_all_ok_policy())
    assert res["status"] == "ok"
    df = pd.read_parquet(res["out_dir"] / "analyst_estimates.parquet")
    assert (df["snapshot_as_of"] == as_of).all()


# --- 2. idempotency = no-op verify, not destructive refetch ------------------
def test_idempotent_rerun_is_noop(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    r1 = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
              policy=_all_ok_policy())
    assert r1["published"] is True
    sha_before = json.loads(
        (r1["out_dir"] / "analyst_estimates.manifest.json").read_text()
    )["sha256"]

    # Second run with a fetch that WOULD raise if called -> proves no refetch.
    def _boom(endpoint_path, sym):
        raise AssertionError("idempotent rerun must not refetch")

    r2 = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
              policy=_boom)
    assert r2["status"] == "skipped"
    assert r2["published"] is False
    sha_after = json.loads(
        (r2["out_dir"] / "analyst_estimates.manifest.json").read_text()
    )["sha256"]
    assert sha_before == sha_after  # untouched


def test_force_republishes(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
         policy=_all_ok_policy(records_per_sym=1))
    r2 = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
              policy=_all_ok_policy(records_per_sym=2), force=True)
    assert r2["status"] == "ok"
    assert r2["published"] is True
    df = pd.read_parquet(r2["out_dir"] / "analyst_estimates.parquet")
    assert len(df) == 2 * len(tickers)  # the re-published (2-record) version


# --- 3. partial-endpoint failure handling + non-destructive ------------------
def test_partial_fetch_not_published(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()

    # Fail most symbols on one endpoint with an HTTP error -> coverage shortfall.
    def policy(endpoint_path, sym):
        if endpoint_path.startswith("analyst-estimates") and sym != tickers[0]:
            return (None, "http_502")
        return ([{"symbol": sym, "epsAvg": 1.0}], None)

    res = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
               policy=policy)
    assert res["status"] == "partial"
    assert res["published"] is False
    assert "analyst_estimates" in res["partial_endpoints"]
    # No final dir was published.
    assert not (tmp_path / as_of).exists()
    # No staging leftovers.
    assert not list(tmp_path.glob(".stage-*"))


def test_any_fetch_error_marks_partial(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()

    # A single network error on one symbol -> partial even if coverage is high.
    def policy(endpoint_path, sym):
        if sym == tickers[0]:
            return (None, "fetch_error:ConnectionError")
        return ([{"symbol": sym}], None)

    res = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
               policy=policy)
    assert res["status"] == "partial"
    assert res["published"] is False


def test_partial_does_not_overwrite_good_prior(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    good = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
                policy=_all_ok_policy(records_per_sym=1))
    good_sha = json.loads(
        (good["out_dir"] / "analyst_estimates.manifest.json").read_text()
    )["sha256"]

    def bad(endpoint_path, sym):
        return (None, "http_500")

    # force=True so it does not short-circuit on "already published"; it must
    # still refuse to publish because the new fetch is partial.
    res = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
               policy=bad, force=True)
    assert res["status"] == "partial"
    after_sha = json.loads(
        (tmp_path / as_of / "analyst_estimates.manifest.json").read_text()
    )["sha256"]
    assert after_sha == good_sha  # prior good snapshot intact


# --- 4. atomic publication ----------------------------------------------------
def test_publish_is_atomic_no_partial_dir(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    res = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
               policy=_all_ok_policy())
    final = res["out_dir"]
    # Every endpoint present in one shot (the rename is all-or-nothing).
    for endpoint in snap.ENDPOINTS:
        assert (final / f"{endpoint}.parquet").exists()
        assert (final / f"{endpoint}.manifest.json").exists()
    # No staging or backup residue left behind.
    assert not list(tmp_path.glob(".stage-*"))
    assert not list(tmp_path.glob(".replaced-*"))


def test_publish_failure_restores_prior_good_snapshot(monkeypatch, tmp_path, tickers):
    """If the stage->final rename FAILS while an old snapshot exists, the old
    snapshot must survive unchanged and the failed stage must be cleaned up /
    quarantined (publish-safety / non-destructive contract; base-data PR #27).
    """
    as_of = date.today().isoformat()
    # 1. Publish a good prior snapshot.
    good = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
                policy=_all_ok_policy(records_per_sym=1))
    assert good["published"] is True
    final_dir = tmp_path / as_of
    good_manifest = (final_dir / "analyst_estimates.manifest.json").read_text()
    good_sha = json.loads(good_manifest)["sha256"]
    good_parquet_bytes = (final_dir / "analyst_estimates.parquet").read_bytes()

    # 2. Force a re-publish, but make ONLY the stage->final os.replace raise.
    #    The (final->backup) move and the (backup->final) RESTORE must still run,
    #    so fail only on the move whose source is the staging dir (".stage-*")
    #    into the final dir -- not on the restore (source ".replaced-*").
    real_replace = snap.os.replace

    def _replace_failing_on_final(src, dst):
        if str(dst) == str(final_dir) and Path(src).name.startswith(".stage-"):
            raise OSError("simulated stage->final failure")
        return real_replace(src, dst)

    monkeypatch.setattr(snap.os, "replace", _replace_failing_on_final)

    with pytest.raises(OSError, match="simulated stage->final failure"):
        _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
             policy=_all_ok_policy(records_per_sym=2), force=True)

    # Restore real os.replace before asserting on the filesystem.
    monkeypatch.setattr(snap.os, "replace", real_replace)

    # (a) The OLD manifest + parquet are STILL present and UNCHANGED.
    assert final_dir.is_dir()
    assert (final_dir / "analyst_estimates.manifest.json").read_text() == good_manifest
    assert json.loads(
        (final_dir / "analyst_estimates.manifest.json").read_text()
    )["sha256"] == good_sha
    assert (final_dir / "analyst_estimates.parquet").read_bytes() == good_parquet_bytes
    # The prior snapshot still has every endpoint (publish never half-applied).
    for endpoint in snap.ENDPOINTS:
        assert (final_dir / f"{endpoint}.parquet").exists()
        assert (final_dir / f"{endpoint}.manifest.json").exists()

    # (b) The failed stage is cleaned up or clearly quarantined (no live .stage-*
    #     residue), and the backup of the prior snapshot was restored (no
    #     dangling .replaced-* backup left holding the only good copy).
    assert not list(tmp_path.glob(".stage-*"))
    assert not list(tmp_path.glob(".replaced-*"))
    quarantined = list(tmp_path.glob(".failed-stage-*"))
    # The failed stage is either removed or moved to an obvious quarantine dir,
    # never left where it could be mistaken for the published snapshot.
    for q in quarantined:
        assert q.name.startswith(".failed-stage-")


# --- 5. canonical-path guard incl symlink behavior ---------------------------
@pytest.mark.parametrize("leaf", sorted(snap._FORBIDDEN_LEAVES))
def test_guard_rejects_forbidden_leaves(leaf):
    assert snap.is_canonical_path(Path("data") / leaf) is True


def test_guard_rejects_non_dedicated_leaf():
    assert snap.is_canonical_path(Path("data/whatever")) is True


def test_guard_allows_dedicated_leaf_and_scratch():
    assert snap.is_canonical_path(Path("data/estimate_snapshots")) is False
    assert snap.is_canonical_path(Path("/tmp/snap_demo")) is False


def test_guard_rejects_symlink_into_forbidden_tree(tmp_path):
    # A symlink named 'estimate_snapshots' that actually points into fmp_harvest
    # must be caught by resolve()-following the link.
    target = tmp_path / "data" / "fmp_harvest"
    target.mkdir(parents=True)
    link = tmp_path / "estimate_snapshots"
    link.symlink_to(target, target_is_directory=True)
    assert snap.is_canonical_path(link) is True


def test_guard_rejects_scratch_symlink_into_data(tmp_path):
    # A /tmp-looking symlink that resolves into a real data tree is NOT scratch.
    real = tmp_path / "data" / "fmp_harvest"
    real.mkdir(parents=True)
    link = Path("/tmp/_rq205_test_symlink_estimate_snapshots")
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(real, target_is_directory=True)
    try:
        assert snap.is_canonical_path(link) is True
    finally:
        link.unlink()


# --- 6. manifest hashes -------------------------------------------------------
def test_manifest_sha256_matches_parquet(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    res = _run(monkeypatch, tickers=tickers, as_of=as_of, out_root=tmp_path,
               policy=_all_ok_policy())
    for endpoint in snap.ENDPOINTS:
        man = json.loads((res["out_dir"] / f"{endpoint}.manifest.json").read_text())
        recomputed = snap._sha256_file(res["out_dir"] / f"{endpoint}.parquet")
        assert man["sha256"] == recomputed
        assert man["as_of"] == as_of
        assert man["status"] == "ok"
        assert man["ticker_count"] == len(tickers)


def test_dry_run_writes_nothing(monkeypatch, tmp_path, tickers):
    as_of = date.today().isoformat()
    monkeypatch.setattr(snap, "fetch_endpoint", _make_fetch(_all_ok_policy()))
    res = snap.collect_snapshot(
        session=_FakeSession(_all_ok_policy()),
        tickers=tickers,
        api_key="",
        as_of=as_of,
        out_root=tmp_path,
        dry_run=True,
    )
    assert res["status"] == "dry_run"
    assert res["published"] is False
    assert not (tmp_path / as_of).exists()
