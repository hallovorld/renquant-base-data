"""Contract tests for the C1 PIT revision-drift feature pipeline.

The load-bearing contracts (M-SIG spec §1.1 + the PIT hard invariant):

  * PIT discipline (ADVERSARIAL): a snapshot dated D+1 — even one carrying a
    10x consensus jump and a brand-new symbol — must NOT influence any feature
    row at as_of <= D;
  * the frozen drift formula, with the |denominator| and the matched
    fiscal-target (roll-artifact) handling;
  * the spec's "no analyst update in the window ==> EXCLUDED, not zero" rule;
  * NO backfill: a lake too young for a window is honestly NaN
    ("no_lag_snapshot"), and no feature row exists before the anchor;
  * missing-day tolerance (a weekday hole and a status=partial day both
    degrade gracefully);
  * incremental idempotence (re-run == byte-stable no-op; a new day == rebuild;
    --force == rebuild);
  * the readiness arithmetic (6/9 calendar months from the REAL anchor,
    end-of-month clamping, busday window maturity);
  * the out-root guard (only a pit_features leaf or /tmp scratch is writable —
    the snapshot lake itself is never a legal output target).

All lakes here are synthetic fixtures under tmp_path — no live data is read.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_base_data import pit_revision_features as feat


# ─────────────────────────── synthetic lake fixture ──────────────────────────

def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_day(
    root: Path,
    day: date,
    symbols: dict,
    *,
    status: str = "ok",
) -> None:
    """Write one published snapshot day.

    ``symbols[sym]`` = {"fiscal": {fiscal_end_iso: (epsAvg, epsHigh, epsLow, nEps)},
                        "tgt": targetConsensus | None,
                        "grades": (strongBuy, buy, hold, sell, strongSell) | None}
    """
    day_dir = root / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    est_rows, tgt_rows, grd_rows, sum_rows = [], [], [], []
    for sym, spec in symbols.items():
        for fe, (avg, high, low, n) in (spec.get("fiscal") or {}).items():
            est_rows.append(
                {"symbol": sym, "date": fe, "epsAvg": avg, "epsHigh": high,
                 "epsLow": low, "numAnalystsEps": n,
                 "snapshot_as_of": day.isoformat()}
            )
        if spec.get("tgt") is not None:
            tgt_rows.append(
                {"symbol": sym, "targetHigh": spec["tgt"] * 1.2,
                 "targetLow": spec["tgt"] * 0.8,
                 "targetConsensus": spec["tgt"], "targetMedian": spec["tgt"],
                 "snapshot_as_of": day.isoformat()}
            )
        if spec.get("grades") is not None:
            sb, b, h, s, ss = spec["grades"]
            grd_rows.append(
                {"symbol": sym, "strongBuy": sb, "buy": b, "hold": h,
                 "sell": s, "strongSell": ss, "consensus": "Buy",
                 "snapshot_as_of": day.isoformat()}
            )
        sum_rows.append({"symbol": sym, "lastMonthCount": 1,
                         "snapshot_as_of": day.isoformat()})

    frames = {
        "analyst_estimates": pd.DataFrame(
            est_rows, columns=["symbol", "date", "epsAvg", "epsHigh", "epsLow",
                               "numAnalystsEps", "snapshot_as_of"]),
        "price_target_consensus": pd.DataFrame(
            tgt_rows, columns=["symbol", "targetHigh", "targetLow",
                               "targetConsensus", "targetMedian", "snapshot_as_of"]),
        "grades_consensus": pd.DataFrame(
            grd_rows, columns=["symbol", "strongBuy", "buy", "hold", "sell",
                               "strongSell", "consensus", "snapshot_as_of"]),
        "price_target_summary": pd.DataFrame(
            sum_rows, columns=["symbol", "lastMonthCount", "snapshot_as_of"]),
    }
    for endpoint, df in frames.items():
        pq = day_dir / f"{endpoint}.parquet"
        df.to_parquet(pq, index=False)
        manifest = {
            "endpoint": endpoint, "as_of": day.isoformat(), "status": status,
            "rows": int(len(df)), "tickers": int(df["symbol"].nunique()) if len(df) else 0,
            "ticker_count": int(df["symbol"].nunique()) if len(df) else 0,
            "requested": len(symbols), "coverage": 1.0,
            "sha256": _sha256(pq),
        }
        (day_dir / f"{endpoint}.manifest.json").write_text(json.dumps(manifest))


def steady_symbol(eps: float = 1.0, fiscal_end: str = "2027-12-31") -> dict:
    return {"fiscal": {fiscal_end: (eps, eps * 1.1, eps * 0.9, 10)},
            "tgt": 100.0, "grades": (2, 1, 1, 0, 0)}


ANCHOR = date(2026, 1, 5)  # a Monday
FY_END = "2027-12-31"


def build_simple_lake(root: Path, n_days: int = 24, *, eps_path=None) -> list[date]:
    """A lake of n_days weekdays; AAA's FY1 epsAvg follows eps_path (default:
    1.0 with a bump to 1.1 halfway), BBB is constant forever (the no-update
    name)."""
    days = _weekdays(ANCHOR, n_days)
    for i, d in enumerate(days):
        eps = eps_path(i) if eps_path else (1.1 if i >= n_days // 2 else 1.0)
        write_day(root, d, {
            "AAA": {"fiscal": {FY_END: (eps, eps + 0.2, eps - 0.2, 10)},
                    "tgt": 100.0 + i, "grades": (2, 1, 1, 0, 0)},
            "BBB": steady_symbol(2.0),
        })
    return days


# ─────────────────────────── PIT discipline (adversarial) ────────────────────

class TestPITDiscipline:
    def test_snapshot_at_d_plus_1_cannot_influence_features_at_d(self, tmp_path):
        root = tmp_path / "lake"
        days = build_simple_lake(root, 24)
        d_last = days[-1]

        truncated = feat.build_features(root, days)

        # adversarial next day: 10x consensus jump, huge target/grade swing,
        # AND a brand-new symbol that never existed before
        next_day = _weekdays(d_last + timedelta(days=1), 1)[0]
        write_day(root, next_day, {
            "AAA": {"fiscal": {FY_END: (11.0, 12.0, 10.0, 99)},
                    "tgt": 999.0, "grades": (0, 0, 0, 0, 9)},
            "BBB": steady_symbol(20.0),
            "NEW": steady_symbol(5.0),
        })

        full = feat.build_features(root)
        full_leq_d = full[full["as_of"] <= d_last].reset_index(drop=True)

        pd.testing.assert_frame_equal(truncated, full_leq_d)
        # and the new symbol must not appear before its first snapshot
        assert "NEW" not in set(full_leq_d["symbol"])

    def test_no_feature_rows_before_the_anchor(self, tmp_path):
        root = tmp_path / "lake"
        days = build_simple_lake(root, 5)
        df = feat.build_features(root)
        assert df["as_of"].min() == days[0]

    def test_young_lake_is_honestly_nan_not_backfilled(self, tmp_path):
        root = tmp_path / "lake"
        build_simple_lake(root, 2)  # far too young for any window
        df = feat.build_features(root)
        assert df["revision_drift_1m"].isna().all()
        assert df["revision_drift_5d"].isna().all()
        assert (df["excluded_reason_1m"] == "no_lag_snapshot").all()


# ─────────────────────────── the frozen formula ───────────────────────────────

class TestFrozenFormula:
    def test_revision_drift_1m_hand_computed(self, tmp_path):
        root = tmp_path / "lake"
        days = build_simple_lake(root, 24)  # eps 1.0 -> 1.1 at i=12
        df = feat.build_features(root)
        d_last = days[-1]  # i=23; tau = 21td back = days[2] (eps 1.0)
        row = df[(df["symbol"] == "AAA") & (df["as_of"] == d_last)].iloc[0]
        assert row["revision_drift_1m"] == pytest.approx((1.1 - 1.0) / 1.0)
        assert bool(row["fy1_updated_1m"]) is True
        assert row["excluded_reason_1m"] == ""
        assert row["fy1_eps_avg"] == pytest.approx(1.1)
        assert row["fy1_eps_avg_lag_1m"] == pytest.approx(1.0)

    def test_denominator_is_absolute_value(self, tmp_path):
        # eps goes -2.0 -> -1.0: drift must be +0.5 (improvement), not -0.5
        root = tmp_path / "lake"
        build_simple_lake(root, 24, eps_path=lambda i: -2.0 if i < 12 else -1.0)
        df = feat.build_features(root)
        row = df[df["symbol"] == "AAA"].iloc[-1]
        assert row["revision_drift_1m"] == pytest.approx((-1.0 - -2.0) / 2.0)

    def test_zero_denominator_is_nan(self, tmp_path):
        root = tmp_path / "lake"
        build_simple_lake(root, 24, eps_path=lambda i: 0.0 if i < 12 else 1.0)
        df = feat.build_features(root)
        row = df[df["symbol"] == "AAA"].iloc[-1]
        assert np.isnan(row["revision_drift_1m"])
        assert row["excluded_reason_1m"] == "zero_denominator"

    def test_usable_from_is_next_business_day(self, tmp_path):
        root = tmp_path / "lake"
        days = build_simple_lake(root, 5)  # Mon..Fri
        df = feat.build_features(root)
        fri = days[4]
        row = df[(df["symbol"] == "AAA") & (df["as_of"] == fri)].iloc[0]
        assert row["usable_from"] == fri + timedelta(days=3)  # Fri -> Mon


class TestFiscalTargetMatching:
    def test_fiscal_roll_does_not_manufacture_drift(self, tmp_path):
        """FY1 at t is matched at the lag date even when the lag date's own
        FY1 was the (since-ended) prior fiscal year — never fiscal time."""
        root = tmp_path / "lake"
        start = date(2026, 6, 1)  # Monday
        days = _weekdays(start, 32)  # through 2026-07-14
        fy_a, fy_b = "2026-06-30", "2027-06-30"
        for i, d in enumerate(days):
            # FY-A (ending mid-window) has a wild path; FY-B revises 2.0 -> 2.2
            eps_b = 2.2 if d >= date(2026, 7, 1) else 2.0
            write_day(root, d, {
                "AAA": {"fiscal": {fy_a: (10.0 - i, 11.0, 9.0, 5),
                                   fy_b: (eps_b, 2.5, 1.5, 7)},
                        "tgt": 50.0, "grades": (1, 1, 1, 0, 0)},
            })
        df = feat.build_features(root)
        d_last = days[-1]  # 2026-07-14: FY1 = FY-B; 21td lag = 2026-06-15 (FY1 was FY-A)
        row = df[df["as_of"] == d_last].iloc[0]
        assert str(row["fy1_fiscal_end"]) == fy_b
        assert row["revision_drift_1m"] == pytest.approx((2.2 - 2.0) / 2.0)
        assert bool(row["fy1_rolled_1m"]) is True


# ─────────────────────────── exclusion rules ──────────────────────────────────

class TestNoUpdateExclusion:
    def test_flat_consensus_is_excluded_not_zero(self, tmp_path):
        root = tmp_path / "lake"
        build_simple_lake(root, 24)
        df = feat.build_features(root)
        row = df[df["symbol"] == "BBB"].iloc[-1]  # BBB never updates
        assert np.isnan(row["revision_drift_1m"])
        assert row["excluded_reason_1m"] == "no_update_in_window"
        assert bool(row["excluded_1m"]) is True
        # the pre-exclusion raw value is kept for transparency, and is 0.0
        assert row["revision_drift_1m_raw"] == pytest.approx(0.0)

    def test_excluded_fraction_is_reportable_per_date(self, tmp_path):
        root = tmp_path / "lake"
        build_simple_lake(root, 24)
        df = feat.build_features(root)
        last = df[df["as_of"] == df["as_of"].max()]
        # AAA updated (included), BBB flat (excluded) -> 0.5
        assert last["excluded_1m"].mean() == pytest.approx(0.5)


# ─────────────────────────── companions (documented choices) ─────────────────

class TestCompanionFeatures:
    def test_breadth_counts_consensus_moves(self, tmp_path):
        # path: 1.0 x10, 1.1, 1.2, 1.15, then flat -> n_up=2 n_down=1 -> 1/3
        def eps_path(i):
            return {10: 1.1, 11: 1.2, 12: 1.15}.get(i, 1.15 if i > 12 else 1.0)

        root = tmp_path / "lake"
        build_simple_lake(root, 24, eps_path=eps_path)
        df = feat.build_features(root)
        row = df[df["symbol"] == "AAA"].iloc[-1]
        assert row["revision_breadth_1m"] == pytest.approx((2 - 1) / 3)

    def test_target_drift_and_grade_migration(self, tmp_path):
        root = tmp_path / "lake"
        days = _weekdays(ANCHOR, 24)
        for i, d in enumerate(days):
            grades = (2, 1, 1, 0, 0) if i < 12 else (1, 1, 2, 0, 0)
            write_day(root, d, {
                "AAA": {"fiscal": {FY_END: (1.0, 1.2, 0.8, 10)},
                        "tgt": 100.0 if i < 12 else 110.0, "grades": grades},
            })
        df = feat.build_features(root)
        row = df[df["as_of"] == days[-1]].iloc[0]
        assert row["target_drift_1m"] == pytest.approx((110.0 - 100.0) / 100.0)
        # score: (2*2+1)/4 = 1.25 -> (2*1+1)/4 = 0.75; delta = -0.5
        assert row["grade_migration_1m"] == pytest.approx(-0.5)


# ─────────────────────────── missing-day tolerance ───────────────────────────

class TestMissingDayTolerance:
    def test_weekday_hole_and_partial_day_degrade_gracefully(self, tmp_path):
        root = tmp_path / "lake"
        days = _weekdays(ANCHOR, 24)
        for i, d in enumerate(days):
            if i == 10:
                continue  # true lapse (no dir at all)
            status = "partial" if i == 15 else "ok"  # failed fetch, unpublished
            eps = 1.1 if i >= 12 else 1.0
            write_day(root, d, {"AAA": {"fiscal": {FY_END: (eps, 1.3, 0.9, 10)},
                                        "tgt": 100.0, "grades": (1, 1, 1, 0, 0)}},
                      status=status)

        published = feat.list_published_days(root)
        assert days[10] not in published and days[15] not in published
        assert len(published) == 22

        df = feat.build_features(root)
        assert set(df["as_of"]) == set(published)  # partial day emits no rows
        # the last day's 21td lag lands where a snapshot exists via <=tau lookup
        row = df[df["as_of"] == days[-1]].iloc[0]
        assert row["revision_drift_1m"] == pytest.approx(0.1)


# ─────────────────────────── incremental build + idempotence ─────────────────

class TestIncrementalBuild:
    def test_idempotent_no_op_then_new_day_rebuild(self, tmp_path):
        root = tmp_path / "lake"
        out = tmp_path / "pit_features"
        days = build_simple_lake(root, 6)

        r1 = feat.build_and_publish(root, out)
        assert r1["status"] == "built"
        pq = out / "c1_revision_drift.parquet"
        mf = out / "c1_revision_drift.manifest.json"
        sha1, manifest1 = _sha256(pq), mf.read_bytes()

        r2 = feat.build_and_publish(root, out)
        assert r2["status"] == "up_to_date"
        assert _sha256(pq) == sha1 and mf.read_bytes() == manifest1

        # a new published day triggers a rebuild that includes it
        nd = _weekdays(days[-1] + timedelta(days=1), 1)[0]
        write_day(root, nd, {"AAA": steady_symbol(1.0), "BBB": steady_symbol(2.0)})
        r3 = feat.build_and_publish(root, out)
        assert r3["status"] == "built"
        manifest3 = json.loads(mf.read_text())
        assert manifest3["processed_days"][-1] == nd.isoformat()
        assert manifest3["as_of_max"] == nd.isoformat()

        r4 = feat.build_and_publish(root, out, force=True)
        assert r4["status"] == "built"

    def test_manifest_carries_input_hashes_and_code_sha(self, tmp_path):
        root = tmp_path / "lake"
        out = tmp_path / "pit_features"
        days = build_simple_lake(root, 3)
        feat.build_and_publish(root, out)
        manifest = json.loads((out / "c1_revision_drift.manifest.json").read_text())
        assert manifest["code_sha256"] == feat.code_sha256()
        assert manifest["sha256"] == _sha256(out / "c1_revision_drift.parquet")
        d0 = days[0].isoformat()
        lake_manifest = json.loads(
            (root / d0 / "analyst_estimates.manifest.json").read_text())
        assert manifest["input_sha256"][d0]["analyst_estimates"] == lake_manifest["sha256"]
        assert manifest["spec"].startswith("renquant-orchestrator doc/design/2026-07-02")

    def test_out_root_guard_refuses_canonical_and_lake_paths(self, tmp_path):
        root = tmp_path / "lake"
        build_simple_lake(root, 2)
        for bad in [root, tmp_path / "estimate_snapshots",
                    tmp_path / "fmp_harvest", tmp_path / "score_db",
                    Path("/Users/renhao/git/github/RenQuant/data/anything")]:
            with pytest.raises(SystemExit):
                feat.build_and_publish(root, bad)
        # NOTE: tmp_path itself is /private/var/folders scratch, so the guard
        # accepts it — the leaf rule is for non-scratch targets (see next test).
        assert feat.is_writable_out_root(Path("/Users/renhao/data/pit_features"))
        assert not feat.is_writable_out_root(Path("/Users/renhao/data/other"))


# ─────────────────────────── readiness arithmetic ────────────────────────────

class TestReadinessArithmetic:
    def test_add_months_clamps_month_end(self):
        assert feat.add_months(date(2026, 7, 2), 6) == date(2027, 1, 2)
        assert feat.add_months(date(2026, 7, 2), 9) == date(2027, 4, 2)
        assert feat.add_months(date(2026, 8, 31), 6) == date(2027, 2, 28)
        assert feat.add_months(date(2026, 11, 30), 3) == date(2027, 2, 28)
        assert feat.add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)

    def test_readiness_block_from_real_anchor(self):
        r = feat.readiness(date(2026, 7, 2))
        assert r["confirmatory_unlock"] == "2027-01-02"
        assert r["second_checkpoint"] == "2027-04-02"
        assert r["monitoring_bound"] == "2027-12-31"
        # hand-computed: 21 busdays after Thu 2026-07-02 is Fri 2026-07-31
        assert r["window_maturity"]["1m"] == "2026-07-31"
        assert r["window_maturity"]["5d"] == "2026-07-09"

    def test_empty_lake_readiness_is_explicitly_not_started(self):
        r = feat.readiness(None)
        assert r["anchor_first_snapshot"] is None
        assert r["confirmatory_unlock"] is None

    def test_coverage_report_readiness_and_gaps(self, tmp_path):
        root = tmp_path / "lake"
        out = tmp_path / "pit_features"
        days = build_simple_lake(root, 24)
        feat.build_and_publish(root, out)
        rep = feat.coverage_report(
            root, out / "c1_revision_drift.parquet", today=days[-1])
        assert rep["days_accrued"] == 24
        assert rep["first_snapshot"] == ANCHOR.isoformat()
        assert rep["missed_weekdays"] == []
        assert rep["readiness"]["confirmatory_unlock"] == feat.add_months(ANCHOR, 6).isoformat()
        stats = rep["exploratory_feature_stats"]
        assert stats["rows"] == 48
        assert "EXPLORATORY" in stats["label"]
        # remove a middle day -> it must show up as missed
        hole = days[10]
        import shutil
        shutil.rmtree(root / hole.isoformat())
        rep2 = feat.coverage_report(root, today=days[-1])
        assert hole.isoformat() in rep2["missed_weekdays"]


# ─────────────────────────── CLI smoke ────────────────────────────────────────

class TestCLI:
    def test_build_then_report_roundtrip(self, tmp_path, capsys):
        root = tmp_path / "lake"
        out = tmp_path / "pit_features"
        build_simple_lake(root, 6)
        rc = feat.main(["build", "--snapshot-root", str(root), "--out", str(out)])
        assert rc == 0
        assert (out / "c1_revision_drift.parquet").is_file()
        capsys.readouterr()
        rc = feat.main([
            "report", "--snapshot-root", str(root),
            "--features", str(out / "c1_revision_drift.parquet"), "--json",
        ])
        assert rc == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["days_accrued"] == 6
        assert rep["readiness"]["confirmatory_unlock"] == feat.add_months(ANCHOR, 6).isoformat()
