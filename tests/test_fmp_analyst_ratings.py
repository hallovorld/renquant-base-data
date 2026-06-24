"""Tests for the FMP historical analyst-ratings fetcher (no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from renquant_base_data.fetchers.fmp_analyst_ratings import (
    FETCH_ERROR,
    NO_COVERAGE,
    QUOTA_ERROR,
    SOURCE,
    WITH_DATA,
    FmpRatingsStore,
    consensus_score,
    fetch_grades_historical,
    parse_grades,
)


def test_consensus_score():
    assert consensus_score(10, 0, 0, 0, 0) == (2.0, 10)
    assert consensus_score(2, 0, 0, 0, 2) == (0.0, 4)
    s, n = consensus_score(0, 0, 0, 0, 0)
    assert np.isnan(s) and n == 0


def _payload():
    return [
        {"symbol": "AAPL", "date": "2026-06-01", "analystRatingsStrongBuy": 7,
         "analystRatingsBuy": 23, "analystRatingsHold": 15, "analystRatingsSell": 1,
         "analystRatingsStrongSell": 2},
        {"symbol": "AAPL", "date": "2026-05-01", "analystRatingsStrongBuy": 6,
         "analystRatingsBuy": 22, "analystRatingsHold": 16, "analystRatingsSell": 2,
         "analystRatingsStrongSell": 2},
    ]


def test_parse_grades_computes_consensus_and_sorts():
    df = parse_grades("AAPL", _payload())
    assert list(df["date"]) == [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")]  # ascending
    last = df.iloc[-1]
    s, n = consensus_score(7, 23, 15, 1, 2)
    assert last["consensus"] == s and last["n_analysts"] == n


def test_fetch_uses_injected_getter():
    res = fetch_grades_historical("AAPL", "k", getter=lambda t: _payload())
    assert res.status == WITH_DATA
    assert len(res.frame) == 2 and res.frame["ticker"].iloc[0] == "AAPL"


def test_fetch_status_distinguishes_coverage_quota_and_error():
    # empty list = the ticker genuinely has no grades (valid, not an error)
    no_cov = fetch_grades_historical("ZZZZ", "k", getter=lambda t: [])
    assert no_cov.status == NO_COVERAGE and no_cov.frame.empty
    # FMP returns {"Error Message": "Limit Reach..."} on a free-tier 429
    quota = fetch_grades_historical("AAPL", "k",
                                    getter=lambda t: {"Error Message": "Limit Reach . upgrade"})
    assert quota.status == QUOTA_ERROR and quota.frame.empty
    # a transient/garbage failure is a fetch_error, NOT no_coverage, and never raises
    boom = fetch_grades_historical("AAPL", "k",
                                   getter=lambda t: (_ for _ in ()).throw(ValueError("net")))
    assert boom.status == FETCH_ERROR
    assert fetch_grades_historical("AAPL", "k", getter=lambda t: None).status == FETCH_ERROR


def test_parse_grades_stamps_source_provenance():
    df = parse_grades("AAPL", _payload())
    assert (df["source"] == SOURCE).all()


def test_store_append_merge_dedup(tmp_path):
    store = FmpRatingsStore(tmp_path / "r.parquet")
    store.upsert([parse_grades("AAPL", _payload())])
    # re-pull with a changed latest month → keep latest, no dup
    p2 = _payload(); p2[0]["analystRatingsStrongBuy"] = 9
    df = store.upsert([parse_grades("AAPL", p2)])
    aapl = df[df["ticker"] == "AAPL"]
    assert len(aapl) == 2  # two distinct months
    jun = aapl[aapl["date"] == pd.Timestamp("2026-06-01")].iloc[0]
    assert jun["analystRatingsStrongBuy"] == 9


def test_refresh_cli_smoke(tmp_path, monkeypatch):
    from renquant_base_data import fmp_analyst_ratings_refresh as R
    out = tmp_path / "r.parquet"
    summary = R.refresh_fmp_ratings(
        watchlist=["AAPL", "ZZZZ"], output=out, api_key="k", sleep_sec=0,
        getter=lambda t: _payload() if t == "AAPL" else [])
    assert summary["with_data"] == 1 and summary["empty"] == 1
    assert summary["tickers_in_store"] == 1


# ── incremental staleness-aware selection (daily small-batch cron) ─────────
def test_select_to_refresh_picks_missing_and_stale_first():
    from renquant_base_data.fmp_analyst_ratings_refresh import select_to_refresh
    wl = ["AAPL", "MSFT", "NVDA", "TSLA"]
    # AAPL fetched today, MSFT 10d ago; NVDA/TSLA never fetched
    existing = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "fetched_at": [pd.Timestamp("2026-06-24"), pd.Timestamp("2026-06-14")],
    })
    pick = select_to_refresh(wl, existing, max_pull=3, today=pd.Timestamp("2026-06-24"))
    assert pick[:2] == ["NVDA", "TSLA"]  # never-fetched first
    assert pick[2] == "MSFT"             # then oldest fetched_at
    assert "AAPL" not in pick            # freshest, skipped this run


def test_select_to_refresh_zero_means_all():
    from renquant_base_data.fmp_analyst_ratings_refresh import select_to_refresh
    wl = ["A", "B", "C"]
    assert select_to_refresh(wl, None, max_pull=0, today=pd.Timestamp("2026-06-24")) == wl


def test_parse_grades_stamps_fetched_at():
    df = parse_grades("AAPL", _payload(), asof=pd.Timestamp("2026-06-24"))
    assert (df["fetched_at"] == pd.Timestamp("2026-06-24")).all()


def test_refresh_incremental_only_pulls_batch(tmp_path, monkeypatch):
    from renquant_base_data import fmp_analyst_ratings_refresh as R
    out = tmp_path / "r.parquet"
    calls = []
    def getter(t):
        calls.append(t); return _payload()
    s = R.refresh_fmp_ratings(watchlist=["AAPL", "MSFT", "NVDA"], output=out,
                              api_key="k", sleep_sec=0, max_pull=2,
                              asof=pd.Timestamp("2026-06-24"), getter=getter)
    assert s["pulled_this_run"] == 2 and len(calls) == 2  # only 2, not all 3


# ── outcome buckets + gates (Codex #24 finding #1+#2): a quota hit must NOT ──
#    be mistaken for empty coverage, and a thin run must be able to fail-closed.
def test_refresh_buckets_quota_separately_from_no_coverage(tmp_path):
    from renquant_base_data import fmp_analyst_ratings_refresh as R
    out = tmp_path / "r.parquet"
    def getter(t):
        if t == "AAPL":
            return _payload()                       # with_data
        if t == "ZZZZ":
            return []                                # no_coverage (valid)
        return {"Error Message": "Limit Reach"}      # quota_error
    s = R.refresh_fmp_ratings(watchlist=["AAPL", "ZZZZ", "MSFT"], output=out,
                              api_key="k", sleep_sec=0, max_pull=0,
                              asof=pd.Timestamp("2026-06-24"), getter=getter)
    assert s["with_data"] == 1 and s["no_coverage"] == 1
    assert s["quota_error"] == 1 and s["errors_total"] == 1
    assert s["coverage_pct"] == round(100 / 3, 1)


def test_evaluate_gates_fail_on_error_and_min_coverage():
    from renquant_base_data.fmp_analyst_ratings_refresh import evaluate_gates
    thin = {"errors_total": 1, "error_samples": ["MSFT:quota_error"],
            "coverage_pct": 33.3, "with_data": 1, "requested": 3}
    # any error trips --fail-on-error
    assert evaluate_gates(thin, min_coverage_pct=0.0, fail_on_error=True)
    # below-bar coverage trips --min-coverage-pct
    assert evaluate_gates(thin, min_coverage_pct=90.0, fail_on_error=False)
    # a clean, full run passes both gates
    full = {"errors_total": 0, "error_samples": [], "coverage_pct": 100.0,
            "with_data": 3, "requested": 3}
    assert evaluate_gates(full, min_coverage_pct=90.0, fail_on_error=True) == []


def test_main_returns_nonzero_when_gate_trips(tmp_path, monkeypatch):
    from renquant_base_data import fmp_analyst_ratings_refresh as R
    from renquant_base_data.fetchers.fmp_analyst_ratings import FetchResult, parse_grades
    wl = tmp_path / "wl.json"
    wl.write_text('{"watchlist": ["AAPL", "MSFT"]}', encoding="utf-8")
    out = tmp_path / "r.parquet"
    monkeypatch.setenv("FMP_API_KEY", "k")
    # every ticker comes back a quota error → --fail-on-error must exit non-zero
    monkeypatch.setattr(R, "fetch_grades_historical",
                        lambda t, key, **kw: FetchResult(parse_grades(t, None), QUOTA_ERROR, "Limit Reach"))
    rc = R.main(["--watchlist", str(wl), "--output", str(out),
                 "--sleep-sec", "0", "--max-pull", "0", "--fail-on-error"])
    assert rc == 1  # quota errors must fail the run, not pass silently
