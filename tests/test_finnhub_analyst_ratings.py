"""Tests for the Finnhub analyst-recommendation fetcher (no network)."""
from __future__ import annotations

import pandas as pd

from renquant_base_data.fetchers.finnhub_analyst_ratings import (
    SOURCE,
    FinnhubRatingsStore,
    fetch_recommendations,
    parse_recommendations,
)
from renquant_base_data.fetchers.fmp_analyst_ratings import (
    FETCH_ERROR,
    NO_COVERAGE,
    WITH_DATA,
)


def _payload():
    return [
        {"symbol": "AAPL", "period": "2026-06-01", "strongBuy": 12, "buy": 20,
         "hold": 8, "sell": 1, "strongSell": 0},
        {"symbol": "AAPL", "period": "2026-05-01", "strongBuy": 10, "buy": 21,
         "hold": 9, "sell": 1, "strongSell": 1},
    ]


def test_parse_computes_consensus_sorts_and_stamps_source():
    df = parse_recommendations("AAPL", _payload())
    assert list(df["period"]) == [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")]
    assert (df["source"] == SOURCE).all()
    last = df.iloc[-1]
    # consensus = (2*12 + 20 - 1 - 0) / 41
    assert abs(last["consensus"] - (2 * 12 + 20 - 1 - 0) / 41) < 1e-9
    assert last["n_analysts"] == 41


def test_fetch_with_data_and_empty_no_coverage():
    ok = fetch_recommendations("AAPL", "k", getter=lambda t: _payload())
    assert ok.status == WITH_DATA and len(ok.frame) == 2
    # an empty list → no_coverage (NOT an error); the symbol may be an ETF/index,
    # delisted/unsupported, or a real stock with no current recs — fetcher can't tell
    empty = fetch_recommendations("SPY", "k", getter=lambda t: [])
    assert empty.status == NO_COVERAGE and empty.frame.empty
    # a thrown error → fetch_error, never raises
    boom = fetch_recommendations("AAPL", "k",
                                 getter=lambda t: (_ for _ in ()).throw(ValueError("net")))
    assert boom.status == FETCH_ERROR


def test_store_append_merge_dedup(tmp_path):
    store = FinnhubRatingsStore(tmp_path / "r.parquet")
    store.upsert([parse_recommendations("AAPL", _payload())])
    p2 = _payload(); p2[0]["strongBuy"] = 15           # same period, changed count
    df = store.upsert([parse_recommendations("AAPL", p2)])
    aapl = df[df["ticker"] == "AAPL"]
    assert len(aapl) == 2                               # two distinct periods, no dup
    jun = aapl[aapl["period"] == pd.Timestamp("2026-06-01")].iloc[0]
    assert jun["strongBuy"] == 15                       # latest write kept


def test_refresh_buckets_empty_as_no_coverage_not_error(tmp_path):
    from renquant_base_data import finnhub_analyst_ratings_refresh as R
    out = tmp_path / "r.parquet"
    def getter(t):
        return _payload() if t in ("AAPL", "MSFT") else []   # SPY/GLD → empty (ambiguous)
    s = R.refresh_finnhub_ratings(watchlist=["AAPL", "MSFT", "SPY", "GLD"], output=out,
                                  api_key="k", sleep_sec=0, max_pull=0,
                                  asof=pd.Timestamp("2026-06-25"), getter=getter)
    assert s["with_data"] == 2 and s["no_coverage"] == 2
    assert s["errors_total"] == 0                       # empty responses are NOT errors
    assert s["coverable"] == 2 and s["coverage_pct"] == 100.0   # over coverable set
    # honesty metrics (Codex #25): coverable cov reads 100% but only HALF the
    # active watchlist actually returned data — the no_coverage gap stays visible
    # (these names may be ETFs OR uncovered stocks; the fetcher can't tell).
    assert s["active_coverage_pct"] == 50.0
    assert s["no_coverage_pct"] == 50.0
    assert set(s["no_coverage_samples"]) == {"SPY", "GLD"}


def _summary(with_data: int, no_coverage: int):
    """A minimal Finnhub-shaped summary for gate tests: ``requested`` names of
    which ``with_data`` returned data and ``no_coverage`` were empty (ambiguous).
    ``coverage_pct`` is over the COVERABLE set (excludes no_coverage) so it reads
    high even when almost nothing returned; ``active_coverage_pct`` is over the
    FULL requested set so the empty bucket counts against it."""
    requested = with_data + no_coverage
    coverable = requested - no_coverage
    return {
        "errors_total": 0, "error_samples": [],
        "requested": requested, "with_data": with_data,
        "no_coverage": no_coverage, "coverable": coverable,
        "coverage_pct": round(100.0 * with_data / coverable, 1) if coverable else 0.0,
        "active_coverage_pct": round(100.0 * with_data / requested, 1) if requested else 0.0,
    }


def test_active_gate_fails_widespread_empty_while_coverable_reads_full():
    """Codex #25: 5/145 return data, 140 empty. coverage_pct (coverable) = 100%
    so a 90% --min-coverage-pct floor would PASS the run — but the active gate
    over the full requested set (3.4%) correctly FAILS it fail-closed."""
    from renquant_base_data.fmp_analyst_ratings_refresh import evaluate_gates
    s = _summary(with_data=5, no_coverage=140)
    assert s["coverage_pct"] == 100.0          # coverable reads full — the trap
    assert s["active_coverage_pct"] == 3.4
    # the diagnostic coverable gate is fooled (no violation at a 90% floor)
    assert evaluate_gates(s, min_coverage_pct=90.0, fail_on_error=False) == []
    # the fail-closed ACTIVE gate is not
    assert evaluate_gates(s, min_coverage_pct=0.0, fail_on_error=False,
                          min_active_coverage_pct=90.0)


def test_active_gate_passes_healthy_baseline():
    """A healthy run (136/145 return data) clears the active gate."""
    from renquant_base_data.fmp_analyst_ratings_refresh import evaluate_gates
    s = _summary(with_data=136, no_coverage=9)
    assert s["active_coverage_pct"] == 93.8
    assert evaluate_gates(s, min_coverage_pct=0.0, fail_on_error=False,
                          min_active_coverage_pct=90.0) == []


def test_active_gate_threshold_boundary():
    """The active gate is a strict '< floor' check: exactly at the floor PASSES,
    one tick below FAILS."""
    from renquant_base_data.fmp_analyst_ratings_refresh import evaluate_gates
    # 90/100 == exactly 90.0% → at the floor, PASS
    at = _summary(with_data=90, no_coverage=10)
    assert at["active_coverage_pct"] == 90.0
    assert evaluate_gates(at, min_coverage_pct=0.0, fail_on_error=False,
                          min_active_coverage_pct=90.0) == []
    # 89/100 == 89.0% → just below the floor, FAIL
    below = _summary(with_data=89, no_coverage=11)
    assert below["active_coverage_pct"] == 89.0
    assert evaluate_gates(below, min_coverage_pct=0.0, fail_on_error=False,
                          min_active_coverage_pct=90.0)


def test_active_gate_off_by_default_preserves_shared_evaluate_gates():
    """min_active_coverage_pct=None (the default) leaves the shared gate's
    FMP behaviour unchanged — the active gate never fires."""
    from renquant_base_data.fmp_analyst_ratings_refresh import evaluate_gates
    s = _summary(with_data=5, no_coverage=140)   # would fail an active floor
    assert evaluate_gates(s, min_coverage_pct=90.0, fail_on_error=False) == []


def test_main_active_gate_trips_nonzero_on_widespread_empty(tmp_path, monkeypatch):
    """End-to-end: the Finnhub CLI exits non-zero when --min-active-coverage-pct
    is breached by a mostly-empty run, even though --min-coverage-pct would pass."""
    from renquant_base_data import finnhub_analyst_ratings_refresh as R
    monkeypatch.setenv("FINNHUB_API_KEY", "k")
    out = tmp_path / "r.parquet"
    monkeypatch.setattr(R, "load_watchlist",
                        lambda _p: ["AAPL", "SPY", "GLD", "QQQ", "DIA"])

    def getter(t):
        return _payload() if t == "AAPL" else []   # 1/5 with data → 20% active

    orig = R.refresh_finnhub_ratings
    monkeypatch.setattr(
        R, "refresh_finnhub_ratings",
        lambda **kw: orig(**{**kw, "output": out, "sleep_sec": 0, "getter": getter}))
    rc = R.main(["--watchlist", "wl.json", "--min-active-coverage-pct", "90"])
    assert rc == 1
    # the diagnostic coverable floor alone would NOT have caught it
    rc_ok = R.main(["--watchlist", "wl.json", "--min-coverage-pct", "90"])
    assert rc_ok == 0
