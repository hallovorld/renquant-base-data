"""CLI to refresh Finnhub analyst recommendation trends (daily cron).

Pulls ``/stock/recommendation`` per watchlist ticker into
``data/analyst_ratings_finnhub.parquet`` and append-merges (dedup by
(ticker, period)) so the recommendation history ACCUMULATES over time — the
free Finnhub window is only ~4 months, but a daily cron grows a multi-month
series for the 3-month REVISION feature. Coverage is BROAD but not proven full —
an empty response is ambiguous ``no_coverage`` (ETF/index, delisted/unsupported,
vendor-empty, or no current recs), so the summary reports ``active_coverage_pct``
/ ``no_coverage_pct`` over the full requested set, never just coverable cov. The
fail-closed coverage gate is ``--min-active-coverage-pct`` (floors
``active_coverage_pct`` = with_data/requested), so a widespread-empty run can't
pass; ``--min-coverage-pct`` (over the coverable set, which excludes the
ambiguous bucket) is a diagnostic floor only (Codex #25). Key from
``FINNHUB_API_KEY`` (.env, gitignored). Free 60 calls/min → throttle ~1s;
~145 names ≈ 2.5 min.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from renquant_base_data.fetchers.finnhub_analyst_ratings import (
    FinnhubRatingsStore,
    fetch_recommendations,
)
from renquant_base_data.fetchers.fmp_analyst_ratings import (
    NO_COVERAGE,
    QUOTA_ERROR,
    WITH_DATA,
)
from renquant_base_data.fmp_analyst_ratings_refresh import (
    evaluate_gates,
    load_watchlist,
    select_to_refresh,
)

log = logging.getLogger("renquant_base_data.finnhub_analyst_ratings_refresh")


def refresh_finnhub_ratings(*, watchlist: list[str], output: str | Path, api_key: str,
                            sleep_sec: float = 1.0, max_pull: int | None = None,
                            asof=None, getter=None) -> dict:
    """Pull this run's batch and append-merge. Buckets each ticker by outcome —
    with_data / no_coverage / quota_error / fetch_error — so a 429 or bad key is
    never silently counted as 'no coverage'.

    Coverage honesty (Codex #25, same class as FMP #24): an empty response is
    AMBIGUOUS — it may be an ETF/index (genuinely no analysts), a delisted/
    unsupported name, a vendor outage, or a real stock with no current
    recommendations. The fetcher cannot tell which. So we do NOT assume
    no_coverage == ETF: `coverage_pct` (over the coverable set, excluding
    no_coverage) is reported ALONGSIDE the honest `active_coverage_pct` (over the
    full requested set) and `no_coverage_pct` + `no_coverage_samples`, so a high
    coverable cov can never be misread as full active-watchlist coverage and the
    gap stays visible."""
    import pandas as pd  # noqa: PLC0415
    asof = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.today().normalize()
    store = FinnhubRatingsStore(Path(output))
    todo = select_to_refresh(watchlist, store.load(), max_pull, asof)
    frames: list = []
    buckets: dict[str, int] = {}
    errors: list[str] = []
    no_cov_names: list[str] = []
    _benign = (WITH_DATA, NO_COVERAGE)
    for i, t in enumerate(todo):
        res = fetch_recommendations(t, api_key, asof=asof, getter=getter)
        buckets[res.status] = buckets.get(res.status, 0) + 1
        if res.status == WITH_DATA:
            frames.append(res.frame)
        elif res.status == NO_COVERAGE:
            no_cov_names.append(t)
        elif res.status not in _benign:
            errors.append(f"{t}:{res.status}")
        if sleep_sec and i < len(todo) - 1:
            time.sleep(sleep_sec)
    df = store.upsert(frames)
    requested = len(todo)
    with_data = buckets.get(WITH_DATA, 0)
    no_cov = buckets.get(NO_COVERAGE, 0)
    errors_total = sum(v for k, v in buckets.items() if k not in _benign)
    coverable = requested - no_cov  # excludes the (ambiguous) no_coverage set
    return {
        "watchlist": len(watchlist), "requested": requested, "pulled_this_run": requested,
        "with_data": with_data, "no_coverage": no_cov,
        "quota_error": buckets.get(QUOTA_ERROR, 0),
        "fetch_error": errors_total - buckets.get(QUOTA_ERROR, 0),
        "errors_total": errors_total, "coverable": coverable,
        # coverable coverage (excludes no_coverage) AND the honest active view
        # over the full requested set, so an empty/ETF/uncovered name can never
        # silently inflate coverage (Codex #25).
        "coverage_pct": round(100.0 * with_data / coverable, 1) if coverable else 0.0,
        "active_coverage_pct": round(100.0 * with_data / requested, 1) if requested else 0.0,
        "no_coverage_pct": round(100.0 * no_cov / requested, 1) if requested else 0.0,
        "no_coverage_samples": no_cov_names[:20],
        "total_rows": int(len(df)),
        "tickers_in_store": int(df["ticker"].nunique()) if len(df) else 0,
        "error_samples": errors[:10], "source": "finnhub_recommendation", "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--watchlist", required=True)
    p.add_argument("--output", default="data/analyst_ratings_finnhub.parquet")
    p.add_argument("--sleep-sec", type=float, default=1.0,
                   help="throttle between calls (free tier 60/min)")
    p.add_argument("--max-pull", type=int, default=0,
                   help="0 = whole watchlist daily (~2.5 min; active coverage is "
                        "whatever Finnhub returns, not assumed full). N = "
                        "incremental most-stale batch.")
    p.add_argument("--min-active-coverage-pct", type=float, default=0.0,
                   help="FAIL-CLOSED coverage control: exit non-zero if "
                        "active_coverage_pct (with_data/requested, over the FULL "
                        "watchlist) falls below this %%. This is the gate to use — "
                        "the ambiguous no_coverage bucket (ETF/index OR delisted/"
                        "uncovered/vendor-empty) counts AGAINST it, so a mostly-"
                        "empty run cannot pass (Codex #25).")
    p.add_argument("--min-coverage-pct", type=float, default=0.0,
                   help="DIAGNOSTIC ONLY — does NOT affect exit status. Compares "
                        "coverage_pct (with_data/coverable), which EXCLUDES the "
                        "ambiguous no_coverage bucket — 5/145 with_data still reads "
                        "100%% coverable cov, so it cannot fail-close on a widespread-"
                        "empty run and is NOT wired to rc. Use --min-active-coverage-pct "
                        "(the only coverage fail-closed control).")
    p.add_argument("--fail-on-error", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        log.error("FINNHUB_API_KEY not set"); return 1
    summary = refresh_finnhub_ratings(
        watchlist=load_watchlist(args.watchlist), output=args.output,
        api_key=key, sleep_sec=args.sleep_sec, max_pull=args.max_pull)
    # The ONLY coverage-related fail-closed control for Finnhub is
    # --min-active-coverage-pct (active_coverage_pct = with_data/requested), so the
    # ambiguous no_coverage bucket counts against it; --fail-on-error is independent.
    # --min-coverage-pct (coverable, excludes no_coverage) is DIAGNOSTIC ONLY and is
    # NOT fed into the exit-status gate — passing it would let a widespread-empty run
    # read 100%% coverable and silently pass (Codex #25 round 2). We hard-pass 0.0 so
    # evaluate_gates never derives a coverable-floor violation for this caller.
    violations = evaluate_gates(summary, min_coverage_pct=0.0,
                                fail_on_error=args.fail_on_error,
                                min_active_coverage_pct=args.min_active_coverage_pct)
    summary["gate_violations"] = violations
    # Diagnostic-only comparison of the coverable metric — logged, recorded, but it
    # can NEVER change rc.
    if args.min_coverage_pct > 0:
        cov = summary.get("coverage_pct", 0.0)
        below = cov < args.min_coverage_pct
        summary["coverage_pct_diagnostic"] = {
            "coverage_pct": cov, "floor": args.min_coverage_pct,
            "below_floor": below, "note": "diagnostic only — does not affect exit status"}
        if below:
            log.info("DIAGNOSTIC (not a gate): coverable coverage_pct %.1f%% < %.1f%% "
                     "— ignored for exit status; --min-active-coverage-pct is the gate",
                     cov, args.min_coverage_pct)
    print(json.dumps(summary))
    if violations:
        for v in violations:
            log.error("gate failed: %s", v)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
