"""C1 PIT revision-drift feature pipeline (M-SIG C1 serving path, sprint D2).

Turns the daily PIT estimate-snapshot lake (written by
``renquant_base_data.fmp_estimate_revisions``, base-data #27; scheduled by
renquant-orchestrator ``ops/pit/``) into the C1 estimate-revision-drift feature
table, per the frozen C1 specification in renquant-orchestrator
``doc/design/2026-07-02-m-sig-signal-stack-spec.md`` §1.1.

This is a PRE-BUILD (flag-off): the snapshot lake started accruing 2026-07-02
and the C1 confirmatory test unlocks only after 6 calendar months of accrual
(the spec's frozen accrual cutoff). Building the serving path now means that
when the snapshots mature, testing + serving C1 is parameter tuning, not new
engineering. The output is a RESEARCH lake (``data/pit_features/``), not a
production input — nothing downstream consumes it yet.

PIT DISCIPLINE (HARD INVARIANT)
-------------------------------
A feature row at ``as_of = D`` is computed ONLY from snapshots with
``snapshot_as_of <= D``. Snapshot days are actual UTC fetch dates (the
collector's own PIT invariant forbids backdating), so the join is natural:
"what did the consensus look like on day D" is exactly the ``<= D`` slice of
the lake. There is NO backfill: the lake starts 2026-07-02, so any feature
needing a lag observation before that date is honestly absent (NaN), never
reconstructed. Tests enforce that a snapshot dated D+1 cannot influence the
feature at D.

FROZEN PRIMARY FEATURE (spec §1.1, may not be tuned)
----------------------------------------------------
``revision_drift_1m(i, t) =
    (FY1_consensus(i, t) - FY1_consensus(i, t - 21td)) / |FY1_consensus(i, t - 21td)|``

where ``t - 21td`` is 21 trading days prior and ``FY1_consensus(i, tau)`` is
the most recent FY1 EPS consensus estimate (``epsAvg``) with
``available_at <= tau``. FY1 only — FY2 and any blending are explicitly out of
scope for the frozen gate. 1m (21td) is the PRIMARY window; the 3m (63td) and
5d variants below are SECONDARY DIAGNOSTICS ONLY and never gate GO/KILL
(running both and picking a winner is the researcher-degrees-of-freedom problem
the spec closes).

Documented choices where the spec leaves freedom
------------------------------------------------
* **Fiscal-target matching (never fiscal time).** All windows are windows of
  SNAPSHOT time. ``FY1`` is selected at the decision date ``t`` (the nearest
  fiscal year end ``>= t`` with a non-null ``epsAvg`` in the latest snapshot
  ``<= t``), and the lag value is the consensus for THE SAME fiscal target read
  from the lag snapshot. This avoids the fiscal-year roll artifact (comparing
  FY2027 now vs FY2026 a month ago would manufacture a huge fake "revision"
  once a year). A ``fy1_rolled_1m`` diagnostic flags rows where the lag
  snapshot's own FY1 differs from ``t``'s, so the confirmatory harness can
  check sensitivity to this interpretation.
* **"No analyst update in the window" exclusion (spec: excluded, not zero).**
  Individual analyst updates are unobservable in a consensus lake; the
  observable proxy is any change in the FY1 row's
  ``(epsAvg, epsHigh, epsLow, numAnalystsEps)`` across consecutive snapshots
  from the lag observation through ``t``. No observed change ==> the name is
  EXCLUDED from that date's cross-section (``revision_drift_1m = NaN``,
  ``excluded_reason_1m = "no_update_in_window"``); the pre-exclusion value is
  kept in ``revision_drift_1m_raw`` for transparency. A true update that
  leaves all four fields bit-identical is undetectable and is conservatively
  treated as no-update.
* **Trading-day calendar.** ``numpy`` busday (Mon-Fri) — a documented
  approximation that ignores NYSE holidays (~9/yr). The snapshot collector
  itself fires on all weekdays (launchd Weekday 1-5, holiday-blind) so the
  weekday axis matches the SNAPSHOT axis better than an exchange calendar
  would; and because every lag lookup is "latest snapshot <= tau", a +-1-day
  calendar wobble only shifts which snapshot is read, never fabricates data.
* **Exploratory companions** (all clearly non-gating, chosen + documented
  here): ``revision_drift_5d`` / ``revision_drift_3m`` (5td / 63td, same
  frozen formula + exclusion rules as the primary); ``revision_breadth_1m``
  (= (n_up - n_down)/(n_up + n_down) over consecutive-snapshot FY1 ``epsAvg``
  changes in the 1m window, NaN when no changes — consensus-level proxy for
  up/down revision breadth); ``target_drift_1m`` (same drift formula on
  ``price_target_consensus.targetConsensus``); ``grade_migration_1m``
  (Delta of the grade score ``(2*strongBuy + buy - sell - 2*strongSell) /
  n_grades`` over the 1m window, from ``grades_consensus``).
* **Publication lag.** The spec's as-of lag (decisions at ``t`` use features
  available at ``t - 1td``) is the CONSUMER's join responsibility; this table
  carries ``usable_from`` (= ``as_of + 1td``) so that join is trivial and
  cannot be silently skipped.

OUTPUT
------
``<out_root>/c1_revision_drift.parquet`` + ``c1_revision_drift.manifest.json``
(hand-rolled evidence manifest matching the snapshot lake's conventions: input
sha256 per consumed snapshot manifest, code sha256, row/symbol counts,
readiness block). Writes are atomic (tempfile + ``os.replace``); an
already-current output is a NO-OP (incremental: the builder detects new
published snapshot days and rebuilds deterministically — the lake is small, a
full deterministic rebuild is the simplest correct "incremental" and makes
idempotence trivial).

The out-root guard mirrors ``fmp_estimate_revisions.is_canonical_path``: only a
``pit_features`` leaf or an explicit /tmp scratch target is writable; the
snapshot lake itself is read-only to this module by construction.

Usage::

    # build/refresh the feature lake (no-op when current)
    python -m renquant_base_data.pit_revision_features build \
        --snapshot-root /Users/renhao/git/github/RenQuant/data/estimate_snapshots \
        --out /Users/renhao/git/github/RenQuant/data/pit_features

    # coverage / C1-test readiness report (add --json for machine-readable)
    python -m renquant_base_data.pit_revision_features report \
        --snapshot-root /Users/renhao/git/github/RenQuant/data/estimate_snapshots
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("renquant_base_data.pit_revision_features")

# The four endpoints the collector publishes per day; a day is consumable only
# when ALL FOUR manifests exist with status == "ok" (the liveness contract).
from .fmp_estimate_revisions import ENDPOINTS  # noqa: E402  (same package)

SPEC_REF = (
    "renquant-orchestrator doc/design/2026-07-02-m-sig-signal-stack-spec.md "
    "SS1.1 (C1 - estimate-revision drift; frozen 2026-07-02 r4)"
)
DATASET_NAME = "c1_revision_drift"
SCHEMA_VERSION = 1

# Trading-day windows (SNAPSHOT time, never fiscal time). "1m" (21td) is the
# spec's frozen PRIMARY; the others are secondary diagnostics and never gate.
WINDOWS_TD: dict[str, int] = {"5d": 5, "1m": 21, "3m": 63}
PRIMARY_WINDOW = "1m"

# Spec §1.1 accrual arithmetic: the first confirmatory read uses data accrued
# through exactly 6 calendar months from the lake's ACTUAL first published
# snapshot date (the anchor); a second checkpoint sits at 9 months; C1's
# monitoring as a G106 candidate is bounded to 2027-Q4 (it never rolls forward
# indefinitely). C1 is INFORMATIVE-ONLY at both checkpoints — it never
# independently votes GO/KILL and sits outside the §2a Bonferroni family.
CONFIRMATORY_MONTHS = 6
SECOND_CHECKPOINT_MONTHS = 9
MONITORING_BOUND = date(2027, 12, 31)

_FY1_FIELDS = ("eps_avg", "eps_high", "eps_low", "n_analysts")

# Mirrors fmp_estimate_revisions._FORBIDDEN_LEAVES in spirit: this builder must
# never write into any canonical/production input, nor into the snapshot lake.
_FORBIDDEN_LEAVES = {
    "fmp_harvest",
    "sec_fundamentals_daily",
    "rawlabel.parquet",
    "score_db",
    "estimate_snapshots",
}


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------

def busday_offset(day: date, offset: int) -> date:
    """Mon-Fri business-day offset (documented weekday approximation)."""
    d64 = np.busday_offset(np.datetime64(day, "D"), offset, roll="backward")
    return d64.astype("datetime64[D]").astype(date)


def add_months(day: date, months: int) -> date:
    """Calendar-month addition with end-of-month clamping (2026-08-31 + 6mo
    -> 2027-02-28), used for the spec's 6/9-calendar-month accrual arithmetic."""
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    # clamp the day-of-month to the target month's length
    next_month_start = date(year + (month == 12), month % 12 + 1, 1)
    last_dom = (next_month_start - timedelta(days=1)).day
    return date(year, month, min(day.day, last_dom))


def _weekdays_between(start: date, end: date) -> list[date]:
    """All Mon-Fri dates in [start, end]."""
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# lake reading (STRICTLY read-only)
# ---------------------------------------------------------------------------

def _parse_day(name: str) -> date | None:
    try:
        return date.fromisoformat(name)
    except ValueError:
        return None


def day_is_published_ok(day_dir: Path) -> bool:
    """A snapshot day is consumable iff every endpoint manifest exists with
    status == "ok" (mirrors the liveness check's contract, stricter than the
    collector's presence-only ``_snapshot_is_published``)."""
    for endpoint in ENDPOINTS:
        mpath = day_dir / f"{endpoint}.manifest.json"
        if not mpath.is_file():
            return False
        try:
            manifest = json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if manifest.get("status") != "ok":
            return False
        parquet = day_dir / f"{endpoint}.parquet"
        if not parquet.is_file() or parquet.stat().st_size == 0:
            return False
    return True


def list_published_days(snapshot_root: Path) -> list[date]:
    """Sorted consumable snapshot days. Non-day dirs and partial/unpublished
    days are skipped with a warning (missing-day tolerance: every lag lookup is
    'latest snapshot <= tau', so a hole degrades gracefully, never errors)."""
    if not snapshot_root.is_dir():
        return []
    days: list[date] = []
    for child in sorted(snapshot_root.iterdir()):
        if not child.is_dir():
            continue
        day = _parse_day(child.name)
        if day is None:
            continue
        if day_is_published_ok(child):
            days.append(day)
        else:
            log.warning("skipping unpublished/partial snapshot day: %s", child)
    return days


def input_fingerprints(snapshot_root: Path, days: Sequence[date]) -> dict[str, dict[str, str]]:
    """Per-day, per-endpoint parquet sha256, read from the lake's own
    manifests (the collector already hashes each published parquet)."""
    out: dict[str, dict[str, str]] = {}
    for day in days:
        per_ep: dict[str, str] = {}
        for endpoint in ENDPOINTS:
            manifest = json.loads(
                (snapshot_root / day.isoformat() / f"{endpoint}.manifest.json").read_text()
            )
            per_ep[endpoint] = manifest.get("sha256") or ""
        out[day.isoformat()] = per_ep
    return out


# ---------------------------------------------------------------------------
# per-symbol PIT lookup structures
# ---------------------------------------------------------------------------

@dataclass
class _SymbolEstimates:
    """Per-symbol snapshot history of the analyst_estimates endpoint."""

    snaps: list[date] = field(default_factory=list)  # sorted snapshot dates
    # rows[snap][fiscal_end] = (eps_avg, eps_high, eps_low, n_analysts)
    rows: dict[date, dict[date, tuple]] = field(default_factory=dict)

    def latest_snap_leq(self, day: date) -> date | None:
        i = bisect.bisect_right(self.snaps, day)
        return self.snaps[i - 1] if i else None

    def snaps_in(self, lo: date, hi: date) -> list[date]:
        i = bisect.bisect_left(self.snaps, lo)
        j = bisect.bisect_right(self.snaps, hi)
        return self.snaps[i:j]

    def fy1_target(self, snap: date, as_of: date) -> date | None:
        """Nearest fiscal year end >= as_of with a non-null epsAvg."""
        candidates = [
            fe
            for fe, vals in self.rows.get(snap, {}).items()
            if fe >= as_of and vals[0] is not None and not _isnan(vals[0])
        ]
        return min(candidates) if candidates else None

    def value(self, snap: date, fiscal_end: date) -> tuple | None:
        return self.rows.get(snap, {}).get(fiscal_end)


@dataclass
class _SymbolSeries:
    """Per-symbol snapshot history of a scalar series (target / grade score)."""

    snaps: list[date] = field(default_factory=list)
    values: dict[date, float] = field(default_factory=dict)

    def latest_leq(self, day: date) -> tuple[date, float] | None:
        i = bisect.bisect_right(self.snaps, day)
        if not i:
            return None
        s = self.snaps[i - 1]
        return s, self.values[s]


def _isnan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _to_date(x: Any) -> date | None:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    try:
        return pd.Timestamp(x).date()
    except (TypeError, ValueError):
        return None


def load_lake(snapshot_root: Path, days: Sequence[date]) -> tuple[
    dict[str, _SymbolEstimates], dict[str, _SymbolSeries], dict[str, _SymbolSeries]
]:
    """Load the published days into per-symbol PIT lookup structures.

    Returns (estimates, target_consensus, grade_score) keyed by symbol."""
    est: dict[str, _SymbolEstimates] = {}
    tgt: dict[str, _SymbolSeries] = {}
    grd: dict[str, _SymbolSeries] = {}

    for day in days:
        day_dir = snapshot_root / day.isoformat()

        est_path = day_dir / "analyst_estimates.parquet"
        if est_path.is_file():
            df = pd.read_parquet(est_path)
            for sym, grp in df.groupby("symbol"):
                se = est.setdefault(str(sym), _SymbolEstimates())
                fiscal_rows: dict[date, tuple] = {}
                for rec in grp.itertuples(index=False):
                    fe = _to_date(getattr(rec, "date", None))
                    if fe is None:
                        continue
                    fiscal_rows[fe] = (
                        _float_or_none(getattr(rec, "epsAvg", None)),
                        _float_or_none(getattr(rec, "epsHigh", None)),
                        _float_or_none(getattr(rec, "epsLow", None)),
                        _float_or_none(getattr(rec, "numAnalystsEps", None)),
                    )
                if fiscal_rows:
                    se.snaps.append(day)
                    se.rows[day] = fiscal_rows

        tgt_path = day_dir / "price_target_consensus.parquet"
        if tgt_path.is_file():
            df = pd.read_parquet(tgt_path)
            for rec in df.itertuples(index=False):
                v = _float_or_none(getattr(rec, "targetConsensus", None))
                if v is None:
                    continue
                ss = tgt.setdefault(str(rec.symbol), _SymbolSeries())
                ss.snaps.append(day)
                ss.values[day] = v

        grd_path = day_dir / "grades_consensus.parquet"
        if grd_path.is_file():
            df = pd.read_parquet(grd_path)
            for rec in df.itertuples(index=False):
                score = _grade_score(rec)
                if score is None:
                    continue
                ss = grd.setdefault(str(rec.symbol), _SymbolSeries())
                ss.snaps.append(day)
                ss.values[day] = score

    return est, tgt, grd


def _float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _grade_score(rec: Any) -> float | None:
    """(2*strongBuy + buy - sell - 2*strongSell) / n_grades in [-2, 2]."""
    try:
        sb = float(rec.strongBuy)
        b = float(rec.buy)
        h = float(rec.hold)
        s = float(rec.sell)
        ss = float(rec.strongSell)
    except (AttributeError, TypeError, ValueError):
        return None
    total = sb + b + h + s + ss
    if not total:
        return None
    return (2.0 * sb + b - s - 2.0 * ss) / total


# ---------------------------------------------------------------------------
# feature computation
# ---------------------------------------------------------------------------

# Stable output schema (also used to keep an empty-lake build well-formed).
FEATURE_COLUMNS = [
    "symbol",
    "as_of",
    "usable_from",
    "revision_drift_5d",
    "revision_drift_1m",
    "revision_drift_3m",
    "revision_drift_1m_raw",
    "revision_breadth_1m",
    "fy1_updated_1m",
    "fy1_rolled_1m",
    "n_est_snapshots_1m",
    "fy1_eps_avg_lag_1m",
    "excluded_1m",
    "excluded_reason_1m",
    "fy1_fiscal_end",
    "fy1_eps_avg",
    "fy1_num_analysts",
    "target_drift_1m",
    "grade_migration_1m",
]


def _drift(now: float | None, lag: float | None) -> float:
    """The frozen drift formula; NaN on missing/zero-denominator."""
    if now is None or lag is None or lag == 0.0:
        return float("nan")
    return (now - lag) / abs(lag)


def _est_window_features(
    se: _SymbolEstimates, as_of: date, window_td: int
) -> dict[str, Any]:
    """The frozen drift + exclusion machinery for one (symbol, as_of, window).

    Returns keys: drift (post-exclusion), raw, updated, n_snaps, rolled,
    lag_eps, reason, breadth."""
    nanrow = {
        "drift": float("nan"), "raw": float("nan"), "breadth": float("nan"),
        "updated": False, "rolled": False, "n_snaps": 0,
        "lag_eps": float("nan"), "reason": "",
    }
    latest = se.latest_snap_leq(as_of)
    if latest is None:
        return {**nanrow, "reason": "no_snapshot"}
    fy1 = se.fy1_target(latest, as_of)
    if fy1 is None:
        return {**nanrow, "reason": "no_fy1"}
    now_vals = se.value(latest, fy1)
    eps_now = now_vals[0] if now_vals else None

    tau = busday_offset(as_of, -window_td)
    lag_snap = se.latest_snap_leq(tau)
    if lag_snap is None:
        # lake does not extend back far enough — honestly absent, no backfill
        return {**nanrow, "reason": "no_lag_snapshot"}
    lag_vals = se.value(lag_snap, fy1)
    if lag_vals is None or lag_vals[0] is None:
        return {**nanrow, "reason": "no_fy1_row_at_lag"}
    eps_lag = lag_vals[0]

    raw = _drift(eps_now, eps_lag)
    reason = "zero_denominator" if math.isnan(raw) else ""

    # fiscal-roll diagnostic: did the lag snapshot's own FY1 differ?
    fy1_at_lag = se.fy1_target(lag_snap, tau)
    rolled = fy1_at_lag is not None and fy1_at_lag != fy1

    # update detection + breadth over [lag_snap] + snaps in [tau, as_of]
    seq_snaps = se.snaps_in(tau, as_of)
    if lag_snap < tau:
        seq_snaps = [lag_snap] + seq_snaps
    seq = [se.value(s, fy1) for s in seq_snaps]
    updated = False
    n_up = n_down = 0
    prev = None
    for vals in seq:
        if vals is None:
            continue
        if prev is not None:
            if any(_ne(prev[k], vals[k]) for k in range(len(_FY1_FIELDS))):
                updated = True
            if prev[0] is not None and vals[0] is not None:
                if vals[0] > prev[0]:
                    n_up += 1
                elif vals[0] < prev[0]:
                    n_down += 1
        prev = vals
    breadth = (n_up - n_down) / (n_up + n_down) if (n_up + n_down) else float("nan")

    drift = raw
    if not updated and not reason:
        # spec §1.1: a name with no analyst update in the window is EXCLUDED
        # from that date's cross-section, not treated as zero drift.
        drift = float("nan")
        reason = "no_update_in_window"

    return {
        "drift": drift, "raw": raw, "breadth": breadth,
        "updated": updated, "rolled": rolled,
        "n_snaps": len(se.snaps_in(tau, as_of)),
        "lag_eps": eps_lag if eps_lag is not None else float("nan"),
        "reason": reason,
    }


def _ne(a: float | None, b: float | None) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return a != b


def _series_drift(ss: _SymbolSeries | None, as_of: date, window_td: int, *, delta: bool = False) -> float:
    """Windowed drift (or plain delta) on a scalar PIT series."""
    if ss is None:
        return float("nan")
    now = ss.latest_leq(as_of)
    if now is None:
        return float("nan")
    lag = ss.latest_leq(busday_offset(as_of, -window_td))
    if lag is None:
        return float("nan")
    if delta:
        return now[1] - lag[1]
    return _drift(now[1], lag[1])


def build_features(
    snapshot_root: Path, days: Sequence[date] | None = None
) -> pd.DataFrame:
    """Compute the C1 feature table over all published snapshot days.

    One row per (symbol, as_of) where as_of ranges over the published snapshot
    days and symbols over every name seen in any endpoint at <= as_of. PIT:
    row (i, D) reads only snapshots dated <= D."""
    days = list(days) if days is not None else list_published_days(snapshot_root)
    est, tgt, grd = load_lake(snapshot_root, days)

    all_symbols = sorted(set(est) | set(tgt) | set(grd))
    records: list[dict[str, Any]] = []
    for as_of in days:
        for sym in all_symbols:
            se = est.get(sym)
            # symbols with no data at all yet as of this day are skipped
            has_any = (
                (se is not None and se.latest_snap_leq(as_of) is not None)
                or (sym in tgt and tgt[sym].latest_leq(as_of) is not None)
                or (sym in grd and grd[sym].latest_leq(as_of) is not None)
            )
            if not has_any:
                continue

            row: dict[str, Any] = {
                "symbol": sym,
                "as_of": as_of,
                "usable_from": busday_offset(as_of, 1),
            }

            per_window = {
                name: (
                    _est_window_features(se, as_of, td)
                    if se is not None
                    else _est_window_features(_SymbolEstimates(), as_of, td)
                )
                for name, td in WINDOWS_TD.items()
            }
            for name, wf in per_window.items():
                row[f"revision_drift_{name}"] = wf["drift"]
            prim = per_window[PRIMARY_WINDOW]
            row["revision_drift_1m_raw"] = prim["raw"]
            row["revision_breadth_1m"] = prim["breadth"]
            row["fy1_updated_1m"] = bool(prim["updated"])
            row["fy1_rolled_1m"] = bool(prim["rolled"])
            row["n_est_snapshots_1m"] = int(prim["n_snaps"])
            row["fy1_eps_avg_lag_1m"] = prim["lag_eps"]
            row["excluded_1m"] = bool(math.isnan(prim["drift"]))
            row["excluded_reason_1m"] = prim["reason"]

            # FY1 level diagnostics at as_of
            fy1_end: date | None = None
            eps_now = float("nan")
            n_analysts = float("nan")
            if se is not None:
                latest = se.latest_snap_leq(as_of)
                if latest is not None:
                    fy1_end = se.fy1_target(latest, as_of)
                    if fy1_end is not None:
                        vals = se.value(latest, fy1_end)
                        if vals is not None:
                            eps_now = vals[0] if vals[0] is not None else float("nan")
                            n_analysts = vals[3] if vals[3] is not None else float("nan")
            row["fy1_fiscal_end"] = fy1_end
            row["fy1_eps_avg"] = eps_now
            row["fy1_num_analysts"] = n_analysts

            row["target_drift_1m"] = _series_drift(tgt.get(sym), as_of, WINDOWS_TD["1m"])
            row["grade_migration_1m"] = _series_drift(
                grd.get(sym), as_of, WINDOWS_TD["1m"], delta=True
            )
            records.append(row)

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    if not df.empty:
        df = df.sort_values(["as_of", "symbol"], kind="mergesort").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# readiness arithmetic (spec §1.1 accrual cutoffs)
# ---------------------------------------------------------------------------

def readiness(anchor: date | None) -> dict[str, Any]:
    """The C1 accrual/readiness block. ``anchor`` is the lake's ACTUAL first
    published snapshot date (the spec: record the real first snapshot, not a
    merge date). None => lake empty, nothing accruing."""
    if anchor is None:
        return {
            "anchor_first_snapshot": None,
            "confirmatory_unlock": None,
            "second_checkpoint": None,
            "monitoring_bound": MONITORING_BOUND.isoformat(),
            "note": "snapshot lake is empty - accrual has not started",
        }
    unlock = add_months(anchor, CONFIRMATORY_MONTHS)
    second = add_months(anchor, SECOND_CHECKPOINT_MONTHS)
    return {
        "anchor_first_snapshot": anchor.isoformat(),
        "confirmatory_unlock": unlock.isoformat(),
        "second_checkpoint": second.isoformat(),
        "monitoring_bound": MONITORING_BOUND.isoformat(),
        "window_maturity": {
            name: busday_offset(anchor, td).isoformat()
            for name, td in WINDOWS_TD.items()
        },
        "note": (
            "C1 is INFORMATIVE-ONLY at both checkpoints (spec SS1.1): it never "
            "independently gates GO/KILL and is excluded from the SS2a "
            "Bonferroni family; any IC read before the confirmatory unlock is "
            "EXPLORATORY and never substitutes for the confirmatory result."
        ),
    }


# ---------------------------------------------------------------------------
# manifest + atomic publish
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def is_writable_out_root(out_root: Path) -> bool:
    """Guard (mirrors the collector's ``is_canonical_path`` logic): the ONLY
    writable targets are a dedicated ``pit_features`` leaf or an explicit /tmp
    scratch tree; canonical leaves (incl. the snapshot lake itself) are always
    refused, judged on both the given and the symlink-resolved path."""
    resolved = out_root.resolve()
    parts = set(out_root.parts) | set(resolved.parts)
    if parts & _FORBIDDEN_LEAVES:
        return False
    from .fmp_estimate_revisions import _is_scratch_arg

    if _is_scratch_arg(out_root) and _is_scratch_arg(resolved):
        return True
    return out_root.name == "pit_features" and resolved.name == "pit_features"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_and_publish(
    snapshot_root: Path, out_root: Path, *, force: bool = False
) -> dict[str, Any]:
    """Incremental build: NO-OP when the output already reflects the current
    set of published snapshot days AND this module's code is unchanged;
    otherwise a full deterministic rebuild published atomically."""
    snapshot_root = Path(snapshot_root)
    if not is_writable_out_root(out_root):
        raise SystemExit(
            f"REFUSING out-root {out_root}: only a dedicated 'pit_features' "
            f"leaf or an explicit /tmp scratch target is writable (the "
            f"snapshot lake and every canonical path are read-only here)"
        )
    # The snapshot lake is READ-ONLY to this builder: refuse any out-root that
    # is, contains, or sits inside the lake — even under a /tmp scratch tree.
    sr, orr = snapshot_root.resolve(), out_root.resolve()
    if orr == sr or sr in orr.parents or orr in sr.parents:
        raise SystemExit(
            f"REFUSING out-root {out_root}: it overlaps the snapshot lake "
            f"{snapshot_root}, which is read-only to this builder"
        )
    days = list_published_days(snapshot_root)
    code_sha = code_sha256()

    parquet_path = out_root / f"{DATASET_NAME}.parquet"
    manifest_path = out_root / f"{DATASET_NAME}.manifest.json"

    if not force and manifest_path.is_file() and parquet_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}
        if (
            prior.get("processed_days") == [d.isoformat() for d in days]
            and prior.get("code_sha256") == code_sha
            and prior.get("sha256") == _sha256_file(parquet_path)
        ):
            return {
                "status": "up_to_date",
                "days": len(days),
                "rows": prior.get("rows"),
                "parquet": str(parquet_path),
            }

    df = build_features(snapshot_root, days)
    out_root.mkdir(parents=True, exist_ok=True)

    # atomic parquet publish
    fd, tmp = tempfile.mkstemp(prefix=f".{DATASET_NAME}.", suffix=".parquet", dir=out_root)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, parquet_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    anchor = days[0] if days else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "spec": SPEC_REF,
        "snapshot_root": str(snapshot_root),
        "processed_days": [d.isoformat() for d in days],
        "input_sha256": input_fingerprints(snapshot_root, days),
        "code_sha256": code_sha,
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()) if not df.empty else 0,
        "as_of_min": days[0].isoformat() if days else None,
        "as_of_max": days[-1].isoformat() if days else None,
        "windows_trading_days": WINDOWS_TD,
        "primary_window": PRIMARY_WINDOW,
        "feature_columns": [c for c in df.columns] if not df.empty else [],
        "readiness": readiness(anchor),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sha256": _sha256_file(parquet_path),
        "status": "ok",
    }
    _atomic_write_bytes(manifest_path, json.dumps(manifest, indent=2).encode() + b"\n")
    return {
        "status": "built",
        "days": len(days),
        "rows": int(len(df)),
        "parquet": str(parquet_path),
        "manifest": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# coverage / readiness report
# ---------------------------------------------------------------------------

def coverage_report(
    snapshot_root: Path,
    features_path: Path | None = None,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Days accrued, coverage, missed weekdays, and the C1-test readiness
    dates computed from the REAL lake (spec §1.1 accrual arithmetic)."""
    snapshot_root = Path(snapshot_root)
    today = today or datetime.now(timezone.utc).date()
    days = list_published_days(snapshot_root)
    anchor = days[0] if days else None

    report: dict[str, Any] = {
        "snapshot_root": str(snapshot_root),
        "today": today.isoformat(),
        "days_accrued": len(days),
        "first_snapshot": anchor.isoformat() if anchor else None,
        "last_snapshot": days[-1].isoformat() if days else None,
        "readiness": readiness(anchor),
    }

    if anchor:
        expected = _weekdays_between(anchor, today)
        published = set(days)
        missed = [d.isoformat() for d in expected if d not in published]
        report["expected_weekdays_since_anchor"] = len(expected)
        report["missed_weekdays"] = missed
        report["missed_weekdays_note"] = (
            "weekday axis; entries here may be NYSE holidays (collector may "
            "legitimately skip), today's not-yet-fired scheduled run, or true "
            "lapses - every true lapse is PERMANENTLY unrecoverable (PIT "
            "invariant: no backfill)"
        )
        unlock = add_months(anchor, CONFIRMATORY_MONTHS)
        report["days_to_confirmatory_unlock"] = max(0, (unlock - today).days)

        # per-endpoint coverage from the latest day's own manifests
        latest_dir = snapshot_root / days[-1].isoformat()
        cov: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            m = json.loads((latest_dir / f"{endpoint}.manifest.json").read_text())
            cov[endpoint] = {
                "tickers": m.get("ticker_count"),
                "requested": m.get("requested"),
                "coverage": m.get("coverage"),
            }
        report["latest_day_endpoint_coverage"] = cov

    # interim EXPLORATORY stats from the feature lake, if built (never gates)
    if features_path and Path(features_path).is_file():
        df = pd.read_parquet(features_path)
        stats: dict[str, Any] = {"rows": int(len(df))}
        if not df.empty:
            stats["symbols"] = int(df["symbol"].nunique())
            stats["as_of_max"] = str(df["as_of"].max())
            per_date_excluded = df.groupby("as_of")["excluded_1m"].mean()
            stats["excluded_fraction_1m_latest"] = float(per_date_excluded.iloc[-1])
            stats["excluded_fraction_1m_mean"] = float(per_date_excluded.mean())
            prim = df[f"revision_drift_{PRIMARY_WINDOW}"]
            stats["revision_drift_1m_nonnull"] = int(prim.notna().sum())
        stats["label"] = (
            "EXPLORATORY ONLY (spec SS1.1): C1 is informative-only; nothing "
            "here is a confirmatory read and nothing here may gate GO/KILL"
        )
        report["exploratory_feature_stats"] = stats

    return report


def _format_report(rep: dict[str, Any]) -> str:
    r = rep.get("readiness", {})
    lines = [
        "C1 PIT revision-drift - coverage / readiness",
        f"  lake:                    {rep['snapshot_root']}",
        f"  today:                   {rep['today']}",
        f"  snapshot days accrued:   {rep['days_accrued']}"
        + (
            f" ({rep['first_snapshot']} .. {rep['last_snapshot']})"
            if rep.get("first_snapshot")
            else ""
        ),
    ]
    if rep.get("first_snapshot"):
        lines += [
            f"  missed weekdays:         {len(rep.get('missed_weekdays', []))} "
            f"of {rep.get('expected_weekdays_since_anchor')}"
            + (f" -> {rep['missed_weekdays']}" if rep.get("missed_weekdays") else ""),
            f"  anchor (first snapshot): {r.get('anchor_first_snapshot')}",
            f"  CONFIRMATORY UNLOCK:     {r.get('confirmatory_unlock')} "
            f"(+{CONFIRMATORY_MONTHS}mo; in {rep.get('days_to_confirmatory_unlock')} days)",
            f"  second checkpoint:       {r.get('second_checkpoint')} (+{SECOND_CHECKPOINT_MONTHS}mo)",
            f"  monitoring bound:        {r.get('monitoring_bound')} (2027-Q4, spec SS1.1)",
            "  window maturity (first as_of with a full lag in-lake):",
        ]
        for name, d in (r.get("window_maturity") or {}).items():
            lines.append(f"    revision_drift_{name}: {d}")
    if rep.get("latest_day_endpoint_coverage"):
        lines.append("  latest-day endpoint coverage:")
        for ep, c in rep["latest_day_endpoint_coverage"].items():
            lines.append(
                f"    {ep}: {c['tickers']}/{c['requested']} tickers "
                f"(coverage {c['coverage']})"
            )
    if rep.get("exploratory_feature_stats"):
        s = rep["exploratory_feature_stats"]
        lines.append("  exploratory feature stats (NEVER gate - spec SS1.1):")
        for k, v in s.items():
            if k != "label":
                lines.append(f"    {k}: {v}")
    lines.append(
        "  NOTE: C1 is informative-only until the confirmatory unlock; any "
        "interim read is exploratory (spec SS1.1)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m renquant_base_data.pit_revision_features",
        description=__doc__.splitlines()[0],
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build/refresh the C1 feature lake (incremental no-op when current)")
    b.add_argument("--snapshot-root", required=True, help="the PIT estimate-snapshot lake (READ-ONLY)")
    b.add_argument("--out", required=True, help="output root; must be a 'pit_features' leaf or /tmp scratch")
    b.add_argument("--force", action="store_true", help="rebuild even if the output is current")

    r = sub.add_parser("report", help="coverage / C1-test readiness report")
    r.add_argument("--snapshot-root", required=True, help="the PIT estimate-snapshot lake (READ-ONLY)")
    r.add_argument("--features", default=None, help="optional built c1_revision_drift.parquet for exploratory stats")
    r.add_argument("--json", action="store_true", help="machine-readable JSON to stdout")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.cmd == "build":
        result = build_and_publish(
            Path(args.snapshot_root), Path(args.out), force=args.force
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "report":
        rep = coverage_report(
            Path(args.snapshot_root),
            Path(args.features) if args.features else None,
        )
        print(json.dumps(rep, indent=2) if args.json else _format_report(rep))
        return 0

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
