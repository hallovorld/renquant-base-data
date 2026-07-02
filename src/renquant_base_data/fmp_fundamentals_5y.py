"""One-shot FMP 5-year fundamentals harvester (research/descriptive substrate).

WHY (2026-07-02): the M-SIG signal stack's quality composite (task C2 in the
renquant-orchestrator unified plan) needs a multi-year fundamentals panel --
levels, ratios, and growth rates with enough fiscal history to compute trailing
quality/stability features. The existing umbrella harvest
(``data/fmp_harvest/``, umbrella PR #409) fetched ``period=annual&limit=20``
only; there is no quarterly history and no dedicated, reproducible 5y panel.
This module is a ONE-SHOT harvester (deliberately NOT scheduled and NOT a
forward snapshotter): run once, it pulls per-symbol annual AND quarterly
history for four FMP ``stable`` endpoints into a NEW dedicated directory.

POINT-IN-TIME LIMITATION (this artifact is NOT the C2 confirmatory substrate):
a one-shot 2026 fetch returns the vendor's CURRENT database values for past
fiscal periods, with NO per-row revision/version identity -- none of the four
endpoints exposes a vendor revision number, an "as originally filed" vs
"latest reported" distinction, or an as-filed vintage timestamp on the values
themselves. Subsequently RESTATED values are therefore indistinguishable from
as-filed values, and joining these rows to old filing dates does NOT
reconstruct what was knowable on those dates. Even where the vendor supplies
filing timestamps (income-statement ``acceptedDate``/``filingDate``, preserved
untouched), the timestamp does not prove the VALUES are as-filed. Every
manifest (per-target and top-level) is stamped ``pit_classification:
research_descriptive_only`` and ``pit_provenance:
vendor_current_values_no_revision_identity``. The admissible C2 CONFIRMATORY
path is fixed by the merged M-SIG spec (renquant-orchestrator PR #243, r4):
observation availability timing must come from a genuine
``acceptedDate``/``filingDate`` or a SEC EDGAR XBRL join
(``sec_fundamentals.build_quarterly_panel``'s ``available_date``); an
observation without one is INADMISSIBLE -- fail closed, never proxy-dated by
the fiscal-period date -- and NO confirmatory claim may rest on this restated
history regardless of timing. This harvester's output may feed C2 only as a
supplementary/exploratory levels-and-ratios panel joined against that
EDGAR-sourced availability date; it must never itself support a confirmatory
C2 result.

House-pattern lineage: modeled closely on
:mod:`renquant_base_data.fmp_estimate_revisions` (base-data PR #27) and reuses
its primitives -- stable-API auth via the umbrella ``.env`` (read-only),
dedicated output dir with per-target parquet + manifest carrying
``fetched_at``, ``--dry-run`` / ``--min-coverage`` / ``--universe`` flags, and
a dependency-injected fetcher so tests never touch the network.

Endpoints x periods (8 targets)::

    key-metrics, ratios, financial-growth, income-statement
      x annual    (limit=10 fiscal years)
      x quarterly (limit=40 fiscal quarters)

Both limits request ~10 fiscal years, comfortably >= 5 years even after vendor
gaps (FMP Starter serves 5y of history). Every published row carries the
harvester-stamped columns ``symbol``, ``fiscal_date`` (the record's fiscal
``date``), ``harvest_period`` (``annual``/``quarterly``), and ``fetched_at``
(run-level UTC timestamp); the vendor's native fields -- including its own
``period`` (FY/Q1..Q4) -- are preserved untouched.

PLAN-GATE NOTE (probed 2026-07-02, single-symbol key-metrics probe): the
``stable`` API recognizes ``period=quarter`` but returned HTTP 402
("Premium Query Parameter ... not available under your current subscription")
for the key then in the umbrella ``.env``. Annual paths are confirmed working
by the existing ``data/fmp_harvest`` manifests. If the quarterly entitlement is
not active at harvest time, every quarterly target 402s, its coverage is 0,
and the run fails the ``--min-coverage`` gate loudly (nothing is published).
``--periods annual`` scopes an annual-only harvest in that case.

Rate limiting: FMP Starter allows 300 req/min. The throttle sleeps 0.5s per
request (~120 req/min), staying well under half the ceiling and leaving
headroom for any other consumer of the key. A full 114-name x 8-target harvest
is ~912 requests, roughly 8 minutes.

Safety contract (same as the house pattern):

* Output goes ONLY to the NEW dedicated ``data/fmp_harvest_5y/`` directory (or
  an explicit /tmp scratch path); a canonical-path guard rejects everything
  else, symlinks included.
* Atomic all-or-nothing publish: every target is staged to a sibling temp dir
  and published via atomic rename ONLY if every target clears its coverage
  floor. A shortfall returns ``status: partial``, publishes NOTHING, and exits
  non-zero.
* Verified idempotency: a re-run against an existing output dir DEEP-VERIFIES
  the published bundle -- the top-level contract manifest (schema + harvester
  version, universe fingerprint, endpoint x period target set incl. path
  templates), every child parquet AND child manifest sha256 against on-disk
  bytes, per-target status/coverage against the requested floor, and the
  guaranteed row columns. All green -> no-op skip. ANY mismatch (truncated or
  corrupt parquet, tampered manifest, stale/incomplete target set, changed
  universe, version drift) -> loud ``verify_failed`` (exit 3) that neither
  skips nor silently re-fetches; ``--force`` is the explicit recovery that
  re-harvests and atomically replaces the bundle.
* ``--dry-run`` lists every planned request group and makes ZERO network calls
  and ZERO writes.
* RUNNING the harvest against the live FMP API is a separate operator-granted
  landing action; this module ships code + tests only.

Usage::

    # see the request plan without any network call or write
    python -m renquant_base_data.fmp_fundamentals_5y --dry-run

    # demo to /tmp scratch (proves fetch+write without touching live data/)
    python -m renquant_base_data.fmp_fundamentals_5y --out /tmp/fmp5y_demo

    # the real one-shot harvest (operator-granted action)
    python -m renquant_base_data.fmp_fundamentals_5y --out data/fmp_harvest_5y

    # annual-only, e.g. while the quarterly entitlement is still plan-gated
    python -m renquant_base_data.fmp_fundamentals_5y --periods annual
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import pandas as pd

from renquant_base_data.fmp_estimate_revisions import (
    DEFAULT_ENV,
    FMP_STABLE_BASE,
    _FORBIDDEN_LEAVES,
    _is_scratch_arg,
    _require_requests,
    _sha256_file,
    fetch_endpoint,
    load_api_key,
    load_universe,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import requests


log = logging.getLogger("renquant_base_data.fmp_fundamentals_5y")

# --- FMP stable endpoints to harvest -------------------------------------------
# Path style matches the proven umbrella harvest manifests
# (data/fmp_harvest/*.manifest.json: "key-metrics?symbol={sym}&period=annual&limit=20").
ENDPOINTS: dict[str, str] = {
    "key_metrics": "key-metrics?symbol={sym}&period={period}&limit={limit}",
    "ratios": "ratios?symbol={sym}&period={period}&limit={limit}",
    "financial_growth": "financial-growth?symbol={sym}&period={period}&limit={limit}",
    "income_statement": "income-statement?symbol={sym}&period={period}&limit={limit}",
}

# period label -> (FMP ``period`` query value, ``limit`` = fiscal rows requested).
# limit 10 annual / 40 quarterly ~= 10 fiscal years, comfortably >= 5y.
PERIODS: dict[str, tuple[str, int]] = {
    "annual": ("annual", 10),
    "quarterly": ("quarter", 40),
}
DEFAULT_PERIODS: tuple[str, ...] = ("annual", "quarterly")

DEFAULT_OUT = "data/fmp_harvest_5y"
DEDICATED_LEAF = "fmp_harvest_5y"
HARVEST_MANIFEST = "harvest.manifest.json"
DEFAULT_MIN_COVERAGE = 0.90
# FMP Starter = 300 req/min; 0.5 s/request ~= 120 req/min, well under half.
THROTTLE_S = 0.50

HARVESTER = "fmp_fundamentals_5y"
# Bundle identity, verified on every re-run before an idempotent skip.
# v1 = the initial PR round (never harvested live); v2 adds the verified
# idempotency contract (universe fingerprint, child hashes) + the PIT stamp.
SCHEMA_VERSION = 2
# Bump on ANY behavior change to fetch/stamp/publish, even schema-compatible
# ones -- a version-drifted bundle must be an explicit --force decision.
# (v1 = the two pre-unification review-round implementations; v2 = the
# unified fail-closed contract.)
HARVESTER_VERSION = 2

# PIT stamp (see POINT-IN-TIME LIMITATION in the module docstring): a one-shot
# fetch of vendor-current history carries no per-row revision identity, so it
# is research/descriptive-only -- NEVER a C2 confirmatory input.
PIT_CLASSIFICATION = "research_descriptive_only"
PIT_PROVENANCE = "vendor_current_values_no_revision_identity"
PIT_NOTE = (
    "One-shot fetch of the vendor's CURRENT database: historical rows may hold "
    "subsequently RESTATED values with no revision/version identity, so joining "
    "them to old filing dates does not reconstruct what was knowable then. "
    "Admissible C2 confirmatory use per renquant-orchestrator PR #243 (r4): "
    "availability timing only from a genuine acceptedDate/filingDate or a SEC "
    "EDGAR XBRL join (sec_fundamentals.build_quarterly_panel available_date); "
    "observations without one are INADMISSIBLE (fail closed, never proxy-dated "
    "by the fiscal-period date), and no confirmatory claim may rest on this "
    "restated history regardless of timing."
)

# Columns this harvester guarantees on every published row; verified on skip.
REQUIRED_ROW_COLUMNS: tuple[str, ...] = (
    "symbol",
    "fiscal_date",
    "harvest_period",
    "fetched_at",
)

# fetch(session, endpoint_path, sym, api_key) -> (records | None, error | None)
FetchFn = Callable[..., "tuple[list[dict[str, Any]] | None, str | None]"]

# Canonical inputs we must never touch: everything the estimate snapshotter
# forbids, plus its own dedicated output tree.
_FORBIDDEN = frozenset(_FORBIDDEN_LEAVES | {"estimate_snapshots"})


def is_canonical_path(out_dir: Path) -> bool:
    """Guard: refuse to write any existing/canonical data path.

    Accept ONLY an out-dir whose leaf is the dedicated ``fmp_harvest_5y`` name,
    or an explicit /tmp scratch target. Judged on the path as given AND fully
    resolved (symlinks followed), mirroring the house-pattern guard in
    :mod:`renquant_base_data.fmp_estimate_revisions`.
    """
    resolved = out_dir.resolve()
    parts = set(out_dir.parts) | set(resolved.parts)
    if parts & _FORBIDDEN:
        return True
    if _is_scratch_arg(out_dir) and _is_scratch_arg(resolved):
        return False
    return out_dir.name != DEDICATED_LEAF or resolved.name != DEDICATED_LEAF


def build_targets(periods: Sequence[str]) -> list[dict[str, str]]:
    """Cross the endpoints with the selected period labels into fetch targets.

    Each target's ``path_template`` has period/limit already substituted and
    only ``{sym}`` left open -- the exact shape ``fetch_endpoint`` consumes and
    the umbrella harvest manifests record.
    """
    unknown = [p for p in periods if p not in PERIODS]
    if unknown:
        raise ValueError(
            f"unknown period label(s) {unknown}; valid: {sorted(PERIODS)}"
        )
    if not periods:
        raise ValueError(f"at least one period label required; valid: {sorted(PERIODS)}")
    targets: list[dict[str, str]] = []
    for endpoint, path_tmpl in ENDPOINTS.items():
        for label in periods:
            fmp_period, limit = PERIODS[label]
            targets.append(
                {
                    "name": f"{endpoint}_{label}",
                    "endpoint": endpoint,
                    "period": label,
                    "path_template": path_tmpl.format(
                        sym="{sym}", period=fmp_period, limit=limit
                    ),
                }
            )
    return targets


def harvest_one_target(
    session: "requests.Session | None",
    target: dict[str, str],
    tickers: Sequence[str],
    api_key: str,
    fetched_at: str,
    stage_dir: Path,
    *,
    fetch: FetchFn,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    throttle_s: float | None = None,
) -> dict[str, Any]:
    """Fetch one (endpoint, period) target across the universe into staging.

    Writes ``<name>.parquet`` + ``<name>.manifest.json`` into ``stage_dir``;
    publication is the caller's atomic rename once EVERY target clears its
    coverage floor. Coverage here is the task-C2 contract: the fraction of the
    requested universe that RETURNED ROWS (plan-locked 402s and empty payloads
    both count against it -- a fundamentals panel with silent holes is exactly
    what the gate must catch).
    """
    if throttle_s is None:
        throttle_s = THROTTLE_S
    started = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    with_data = no_data = http_error = fetch_error = 0
    error_samples: list[str] = []

    for sym in tickers:
        records, err = fetch(session, target["path_template"], sym, api_key)
        if err is not None:
            if err.startswith("http_"):
                http_error += 1
            else:
                fetch_error += 1
            if len(error_samples) < 5:
                error_samples.append(f"{sym}:{err}")
        elif not records:
            no_data += 1
        else:
            with_data += 1
            for rec in records:
                rec = dict(rec)
                rec.setdefault("symbol", sym)
                # Guaranteed harvester columns; the vendor's own fields
                # (incl. its native ``period`` = FY/Q1..Q4) stay untouched.
                rec["fiscal_date"] = rec.get("date")
                rec["harvest_period"] = target["period"]
                rec["fetched_at"] = fetched_at
                rows.append(rec)
        if throttle_s:
            time.sleep(throttle_s)

    requested = len(tickers)
    coverage = (with_data / requested) if requested else 1.0
    status = "ok" if coverage >= min_coverage else "partial"

    parquet_path = stage_dir / f"{target['name']}.parquet"
    manifest_path = stage_dir / f"{target['name']}.manifest.json"
    finished = datetime.now(timezone.utc)

    manifest: dict[str, Any] = {
        "name": target["name"],
        "endpoint": target["endpoint"],
        "period": target["period"],
        "path_template": target["path_template"],
        "url_base": FMP_STABLE_BASE,
        "schema_version": SCHEMA_VERSION,
        "harvester_version": HARVESTER_VERSION,
        "pit_classification": PIT_CLASSIFICATION,
        "pit_provenance": PIT_PROVENANCE,
        "requested": requested,
        "with_data": with_data,
        "no_data": no_data,
        "http_error": http_error,
        "fetch_error": fetch_error,
        "error_samples": error_samples,
        "coverage": round(coverage, 4),
        "min_coverage": min_coverage,
        "rows": len(rows),
        "output": f"{target['name']}.parquet",
        "fetched_at": fetched_at,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "status": status,
    }

    stage_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    manifest["sha256"] = _sha256_file(parquet_path)
    # The exact published column set; lets a consumer (and the verifier) know
    # the schema without a parquet read (the sha256 binds it to the bytes).
    manifest["schema_columns"] = sorted(df.columns.tolist())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def universe_fingerprint(tickers: Sequence[str]) -> str:
    """Order- and duplication-insensitive identity of the requested universe."""
    canon = "\n".join(sorted({t.strip() for t in tickers if t.strip()}))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def verify_published_bundle(
    out_dir: Path,
    targets: Sequence[dict[str, str]],
    tickers: Sequence[str],
    min_coverage: float,
) -> list[str]:
    """Deep-verify a published bundle against the CURRENT request.

    Returns a list of human-readable problems; empty == verified. An idempotent
    skip is allowed ONLY on an empty list -- manifest-file existence alone says
    nothing about a truncated parquet, a tampered manifest, a stale target set,
    or a bundle harvested for a different universe/schema. Checked:

      * top-level contract manifest: parseable, harvester name, schema_version,
        harvester_version, universe fingerprint (sha256 of the sorted deduped
        ticker set), per-target contract entries;
      * target set: stamped names == the requested endpoint x period targets
        (both directions), and each stamped ``path_template`` matches the
        code's current endpoint/period/limit config;
      * per target: child manifest exists, its sha256 matches the top-level
        ``manifest_sha256`` (tamper evidence), parquet exists, its sha256
        matches BOTH recorded hashes, child status is ``ok``, recorded
        coverage clears the CURRENTLY requested floor, and the parquet reads
        back with every guaranteed row column.
    """
    top_path = out_dir / HARVEST_MANIFEST
    if not top_path.is_file():
        return [f"missing top-level manifest {HARVEST_MANIFEST}"]
    try:
        top = json.loads(top_path.read_text())
    except (OSError, ValueError) as exc:
        return [f"unreadable top-level manifest: {exc}"]

    problems: list[str] = []
    if top.get("harvester") != HARVESTER:
        problems.append(f"harvester mismatch: bundle={top.get('harvester')!r}")
    if top.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version mismatch: bundle={top.get('schema_version')!r} "
            f"code={SCHEMA_VERSION}"
        )
    if top.get("harvester_version") != HARVESTER_VERSION:
        problems.append(
            f"harvester_version mismatch: bundle={top.get('harvester_version')!r} "
            f"code={HARVESTER_VERSION!r}"
        )
    if top.get("universe_fingerprint") != universe_fingerprint(tickers):
        problems.append(
            "universe fingerprint mismatch: bundle was harvested for a "
            "different ticker universe than the one now requested"
        )

    stamped = top.get("targets")
    if not isinstance(stamped, dict):
        problems.append("top-level manifest carries no per-target contract")
        return problems
    expected = {t["name"]: t for t in targets}
    missing = sorted(set(expected) - set(stamped))
    extra = sorted(set(stamped) - set(expected))
    if missing:
        problems.append(f"bundle lacks requested target(s): {missing}")
    if extra:
        problems.append(f"bundle carries unrequested target(s): {extra}")

    for name in sorted(set(expected) & set(stamped)):
        entry = stamped[name]
        if entry.get("path_template") != expected[name]["path_template"]:
            problems.append(
                f"{name}: endpoint/period config drift: bundle fetched "
                f"{entry.get('path_template')!r}, code now requests "
                f"{expected[name]['path_template']!r}"
            )
        man_path = out_dir / f"{name}.manifest.json"
        if not man_path.is_file():
            problems.append(f"{name}: missing child manifest")
            continue
        if entry.get("manifest_sha256") != _sha256_file(man_path):
            problems.append(
                f"{name}: child manifest sha256 mismatch (tampered or partial write)"
            )
        try:
            child = json.loads(man_path.read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: unreadable child manifest: {exc}")
            continue
        parquet_path = out_dir / str(entry.get("output") or f"{name}.parquet")
        if not parquet_path.is_file():
            problems.append(f"{name}: missing parquet {parquet_path.name}")
            continue
        actual_sha = _sha256_file(parquet_path)
        if actual_sha != entry.get("sha256") or actual_sha != child.get("sha256"):
            problems.append(
                f"{name}: parquet sha256 mismatch (on-disk bytes vs recorded "
                "hashes -- truncated, corrupt, or replaced)"
            )
            continue  # bytes are wrong; do not attempt to parse them
        if child.get("status") != "ok":
            problems.append(f"{name}: child status {child.get('status')!r} != 'ok'")
        coverage = child.get("coverage")
        if not isinstance(coverage, (int, float)) or coverage < min_coverage:
            problems.append(
                f"{name}: recorded coverage {coverage!r} below the requested "
                f"floor {min_coverage}"
            )
        recorded_cols = child.get("schema_columns")
        if not recorded_cols:
            problems.append(f"{name}: child manifest records no schema_columns")
        else:
            missing_recorded = [
                c for c in REQUIRED_ROW_COLUMNS if c not in set(recorded_cols)
            ]
            if missing_recorded:
                problems.append(
                    f"{name}: recorded schema_columns missing required "
                    f"column(s) {missing_recorded}"
                )
        try:
            columns = set(pd.read_parquet(parquet_path).columns)
        except Exception as exc:  # noqa: BLE001 - any parse failure = corrupt
            problems.append(f"{name}: unreadable parquet: {exc}")
            continue
        missing_cols = [c for c in REQUIRED_ROW_COLUMNS if c not in columns]
        if missing_cols:
            problems.append(
                f"{name}: parquet missing required column(s) {missing_cols}"
            )
    return problems


def harvest(
    *,
    session: "requests.Session | None",
    tickers: Sequence[str],
    api_key: str,
    out_dir: Path,
    periods: Sequence[str] = DEFAULT_PERIODS,
    dry_run: bool = False,
    force: bool = False,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    throttle_s: float | None = None,
    fetch: FetchFn | None = None,
) -> dict[str, Any]:
    """Run the one-shot harvest and atomically publish on full success.

    Safety contract (house pattern):
      * ``dry_run`` returns the request plan -- ZERO fetch calls, ZERO writes.
      * Verified idempotency -- if the output dir exists, the published bundle
        is DEEP-VERIFIED (:func:`verify_published_bundle`: hashes, target set,
        universe fingerprint, schema/harvester version, coverage floor, row
        columns). Fully verified -> no-op ``skipped``. ANY mismatch ->
        ``verify_failed``: nothing is skipped, nothing is re-fetched, the
        problems are surfaced, and re-harvesting over the bad bundle requires
        an explicit ``force``.
      * All-or-nothing atomic publish -- every target is staged into a sibling
        temp dir; only if EVERY target clears ``min_coverage`` is the dir
        published via atomic ``os.replace``. A shortfall publishes NOTHING
        (``status: partial``) and any prior good harvest survives untouched.
    """
    if fetch is None:
        fetch = fetch_endpoint
    targets = build_targets(list(periods))
    fetched_at = datetime.now(timezone.utc).isoformat()

    if dry_run:
        planned = [
            {
                "name": t["name"],
                "endpoint": t["endpoint"],
                "period": t["period"],
                "path_template": t["path_template"],
                "symbols": len(tickers),
                "output": f"{t['name']}.parquet",
            }
            for t in targets
        ]
        total = len(tickers) * len(targets)
        effective_throttle = THROTTLE_S if throttle_s is None else throttle_s
        return {
            "status": "dry_run",
            "published": False,
            "out_dir": out_dir,
            "planned_requests": total,
            "estimated_minutes": round(total * effective_throttle / 60.0, 1),
            "targets": planned,
            "manifests": [],
        }

    if out_dir.exists() and not force:
        problems = verify_published_bundle(out_dir, targets, tickers, min_coverage)
        if problems:
            return {
                "status": "verify_failed",
                "published": False,
                "out_dir": out_dir,
                "problems": problems,
                "manifests": [],
            }
        return {
            "status": "skipped",
            "published": False,
            "out_dir": out_dir,
            "reason": "already_published_verified",
            "manifests": [],
        }

    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".stage-{DEDICATED_LEAF}-", dir=parent))
    try:
        manifests = [
            harvest_one_target(
                session,
                target,
                tickers,
                api_key,
                fetched_at,
                stage_dir,
                fetch=fetch,
                min_coverage=min_coverage,
                throttle_s=throttle_s,
            )
            for target in targets
        ]
        partial = [m["name"] for m in manifests if m["status"] != "ok"]
        # Top-level contract manifest: binds the universe (fingerprint), the
        # endpoint x period config (path templates), the code/schema version,
        # the PIT stamp, and EVERY child hash (parquet + child manifest), so a
        # later idempotent skip can deep-verify the whole bundle.
        summary: dict[str, Any] = {
            "harvester": HARVESTER,
            "harvester_version": HARVESTER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "pit_classification": PIT_CLASSIFICATION,
            "pit_provenance": PIT_PROVENANCE,
            "pit_note": PIT_NOTE,
            "fetched_at": fetched_at,
            "universe": len(tickers),
            "universe_fingerprint": universe_fingerprint(tickers),
            "periods": list(periods),
            "min_coverage": min_coverage,
            "required_row_columns": list(REQUIRED_ROW_COLUMNS),
            "targets": {
                m["name"]: {
                    "endpoint": m["endpoint"],
                    "period": m["period"],
                    "path_template": m["path_template"],
                    "output": m["output"],
                    "rows": m["rows"],
                    "with_data": m["with_data"],
                    "coverage": m["coverage"],
                    "status": m["status"],
                    "sha256": m["sha256"],
                    "schema_columns": m["schema_columns"],
                    "manifest_sha256": _sha256_file(
                        stage_dir / f"{m['name']}.manifest.json"
                    ),
                }
                for m in manifests
            },
            "status": "partial" if partial else "ok",
        }
        if partial:
            # Coverage gate: publish NOTHING; a prior good harvest survives.
            return {
                "status": "partial",
                "published": False,
                "out_dir": out_dir,
                "partial_targets": partial,
                "manifests": manifests,
                "summary": summary,
            }
        (stage_dir / HARVEST_MANIFEST).write_text(json.dumps(summary, indent=2) + "\n")
        # Atomic-and-recoverable publish (house pattern, base-data PR #27): a
        # prior harvest (if any) is moved aside and only deleted AFTER the new
        # dir is definitely in place; a failed rename restores it, so a publish
        # error never leaves the path with NO harvest.
        backup: Path | None = None
        if out_dir.exists():
            backup = parent / f".replaced-{DEDICATED_LEAF}-{int(time.time())}"
            os.replace(out_dir, backup)
        try:
            os.replace(stage_dir, out_dir)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, out_dir)
            if stage_dir.exists():
                quarantine = parent / f".failed-stage-{DEDICATED_LEAF}-{int(time.time())}"
                try:
                    os.replace(stage_dir, quarantine)
                except OSError:
                    shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        stage_dir = out_dir  # consumed by the rename; nothing to clean up
        return {
            "status": "ok",
            "published": True,
            "out_dir": out_dir,
            "manifests": manifests,
            "summary": summary,
        }
    finally:
        if stage_dir != out_dir and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--universe",
        default=None,
        help="strategy_config.json (reads 'watchlist') or a one-ticker-per-line "
        "file; default = renquant_104 golden config",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"dedicated output dir (default {DEFAULT_OUT}); must be a "
        f"'{DEDICATED_LEAF}' dir or a /tmp demo path",
    )
    ap.add_argument(
        "--periods",
        default=",".join(DEFAULT_PERIODS),
        help="comma-separated period labels to harvest "
        f"(default {','.join(DEFAULT_PERIODS)}; valid: {','.join(sorted(PERIODS))})",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="fraction of the universe that must RETURN ROWS per target to "
        f"publish (default {DEFAULT_MIN_COVERAGE}); below it the run fails, "
        "publishes nothing, and exits non-zero",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-harvest and atomically replace whatever is at the output dir "
        "(default: an existing dir is deep-verified -- a fully verified bundle "
        "is a no-op skip, ANY mismatch fails with exit 3 and requires --force)",
    )
    ap.add_argument(
        "--env",
        default=str(DEFAULT_ENV),
        help="path to a .env holding FMP_API_KEY (read-only)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list every planned request group, but make NO network call or write",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out)
    if is_canonical_path(out_dir):
        print(
            f"error: refusing to write canonical/non-dedicated path {out_dir!r}; "
            f"use a '{DEDICATED_LEAF}' dir or a /tmp demo path",
            file=sys.stderr,
        )
        return 2

    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    try:
        targets = build_targets(periods)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        tickers = load_universe(args.universe)
    except (OSError, ValueError) as exc:
        print(f"error: could not load universe: {exc}", file=sys.stderr)
        return 2

    print(
        f"fmp_fundamentals_5y: universe={len(tickers)} tickers "
        f"targets={[t['name'] for t in targets]} out={out_dir}"
        + ("  [DRY-RUN]" if args.dry_run else ""),
        file=sys.stderr,
    )

    api_key = None
    if not args.dry_run:
        api_key = load_api_key(Path(args.env))
        if not api_key:
            print(f"error: FMP_API_KEY not found (env or {args.env})", file=sys.stderr)
            return 2

    # ``requests`` (lazy) is needed only for a live fetch; a dry-run plans the
    # requests without any network session or the dependency installed.
    session = None if args.dry_run else _require_requests().Session()
    result = harvest(
        session=session,
        tickers=tickers,
        api_key=api_key or "",
        out_dir=out_dir,
        periods=periods,
        dry_run=args.dry_run,
        force=args.force,
        min_coverage=args.min_coverage,
        fetch=fetch_endpoint,
    )

    if result["status"] == "dry_run":
        for t in result["targets"]:
            print(
                f"  PLAN {t['name']:32s} {t['symbols']:4d} symbols  "
                f"{FMP_STABLE_BASE}/{t['path_template']}",
                file=sys.stderr,
            )
        print(
            f"  {result['planned_requests']} requests total, "
            f"~{result['estimated_minutes']} min at the courteous throttle",
            file=sys.stderr,
        )
    for m in result.get("manifests", []):
        print(
            f"  {m['name']:32s} rows={m['rows']:6d} with_data={m['with_data']:4d} "
            f"no_data={m['no_data']:3d} http_err={m['http_error']:3d} "
            f"fetch_err={m['fetch_error']:3d} coverage={m['coverage']:.2%} "
            f"status={m['status']}",
            file=sys.stderr,
        )
    if result["status"] == "skipped":
        print(
            f"  already published AND fully verified at {out_dir} (hashes, target "
            f"set, universe, schema v{SCHEMA_VERSION}) -- no-op "
            f"(pass --force to re-harvest)",
            file=sys.stderr,
        )
    elif result["status"] == "verify_failed":
        print(
            f"  VERIFY FAILED: {out_dir} exists but is NOT a verified published "
            f"bundle:",
            file=sys.stderr,
        )
        for problem in result["problems"]:
            print(f"    - {problem}", file=sys.stderr)
        print(
            "  refusing BOTH the silent skip and a silent re-fetch; re-run with "
            "--force to re-harvest and atomically replace it",
            file=sys.stderr,
        )
    elif result["status"] == "partial":
        print(
            f"  PARTIAL: targets below the coverage floor: {result['partial_targets']}; "
            f"NOTHING published (prior harvest, if any, left intact)",
            file=sys.stderr,
        )
    elif result["status"] == "ok":
        print(
            f"  STAMP: pit_classification={PIT_CLASSIFICATION} pit_provenance="
            f"{PIT_PROVENANCE} -- research/descriptive only, NOT a C2 "
            f"confirmatory (PIT) input; see the manifest pit_note",
            file=sys.stderr,
        )

    payload: dict[str, Any] = {
        "out_dir": str(out_dir),
        "universe": len(tickers),
        "periods": periods,
        "dry_run": args.dry_run,
        "status": result["status"],
        "published": result.get("published", False),
        "targets": {m["name"]: m["rows"] for m in result.get("manifests", [])},
    }
    if result.get("problems"):
        payload["problems"] = result["problems"]
    print(json.dumps(payload, indent=2))
    return {"partial": 1, "verify_failed": 3}.get(result["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
