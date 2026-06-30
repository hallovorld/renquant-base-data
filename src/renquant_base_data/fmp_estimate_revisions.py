"""Forward snapshotter for FMP analyst estimates / consensus / targets.

WHY (2026-06-27): the analyst estimate-*revision* signal is the literature's
best large-cap orthogonal lead (post-revision drift; Womack 1996, Gleason-Lee
2003). It is currently **un-buildable** because the FMP harvest
(``data/fmp_harvest/`` in the umbrella, umbrella PR #409) is a single *current*
snapshot with **no revision history**: every parquet there reflects the
consensus as it was on the harvest day only. A point-in-time audit confirmed we
cannot reconstruct "what the consensus was 1m/3m ago" from a one-shot snapshot.

The fix is cheap and only needs *time*: start snapshotting the estimates
**forward from today** so a real as-of revision history accrues. Run this once a
day; after ~3-6 months you have a dated series and can compute trailing
Δ(consensus estimate) as-of each date with no look-ahead.

This collector ONLY fetches and writes dated snapshots to a NEW dedicated path
(default ``data/estimate_snapshots/<YYYY-MM-DD>/<endpoint>.parquet``). It never
writes a canonical/existing data path and never schedules itself -- wiring a
cron/launchd job is a separate operator deploy decision (and belongs to the
orchestrator, which schedules base-data primitives and persists their
fingerprints; see renquant-orchestrator PR #205).

Auth/endpoint pattern matches the existing harvest (the FMP ``stable`` API,
``apikey`` query param, key read read-only from the umbrella ``.env``).

PIT PROVENANCE (HARD INVARIANT): every row is fetched NOW. ``snapshot_as_of`` is
therefore ALWAYS the actual UTC fetch date, and each manifest also records the
``fetched_at`` UTC timestamp. ``--as-of`` is accepted ONLY when it equals today's
UTC date (a redundant, self-documenting assertion that also fails loudly on host
clock drift); BOTH a PAST and a FUTURE ``--as-of`` are REJECTED. A past date
would fabricate point-in-time history that never existed; a future date would
stamp today's freshly fetched data as future data -- equally fake provenance.
Scheduling picks the date directory at RUN TIME from the real fetch date; it does
NOT pre-name a future slot. Historical backfill is valid *only* from an immutable
source that was actually captured at that historical time (with its own
provenance) -- this forward collector cannot and must not manufacture it.

ATOMIC PUBLISH: each date is written to a sibling temp dir, validated against a
coverage floor, and published with an atomic rename ONLY on success. A
partial/error fetch is marked ``status: partial`` and is NOT published over an
existing good snapshot. By default an existing successful snapshot is a no-op
(verify), never a destructive refetch -- pass ``--force`` to deliberately
re-publish.

Usage::

    # demo to /tmp (proves fetch+write without touching live data/)
    python -m renquant_base_data.fmp_estimate_revisions --out /tmp/snap_demo

    # explicit universe file (one ticker per line, or a strategy_config.json)
    python -m renquant_base_data.fmp_estimate_revisions --universe /path/to/universe.txt

    # assert today's date explicitly (must EQUAL today's UTC date; past/future error)
    python -m renquant_base_data.fmp_estimate_revisions --as-of 2026-06-27

    # see what would be fetched/written without any network call or write
    python -m renquant_base_data.fmp_estimate_revisions --dry-run

Cron-safety: re-running an already-published date is a no-op verify (NOT a
destructive refetch). For a real scheduled deploy, wrap the invocation in a
``flock`` guard so two runs can't race the same date dir, e.g.::

    flock -n /tmp/snapshot_fmp_estimates.lock \\
        python -m renquant_base_data.fmp_estimate_revisions --out data/estimate_snapshots

Scheduling itself is intentionally NOT done here (it is the orchestrator's job).
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
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import requests


log = logging.getLogger("renquant_base_data.fmp_estimate_revisions")

# --- FMP endpoints to snapshot ------------------------------------------------
# Mirrors the umbrella harvest manifests (data/fmp_harvest/*.manifest.json):
#   - base = the FMP "stable" API
#   - each endpoint is a per-symbol GET with ?symbol={sym}&apikey={key}
# These four carry the consensus / estimate / target fields the revision signal
# is built from. analyst_estimates is THE one that revises (mean EPS/revenue
# estimate); the others give consensus rating + target drift.
FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
ENDPOINTS: dict[str, str] = {
    "analyst_estimates": "analyst-estimates?symbol={sym}&period=annual",
    "grades_consensus": "grades-consensus?symbol={sym}",
    "price_target_consensus": "price-target-consensus?symbol={sym}",
    "price_target_summary": "price-target-summary?symbol={sym}",
}

DEFAULT_OUT = "data/estimate_snapshots"
DEFAULT_ENV = Path("/Users/renhao/git/github/RenQuant/.env")
DEFAULT_UNIVERSE_CONFIG = Path(
    "/Users/renhao/git/github/RenQuant/backtesting/renquant_104/strategy_config.golden.json"
)
REQUEST_TIMEOUT_S = 30
THROTTLE_S = 0.20  # gentle on the rate limit (FMP Starter = 300/min)
# Coverage floor: the fraction of the requested universe that must return data
# (per endpoint) for the snapshot to be considered complete and publishable.
# Free FMP returns ~134/142 (~94%) for these endpoints; the ~8 misses are
# plan-locked, not transient. A real run below this floor signals a fetch
# outage (HTTP/network) rather than the known plan gaps, so we mark it partial
# and refuse to overwrite a good prior snapshot.
DEFAULT_MIN_COVERAGE = 0.90


# ``requests`` is needed ONLY on the live network path (``fetch_endpoint`` /
# ``main``). It is imported lazily via :func:`_require_requests` so that merely
# importing this module -- e.g. CI test collection, which exercises the pure
# contract functions with a fake fetch and never hits the network -- does not
# require the dependency to be installed. The live path raises a clear error if
# it is genuinely missing.
def _require_requests() -> "requests":
    """Import and return the ``requests`` module on demand (live path only)."""
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise SystemExit(
            "error: the 'requests' package is required for a live fetch but is not "
            "installed. Install it (declared in pyproject [project].dependencies) "
            "or run with --dry-run, which makes no network call."
        ) from exc
    return requests


# --- helpers ------------------------------------------------------------------
def _read_env_value(env_path: Path, key: str) -> str | None:
    """Read a single KEY=VALUE from a .env file, read-only. Never logs the value."""
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def load_api_key(env_path: Path) -> str | None:
    """FMP key: env var first, then read-only from the umbrella .env."""
    return os.environ.get("FMP_API_KEY") or _read_env_value(env_path, "FMP_API_KEY")


class AsOfError(ValueError):
    """Raised when a user-supplied --as-of would misdate a live fetch."""


def resolve_as_of(as_of_arg: str | None, *, today: date | None = None) -> str:
    """Resolve the snapshot as-of date for a LIVE fetch.

    HARD INVARIANT: every row is fetched NOW, so the ONLY honest as-of is the
    actual UTC fetch date. We therefore:

    * default to today's UTC date when ``--as-of`` is omitted;
    * accept an explicit ``--as-of`` ONLY if it equals today's UTC date -- it is
      a redundant assertion of the fetch date, useful only to make a script
      invocation self-documenting / to fail loudly if the host clock has drifted;
    * REJECT a PAST date -- stamping today's freshly fetched rows with a past
      date fabricates point-in-time history that never existed;
    * REJECT a FUTURE date -- pre-labelling today's fetch with a future slot
      stamps today's data as future data, which is equally fake provenance.
      Scheduling chooses the directory at RUN TIME from the real fetch date; it
      does not pre-name a future slot here.

    Legitimate historical backfill must come from an immutable source captured at
    that time (with its own provenance), not from this forward collector.

    Returns the validated ``YYYY-MM-DD`` string (always today's UTC date); raises
    :class:`AsOfError` on a malformed, past, or future value.
    """
    utc_today = today or datetime.now(timezone.utc).date()
    if as_of_arg is None:
        return utc_today.isoformat()
    try:
        requested = datetime.strptime(as_of_arg, "%Y-%m-%d").date()
    except ValueError as exc:
        raise AsOfError(f"--as-of must be YYYY-MM-DD, got {as_of_arg!r}") from exc
    if requested != utc_today:
        when = "the past" if requested < utc_today else "the future"
        raise AsOfError(
            f"--as-of {as_of_arg} is in {when} (UTC today is {utc_today.isoformat()}). "
            "A live fetch returns data as of NOW, so snapshot_as_of must equal the "
            "actual UTC fetch date. Backdating fabricates point-in-time history; "
            "pre-labelling a future slot stamps today's data as future data. "
            "Scheduling picks the directory at run time from the real fetch date; "
            "it does not pre-name a future slot. Historical backfill is valid only "
            "from an immutable source captured at that historical time."
        )
    return requested.isoformat()


def load_universe(universe_arg: str | None) -> list[str]:
    """Resolve the ticker universe.

    --universe accepts either:
      * a strategy_config.json (reads the ``watchlist`` list), or
      * a plain text/CSV file (one ticker per line; '#' comments ignored).
    Default: the renquant_104 golden config watchlist (read-only).
    """
    path = Path(universe_arg) if universe_arg else DEFAULT_UNIVERSE_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"universe source not found: {path}")
    text = path.read_text()
    if path.suffix == ".json":
        cfg = json.loads(text)
        wl = cfg.get("watchlist") or cfg.get("universe") or cfg.get("tickers")
        if not isinstance(wl, list) or not wl:
            raise ValueError(f"no 'watchlist' list found in {path}")
        tickers = [str(t).strip().upper() for t in wl if str(t).strip()]
    else:
        tickers = []
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                tickers.append(line.upper())
    # dedupe, stable order
    seen: dict[str, None] = {}
    for t in tickers:
        seen.setdefault(t, None)
    return list(seen)


def fetch_endpoint(
    session: "requests.Session", endpoint_path: str, sym: str, api_key: str
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Fetch one (endpoint, symbol). Returns (records, error). Never raises."""
    # ``requests`` is imported lazily, so resolve its exception base here (the
    # live path has already imported it via ``main``/``_require_requests``). Any
    # transport error becomes a clean ``fetch_error`` rather than propagating.
    request_exception = _require_requests().RequestException
    url = f"{FMP_STABLE_BASE}/{endpoint_path.format(sym=sym)}&apikey={api_key}"
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT_S)
    except request_exception as exc:
        return None, f"fetch_error:{type(exc).__name__}"
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"
    try:
        data = resp.json()
    except ValueError:
        return None, "bad_json"
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None, "unexpected_shape"
    return data, None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_one_endpoint(
    session: "requests.Session",
    endpoint: str,
    endpoint_path: str,
    tickers: Sequence[str],
    api_key: str,
    as_of: str,
    stage_dir: Path,
    dry_run: bool,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Fetch one endpoint across the universe into a STAGING dir.

    Writes the parquet + manifest into ``stage_dir`` (a temp area), never the
    final published dir -- publication is an atomic rename done by the caller
    once every endpoint clears its coverage floor. The per-endpoint manifest
    carries a ``status`` of ``ok``/``partial``/``dry_run`` so a shortfall is
    visible and blocks publication.
    """
    started = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    with_data = no_data = http_error = fetch_error = 0
    error_samples: list[str] = []

    for sym in tickers:
        if dry_run:
            continue
        records, err = fetch_endpoint(session, endpoint_path, sym, api_key)
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
                # stamp the as-of so the accrued series is self-describing PIT
                rec["snapshot_as_of"] = as_of
                rows.append(rec)
        time.sleep(THROTTLE_S)

    requested = len(tickers)
    # A fetch error (HTTP/network) is the failure mode that must block
    # publication; "no_data" (plan-locked names) is expected and OK. Coverage
    # is the share of names that either returned data OR cleanly had none.
    reached = with_data + no_data
    coverage = (reached / requested) if requested else 1.0
    had_fetch_failure = (http_error + fetch_error) > 0
    status = "ok"
    if not dry_run and (coverage < min_coverage or had_fetch_failure):
        status = "partial"

    out_path = stage_dir / endpoint
    parquet_path = out_path.with_suffix(".parquet")
    manifest_path = out_path.with_suffix(".manifest.json")
    finished = datetime.now(timezone.utc)

    manifest: dict[str, Any] = {
        "endpoint": endpoint,
        "path_template": endpoint_path,
        "url_base": FMP_STABLE_BASE,
        "as_of": as_of,
        "requested": requested,
        "with_data": with_data,
        "no_data": no_data,
        "http_error": http_error,
        "fetch_error": fetch_error,
        "error_samples": error_samples,
        "coverage": round(coverage, 4),
        "min_coverage": min_coverage,
        "rows": len(rows),
        "tickers": with_data,
        "ticker_count": with_data,
        "output": f"{endpoint}.parquet",
        "fetched_at": finished.isoformat(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "status": "dry_run" if dry_run else status,
    }

    if dry_run:
        manifest["sha256"] = None
        return manifest

    stage_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    manifest["sha256"] = _sha256_file(parquet_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _snapshot_is_published(final_dir: Path) -> bool:
    """A date is 'published' if its dir holds a manifest for every endpoint."""
    if not final_dir.is_dir():
        return False
    for endpoint in ENDPOINTS:
        if not (final_dir / f"{endpoint}.manifest.json").exists():
            return False
    return True


def collect_snapshot(
    *,
    session: "requests.Session",
    tickers: Sequence[str],
    api_key: str,
    as_of: str,
    out_root: Path,
    dry_run: bool,
    force: bool = False,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> dict[str, Any]:
    """Fetch every endpoint for ``as_of`` and atomically publish on success.

    Safety contract:
      * Idempotent verify, NOT destructive refetch -- if the date is already
        fully published and ``force`` is False, this is a no-op (``skipped``)
        and the existing snapshot is left untouched.
      * Staged then atomically renamed -- all endpoints are written into a
        sibling temp dir; only if EVERY endpoint clears its coverage floor is
        the date dir published via an atomic ``os.replace``. A partial fetch
        leaves any prior good snapshot intact and returns ``status: partial``.
    """
    final_dir = out_root / as_of

    if dry_run:
        manifests = [
            snapshot_one_endpoint(
                session=session,
                endpoint=endpoint,
                endpoint_path=endpoint_path,
                tickers=tickers,
                api_key=api_key,
                as_of=as_of,
                stage_dir=final_dir,  # unused on dry-run (no writes)
                dry_run=True,
                min_coverage=min_coverage,
            )
            for endpoint, endpoint_path in ENDPOINTS.items()
        ]
        return {"status": "dry_run", "published": False, "out_dir": final_dir,
                "manifests": manifests}

    if _snapshot_is_published(final_dir) and not force:
        return {"status": "skipped", "published": False, "out_dir": final_dir,
                "reason": "already_published", "manifests": []}

    out_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".stage-{as_of}-", dir=out_root))
    try:
        manifests = [
            snapshot_one_endpoint(
                session=session,
                endpoint=endpoint,
                endpoint_path=endpoint_path,
                tickers=tickers,
                api_key=api_key,
                as_of=as_of,
                stage_dir=stage_dir,
                dry_run=False,
                min_coverage=min_coverage,
            )
            for endpoint, endpoint_path in ENDPOINTS.items()
        ]
        partial = [m["endpoint"] for m in manifests if m["status"] != "ok"]
        if partial:
            # Do NOT publish over a (possibly good) existing snapshot.
            return {"status": "partial", "published": False, "out_dir": final_dir,
                    "partial_endpoints": partial, "manifests": manifests}
        # Atomic publish: replace the final dir in one rename.
        if final_dir.exists():
            backup = out_root / f".replaced-{as_of}-{int(time.time())}"
            os.replace(final_dir, backup)
            shutil.rmtree(backup, ignore_errors=True)
        os.replace(stage_dir, final_dir)
        stage_dir = final_dir  # consumed by the rename; nothing to clean up
        return {"status": "ok", "published": True, "out_dir": final_dir,
                "manifests": manifests}
    finally:
        if stage_dir != final_dir and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


# Canonical inputs we must never touch.
_FORBIDDEN_LEAVES = frozenset(
    {
        "fmp_harvest",
        "fmp_harvest_finnhub",
        "sec_fundamentals_daily",
        "rawlabel.parquet",
        "score_db",
    }
)


def _is_scratch_arg(out_root: Path) -> bool:
    """True only when the user EXPLICITLY pointed at a scratch tree.

    Judged on the argument as given (not the cwd-resolved path), so a relative
    target like ``data/whatever`` is never mistaken for scratch just because the
    process happens to run from under /tmp.
    """
    s = str(out_root)
    return (
        s.startswith("/tmp/")
        or s == "/tmp"
        or s.startswith("/private/tmp/")  # macOS resolves /tmp -> /private/tmp
        or s == "/private/tmp"
        or s.startswith("/private/var/folders")  # tempfile default on macOS
        or s.startswith("/var/folders")
    )


def is_canonical_path(out_root: Path) -> bool:
    """Guard: refuse to write any existing/canonical data path.

    The whole point is a NEW dedicated directory. We accept ONLY an out-root
    whose leaf is the dedicated ``estimate_snapshots`` name, or an explicit
    scratch (/tmp) target. Judged on the path *as given* AND on its fully
    resolved form, so neither a relative arg, the cwd, NOR a symlink can sneak a
    canonical leaf through: ``resolve()`` follows every symlink component, so a
    benign-looking ``estimate_snapshots`` that is actually a symlink into
    ``fmp_harvest`` resolves to the forbidden leaf and is rejected. This makes
    it structurally hard to clobber, e.g., the fmp_harvest, rawlabel, or
    sec_fundamentals canonical inputs.
    """
    resolved = out_root.resolve()
    # Forbidden-leaf check FIRST and on the resolved (symlink-followed) path, so a
    # symlink -- even one whose own path looks like /tmp scratch -- that resolves
    # into fmp_harvest / rawlabel / sec_fundamentals / score_db is always caught.
    parts = set(out_root.parts) | set(resolved.parts)
    if parts & _FORBIDDEN_LEAVES:
        return True
    # A scratch target is allowed only if it is scratch BOTH as given and once
    # fully resolved -- a symlink under /tmp pointing at a real data tree must
    # not be waved through.
    if _is_scratch_arg(out_root) and _is_scratch_arg(resolved):
        return False
    # require the dedicated leaf name for any non-scratch target, judged on the
    # resolved leaf too (a symlink can rename the leaf)
    return out_root.name != "estimate_snapshots" or resolved.name != "estimate_snapshots"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--universe",
        default=None,
        help="strategy_config.json (reads 'watchlist') or a one-ticker-per-line file; "
        "default = renquant_104 golden config",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"output root (default {DEFAULT_OUT}); a dated subdir is created under it",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="redundant assertion of today's UTC date YYYY-MM-DD for the snapshot "
        "dir (default = today, UTC). It must EQUAL today's UTC date -- BOTH a past "
        "and a future date are rejected, because a live fetch returns data as of NOW "
        "(snapshot_as_of = the actual fetch date). Backdating fabricates PIT history; "
        "a future slot stamps today's data as future data.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-publish even if this date is already fully published "
        "(default: an already-published date is a no-op verify, not a refetch)",
    )
    ap.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="fraction of the universe that must be reached per endpoint to "
        f"publish (default {DEFAULT_MIN_COVERAGE}); below it the snapshot is "
        "marked partial and NOT published",
    )
    ap.add_argument(
        "--env",
        default=str(DEFAULT_ENV),
        help="path to a .env holding FMP_API_KEY (read-only)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve universe + plan paths and print, but make NO network call or write",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        as_of = resolve_as_of(args.as_of)
    except AsOfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    if is_canonical_path(out_root):
        print(
            f"error: refusing to write canonical/non-dedicated path {out_root!r}; "
            f"use a 'estimate_snapshots' dir or a /tmp demo path",
            file=sys.stderr,
        )
        return 2

    try:
        tickers = load_universe(args.universe)
    except (OSError, ValueError) as exc:
        print(f"error: could not load universe: {exc}", file=sys.stderr)
        return 2

    out_dir = out_root / as_of
    print(
        f"fmp_estimate_revisions: as_of={as_of} universe={len(tickers)} tickers "
        f"endpoints={list(ENDPOINTS)} out={out_dir}"
        + ("  [DRY-RUN]" if args.dry_run else ""),
        file=sys.stderr,
    )

    api_key = None
    if not args.dry_run:
        api_key = load_api_key(Path(args.env))
        if not api_key:
            print(
                f"error: FMP_API_KEY not found (env or {args.env})",
                file=sys.stderr,
            )
            return 2

    # ``requests`` (lazy) is needed only for a live fetch; a dry-run plans paths
    # without any network session, so it must work without the dependency.
    session = None if args.dry_run else _require_requests().Session()
    result = collect_snapshot(
        session=session,
        tickers=tickers,
        api_key=api_key or "",
        as_of=as_of,
        out_root=out_root,
        dry_run=args.dry_run,
        force=args.force,
        min_coverage=args.min_coverage,
    )
    manifests = result.get("manifests", [])
    for m in manifests:
        print(
            f"  {m['endpoint']:24s} rows={m['rows']:6d} with_data={m['with_data']:4d} "
            f"no_data={m['no_data']:3d} http_err={m['http_error']:3d} "
            f"fetch_err={m['fetch_error']:3d} status={m['status']}",
            file=sys.stderr,
        )
    if result["status"] == "skipped":
        print(
            f"  {as_of} already published at {out_dir} -- no-op (pass --force to re-publish)",
            file=sys.stderr,
        )
    elif result["status"] == "partial":
        print(
            f"  PARTIAL: endpoints below coverage floor: {result['partial_endpoints']}; "
            f"NOT published (prior snapshot, if any, left intact)",
            file=sys.stderr,
        )

    print(
        json.dumps(
            {
                "as_of": as_of,
                "out_dir": str(out_dir),
                "universe": len(tickers),
                "dry_run": args.dry_run,
                "status": result["status"],
                "published": result.get("published", False),
                "endpoints": {m["endpoint"]: m["rows"] for m in manifests},
            },
            indent=2,
        )
    )
    # Non-zero exit on a real shortfall so a scheduler/alert can react.
    return 1 if result["status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
