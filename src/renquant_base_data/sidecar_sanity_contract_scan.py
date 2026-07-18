"""AC-1 (x) migration precondition: no sanity contract may name the sentiment cols.

Context (doc/design/2026-07-18-rawlabel-sidecar-sentiment-reconciliation.md, AC-1):
the WF gate (renquant-backtesting ``wf_gate/runner.py``) reads the served
rawlabel sidecar with a bare ``pd.read_parquet`` and resolves the needed
columns DYNAMICALLY from the candidate artifact's sanity contract
(``feature_cols``). A model whose contract names the three sentiment columns
would, after the sidecar migrates 179 -> 176 columns, silently flip that
sanity run from the direct path into the supplement/merge path
(``feature_panel_merge: True``). AC-1 option (x) therefore requires a
mechanical check that NO active/candidate sanity contract names
``sentiment_pos_share`` / ``mean_sentiment`` / ``n_articles_log`` before any
migration is allowed to proceed.

This module IS that check. It is read-only: it enumerates artifact JSON
payloads under the given roots, extracts each payload's ``feature_cols``
(the sanity contract the WF gate consumes) plus ``training_contract.dataset``
(when recorded, the sanity run scores on that dataset and never touches the
sidecar), and FAILS (exit 1) when any contract names a sentiment column —
or when a payload cannot be parsed (fail closed: an unreadable contract is
an unverified contract).

Scope notes:
- ``feature_cols``-less JSONs (calibrations, thresholds, calendars) are
  counted but are not contracts; they cannot flip the sanity path.
- Walk-forward MANIFEST payloads (a ``retrains`` list) are chased: each
  entry's ``artifact_uri`` is resolved to its metadata payload
  (``<uri>.metadata.json`` for ``.pt`` checkpoints, the JSON itself
  otherwise) so PatchTST-style candidates are covered.
- ``dataset_recorded`` is reported per contract: a contract WITH a recorded
  training dataset resolves sanity on that dataset (sidecar path not
  taken); the verdict is still strict per the RFC's (x) wording — ANY
  contract naming the columns fails the precondition, and the report gives
  the reviewer the sidecar-path exposure split.

Exit status: 0 = precondition HOLDS (migration may proceed under (x));
1 = precondition FAILS or a payload was unparseable (fail closed).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from renquant_base_data.rawlabel_sidecar import SENTIMENT_COLS

#: Default active + candidate sanity-contract surfaces, relative to the
#: umbrella repo root. Diagnostics / cache / modal_sweep_* bundle archives are
#: deliberately OUT of scope (frozen archives, not live consumers — AC-1 sweep
#: exemption (iii)).
DEFAULT_SURFACES = (
    "backtesting/renquant_104/artifacts/*.json",
    "backtesting/renquant_104/artifacts/prod/*.json",
    "backtesting/renquant_104/artifacts/shadow/*.json",
    "backtesting/renquant_104/artifacts/walkforward_*/**/*.json",
    "backtesting/renquant_104/artifacts/sim/**/*.json",
    "artifacts/walkforward_manifest_*.json",
)


def _load_json(path: Path) -> "dict | list | None":
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _resolve_manifest_artifact_payload(
    uri: str, manifest_path: Path
) -> "tuple[str, dict | None] | None":
    """Resolve a walk-forward manifest ``artifact_uri`` to its metadata payload.

    ``.pt`` checkpoints carry their sanity contract in a ``<uri>.metadata.json``
    sidecar; JSON artifacts carry it inline. Relative URIs are anchored the way
    the WF runner anchors them (``_manifest_uri_to_path``): against the
    manifest's own directory and its ancestors (manifests under
    ``artifacts/sim/`` record strategy-home-relative URIs like
    ``artifacts/walkforward_v2_.../panel-ltr.json``). Returns
    ``(resolved_path, payload)`` with ``payload=None`` when the metadata file
    exists but cannot be parsed, and ``None`` when there is no metadata surface
    to scan (reported upstream as unresolved — fail closed).
    """
    p = Path(uri)
    anchors: "list[Path]" = [Path("")] if p.is_absolute() else list(
        manifest_path.resolve().parents
    )
    for anchor in anchors:
        resolved = p if p.is_absolute() else anchor / p
        candidates = (
            [resolved.with_name(resolved.name + ".metadata.json")]
            if resolved.suffix != ".json"
            else [resolved]
        )
        for cand in candidates:
            if cand.exists():
                payload = _load_json(cand)
                return str(cand), payload if isinstance(payload, dict) else None
    return None


def scan_contracts(roots: "list[str]") -> dict:
    """Scan artifact payloads under ``roots`` (files, dirs, or globs).

    Returns a report dict with per-file records and a strict verdict.
    """
    seen: "dict[str, dict]" = {}
    unresolved: "list[str]" = []
    sentiment = set(SENTIMENT_COLS)

    paths: "list[str]" = []
    for root in roots:
        rp = Path(root)
        if rp.is_dir():
            paths.extend(sorted(glob.glob(str(rp / "**/*.json"), recursive=True)))
        else:
            paths.extend(sorted(glob.glob(root, recursive=True)))

    def _record(path: str, payload: "dict | list | None") -> None:
        if path in seen:
            return
        if payload is None:
            seen[path] = {"status": "unparseable"}
            return
        if not isinstance(payload, dict):
            seen[path] = {"status": "no_feature_cols"}
            return
        # Chase walk-forward manifests to their per-cut artifact contracts.
        retrains = payload.get("retrains")
        if isinstance(retrains, list) and retrains and "feature_cols" not in payload:
            seen[path] = {"status": "manifest", "n_retrains": len(retrains)}
            for entry in retrains:
                uri = (entry or {}).get("artifact_uri") if isinstance(entry, dict) else None
                if not uri:
                    unresolved.append(f"{path}: retrain entry without artifact_uri")
                    continue
                resolved = _resolve_manifest_artifact_payload(str(uri), Path(path))
                if resolved is None:
                    unresolved.append(f"{path}: no metadata surface for {uri}")
                    continue
                _record(*resolved)
            return
        feature_cols = payload.get("feature_cols")
        contract = (
            payload.get("training_contract")
            if isinstance(payload.get("training_contract"), dict)
            else {}
        )
        dataset = (contract or {}).get("dataset") or payload.get("dataset")
        if not isinstance(feature_cols, list):
            seen[path] = {
                "status": "no_feature_cols",
                "dataset_recorded": bool(dataset),
            }
            return
        named = sorted(sentiment & {str(c) for c in feature_cols})
        seen[path] = {
            "status": "contract",
            "n_feature_cols": len(feature_cols),
            "sentiment_named": named,
            "dataset_recorded": bool(dataset),
            "label_col": payload.get("label_col") or payload.get("label"),
        }

    for path in paths:
        _record(path, _load_json(Path(path)))

    contracts = {p: r for p, r in seen.items() if r["status"] == "contract"}
    violations = {p: r for p, r in contracts.items() if r["sentiment_named"]}
    unparseable = sorted(p for p, r in seen.items() if r["status"] == "unparseable")
    # Sidecar-path exposure: contracts with NO recorded training dataset fall
    # back to the served rawlabel sidecar in wf_gate _load_sanity_panel.
    sidecar_exposed_violations = sorted(
        p for p, r in violations.items() if not r["dataset_recorded"]
    )
    return {
        "sentiment_cols": sorted(sentiment),
        "n_scanned": len(seen),
        "n_contracts": len(contracts),
        "n_no_feature_cols": sum(
            1 for r in seen.values() if r["status"] == "no_feature_cols"
        ),
        "n_manifests": sum(1 for r in seen.values() if r["status"] == "manifest"),
        "violations": {p: contracts[p] for p in sorted(violations)},
        "n_violations": len(violations),
        "sidecar_exposed_violations": sidecar_exposed_violations,
        "unparseable": unparseable,
        "unresolved_manifest_entries": unresolved,
        "precondition_holds": not violations and not unparseable and not unresolved,
    }


def parse_args(argv: "list | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--umbrella-root",
        type=Path,
        default=None,
        help="Umbrella repo root; expands the default active+candidate surfaces.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Extra file/dir/glob to scan (repeatable).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: "list | None" = None) -> int:
    args = parse_args(argv)
    roots = list(args.root)
    if args.umbrella_root:
        roots.extend(str(args.umbrella_root / s) for s in DEFAULT_SURFACES)
    if not roots:
        raise SystemExit("no roots: pass --umbrella-root and/or --root")
    report = scan_contracts(roots)
    print(json.dumps({k: v for k, v in report.items() if k != "violations"}, indent=2))
    for path, rec in report["violations"].items():
        print(
            f"VIOLATION {path} names {rec['sentiment_named']} "
            f"(n_feature_cols={rec['n_feature_cols']}, "
            f"dataset_recorded={rec['dataset_recorded']})"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["precondition_holds"]:
        print("PRECONDITION HOLDS: no active/candidate sanity contract names the sentiment columns")
        return 0
    print(
        f"PRECONDITION FAILS: {report['n_violations']} contract(s) name sentiment columns; "
        f"{len(report['unparseable'])} unparseable; "
        f"{len(report['unresolved_manifest_entries'])} unresolved manifest entries "
        "(AC-1 (x) blocked — see the RFC: disposition flips to (y) or migration is blocked)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
