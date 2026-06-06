"""Track B BULL_CALM feature-readiness manifest and validation.

This module is intentionally a lightweight gate for the Track B PR chain. It
does not materialize data or run walk-forward training; it only checks that a
candidate feature surface exposes the four Track B columns needed by the
BULL_CALM full WF retrain.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from renquant_base_data.track_b_features import TRACK_B_FEATURES

TRACK_B_READINESS_SCHEMA_VERSION = "track-b-feature-readiness-v1"
TRACK_B_WF_TARGET = "BULL_CALM Track B full WF retrain"

TRACK_B_READINESS_CHECKLIST: tuple[dict[str, str], ...] = (
    {
        "slug": "feature_builders_registered",
        "description": (
            "Track B feature builders are importable from "
            "renquant_base_data.track_b_features."
        ),
    },
    {
        "slug": "manifest_features_match_code",
        "description": "The readiness manifest required_features exactly matches TRACK_B_FEATURES.",
    },
    {
        "slug": "candidate_panel_contains_required_columns",
        "description": (
            "The retrain candidate panel contains mom_carry_12_1, beta_dm, "
            "rvar_total, and idio_vol_market."
        ),
    },
    {
        "slug": "no_long_training_in_base_data",
        "description": (
            "This repo only validates feature readiness; full WF retrain runs "
            "in the strategy repo."
        ),
    },
)

TRACK_B_BULL_CALM_FEATURE_MANIFEST: dict[str, Any] = {
    "manifest_id": "track-b-bull-calm-feature-readiness",
    "schema_version": TRACK_B_READINESS_SCHEMA_VERSION,
    "target": TRACK_B_WF_TARGET,
    "required_features": list(TRACK_B_FEATURES),
    "source_module": "renquant_base_data.track_b_features",
    "validation_entrypoint": (
        "python -m renquant_base_data.track_b_readiness "
        "--columns <feature columns>"
    ),
    "long_training": False,
    "checklist": list(TRACK_B_READINESS_CHECKLIST),
}


def validate_track_b_feature_readiness(
    columns: Iterable[str],
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that *columns* are ready for Track B BULL_CALM retrain.

    Returns a JSON-serializable report instead of raising, so callers can use
    it in CI, notebooks, or preflight logs.
    """
    manifest_in = (
        manifest if manifest is not None else TRACK_B_BULL_CALM_FEATURE_MANIFEST
    )
    column_set = {str(col) for col in columns}
    manifest_features = tuple(str(f) for f in manifest_in.get("required_features", ()))
    checklist = manifest_in.get("checklist", ())
    checklist_slugs = {
        str(item.get("slug", ""))
        for item in checklist
        if isinstance(item, Mapping)
    }
    required_slugs = {item["slug"] for item in TRACK_B_READINESS_CHECKLIST}

    missing_features = [
        feature for feature in TRACK_B_FEATURES if feature not in column_set
    ]
    manifest_missing = [
        feature for feature in TRACK_B_FEATURES if feature not in manifest_features
    ]
    manifest_extra = [
        feature for feature in manifest_features if feature not in TRACK_B_FEATURES
    ]
    missing_checklist = sorted(required_slugs - checklist_slugs)
    schema_ok = manifest_in.get("schema_version") == TRACK_B_READINESS_SCHEMA_VERSION
    target_ok = manifest_in.get("target") == TRACK_B_WF_TARGET
    long_training_disabled = manifest_in.get("long_training") is False

    ok = (
        not missing_features
        and not manifest_missing
        and not manifest_extra
        and not missing_checklist
        and schema_ok
        and target_ok
        and long_training_disabled
    )
    return {
        "ok": bool(ok),
        "target": TRACK_B_WF_TARGET,
        "required_features": list(TRACK_B_FEATURES),
        "missing_features": missing_features,
        "present_features": [
            feature for feature in TRACK_B_FEATURES if feature in column_set
        ],
        "manifest_missing_features": manifest_missing,
        "manifest_extra_features": manifest_extra,
        "missing_checklist_slugs": missing_checklist,
        "schema_ok": schema_ok,
        "target_ok": target_ok,
        "long_training_disabled": long_training_disabled,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate BULL_CALM Track B feature readiness without running training.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=(),
        help="Feature column names from the candidate retrain panel.",
    )
    args = parser.parse_args(argv)
    report = validate_track_b_feature_readiness(args.columns)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TRACK_B_BULL_CALM_FEATURE_MANIFEST",
    "TRACK_B_READINESS_CHECKLIST",
    "TRACK_B_READINESS_SCHEMA_VERSION",
    "TRACK_B_WF_TARGET",
    "main",
    "validate_track_b_feature_readiness",
]
