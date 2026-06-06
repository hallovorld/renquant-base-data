from __future__ import annotations

import json
from pathlib import Path

from renquant_base_data.track_b_features import TRACK_B_FEATURES
from renquant_base_data.track_b_readiness import (
    TRACK_B_BULL_CALM_FEATURE_MANIFEST,
    TRACK_B_READINESS_CHECKLIST,
    TRACK_B_READINESS_SCHEMA_VERSION,
    TRACK_B_WF_TARGET,
    main,
    validate_track_b_feature_readiness,
)


def _manifest_json() -> dict:
    path = (
        Path(__file__).parents[1]
        / "manifests"
        / "track-b-bull-calm-feature-readiness.json"
    )
    return json.loads(path.read_text())


def test_track_b_readiness_manifest_matches_code_constants() -> None:
    manifest = _manifest_json()

    assert manifest["schema_version"] == TRACK_B_READINESS_SCHEMA_VERSION
    assert manifest["target"] == TRACK_B_WF_TARGET
    assert tuple(manifest["required_features"]) == TRACK_B_FEATURES
    assert manifest["long_training"] is False
    assert (
        tuple(TRACK_B_BULL_CALM_FEATURE_MANIFEST["required_features"])
        == TRACK_B_FEATURES
    )


def test_track_b_readiness_checklist_has_required_slugs() -> None:
    manifest_slugs = {item["slug"] for item in _manifest_json()["checklist"]}
    code_slugs = {item["slug"] for item in TRACK_B_READINESS_CHECKLIST}

    assert manifest_slugs == code_slugs
    assert "candidate_panel_contains_required_columns" in manifest_slugs
    assert "no_long_training_in_base_data" in manifest_slugs


def test_validate_track_b_feature_readiness_passes_for_all_four_features() -> None:
    report = validate_track_b_feature_readiness(
        ["date", "ticker", *TRACK_B_FEATURES],
        manifest=_manifest_json(),
    )

    assert report["ok"] is True
    assert report["missing_features"] == []
    assert report["long_training_disabled"] is True


def test_validate_track_b_feature_readiness_reports_missing_features() -> None:
    report = validate_track_b_feature_readiness(["mom_carry_12_1", "beta_dm"])

    assert report["ok"] is False
    assert report["missing_features"] == ["rvar_total", "idio_vol_market"]


def test_track_b_readiness_cli_exit_codes(capsys) -> None:
    assert main(["--columns", *TRACK_B_FEATURES]) == 0
    ok_out = json.loads(capsys.readouterr().out)
    assert ok_out["ok"] is True

    assert main(["--columns", "mom_carry_12_1"]) == 2
    bad_out = json.loads(capsys.readouterr().out)
    assert bad_out["missing_features"] == ["beta_dm", "rvar_total", "idio_vol_market"]
