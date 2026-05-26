from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_base_data import load_data_manifest, resolve_data_manifest


def _write_manifest(path: Path, **overrides) -> dict:
    payload = {
        "dataset_id": "alpha158-fund-prod",
        "schema_version": "v1",
        "fingerprint": "sha256:data",
        "uri": "object://renquant-data/alpha158-fund-prod.parquet",
        "asset_class": "equity",
        "retention_class": "prod",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_resolve_data_manifest_selects_exact_dataset(tmp_path: Path) -> None:
    expected = _write_manifest(tmp_path / "prod.json")
    _write_manifest(
        tmp_path / "shadow.json",
        dataset_id="alpha158-fund-shadow",
        retention_class="shadow",
        fingerprint="sha256:shadow",
    )

    resolved = resolve_data_manifest(tmp_path, dataset_id="alpha158-fund-prod")

    assert resolved == expected


def test_resolve_data_manifest_fails_closed_on_ambiguity(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "a.json", dataset_id="same")
    _write_manifest(tmp_path / "b.json", dataset_id="same", fingerprint="sha256:b")

    with pytest.raises(ValueError, match="ambiguous data manifest selection"):
        resolve_data_manifest(tmp_path, dataset_id="same")


def test_load_data_manifest_validates_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path)

    assert load_data_manifest(path)["dataset_id"] == "alpha158-fund-prod"
