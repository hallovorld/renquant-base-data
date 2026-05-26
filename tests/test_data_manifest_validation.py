from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_base_data import DataManifestContext, DataManifestValidationPipeline


def test_example_manifest_validates() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "manifests" / "example-dataset.json").read_text())
    ctx = DataManifestContext(manifest)
    result = DataManifestValidationPipeline().run(ctx)

    assert result.ok is True
    assert ctx.validation_report["ok"] is True


def test_local_absolute_data_uri_is_rejected() -> None:
    ctx = DataManifestContext({
        "dataset_id": "bad",
        "schema_version": "v1",
        "fingerprint": "sha256:bad",
        "uri": "/Users/renhao/git/github/RenQuant/data/file.parquet",
        "asset_class": "equity",
    })
    with pytest.raises(ValueError, match="developer-local"):
        DataManifestValidationPipeline().run(ctx)
