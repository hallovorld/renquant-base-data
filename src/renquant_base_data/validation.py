"""Data-manifest validation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from renquant_common import Job, Pipeline, Task


@dataclass
class DataManifestContext:
    manifest: dict[str, Any]
    validation_report: dict[str, Any] = field(default_factory=dict)


class ValidateDataManifestTask(Task):
    def run(self, ctx: DataManifestContext) -> bool | None:
        required = ("dataset_id", "schema_version", "fingerprint", "uri", "asset_class")
        missing = [key for key in required if not ctx.manifest.get(key)]
        if missing:
            raise ValueError(f"data manifest missing required keys: {missing}")
        if ctx.manifest["uri"].startswith("/Users/"):
            raise ValueError("data manifest uri must not be developer-local absolute path")
        ctx.validation_report = {
            "dataset_id": ctx.manifest["dataset_id"],
            "fingerprint": ctx.manifest["fingerprint"],
            "ok": True,
        }
        return True


class DataManifestValidationJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [ValidateDataManifestTask()]


class DataManifestValidationPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__([DataManifestValidationJob()], name="data-manifest-validation")
