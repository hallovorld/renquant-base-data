"""Dataset manifest registry and resolver pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renquant_common import Job, Pipeline, Task

from .validation import validate_data_manifest


@dataclass
class DataRegistryContext:
    """Mutable context for resolving one dataset manifest from a registry."""

    registry_dir: Path
    dataset_id: str | None = None
    asset_class: str | None = None
    retention_class: str | None = None
    candidates: list[tuple[Path, dict[str, Any]]] = field(default_factory=list)
    selected_path: Path | None = None
    manifest: dict[str, Any] | None = None
    validation_report: dict[str, Any] = field(default_factory=dict)


class LoadDataRegistryTask(Task):
    def run(self, ctx: DataRegistryContext) -> bool | None:
        if not ctx.registry_dir.exists():
            raise FileNotFoundError(f"data registry does not exist: {ctx.registry_dir}")
        if not ctx.registry_dir.is_dir():
            raise NotADirectoryError(f"data registry is not a directory: {ctx.registry_dir}")
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(ctx.registry_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            candidates.append((path, payload))
        if not candidates:
            raise ValueError(f"data registry has no JSON manifests: {ctx.registry_dir}")
        ctx.candidates = candidates
        return True


class SelectDataManifestTask(Task):
    def run(self, ctx: DataRegistryContext) -> bool | None:
        matches = []
        for path, manifest in ctx.candidates:
            if ctx.dataset_id is not None and manifest.get("dataset_id") != ctx.dataset_id:
                continue
            if ctx.asset_class is not None and manifest.get("asset_class") != ctx.asset_class:
                continue
            if ctx.retention_class is not None and manifest.get("retention_class") != ctx.retention_class:
                continue
            matches.append((path, manifest))
        if not matches:
            raise ValueError(
                "no data manifest matched "
                f"dataset_id={ctx.dataset_id!r} asset_class={ctx.asset_class!r} "
                f"retention_class={ctx.retention_class!r}"
            )
        if len(matches) > 1:
            names = [str(path.name) for path, _ in matches]
            raise ValueError(f"ambiguous data manifest selection: {names}")
        ctx.selected_path, ctx.manifest = matches[0]
        return True


class ValidateSelectedDataManifestTask(Task):
    def run(self, ctx: DataRegistryContext) -> bool | None:
        if ctx.manifest is None:
            raise ValueError("manifest must be selected before validation")
        ctx.validation_report = validate_data_manifest(ctx.manifest)
        ctx.validation_report["path"] = str(ctx.selected_path)
        return True


class DataManifestResolverJob(Job):
    @property
    def tasks(self) -> list[Task]:
        return [
            LoadDataRegistryTask(),
            SelectDataManifestTask(),
            ValidateSelectedDataManifestTask(),
        ]


class DataManifestResolverPipeline(Pipeline):
    def __init__(self) -> None:
        super().__init__([DataManifestResolverJob()], name="data-manifest-resolver")


def resolve_data_manifest(
    registry_dir: str | Path,
    *,
    dataset_id: str | None = None,
    asset_class: str | None = None,
    retention_class: str | None = None,
) -> dict[str, Any]:
    """Resolve and validate exactly one dataset manifest from a registry."""
    ctx = DataRegistryContext(
        registry_dir=Path(registry_dir),
        dataset_id=dataset_id,
        asset_class=asset_class,
        retention_class=retention_class,
    )
    DataManifestResolverPipeline().run(ctx)
    if ctx.manifest is None:
        raise ValueError("data manifest resolver finished without a manifest")
    return ctx.manifest


def load_data_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a single dataset manifest file."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_data_manifest(manifest)
    return manifest
