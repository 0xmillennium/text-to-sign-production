"""Runtime input verification for the tiers workflow."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.catalog import (
    iter_samples,
    load_passed_samples_catalog,
)
from text_to_sign_production.artifacts.store import ArtifactStores, build_artifact_stores
from text_to_sign_production.foundation.roots import build_repo_roots
from text_to_sign_production.workflows.tiers.types import (
    TiersWorkflowConfig,
    TiersWorkflowInputError,
)


def verify_tiers_runtime_inputs(config: TiersWorkflowConfig) -> None:
    stores = build_artifact_stores(
        build_repo_roots(config.project_root),
        build_repo_roots(config.drive_project_root),
    )
    missing: list[str] = []

    _require_file(
        config.project_root / config.filters_config_relative_path,
        "filters config",
        missing,
    )
    _require_file(
        config.project_root / config.tiers_config_relative_path,
        "tiers config",
        missing,
    )

    for split in config.splits:
        manifest_path = stores.runtime.manifests.untiered_passed_manifest(split).path
        sample_dir = stores.runtime.samples.passed_split_dir(split).path
        _require_file(manifest_path, f"{split.value} untiered passed manifest", missing)
        _require_dir(sample_dir, f"{split.value} passed sample directory", missing)
        if manifest_path.is_file() and sample_dir.is_dir() and _jsonl_has_records(manifest_path):
            if not any(sample_dir.glob("*.npz")):
                missing.append(
                    f"{split.value} passed sample directory has no .npz files: {sample_dir}"
                )

    if not missing:
        _require_manifest_declared_samples(stores, config, missing)

    if missing:
        raise TiersWorkflowInputError(
            "Tiers runtime inputs are incomplete: " + "; ".join(missing)
        )


def _require_manifest_declared_samples(
    stores: ArtifactStores,
    config: TiersWorkflowConfig,
    missing: list[str],
) -> None:
    try:
        catalog = load_passed_samples_catalog(stores, splits=config.splits)
    except Exception as exc:
        missing.append(f"passed samples catalog is invalid: {exc}")
        return

    for handle in iter_samples(catalog):
        runtime_sample = handle.runtime_sample
        if runtime_sample is None:
            missing.append(
                "passed manifest entry has no runtime sample path: "
                f"{handle.ref.split.value}/{handle.ref.sample_id}"
            )
            continue

        if not runtime_sample.path.is_file():
            missing.append(
                "passed manifest entry sample file is missing: "
                f"{handle.ref.split.value}/{handle.ref.sample_id}: "
                f"{runtime_sample.path}"
            )


def _require_file(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_file():
        missing.append(f"{label} is not a file: {path}")


def _require_dir(path: Path, label: str, missing: list[str]) -> None:
    if not path.is_dir():
        missing.append(f"{label} is not a directory: {path}")


def _jsonl_has_records(path: Path) -> bool:
    with path.open(encoding="utf-8") as handle:
        return any(bool(line.strip()) for line in handle)


__all__ = ["verify_tiers_runtime_inputs"]
