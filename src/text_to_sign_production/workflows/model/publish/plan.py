"""Plan file-only publication of model workflow outputs to Drive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.modeling.candidates import read_runtime_support_manifest
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME,
)
from text_to_sign_production.workflows.foundation.execution import (
    FileCopyOperation,
    OperationProgressSpec,
)
from text_to_sign_production.workflows.model.constants import MODEL_STAGE_PUBLISH_EXECUTE
from text_to_sign_production.workflows.model.contracts import (
    ModelPublishPlan,
    ModelPublishSkippedSource,
    ModelPublishSourceBundle,
    ModelPublishTarget,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout
from text_to_sign_production.workflows.model.progress import model_progress_stage


@dataclass(frozen=True, slots=True)
class _PublishSource:
    label: str
    kind: str
    path: Path
    sha256: str | None
    execution_id: str | None


def build_model_publish_plan(
    *,
    layout: ModelLayout,
    sources: ModelPublishSourceBundle,
) -> ModelPublishPlan:
    """Map materialized runtime files to matching Drive-relative targets."""

    targets: list[ModelPublishTarget] = []
    skipped: list[ModelPublishSkippedSource] = []
    seen_targets: dict[Path, _PublishSource] = {}
    publish_sources = (
        *_sources(sources),
        *_checkpoint_sources(sources),
        *_runtime_support_sources(sources),
        *_calibration_sources(sources),
        *_generated_pose_archive_sources(layout),
    )
    archive_roots = _generated_pose_archive_roots(publish_sources)
    for source in publish_sources:
        path = source.path.resolve(strict=False)
        if _is_individual_generated_pose_sample_source(source, archive_roots=archive_roots):
            skipped.append(
                ModelPublishSkippedSource(
                    label=source.label,
                    path=path,
                    reason=(
                        "individual generated-pose sample publication is disabled "
                        "because a generated samples archive is available"
                    ),
                )
            )
            continue
        try:
            relative = path.relative_to(layout.stores.runtime.repo_root)
        except ValueError as exc:
            raise ModelWorkflowInvariantError(
                f"publish source is outside runtime repository root: {path}"
            ) from exc
        if not path.exists():
            skipped.append(
                ModelPublishSkippedSource(
                    label=source.label,
                    path=path,
                    reason="source path does not exist",
                )
            )
            continue
        if path.is_dir():
            skipped.append(
                ModelPublishSkippedSource(
                    label=source.label,
                    path=path,
                    reason="directory publication is not supported; publish materialized files",
                )
            )
            continue
        if not path.is_file():
            skipped.append(
                ModelPublishSkippedSource(
                    label=source.label,
                    path=path,
                    reason="source path is not a regular file",
                )
            )
            continue
        target_path = (layout.stores.drive.repo_root / relative).resolve(strict=False)
        if target_path in seen_targets:
            previous = seen_targets[target_path]

            previous_source_path = previous.path.resolve(strict=False)
            current_source_path = path.resolve(strict=False)

            if (
                previous_source_path == current_source_path
                and previous.kind == source.kind
            ):
                skipped.append(
                    ModelPublishSkippedSource(
                        label=source.label,
                        path=path,
                        reason=(
                            "duplicate publish source already covered by an earlier "
                            f"source with the same target, kind={source.kind!r}, and source path"
                        ),
                    )
                )
                continue

            raise ModelWorkflowInvariantError(
                "duplicate model publish target: "
                f"target_path={target_path}; "
                f"previous_label={previous.label!r}; "
                f"previous_kind={previous.kind!r}; "
                f"previous_source_path={previous.path}; "
                f"current_label={source.label!r}; "
                f"current_kind={source.kind!r}; "
                f"current_source_path={path}"
            )
        seen_targets[target_path] = source
        targets.append(
            ModelPublishTarget(
                label=source.label,
                kind=source.kind,
                source_path=path,
                target_path=target_path,
                source_sha256=source.sha256 or sha256_file(path),
                source_execution_id=source.execution_id,
            )
        )
    operations = tuple(_copy_operation(target) for target in targets)
    return ModelPublishPlan(
        targets=tuple(targets),
        skipped_sources=tuple(skipped),
        operations=operations,
    )


def _sources(bundle: ModelPublishSourceBundle) -> tuple[_PublishSource, ...]:
    sources: list[_PublishSource] = []
    if bundle.metadata_artifacts is not None:
        for receipt in (
            bundle.metadata_artifacts.effective_config,
            bundle.metadata_artifacts.research_spec,
            bundle.metadata_artifacts.run_metadata,
            bundle.metadata_artifacts.runtime_support_manifest,
        ):
            sources.append(
                _PublishSource(
                    label=f"publish {receipt.label}",
                    kind=receipt.kind,
                    path=receipt.path,
                    sha256=receipt.sha256,
                    execution_id=receipt.execution_id,
                )
            )
    if bundle.report_artifacts is not None:
        for receipt in bundle.report_artifacts.artifacts:
            sources.append(
                _PublishSource(
                    label=f"publish {receipt.label}",
                    kind=receipt.kind,
                    path=receipt.path,
                    sha256=receipt.sha256,
                    execution_id=receipt.execution_id,
                )
            )
    if bundle.validation_artifacts is not None:
        for receipt in bundle.validation_artifacts.receipts:
            sources.append(
                _PublishSource(
                    label=f"publish {receipt.label}",
                    kind=receipt.kind,
                    path=receipt.path,
                    sha256=receipt.sha256,
                    execution_id=receipt.execution_id,
                )
            )
    for artifact_result in bundle.objective_artifacts:
        if artifact_result.ablation_readiness_path is None:
            raise ModelWorkflowInvariantError(
                "semantic objective publication requires an ablation readiness artifact."
            )
        for receipt in artifact_result.receipts:
            sources.append(
                _PublishSource(
                    label=f"publish {receipt.label}",
                    kind=receipt.kind,
                    path=receipt.path,
                    sha256=receipt.sha256,
                    execution_id=receipt.execution_id,
                )
            )
    if bundle.stage_artifact_receipts is not None:
        for receipt in (bundle.stage_artifact_receipts.index_receipt,):
            sources.append(
                _PublishSource(
                    label=f"publish {receipt.label}",
                    kind=receipt.kind,
                    path=receipt.path,
                    sha256=receipt.sha256,
                    execution_id=receipt.execution_id,
                )
            )
    return tuple(sources)


def _checkpoint_sources(bundle: ModelPublishSourceBundle) -> tuple[_PublishSource, ...]:
    if bundle.metadata_artifacts is None:
        return ()
    metadata_path = bundle.metadata_artifacts.run_metadata_path
    if not metadata_path.is_file():
        return ()
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelWorkflowInvariantError(
            f"model run metadata is not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(raw, dict):
        raise ModelWorkflowInvariantError(
            f"model run metadata root must be an object: {metadata_path}"
        )
    sources: list[_PublishSource] = []
    for role, expected_name, value in (
        ("best", "best.pt", _metadata_checkpoint_value(raw, "best")),
        ("last", "last.pt", _metadata_checkpoint_value(raw, "last")),
    ):
        path = _metadata_checkpoint_path(value, metadata_path=metadata_path, role=role)
        if path is None:
            continue
        if path.name != expected_name:
            raise ModelWorkflowInvariantError(
                f"canonical {role} checkpoint path must end with {expected_name!r}: {path}"
            )
        if path.parent.name != "checkpoints":
            raise ModelWorkflowInvariantError(
                f"canonical {role} checkpoint path must be under a checkpoints directory: {path}"
            )
        if not path.is_file():
            raise ModelWorkflowInvariantError(
                f"canonical {role} checkpoint path is missing: {path}"
            )
        sources.append(
            _PublishSource(
                label=f"publish {role} checkpoint",
                kind="model_checkpoint",
                path=path,
                sha256=None,
                execution_id=bundle.metadata_artifacts.execution_id,
            )
        )
    return tuple(sources)


def _calibration_sources(bundle: ModelPublishSourceBundle) -> tuple[_PublishSource, ...]:
    if bundle.metadata_artifacts is None:
        return ()
    metadata_path = bundle.metadata_artifacts.run_metadata_path
    if not metadata_path.is_file():
        return ()
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelWorkflowInvariantError(
            f"model run metadata is not valid JSON: {metadata_path}"
        ) from exc
    if not isinstance(raw, dict):
        return ()
    calibration = raw.get("calibration")
    if not isinstance(calibration, dict):
        return ()
    refs: list[tuple[str, str, str]] = []
    for key, kind in (
        ("compute_calibration_path", "compute_calibration"),
        ("compute_calibration_report_path", "compute_calibration_report"),
        ("selected_overrides_path", "selected_overrides")
    ):
        value = calibration.get(key)
        if isinstance(value, str) and value.strip():
            refs.append((key, kind, value))
    for item in calibration.get("representative_surfaces", []):
        if not isinstance(item, dict):
            continue
        for key in ("surface_metadata_path", "source_manifest_path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                refs.append((key, "representative_calibration_surface", value))
    sources: list[_PublishSource] = []
    seen: set[Path] = set()
    already_published_report_paths = (
        {
            receipt.path.resolve(strict=False)
            for receipt in bundle.report_artifacts.artifacts
        }
        if bundle.report_artifacts is not None
        else set()
    )

    for label, kind, value in refs:
        path = Path(value)
        resolved = path.resolve(strict=False)

        if resolved in already_published_report_paths:
            continue

        if resolved in seen or not path.is_file():
            continue

        seen.add(resolved)
        sources.append(
            _PublishSource(
                label=f"publish model calibration {label}",
                kind=kind,
                path=path,
                sha256=sha256_file(path),
                execution_id=bundle.metadata_artifacts.run_metadata.execution_id,
            )
        )

    return tuple(sources)


def _metadata_checkpoint_value(metadata: dict[str, Any], role: str) -> object:
    direct_key = f"{role}_checkpoint_path"
    if direct_key in metadata:
        return metadata[direct_key]
    checkpoints = metadata.get("checkpoints")
    if isinstance(checkpoints, dict):
        role_entry = checkpoints.get(role)
        if isinstance(role_entry, dict) and "path" in role_entry:
            return role_entry["path"]
    return None


def _metadata_checkpoint_path(
    value: object,
    *,
    metadata_path: Path,
    role: str,
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInvariantError(
            f"model run metadata {role} checkpoint path must be a non-empty string or null"
        )
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return path


def _runtime_support_sources(bundle: ModelPublishSourceBundle) -> tuple[_PublishSource, ...]:
    if bundle.metadata_artifacts is None:
        return ()
    manifest_path = bundle.metadata_artifacts.runtime_support_manifest_path
    if not manifest_path.is_file():
        return ()
    manifest = read_runtime_support_manifest(manifest_path)
    paths = _runtime_support_source_path_sequence(bundle)
    return tuple(
        _PublishSource(
            label=f"publish runtime support {artifact.role}",
            kind=artifact.kind,
            path=path,
            sha256=artifact.sha256,
            execution_id=bundle.metadata_artifacts.execution_id,
        )
        for artifact, path in zip(manifest.artifacts, paths, strict=True)
    )


def _runtime_support_source_paths(bundle: ModelPublishSourceBundle) -> set[Path]:
    return set(_runtime_support_source_path_sequence(bundle))


def _runtime_support_source_path_sequence(bundle: ModelPublishSourceBundle) -> tuple[Path, ...]:
    if bundle.metadata_artifacts is None:
        return ()
    manifest_path = bundle.metadata_artifacts.runtime_support_manifest_path
    if not manifest_path.is_file():
        return ()
    manifest = read_runtime_support_manifest(manifest_path)
    run_root = manifest_path.parent
    return tuple(
        (run_root / artifact.relative_path).resolve(strict=False)
        for artifact in manifest.artifacts
    )


def _generated_pose_archive_sources(layout: ModelLayout) -> tuple[_PublishSource, ...]:
    sources: list[_PublishSource] = []
    for split_output in layout.outputs.generated_pose_split_outputs:
        split_root = split_output.manifest_path.parent
        for label, kind, path in (
            (
                f"publish generated pose manifest [{split_output.split.value}]",
                "generated_pose_manifest",
                split_output.manifest_path,
            ),
            (
                f"publish generated pose samples archive [{split_output.split.value}]",
                "generated_pose_samples_archive",
                split_root / GENERATED_POSE_SAMPLE_ARCHIVE_NAME,
            ),
            (
                f"publish generated pose samples archive manifest [{split_output.split.value}]",
                "generated_pose_samples_archive_manifest",
                split_root / GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME,
            ),
            (
                f"publish generated pose samples archive sha256 [{split_output.split.value}]",
                "generated_pose_samples_archive_sha256",
                split_root / GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME,
            ),
        ):
            if path.exists():
                sources.append(
                    _PublishSource(
                        label=label,
                        kind=kind,
                        path=path,
                        sha256=None,
                        execution_id=None,
                    )
                )
    return tuple(sources)


def _generated_pose_archive_roots(sources: tuple[_PublishSource, ...]) -> set[Path]:
    return {
        source.path.resolve(strict=False).parent
        for source in sources
        if source.path.name == GENERATED_POSE_SAMPLE_ARCHIVE_NAME
    }


def _is_individual_generated_pose_sample_source(
    source: _PublishSource,
    *,
    archive_roots: set[Path],
) -> bool:
    if source.kind != "generated_pose_sample":
        return False
    path = source.path.resolve(strict=False)
    if path.suffix != ".npz":
        return False
    parts = path.as_posix()
    if "/generated_pose/" in parts and "/samples/" in parts:
        return True
    try:
        return path.parent.parent in archive_roots
    except IndexError:
        return False


def _copy_operation(target: ModelPublishTarget) -> FileCopyOperation:
    expected_bytes = _maybe_input_bytes(target.source_path)
    return FileCopyOperation(
        label=target.label,
        source_path=target.source_path,
        target_path=target.target_path,
        failure_message=f"Failed to {target.label}",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_bytes,
        progress=OperationProgressSpec(
            stage=model_progress_stage(
                stage_id=MODEL_STAGE_PUBLISH_EXECUTE,
                label="Publish model workflow artifacts",
                unit="bytes",
                owner_module=__name__,
                operation_kind="publish",
                total_semantics="publish input bytes when known",
                bar_eligible=False,
            ),
            expected_total=expected_bytes,
            live_owner="shell",
        ),
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


__all__ = ["build_model_publish_plan"]
