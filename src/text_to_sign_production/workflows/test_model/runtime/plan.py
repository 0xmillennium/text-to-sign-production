"""Build restore plans for the single-sample test-model workflow."""

from __future__ import annotations

import json
from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import read_runtime_support_manifest
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
    resolve_modeling_manifest_path,
)
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
    OperationProgressSpec,
)
from text_to_sign_production.workflows.test_model.constants import (
    TEST_MODEL_STAGE_RUNTIME_RESTORE,
)
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelIssue,
    TestModelRequest,
    TestModelRestoreOperation,
    TestModelRestorePlan,
    TestModelRestorePlanValidation,
    TestModelWorkflowConfig,
)
from text_to_sign_production.workflows.test_model.layout import TestModelLayout
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage


def build_test_model_restore_plan(
    config: TestModelWorkflowConfig,
    request: TestModelRequest,
    *,
    layout: TestModelLayout,
) -> TestModelRestorePlan:
    del config
    errors: list[TestModelIssue] = []
    warnings: list[TestModelIssue] = []
    metadata_sources = _find_model_run_metadata(layout, request.model_run_name)
    if not metadata_sources:
        errors.append(
            TestModelIssue(
                "model_run_metadata_missing",
                f"could not find Drive model run metadata for MODEL_RUN_NAME={request.model_run_name}",
            )
        )
        return TestModelRestorePlan(request.model_run_name, None, (), tuple(errors), ())
    if len(metadata_sources) > 1:
        errors.append(
            TestModelIssue(
                "model_run_metadata_ambiguous",
                "multiple Drive model run metadata files matched MODEL_RUN_NAME",
            )
        )
        return TestModelRestorePlan(request.model_run_name, None, (), tuple(errors), ())
    metadata_source = metadata_sources[0]
    metadata = _read_json(metadata_source, errors)
    if not metadata:
        return TestModelRestorePlan(request.model_run_name, metadata_source, (), tuple(errors), ())
    model_key = str(metadata.get("model_key", "")).strip()
    manifest_family_value = str(metadata.get("manifest_family", "")).strip()
    operations: list[TestModelRestoreOperation] = []
    if not model_key or not manifest_family_value:
        errors.append(
            TestModelIssue(
                "model_run_metadata_incomplete",
                "model run metadata must include model_key and manifest_family",
                path=metadata_source,
            )
        )
        return TestModelRestorePlan(request.model_run_name, metadata_source, (), tuple(errors), ())
    operations.extend(
        _model_run_operations(
            layout,
            model_key,
            request.model_run_name,
            selected_checkpoint_role=request.checkpoint_policy.value,
            errors=errors,
        )
    )
    try:
        manifest_family = parse_modeling_manifest_family(manifest_family_value)
        operations.extend(_test_sample_operations(layout, manifest_family))
    except (TypeError, ValueError) as exc:
        errors.append(
            TestModelIssue(
                "manifest_family_invalid",
                f"model run metadata manifest_family is invalid: {exc}",
                path=metadata_source,
            )
        )
    return TestModelRestorePlan(
        model_run_name=request.model_run_name,
        model_run_metadata_source=metadata_source,
        operations=tuple(operations),
        blocking_errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_test_model_restore_plan(
    layout: TestModelLayout,
    plan: TestModelRestorePlan,
) -> TestModelRestorePlanValidation:
    errors = list(plan.blocking_errors)
    warnings = list(plan.warnings)
    labels: set[str] = set()
    targets: set[Path] = set()
    drive_root = layout.config.drive_project_root.resolve(strict=False)
    runtime_root = layout.config.runtime_root.resolve(strict=False)
    if not plan.operations and not errors:
        errors.append(TestModelIssue("empty_restore_plan", "restore plan is empty"))
    for operation in plan.operations:
        label = operation.workflow_operation.label
        if label in labels:
            errors.append(TestModelIssue("duplicate_operation_label", label, label=label))
        labels.add(label)
        source = operation.source.resolve(strict=False)
        target = operation.target.resolve(strict=False)
        if target in targets:
            errors.append(
                TestModelIssue(
                    "duplicate_target_path",
                    f"multiple restore operations target the same path: {operation.target}",
                    path=operation.target,
                    label=label,
                )
            )
        targets.add(target)
        if source == target:
            errors.append(
                TestModelIssue(
                    "source_equals_target",
                    f"restore source and target are the same path: {operation.source}",
                    path=operation.source,
                    label=label,
                )
            )
        try:
            source.relative_to(drive_root)
        except ValueError:
            errors.append(
                TestModelIssue(
                    "restore_source_outside_drive_root",
                    f"restore source must be under Drive project root: {operation.source}",
                    path=operation.source,
                    label=label,
                )
            )
        try:
            target.relative_to(runtime_root)
        except ValueError:
            errors.append(
                TestModelIssue(
                    "restore_target_outside_runtime_root",
                    f"restore target must be under runtime root: {operation.target}",
                    path=operation.target,
                    label=label,
                )
            )
        if ".." in operation.target.parts:
            errors.append(
                TestModelIssue(
                    "restore_target_parent_traversal",
                    f"restore target must not contain parent traversal: {operation.target}",
                    path=operation.target,
                    label=label,
                )
            )
        if not operation.source.exists():
            issue = TestModelIssue(
                "required_source_missing" if operation.required else "optional_source_missing",
                f"{operation.description}: source does not exist: {operation.source}",
                path=operation.source,
                label=label,
            )
            if operation.required:
                errors.append(issue)
            else:
                warnings.append(issue)
        elif operation.expected_sha256 is not None and operation.source.is_file():
            observed_sha256 = sha256_file(operation.source)
            if observed_sha256 != operation.expected_sha256:
                errors.append(
                    TestModelIssue(
                        "restore_source_sha256_mismatch",
                        (
                            f"{operation.description}: source sha256 does not match "
                            "runtime support manifest"
                        ),
                        path=operation.source,
                        label=label,
                    )
                )
    return TestModelRestorePlanValidation(
        plan=plan,
        valid=not errors,
        blocking_errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _find_model_run_metadata(layout: TestModelLayout, run_name: str) -> tuple[Path, ...]:
    models_root = layout.stores.drive.models.root
    if not models_root.is_dir():
        return ()
    return tuple(sorted(models_root.glob(f"*/{run_name}/run_metadata.json")))


def _read_json(path: Path, errors: list[TestModelIssue]) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(
            TestModelIssue("model_run_metadata_unreadable", str(exc), path=path)
        )
        return {}
    if not isinstance(loaded, dict):
        errors.append(
            TestModelIssue("model_run_metadata_invalid", "metadata root must be an object", path=path)
        )
        return {}
    return loaded


def _model_run_operations(
    layout: TestModelLayout,
    model_key: str,
    run_name: str,
    *,
    selected_checkpoint_role: str,
    errors: list[TestModelIssue],
) -> tuple[TestModelRestoreOperation, ...]:
    drive = layout.stores.drive.models
    runtime = layout.stores.runtime.models
    pairs = [
        ("model_run_metadata", drive.model_run_metadata_file(model_key, run_name).path, runtime.model_run_metadata_file(model_key, run_name).path, True),
        ("effective_config", drive.model_effective_config_file(model_key, run_name).path, runtime.model_effective_config_file(model_key, run_name).path, True),
        ("research_spec", drive.model_research_spec_file(model_key, run_name).path, runtime.model_research_spec_file(model_key, run_name).path, True),
        ("runtime_support_manifest", drive.model_runtime_support_manifest_file(model_key, run_name).path, runtime.model_runtime_support_manifest_file(model_key, run_name).path, True),
        ("best_checkpoint", drive.model_checkpoint_file(model_key, run_name, "best.pt").path, runtime.model_checkpoint_file(model_key, run_name, "best.pt").path, selected_checkpoint_role == "best"),
        ("last_checkpoint", drive.model_checkpoint_file(model_key, run_name, "last.pt").path, runtime.model_checkpoint_file(model_key, run_name, "last.pt").path, selected_checkpoint_role == "last"),
    ]
    operations = [
        _copy_operation(group=group, source=source, target=target, required=required)
        for group, source, target, required in pairs
    ]
    support_manifest_source = drive.model_runtime_support_manifest_file(model_key, run_name).path
    if not support_manifest_source.is_file():
        errors.append(
            TestModelIssue(
                "runtime_support_manifest_missing",
                "runtime_support_manifest.json is required for test_model restore.",
                path=support_manifest_source,
            )
        )
        return tuple(operations)
    try:
        support_manifest = read_runtime_support_manifest(support_manifest_source)
    except Exception as exc:
        errors.append(
            TestModelIssue(
                "runtime_support_manifest_unreadable",
                f"runtime_support_manifest.json could not be read: {exc}",
                path=support_manifest_source,
            )
        )
        return tuple(operations)
    drive_root = drive.model_run_root(model_key, run_name).path
    runtime_root = runtime.model_run_root(model_key, run_name).path
    for artifact in support_manifest.artifacts:
        if not artifact.required_for_test_model:
            continue
        operations.append(
            _copy_operation(
                group=artifact.role,
                source=drive_root / artifact.relative_path,
                target=runtime_root / artifact.relative_path,
                required=True,
                expected_sha256=artifact.sha256,
            )
        )
    return tuple(operations)


def _test_sample_operations(
    layout: TestModelLayout,
    manifest_family: ModelingManifestFamily,
) -> tuple[TestModelRestoreOperation, ...]:
    split = SampleSplit.TEST
    drive = layout.stores.drive
    runtime = layout.stores.runtime
    return (
        _copy_operation(
            group="test_manifest",
            source=resolve_modeling_manifest_path(drive, manifest_family, split),
            target=resolve_modeling_manifest_path(runtime, manifest_family, split),
            required=True,
        ),
        _extract_operation(
            group="test_passed_samples",
            source=drive.samples.split_archive("passed", split).path,
            target=runtime.samples.split_extract_root("passed").path,
            required=True,
        ),
        _copy_operation(
            group="test_translations",
            source=drive.assets.translation_csv(split).path,
            target=runtime.assets.translation_csv(split).path,
            required=False,
        ),
        _extract_operation(
            group="test_raw_sources",
            source=drive.assets.keypoint_archive(split).path,
            target=runtime.assets.keypoint_extract_root().path,
            required=False,
        ),
    )


def _copy_operation(
    *,
    group: str,
    source: Path,
    target: Path,
    required: bool,
    expected_sha256: str | None = None,
) -> TestModelRestoreOperation:
    description = f"restore {group}"
    expected_bytes = _maybe_input_bytes(source)
    return TestModelRestoreOperation(
        group=group,
        source=source,
        target=target,
        required=required,
        description=description,
        expected_sha256=expected_sha256,
        workflow_operation=FileCopyOperation(
            label=description,
            source_path=source,
            target_path=target,
            failure_message=f"Failed to {description}",
            overwrite_policy="atomic_replace",
            expected_input_bytes=expected_bytes,
            progress=_restore_progress_spec(expected_bytes),
        ),
    )


def _extract_operation(
    *,
    group: str,
    source: Path,
    target: Path,
    required: bool,
) -> TestModelRestoreOperation:
    description = f"restore {group}"
    expected_bytes = _maybe_input_bytes(source)
    return TestModelRestoreOperation(
        group=group,
        source=source,
        target=target,
        required=required,
        description=description,
        workflow_operation=ArchiveExtractOperation(
            label=description,
            archive_path=source,
            extraction_root=target,
            failure_message=f"Failed to {description}",
            compression_kind="tar_zst",
            strip_components=None,
            members=(),
            expected_input_bytes=expected_bytes,
            progress=_restore_progress_spec(expected_bytes),
        ),
    )


def _restore_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=test_model_progress_stage(
            stage_id=TEST_MODEL_STAGE_RUNTIME_RESTORE,
            label="Restore test_model runtime artifacts",
            unit="bytes",
            owner_module=__name__,
            operation_kind="restore",
            total_semantics="runtime restore bytes when known",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


__all__ = ["build_test_model_restore_plan", "validate_test_model_restore_plan"]
