"""Read-only preflight validation for model notebook execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import (
    ModelCandidateError,
    ObjectiveAttachmentError,
    default_stage_plan_for_request,
    validate_objective_attachments,
)
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.errors import ModelProviderLookupError
from text_to_sign_production.modeling.data import (
    ModelingDataError,
    read_modeling_manifest,
    resolve_modeling_manifest_path,
)
from text_to_sign_production.modeling.objectives.semantic_consistency import (
    load_semantic_consistency_config,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveExtractOperation,
    FileCopyOperation,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelExpectedInput,
    ModelExpectedOutput,
    ModelPreflightCheck,
    ModelPreflightResult,
    ModelRuntimePlan,
    ModelWorkflowConfig,
)
from text_to_sign_production.workflows.model.layout import ModelLayout, required_model_splits
from text_to_sign_production.workflows.model.runtime import validate_model_runtime_plan


def run_model_preflight(
    *,
    config: ModelWorkflowConfig,
    layout: ModelLayout,
    runtime_plan: ModelRuntimePlan,
) -> ModelPreflightResult:
    """Validate notebook readiness before runtime restore, training, or publish."""

    checks: list[ModelPreflightCheck] = []
    checks.extend(_config_checks(config, layout))
    checks.extend(_runtime_plan_checks(config, layout, runtime_plan))
    checks.extend(_drive_input_checks(layout))
    checks.extend(_test_metadata_checks(config, layout))
    checks.extend(_provider_checks(config, layout, runtime_plan))
    checks.extend(_objective_checks(config, layout, runtime_plan))
    checks.extend(_publish_checks(config, layout))
    return ModelPreflightResult(
        model_key=config.model_key,
        run_name=config.run_name,
        manifest_family=config.manifest_family,
        run_mode=config.run_mode,
        auxiliary_objectives=config.auxiliary_objectives,
        ready=not any(check.status == "fail" for check in checks),
        checks=tuple(checks),
        expected_inputs=_expected_inputs(runtime_plan),
        expected_outputs=_expected_outputs(config, layout),
    )


def _config_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> tuple[ModelPreflightCheck, ...]:
    checks: list[ModelPreflightCheck] = []
    checks.append(_path_check("project_root", "config", config.project_root, must_exist=True))
    checks.append(
        _path_check(
            "drive_project_root",
            "config",
            config.drive_project_root,
            must_exist=False,
            parent_must_exist=True,
        )
    )
    checks.append(
        _pass(
            "model_key",
            "config",
            f"model key is recognized: {config.model_key.value}",
        )
    )
    checks.append(
        _pass(
            "manifest_family",
            "config",
            f"manifest family parsed: {config.manifest_family.family_id}",
        )
    )
    checks.append(_pass("run_mode", "config", f"run mode parsed: {config.run_mode.value}"))
    if config.model_config_relpath is None:
        checks.append(_pass("model_config", "config", "no model config source requested"))
    else:
        path = layout.runtime.model_config_original_path
        checks.append(
            _file_check(
                "model_config",
                "config",
                path,
                "model config file exists",
                (
                    "model config file is missing. Fix MODEL_CONFIG_RELATIVE_PATH or add "
                    "the expected config under PROJECT_ROOT."
                ),
            )
        )
    if ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives:
        checks.append(
            _file_check(
                "semantic_objective_config",
                "config",
                layout.runtime.semantic_objective_config_original_path,
                "semantic objective config source exists",
                (
                    "semantic_consistency was requested but "
                    "configs/modeling/objectives/semantic_consistency.yaml is missing."
                ),
            )
        )
    return tuple(checks)


def _runtime_plan_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
    runtime_plan: ModelRuntimePlan,
) -> tuple[ModelPreflightCheck, ...]:
    checks: list[ModelPreflightCheck] = []
    try:
        validate_model_runtime_plan(config, layout, runtime_plan)
    except Exception as exc:
        checks.append(
            _fail(
                "runtime_plan_alignment",
                "runtime_plan",
                f"runtime plan is not aligned with workflow config/layout: {exc}",
            )
        )
        return tuple(checks)
    checks.append(
        _pass(
            "runtime_plan_alignment",
            "runtime_plan",
            "runtime plan matches workflow config and layout",
            details={
                "operation_count": len(runtime_plan.restore_operations),
                "required_splits": [split.value for split in required_model_splits(config)],
            },
        )
    )
    file_copies = sum(isinstance(operation, FileCopyOperation) for operation in runtime_plan.restore_operations)
    archives = sum(isinstance(operation, ArchiveExtractOperation) for operation in runtime_plan.restore_operations)
    checks.append(
        _pass(
            "runtime_plan_operation_types",
            "runtime_plan",
            "runtime restore operation types match expected config/split semantics",
            details={"file_copies": file_copies, "archive_extracts": archives},
        )
    )
    return tuple(checks)


def _drive_input_checks(layout: ModelLayout) -> tuple[ModelPreflightCheck, ...]:
    checks: list[ModelPreflightCheck] = []
    drive_root = layout.stores.drive.repo_root.resolve(strict=False)
    runtime_root = layout.stores.runtime.repo_root.resolve(strict=False)
    for split_layout in layout.runtime.splits:
        split = split_layout.split.value
        checks.append(
            _file_check(
                f"manifest_source_{split}",
                "drive_input",
                split_layout.drive_manifest_path,
                f"Drive modeling manifest exists for {split}",
                f"Drive modeling manifest is missing for split {split}: {split_layout.drive_manifest_path}",
            )
        )
        try:
            read_modeling_manifest(
                layout.stores.drive,
                layout.config.manifest_family,
                split_layout.split,
            )
            checks.append(
                _pass(
                    f"manifest_schema_{split}",
                    "drive_input",
                    f"Drive modeling manifest schema is readable for {split}",
                    path=split_layout.drive_manifest_path,
                )
            )
        except (FileNotFoundError, OSError, ModelingDataError, ValueError, TypeError) as exc:
            checks.append(
                _fail(
                    f"manifest_schema_{split}",
                    "drive_input",
                    f"Drive modeling manifest is not readable for {split}: {exc}",
                    path=split_layout.drive_manifest_path,
                )
            )
        checks.append(
            _file_check(
                f"passed_archive_source_{split}",
                "drive_input",
                split_layout.drive_passed_archive_path,
                f"Drive passed-sample archive exists for {split}",
                (
                    f"Drive passed-sample archive is missing for split {split}: "
                    f"{split_layout.drive_passed_archive_path}"
                ),
            )
        )
        for label, source, target in (
            ("manifest", split_layout.drive_manifest_path, split_layout.runtime_manifest_path),
            (
                "passed_archive",
                split_layout.drive_passed_archive_path,
                split_layout.runtime_passed_samples_split_root,
            ),
        ):
            checks.extend(_source_target_checks(label, source, target, drive_root, runtime_root))
    return tuple(checks)


def _test_metadata_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> tuple[ModelPreflightCheck, ...]:
    manifest_path = resolve_modeling_manifest_path(
        layout.stores.drive,
        config.manifest_family,
        SampleSplit.TEST,
    )
    checks = [
        _file_check(
            "test_manifest_source_exists",
            "test_metadata",
            manifest_path,
            "Drive modeling manifest exists for downstream test split compatibility",
            (
                "Drive modeling manifest is missing for downstream test split "
                f"compatibility: {manifest_path}"
            ),
        )
    ]
    try:
        read_modeling_manifest(
            layout.stores.drive,
            config.manifest_family,
            SampleSplit.TEST,
        )
        checks.append(
            _pass(
                "test_manifest_schema_readable",
                "test_metadata",
                "Drive test split modeling manifest schema is readable",
                path=manifest_path,
            )
        )
    except (FileNotFoundError, OSError, ModelingDataError, ValueError, TypeError) as exc:
        checks.append(
            _fail(
                "test_manifest_schema_readable",
                "test_metadata",
                f"Drive test split modeling manifest is not readable: {exc}",
                path=manifest_path,
            )
        )
    return tuple(checks)


def _provider_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
    runtime_plan: ModelRuntimePlan,
) -> tuple[ModelPreflightCheck, ...]:
    del runtime_plan
    checks: list[ModelPreflightCheck] = []
    try:
        ensure_model_provider_registered(config.model_key)
        default_stage_plan_for_request(config.to_model_run_request(config_path=None))
    except (ModelProviderLookupError, ModelCandidateError, ObjectiveAttachmentError, ValueError, TypeError) as exc:
        return (
            _fail(
                "provider_readiness",
                "provider",
                (
                    f"provider/stage plan is not ready for {config.model_key.value}: {exc}. "
                    "Select a registered model provider or fix objective compatibility."
                ),
            ),
        )
    checks.append(
        _pass(
            "provider_readiness",
            "provider",
            f"provider bootstrap and lightweight stage plan are available for {config.model_key.value}",
        )
    )
    checks.extend(_base_direct_dependency_checks(config, layout))
    return tuple(checks)


def _base_direct_dependency_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> tuple[ModelPreflightCheck, ...]:
    if (
        config.model_key is not ModelKey.BASE_DIRECT
        or config.run_mode.value != "full"
        or config.compute_profile != "colab_a100_80gb"
    ):
        return ()
    from text_to_sign_production.modeling.candidates.base_direct.config import (
        load_base_direct_config,
    )

    config_path = layout.runtime.model_config_original_path
    if config_path is None or not config_path.is_file():
        return (
            _fail(
                "base_direct_text_encoder_config",
                "provider",
                "base_direct full A100 preflight cannot inspect text_encoder config.",
                path=config_path,
            ),
        )
    try:
        loaded = load_base_direct_config(
            config_path,
            request=config.to_model_run_request(config_path=config_path),
        )
    except Exception as exc:
        return (
            _fail(
                "base_direct_text_encoder_config",
                "provider",
                f"base_direct text_encoder config could not be loaded: {exc}",
                path=config_path,
            ),
        )
    if not _base_direct_text_encoder_requires_transformers(
        loaded.text_encoder.model_name
    ):
        return ()
    if importlib.util.find_spec("transformers") is None:
        return (
            _fail(
                "base_direct_transformers_missing",
                "provider",
                (
                    "base_direct full A100 requires transformers before provider-real "
                    f"calibration can build text_encoder.model_name={loaded.text_encoder.model_name!r}."
                ),
            ),
        )
    checks = [
        _pass(
            "base_direct_transformers_available",
            "provider",
            "transformers is importable for base_direct text encoder preflight",
        )
    ]

    if not loaded.text_encoder.local_files_only:
        checks.append(
            _pass(
                "base_direct_text_encoder_remote_access_allowed",
                "provider",
                (
                    "base_direct text_encoder.local_files_only=false; provider-real "
                    "calibration may download the configured text encoder if it is not "
                    "already cached."
                ),
            )
        )
        return tuple(checks)

    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(
            loaded.text_encoder.model_name,
            revision=loaded.text_encoder.revision,
            local_files_only=True,
        )
    except Exception as exc:
        checks.append(
            _fail(
                "base_direct_calibration_text_encoder_cache_missing",
                "provider",
                (
                    "base_direct text_encoder.local_files_only=true, but the local "
                    f"transformers cache does not contain {loaded.text_encoder.model_name!r}: {exc}"
                ),
            )
        )
    else:
        checks.append(
            _pass(
                "base_direct_calibration_text_encoder_cache",
                "provider",
                "base_direct local transformers cache contains the configured text encoder",
            )
        )

    return tuple(checks)


def _base_direct_text_encoder_requires_transformers(model_name: str) -> bool:
    return bool(isinstance(model_name, str) and model_name.strip())


def _objective_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
    runtime_plan: ModelRuntimePlan,
) -> tuple[ModelPreflightCheck, ...]:
    checks: list[ModelPreflightCheck] = []
    try:
        validate_objective_attachments(
            require_model_spec(config.model_key),
            config.auxiliary_objectives,
        )
        checks.append(_pass("objective_compatibility", "objective", "auxiliary objectives are compatible"))
    except Exception as exc:
        checks.append(
            _fail(
                "objective_compatibility",
                "objective",
                f"auxiliary objective compatibility failed: {exc}",
            )
        )
    semantic_requested = ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives
    if not semantic_requested:
        return tuple(checks)
    snapshot_operation_present = any(
        isinstance(operation, FileCopyOperation)
        and operation.source_path == layout.runtime.semantic_objective_config_original_path
        and operation.target_path == layout.runtime.semantic_objective_config_path
        for operation in runtime_plan.restore_operations
    )
    if snapshot_operation_present:
        checks.append(
            _pass(
                "semantic_snapshot_operation",
                "objective",
                "runtime plan snapshots the semantic objective config before execution",
                path=layout.runtime.semantic_objective_config_path,
            )
        )
    else:
        checks.append(
            _fail(
                "semantic_snapshot_operation",
                "objective",
                "semantic_consistency requires a runtime plan operation that snapshots its config.",
                path=layout.runtime.semantic_objective_config_path,
            )
        )
    try:
        if layout.runtime.semantic_objective_config_original_path is None:
            raise FileNotFoundError("semantic objective config source path is not configured")
        load_semantic_consistency_config(layout.runtime.semantic_objective_config_original_path)
        checks.append(
            _pass(
                "semantic_config_parse",
                "objective",
                "semantic objective config source parses strictly",
                path=layout.runtime.semantic_objective_config_original_path,
            )
        )
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        checks.append(
            _fail(
                "semantic_config_parse",
                "objective",
                f"semantic objective config source is invalid: {exc}",
                path=layout.runtime.semantic_objective_config_original_path,
            )
        )
    checks.append(
        _warn(
            "semantic_ablation_baseline",
            "objective",
            (
                "semantic_consistency preflight cannot find a paired baseline here; "
                "ablation comparison will remain not ready until a compatible baseline "
                "run metadata artifact is provided."
            ),
            path=layout.outputs.run_metadata_path,
        )
    )
    return tuple(checks)


def _publish_checks(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> tuple[ModelPreflightCheck, ...]:
    checks: list[ModelPreflightCheck] = []
    drive_root = config.drive_project_root.resolve(strict=False)
    publish_root = layout.publish.model_run_root.resolve(strict=False)
    try:
        publish_root.relative_to(drive_root)
        checks.append(
            _pass(
                "publish_root_under_drive",
                "publish",
                "publish target root is under Drive project root",
                path=layout.publish.model_run_root,
            )
        )
    except ValueError:
        checks.append(
            _fail(
                "publish_root_under_drive",
                "publish",
                f"publish target root must be under Drive project root: {layout.publish.model_run_root}",
                path=layout.publish.model_run_root,
            )
        )
    if layout.publish.model_run_root.exists():
        checks.append(
            _warn(
                "publish_target_exists",
                "publish",
                (
                    "publish target root already exists. The publish step uses its own "
                    "overwrite policy; inspect this path before reusing the run name."
                ),
                path=layout.publish.model_run_root,
            )
        )
    return tuple(checks)


def _expected_outputs(
    config: ModelWorkflowConfig,
    layout: ModelLayout,
) -> tuple[ModelExpectedOutput, ...]:
    outputs = [
        ModelExpectedOutput("runtime model run root", layout.outputs.model_run_root, "runtime_root", True, False),
        ModelExpectedOutput("checkpoint root", layout.outputs.checkpoints_root, "checkpoint", True, False),
        ModelExpectedOutput("model metadata", layout.outputs.run_metadata_path, "metadata", True, False),
        ModelExpectedOutput("validation reports root", layout.reports.validation_root, "report", True, False),
        ModelExpectedOutput("provider reports root", layout.reports.root, "provider_report", True, False),
        ModelExpectedOutput("publish target root", layout.publish.model_run_root, "publish", True, False),
    ]
    for split_output in layout.outputs.generated_pose_split_outputs:
        outputs.append(
            ModelExpectedOutput(
                f"generated pose manifest [{split_output.split.value}]",
                split_output.manifest_path,
                "generated_pose_manifest",
                True,
                False,
            )
        )
    if ObjectiveKey.SEMANTIC_CONSISTENCY in config.auxiliary_objectives:
        outputs.append(
            ModelExpectedOutput(
                "semantic objective reports root",
                layout.reports.semantic_objective_root,
                "semantic_objective",
                True,
                False,
            )
        )
    return tuple(outputs)


def _expected_inputs(runtime_plan: ModelRuntimePlan) -> tuple[ModelExpectedInput, ...]:
    inputs: list[ModelExpectedInput] = []
    for operation in runtime_plan.restore_operations:
        if isinstance(operation, FileCopyOperation):
            inputs.append(
                ModelExpectedInput(
                    label=operation.label,
                    source_path=operation.source_path,
                    target_path=operation.target_path,
                    artifact_family=_artifact_family_from_label(operation.label),
                    required_for_restore=True,
                )
            )
        elif isinstance(operation, ArchiveExtractOperation):
            inputs.append(
                ModelExpectedInput(
                    label=operation.label,
                    source_path=operation.archive_path,
                    target_path=operation.extraction_root,
                    artifact_family=_artifact_family_from_label(operation.label),
                    required_for_restore=True,
                )
            )
    return tuple(inputs)


def _artifact_family_from_label(label: str) -> str:
    return label.removeprefix("snapshot ").removeprefix("restore ").replace(" ", "_")


def _path_check(
    name: str,
    scope: str,
    path: Path,
    *,
    must_exist: bool,
    parent_must_exist: bool = False,
) -> ModelPreflightCheck:
    if not path.is_absolute():
        return _fail(name, scope, f"{name} must be an absolute path: {path}", path=path)
    if must_exist and not path.exists():
        return _fail(name, scope, f"{name} does not exist: {path}", path=path)
    if parent_must_exist and not path.parent.exists():
        return _fail(name, scope, f"{name} parent does not exist: {path.parent}", path=path)
    return _pass(name, scope, f"{name} path is usable", path=path)


def _file_check(
    name: str,
    scope: str,
    path: Path | None,
    pass_message: str,
    fail_message: str,
) -> ModelPreflightCheck:
    if path is None:
        return _fail(name, scope, fail_message)
    if not path.is_file():
        return _fail(name, scope, fail_message, path=path)
    return _pass(name, scope, pass_message, path=path)


def _source_target_checks(
    label: str,
    source: Path,
    target: Path,
    drive_root: Path,
    runtime_root: Path,
) -> tuple[ModelPreflightCheck, ...]:
    checks: list[ModelPreflightCheck] = []
    resolved_source = source.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    try:
        resolved_source.relative_to(drive_root)
        checks.append(_pass(f"{label}_source_root", "drive_input", "source is under Drive root", path=source))
    except ValueError:
        checks.append(_fail(f"{label}_source_root", "drive_input", f"source is outside Drive root: {source}", path=source))
    try:
        resolved_target.relative_to(runtime_root)
        checks.append(_pass(f"{label}_target_root", "drive_input", "target is under runtime root", path=target))
    except ValueError:
        checks.append(_fail(f"{label}_target_root", "drive_input", f"target is outside runtime root: {target}", path=target))
    if resolved_source == resolved_target:
        checks.append(_fail(f"{label}_source_target_distinct", "drive_input", "restore source and target must differ", path=source))
    else:
        checks.append(_pass(f"{label}_source_target_distinct", "drive_input", "restore source and target differ"))
    return tuple(checks)


def _pass(
    name: str,
    scope: str,
    message: str,
    *,
    path: Path | None = None,
    details: dict[str, object] | None = None,
) -> ModelPreflightCheck:
    return ModelPreflightCheck(name, scope, "pass", message, path, details or {})


def _warn(
    name: str,
    scope: str,
    message: str,
    *,
    path: Path | None = None,
    details: dict[str, object] | None = None,
) -> ModelPreflightCheck:
    return ModelPreflightCheck(name, scope, "warn", message, path, details or {})


def _fail(
    name: str,
    scope: str,
    message: str,
    *,
    path: Path | None = None,
    details: dict[str, object] | None = None,
) -> ModelPreflightCheck:
    return ModelPreflightCheck(name, scope, "fail", message, path, details or {})


__all__ = ["run_model_preflight"]
