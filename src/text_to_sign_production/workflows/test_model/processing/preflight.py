"""Read-only preflight validation for test_model notebook execution."""

from __future__ import annotations

import json
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
from text_to_sign_production.modeling.candidates.errors import ModelProviderLookupError
from text_to_sign_production.modeling.data import (
    ModelingDataError,
    parse_modeling_manifest_family,
    read_modeling_manifest,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.workflows.test_model.contracts import (
    CheckpointPolicy,
    TestModelExpectedInput,
    TestModelExpectedOutput,
    TestModelPreflightCheck,
    TestModelPreflightResult,
    TestModelRequest,
    TestModelWorkflowConfig,
)
from text_to_sign_production.workflows.test_model.layout import (
    build_test_model_layout,
    drive_test_model_sample_run_root,
    runtime_test_model_sample_run_root,
)
from text_to_sign_production.workflows.test_model.runtime import (
    build_test_model_restore_plan,
    validate_test_model_restore_plan,
)

_MODEL_RUN_PLACEHOLDERS = frozenset(
    {
        "paste-generated-model-run-here",
        "paste-model-run-here",
        "replace-with-model-run",
    }
)
_TARGET_PLACEHOLDERS = frozenset(
    {
        "paste-test-source-sentence-name-here",
        "paste-target-sentence-name-here",
        "replace-with-target-sentence",
    }
)


def run_test_model_preflight(
    *,
    config: TestModelWorkflowConfig,
    request: TestModelRequest,
    execution_id: str,
) -> TestModelPreflightResult:
    """Validate request and Drive-side readiness without restore or inference."""

    layout = build_test_model_layout(config)
    checks: list[TestModelPreflightCheck] = []
    checks.extend(_request_checks(config, request))
    restore_plan = build_test_model_restore_plan(config, request, layout=layout)
    validation = validate_test_model_restore_plan(layout, restore_plan)
    metadata = _read_metadata(restore_plan.model_run_metadata_source, checks)
    resolved_model_key = _model_key(metadata, checks)
    resolved_family = _manifest_family(metadata, checks)
    semantic_summary = _semantic_summary(metadata)
    checks.extend(_restore_plan_checks(restore_plan, validation))
    checks.extend(_metadata_checks(request, metadata, restore_plan.model_run_metadata_source))
    checks.extend(_provider_checks(resolved_model_key))
    checks.extend(_target_checks(layout, request, resolved_family))
    resolved_sample_id = _resolved_sample_id(layout, request, resolved_family)
    return TestModelPreflightResult(
        model_run_name=request.model_run_name,
        checkpoint_policy=request.checkpoint_policy,
        target_sentence_name=request.target_sentence_name,
        ready=not any(check.status == "fail" for check in checks),
        resolved_model_key=resolved_model_key,
        resolved_manifest_family=resolved_family,
        resolved_sample_id=resolved_sample_id,
        auxiliary_objectives=semantic_summary["auxiliary_objectives"],
        semantic_objective_attached=semantic_summary["semantic_objective_attached"],
        semantic_ready_for_comparison=semantic_summary["semantic_ready_for_comparison"],
        semantic_required_baseline_missing=semantic_summary[
            "semantic_required_baseline_missing"
        ],
        semantic_ablation_status=semantic_summary["semantic_ablation_status"],
        checks=tuple(checks),
        expected_inputs=_expected_inputs(restore_plan, resolved_model_key),
        expected_outputs=_expected_outputs(layout, request, execution_id),
    )


def _request_checks(
    config: TestModelWorkflowConfig,
    request: TestModelRequest,
) -> tuple[TestModelPreflightCheck, ...]:
    checks = [
        _path_check("project_root", "request", config.project_root, must_exist=True),
        _path_check(
            "drive_project_root",
            "request",
            config.drive_project_root,
            must_exist=False,
            parent_must_exist=True,
        ),
    ]
    if request.model_run_name.strip().lower() in _MODEL_RUN_PLACEHOLDERS:
        checks.append(
            _fail(
                "model_run_name",
                "request",
                "MODEL_RUN_NAME is still a placeholder; paste a published model run name from model.ipynb.",
            )
        )
    else:
        checks.append(_pass("model_run_name", "request", "MODEL_RUN_NAME is populated"))
    if request.target_sentence_name.strip().lower() in _TARGET_PLACEHOLDERS:
        checks.append(
            _fail(
                "target_sentence_name",
                "request",
                (
                    "TARGET_SENTENCE_NAME is still a placeholder; choose an exact "
                    "source_sentence_name from the restored test manifest."
                ),
            )
        )
    else:
        checks.append(
            _pass("target_sentence_name", "request", "TARGET_SENTENCE_NAME is populated")
        )
    try:
        CheckpointPolicy(request.checkpoint_policy)
        checks.append(
            _pass(
                "checkpoint_policy",
                "request",
                f"checkpoint policy is valid: {request.checkpoint_policy.value}",
            )
        )
    except (TypeError, ValueError) as exc:
        checks.append(
            _fail(
                "checkpoint_policy",
                "request",
                f"CHECKPOINT_POLICY must be 'best' or 'last': {exc}",
            )
        )
    return tuple(checks)


def _semantic_summary(metadata: dict[str, object]) -> dict[str, object]:
    objectives = _auxiliary_objectives(metadata.get("auxiliary_objectives"))
    semantic_attached = ObjectiveKey.SEMANTIC_CONSISTENCY in objectives
    semantic_metadata = metadata.get("semantic_objective")
    ready_for_comparison: bool | None = None
    required_baseline_missing: bool | None = None
    status = "not_attached"
    if semantic_attached:
        status = "attached_without_readiness_metadata"
        if isinstance(semantic_metadata, dict):
            status_value = semantic_metadata.get("ablation_status")
            ready_value = semantic_metadata.get("ready_for_comparison")
            baseline_value = semantic_metadata.get("required_baseline_missing")
            ready_for_comparison = (
                ready_value if isinstance(ready_value, bool) else None
            )
            required_baseline_missing = (
                baseline_value if isinstance(baseline_value, bool) else None
            )
            if isinstance(status_value, str) and status_value.strip():
                status = status_value
            elif ready_for_comparison is True:
                status = "ready_for_comparison"
            elif required_baseline_missing is True:
                status = "required_baseline_missing"
            else:
                status = "not_ready_for_comparison"
    return {
        "auxiliary_objectives": objectives,
        "semantic_objective_attached": semantic_attached,
        "semantic_ready_for_comparison": ready_for_comparison,
        "semantic_required_baseline_missing": required_baseline_missing,
        "semantic_ablation_status": status,
    }


def _auxiliary_objectives(value: object) -> tuple[ObjectiveKey, ...]:
    if not isinstance(value, list | tuple):
        return ()
    objectives: list[ObjectiveKey] = []
    for item in value:
        try:
            objective = ObjectiveKey(item)
        except (TypeError, ValueError):
            continue
        if objective not in objectives:
            objectives.append(objective)
    return tuple(objectives)


def _restore_plan_checks(restore_plan, validation) -> tuple[TestModelPreflightCheck, ...]:
    checks: list[TestModelPreflightCheck] = []
    if restore_plan.model_run_metadata_source is None:
        checks.append(
            _fail(
                "model_run_metadata",
                "model_run",
                (
                    "published model run metadata was not found under Drive models. "
                    "Fix MODEL_RUN_NAME or publish the model run first."
                ),
            )
        )
    else:
        checks.append(
            _pass(
                "model_run_metadata",
                "model_run",
                "published model run metadata path resolved",
                path=restore_plan.model_run_metadata_source,
            )
        )
    support_manifest_ops = [
        operation
        for operation in restore_plan.operations
        if operation.group == "runtime_support_manifest"
    ]
    required_support_ops = [
        operation
        for operation in restore_plan.operations
        if _is_provider_input(operation.group)
    ]
    missing_support_count = sum(
        1
        for operation in required_support_ops
        if operation.required and not operation.source.exists()
    )
    checks.append(
        (
            _pass
            if support_manifest_ops and support_manifest_ops[0].source.is_file()
            else _fail
        )(
            "runtime_support_manifest",
            "restore_plan",
            (
                "runtime support manifest found"
                if support_manifest_ops and support_manifest_ops[0].source.is_file()
                else "runtime support manifest missing"
            ),
            path=(support_manifest_ops[0].source if support_manifest_ops else None),
            details={
                "required_support_artifact_count": len(required_support_ops),
                "missing_support_artifact_count": missing_support_count,
            },
        )
    )
    for issue in validation.blocking_errors:
        checks.append(
            _fail(
                issue.code,
                "restore_plan",
                issue.message,
                path=issue.path,
            )
        )
    for issue in validation.warnings:
        checks.append(_warn(issue.code, "restore_plan", issue.message, path=issue.path))
    if validation.valid:
        checks.append(
            _pass(
                "restore_plan_sources",
                "restore_plan",
                "all required Drive restore sources exist and target runtime paths are aligned",
                details={"operation_count": len(restore_plan.operations)},
            )
        )
    return tuple(checks)


def _metadata_checks(
    request: TestModelRequest,
    metadata: dict[str, object],
    path: Path | None,
) -> tuple[TestModelPreflightCheck, ...]:
    checks: list[TestModelPreflightCheck] = []
    if not metadata:
        return tuple(checks)
    run_name = metadata.get("model_run_name") or metadata.get("run_name")
    if isinstance(run_name, str) and run_name != request.model_run_name:
        checks.append(
            _fail(
                "metadata_run_name",
                "model_run",
                f"metadata run_name {run_name!r} does not match MODEL_RUN_NAME {request.model_run_name!r}.",
                path=path,
            )
        )
    status = metadata.get("status")
    if status != "completed":
        checks.append(
            _fail(
                "metadata_status",
                "model_run",
                f"model run metadata status must be 'completed'; observed {status!r}.",
                path=path,
            )
        )
    for field_name in ("model_key", "manifest_family", "train_split", "validation_split", "test_split"):
        if not isinstance(metadata.get(field_name), str) or not str(metadata.get(field_name)).strip():
            checks.append(
                _fail(
                    f"metadata_{field_name}",
                    "model_run",
                    f"model run metadata must include non-empty {field_name}.",
                    path=path,
                )
            )
    role = request.checkpoint_policy.value
    checkpoint_key = f"{role}_checkpoint_path"
    if not isinstance(metadata.get(checkpoint_key), str) or not str(metadata.get(checkpoint_key)).strip():
        checks.append(
            _fail(
                "metadata_checkpoint_reference",
                "checkpoint",
                f"checkpoint policy {role!r} requires {checkpoint_key} in model run metadata.",
                path=path,
            )
        )
    return tuple(checks)


def _provider_checks(model_key: ModelKey | None) -> tuple[TestModelPreflightCheck, ...]:
    if model_key is None:
        return ()
    try:
        ensure_model_provider_registered(model_key)
    except (ModelProviderLookupError, TypeError, ValueError) as exc:
        return (
            _fail(
                "provider_readiness",
                "provider",
                f"provider for model_key {model_key.value!r} is not available: {exc}",
            ),
        )
    checks: list[TestModelPreflightCheck] = [
        _pass(
            "provider_readiness",
            "provider",
            f"provider bootstrap is available for {model_key.value}",
        )
    ]
    if model_key is ModelKey.BASE_DIRECT:
        checks.append(
            _warn(
                "provider_inference_config_paths",
                "provider",
                (
                    "base_direct single-sample inference uses inference-only config loading; "
                    "training manifest path existence is not required."
                ),
            )
        )
    return tuple(checks)


def _target_checks(layout, request: TestModelRequest, manifest_family) -> tuple[TestModelPreflightCheck, ...]:
    if manifest_family is None:
        return (
            _fail(
                "target_manifest_family",
                "target",
                "target sentence cannot be resolved because model run manifest_family is missing or invalid.",
            ),
        )
    try:
        manifest = read_modeling_manifest(layout.stores.drive, manifest_family, SampleSplit.TEST)
    except (FileNotFoundError, OSError, ModelingDataError, ValueError, TypeError) as exc:
        return (
            _fail(
                "test_manifest",
                "target",
                f"Drive test split manifest is missing or invalid: {exc}",
            ),
        )
    matches = tuple(
        entry
        for entry in manifest.entries
        if entry.source_sentence_name == request.target_sentence_name
    )
    if not matches:
        candidates = _target_candidate_suggestions(manifest.entries)
        return (
            _fail(
                "target_sentence",
                "target",
                (
                    "TARGET_SENTENCE_NAME was not found as an exact source_sentence_name "
                    "in the model run test split manifest."
                ),
                path=manifest.manifest_path,
                details={"candidate_suggestions": candidates},
            ),
        )
    if len(matches) > 1:
        return (
            _fail(
                "target_sentence",
                "target",
                (
                    "TARGET_SENTENCE_NAME matched multiple source_sentence_name rows in "
                    "the test split manifest; choose an unambiguous sentence name."
                ),
                path=manifest.manifest_path,
                details={"match_count": len(matches)},
            ),
        )
    return (
        _pass(
            "target_sentence",
            "target",
            f"target sentence resolved to sample_id {matches[0].sample_id}",
            path=manifest.manifest_path,
            details={"sample_id": matches[0].sample_id},
        ),
    )


def _read_metadata(path: Path | None, checks: list[TestModelPreflightCheck]) -> dict[str, object]:
    if path is None:
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        checks.append(
            _fail(
                "metadata_json",
                "model_run",
                f"model run metadata JSON could not be read strictly: {exc}",
                path=path,
            )
        )
        return {}
    if not isinstance(loaded, dict):
        checks.append(
            _fail(
                "metadata_json",
                "model_run",
                "model run metadata JSON root must be an object.",
                path=path,
            )
        )
        return {}
    checks.append(_pass("metadata_json", "model_run", "model run metadata JSON is readable", path=path))
    return loaded


def _model_key(
    metadata: dict[str, object],
    checks: list[TestModelPreflightCheck],
) -> ModelKey | None:
    value = metadata.get("model_key")
    try:
        model_key = ModelKey(value)
    except (TypeError, ValueError):
        checks.append(
            _fail("metadata_model_key", "model_run", f"model_key is invalid: {value!r}")
        )
        return None
    return model_key


def _manifest_family(
    metadata: dict[str, object],
    checks: list[TestModelPreflightCheck],
):
    value = metadata.get("manifest_family")
    try:
        return parse_modeling_manifest_family(str(value))
    except (TypeError, ValueError, ModelingDataError) as exc:
        checks.append(
            _fail("metadata_manifest_family", "model_run", f"manifest_family is invalid: {exc}")
        )
        return None


def _resolved_sample_id(layout, request: TestModelRequest, manifest_family) -> str | None:
    if manifest_family is None:
        return None
    try:
        manifest = read_modeling_manifest(layout.stores.drive, manifest_family, SampleSplit.TEST)
    except (FileNotFoundError, OSError, ModelingDataError, ValueError, TypeError):
        return None
    matches = tuple(
        entry
        for entry in manifest.entries
        if entry.source_sentence_name == request.target_sentence_name
    )
    return matches[0].sample_id if len(matches) == 1 else None


def _target_candidate_suggestions(entries) -> tuple[dict[str, object], ...]:
    preferred = [entry for entry in entries if 60 <= entry.frame_count <= 140]
    selected = list(preferred[:10])
    if len(selected) < 10:
        for entry in entries:
            if entry in selected:
                continue
            selected.append(entry)
            if len(selected) == 10:
                break
    return tuple(
        {
            "source_sentence_name": entry.source_sentence_name,
            "sample_id": entry.sample_id,
            "frame_count": entry.frame_count,
        }
        for entry in selected
    )


def _expected_inputs(
    restore_plan,
    resolved_model_key: ModelKey | None,
) -> tuple[TestModelExpectedInput, ...]:
    provider_key = None if resolved_model_key is None else resolved_model_key.value
    return tuple(
        TestModelExpectedInput(
            label=operation.description,
            source_path=operation.source,
            target_path=operation.target,
            artifact_family=operation.group,
            required_for_restore=operation.required,
            provider_key=provider_key if _is_provider_input(operation.group) else None,
        )
        for operation in restore_plan.operations
    )


def _is_provider_input(group: str) -> bool:
    return group not in {
        "model_run_metadata",
        "effective_config",
        "research_spec",
        "runtime_support_manifest",
        "best_checkpoint",
        "last_checkpoint",
        "test_manifest",
        "test_passed_samples",
        "test_translations",
        "test_raw_sources",
    }


def _expected_outputs(
    layout,
    request: TestModelRequest,
    execution_id: str,
) -> tuple[TestModelExpectedOutput, ...]:
    runtime_root = runtime_test_model_sample_run_root(
        layout,
        request.model_run_name,
        request.target_sentence_name,
        execution_id,
    )
    drive_root = drive_test_model_sample_run_root(
        layout,
        request.model_run_name,
        request.target_sentence_name,
        execution_id,
    )
    return (
        TestModelExpectedOutput("execution output root", runtime_root, "runtime_root", True),
        TestModelExpectedOutput(
            "generated pose manifest",
            runtime_root / "generated_pose" / "manifest.jsonl",
            "generated_pose_manifest",
            True,
        ),
        TestModelExpectedOutput(
            "generated pose sample directory",
            runtime_root / "generated_pose" / "samples",
            "generated_pose_sample",
            True,
        ),
        TestModelExpectedOutput(
            "skeleton video",
            runtime_root / "videos" / "generated_pose_skeleton.mp4",
            "visualization",
            True,
        ),
        TestModelExpectedOutput(
            "source_vs_generated video",
            runtime_root / "videos" / "source_vs_generated_pose.mp4",
            "visualization",
            False,
        ),
        TestModelExpectedOutput(
            "reference_vs_generated video",
            runtime_root / "videos" / "reference_vs_generated_pose.mp4",
            "visualization",
            True,
        ),
        TestModelExpectedOutput(
            "reference comparison JSON",
            runtime_root / "reference_comparison.json",
            "report",
            True,
        ),
        TestModelExpectedOutput(
            "reference comparison summary",
            runtime_root / "reference_comparison_summary.md",
            "report",
            True,
        ),
        TestModelExpectedOutput(
            "test_model summary",
            runtime_root / "test_model_summary.md",
            "report",
            True,
        ),
        TestModelExpectedOutput(
            "test_model result",
            runtime_root / "test_model_result.json",
            "report",
            True,
        ),
        TestModelExpectedOutput(
            "visual artifacts",
            runtime_root / "visual_artifacts.json",
            "report",
            True,
        ),
        TestModelExpectedOutput(
            "report index",
            runtime_root / "index.json",
            "report",
            True,
        ),
        TestModelExpectedOutput("report root", runtime_root, "report", True),
        TestModelExpectedOutput("publish target root", drive_root, "publish", True),
    )


def _path_check(name: str, scope: str, path: Path, *, must_exist: bool, parent_must_exist: bool = False) -> TestModelPreflightCheck:
    if not path.is_absolute():
        return _fail(name, scope, f"{name} must be an absolute path: {path}", path=path)
    if must_exist and not path.exists():
        return _fail(name, scope, f"{name} does not exist: {path}", path=path)
    if parent_must_exist and not path.parent.exists():
        return _fail(name, scope, f"{name} parent does not exist: {path.parent}", path=path)
    return _pass(name, scope, f"{name} path is usable", path=path)


def _pass(name: str, scope: str, message: str, *, path: Path | None = None, details: dict[str, object] | None = None) -> TestModelPreflightCheck:
    return TestModelPreflightCheck(name, scope, "pass", message, path, details or {})


def _warn(name: str, scope: str, message: str, *, path: Path | None = None, details: dict[str, object] | None = None) -> TestModelPreflightCheck:
    return TestModelPreflightCheck(name, scope, "warn", message, path, details or {})


def _fail(name: str, scope: str, message: str, *, path: Path | None = None, details: dict[str, object] | None = None) -> TestModelPreflightCheck:
    return TestModelPreflightCheck(name, scope, "fail", message, path, details or {})


__all__ = ["run_test_model_preflight"]
