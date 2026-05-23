"""Structural runtime verification for model workflow inputs."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows.foundation.execution import ReadinessLevel
from text_to_sign_production.modeling.data import ModelingDataError, read_modeling_manifest
from text_to_sign_production.workflows.model.contracts import (
    ModelRuntimeAssetCheck,
    ModelRuntimePlan,
    ModelRuntimeVerification,
)


def verify_model_runtime(plan: ModelRuntimePlan) -> ModelRuntimeVerification:
    """Verify model runtime manifests and roots without loading pose payloads."""

    checks: list[ModelRuntimeAssetCheck] = []
    if plan.execution_inputs.model_config_path is not None:
        checks.append(_readable_file_check("model config", plan.execution_inputs.model_config_path))
    if plan.execution_inputs.semantic_objective_config_path is not None:
        checks.append(
            _readable_file_check(
                "semantic objective config snapshot",
                plan.execution_inputs.semantic_objective_config_path,
                missing_message=(
                    "semantic_consistency was requested; restore runtime to snapshot "
                    "configs/modeling/objectives/semantic_consistency.yaml before processing."
                ),
            )
        )
    for split_input in plan.execution_inputs.split_inputs:
        split = split_input.split.value
        checks.append(
            _manifest_check(
                f"modeling manifest [{split}]",
                plan=plan,
                split=split_input.split,
            )
        )
        checks.append(
            _directory_check(
                f"passed samples root [{split}]",
                split_input.passed_samples_split_root,
            )
        )
    checked_semantics = [
        "manifest_family_schema",
        "split_identity",
        "passed_samples_root_presence",
    ]
    if plan.execution_inputs.model_config_path is not None:
        checked_semantics.append("config_snapshot_presence")
    if plan.execution_inputs.semantic_objective_config_path is not None:
        checked_semantics.append("semantic_objective_config_snapshot_presence")
    return ModelRuntimeVerification(
        checks=tuple(checks),
        readiness_level=(
            ReadinessLevel.FAST_READINESS
            if all(check.exists and check.valid for check in checks)
            else ReadinessLevel.STRUCTURAL_READINESS
        ),
        checked_semantics=tuple(checked_semantics),
        limitations=(
            "modeling manifest rows are schema-read but sample payload arrays are not loaded",
            "passed sample directories are checked for presence, not exhaustive row coverage",
        ),
    )


def _readable_file_check(
    label: str,
    path: Path,
    *,
    missing_message: str | None = None,
) -> ModelRuntimeAssetCheck:
    if not path.exists():
        return ModelRuntimeAssetCheck(
            label=label,
            path=path,
            exists=False,
            message=missing_message,
        )
    if not path.is_file():
        return ModelRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message="path is not a file",
        )
    try:
        path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ModelRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=f"file is not readable as UTF-8 text: {exc}",
        )
    return ModelRuntimeAssetCheck(label=label, path=path, exists=True)


def _manifest_check(
    label: str,
    *,
    plan: ModelRuntimePlan,
    split,
) -> ModelRuntimeAssetCheck:
    path = next(
        item.manifest_path
        for item in plan.execution_inputs.split_inputs
        if item.split is split
    )
    if not path.exists():
        return ModelRuntimeAssetCheck(label=label, path=path, exists=False, scope="domain")
    if not path.is_file():
        return ModelRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message="path is not a file",
            scope="domain",
        )
    try:
        manifest = read_modeling_manifest(
            plan.execution_inputs.runtime_topology,
            plan.execution_inputs.request.manifest_family,
            split,
        )
        if manifest.manifest_path.resolve(strict=False) != path.resolve(strict=False):
            raise ModelingDataError(
                "resolved modeling manifest path does not match runtime plan path"
            )
        if manifest.split is not split:
            raise ModelingDataError(
                f"modeling manifest split mismatch: expected {split.value}, "
                f"observed {manifest.split.value}"
            )
    except (OSError, FileNotFoundError, ModelingDataError, ValueError, TypeError) as exc:
        return ModelRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
            scope="domain",
        )
    return ModelRuntimeAssetCheck(label=label, path=path, exists=True, valid=True, scope="domain")


def _directory_check(label: str, path: Path) -> ModelRuntimeAssetCheck:
    if not path.exists():
        return ModelRuntimeAssetCheck(label=label, path=path, exists=False)
    if not path.is_dir():
        return ModelRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message="path is not a directory",
        )
    return ModelRuntimeAssetCheck(label=label, path=path, exists=True)


__all__ = ["verify_model_runtime"]
