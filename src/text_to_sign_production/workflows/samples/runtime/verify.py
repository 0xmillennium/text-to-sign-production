from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.gate.sources import validate_translation_columns
from text_to_sign_production.workflows.samples.contracts import (
    SamplesRuntimeAssetCheck,
    SamplesRuntimePlan,
    SamplesRuntimeVerification,
    SamplesSplitRuntimeInputs,
)


def verify_samples_runtime(
    plan: SamplesRuntimePlan,
) -> SamplesRuntimeVerification:
    return SamplesRuntimeVerification(checks=_build_runtime_asset_checks(plan))


def _build_runtime_asset_checks(
    plan: SamplesRuntimePlan,
) -> tuple[SamplesRuntimeAssetCheck, ...]:
    checks = [
        SamplesRuntimeAssetCheck(
            label="gates config",
            path=plan.execution_inputs.gates_config_path,
            exists=plan.execution_inputs.gates_config_path.exists(),
        )
    ]
    for split_input in plan.execution_inputs.split_inputs:
        checks.extend(
            _split_runtime_checks(
                split_input=split_input,
                canonical_text_column=plan.execution_inputs.translation_canonical_text_column,
            )
        )
    return tuple(checks)


def _split_runtime_checks(
    *,
    split_input: SamplesSplitRuntimeInputs,
    canonical_text_column: str,
) -> tuple[SamplesRuntimeAssetCheck, ...]:
    return (
        _translation_check(
            f"translation csv [{split_input.split}]",
            split_input.translation_csv_path,
            canonical_text_column=canonical_text_column,
        ),
        _asset_check(f"keypoint root [{split_input.split}]", split_input.keypoint_root),
        _asset_check(f"keypoint json root [{split_input.split}]", split_input.keypoint_json_root),
        _asset_check(
            f"keypoint video root [{split_input.split}]",
            split_input.keypoint_video_root,
        ),
    )


def _asset_check(label: str, path: Path) -> SamplesRuntimeAssetCheck:
    return SamplesRuntimeAssetCheck(label=label, path=path, exists=path.exists())


def _translation_check(
    label: str,
    path: Path,
    *,
    canonical_text_column: str,
) -> SamplesRuntimeAssetCheck:
    if not path.exists():
        return SamplesRuntimeAssetCheck(label=label, path=path, exists=False)
    try:
        validate_translation_columns(path, canonical_text_column=canonical_text_column)
    except (OSError, ValueError, TypeError) as exc:
        return SamplesRuntimeAssetCheck(
            label=label,
            path=path,
            exists=True,
            valid=False,
            message=str(exc),
        )
    return SamplesRuntimeAssetCheck(label=label, path=path, exists=True, valid=True)
