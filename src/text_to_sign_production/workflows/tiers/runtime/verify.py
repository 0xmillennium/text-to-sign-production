from __future__ import annotations

from pathlib import Path

from text_to_sign_production.workflows.tiers.contracts import (
    TiersRuntimeAssetCheck,
    TiersRuntimePlan,
    TiersRuntimeVerification,
    TiersSplitRuntimeInputs,
)


def verify_tiers_runtime(
    plan: TiersRuntimePlan,
) -> TiersRuntimeVerification:
    return TiersRuntimeVerification(
        checks=_build_runtime_asset_checks(plan),
    )


def _build_runtime_asset_checks(
    plan: TiersRuntimePlan,
) -> tuple[TiersRuntimeAssetCheck, ...]:
    execution_inputs = plan.execution_inputs
    return (
        _runtime_asset_check("filters config", execution_inputs.filters_config_path),
        _runtime_asset_check("tiers config", execution_inputs.tiers_config_path),
        _runtime_asset_check("passed samples root", execution_inputs.passed_samples_root),
        *(
            check
            for split_input in execution_inputs.split_inputs
            for check in _split_runtime_checks(split_input)
        ),
    )


def _split_runtime_checks(
    split_input: TiersSplitRuntimeInputs,
) -> tuple[TiersRuntimeAssetCheck, ...]:
    return (
        _runtime_asset_check(
            f"passed manifest [{split_input.split}]",
            split_input.passed_manifest_path,
        ),
        _runtime_asset_check(
            f"passed samples root [{split_input.split}]",
            split_input.passed_samples_split_root,
        ),
    )


def _runtime_asset_check(label: str, path: Path) -> TiersRuntimeAssetCheck:
    return TiersRuntimeAssetCheck(
        label=label,
        path=path,
        exists=path.exists(),
    )
