from __future__ import annotations

from text_to_sign_production.data.gate.policies import load_gates_config
from text_to_sign_production.data.gate.sources.translations import validate_translation_columns
from text_to_sign_production.data.tier.policies.filters import load_tier_filters_config
from text_to_sign_production.data.tier.policies.policies import load_tier_policies_config
from text_to_sign_production.workflows.debug.contracts import (
    DebugRestorePlan,
    DebugRuntimeCheck,
    DebugRuntimeVerification,
    DebugWorkflowConfig,
)
from text_to_sign_production.workflows.debug.layout import DebugLayout


def verify_debug_runtime(
    config: DebugWorkflowConfig,
    plan: DebugRestorePlan,
    *,
    layout: DebugLayout,
) -> DebugRuntimeVerification:
    runtime = layout.runtime
    checks: list[DebugRuntimeCheck] = [
        _file_check(
            "gates config",
            config.gates_config_runtime_path,
            required=True,
            parser=load_gates_config,
        ),
        _file_check(
            "filters config",
            config.filters_config_runtime_path,
            required=True,
            parser=load_tier_filters_config,
        ),
        _file_check(
            "tiers config",
            config.tiers_config_runtime_path,
            required=True,
            parser=load_tier_policies_config,
        ),
    ]
    for split in plan.debug_splits:
        checks.extend(
            (
                _file_check(
                    f"translation [{split.value}]",
                    runtime.assets.translation_csv(split).path,
                    required=True,
                    parser=validate_translation_columns,
                ),
                _dir_check(
                    f"keypoint split root [{split.value}]",
                    runtime.assets.keypoint_split_root(split).path,
                    required=False,
                ),
                _dir_check(
                    f"keypoint json root [{split.value}]",
                    runtime.assets.keypoint_json_dir(split).path,
                    required=False,
                ),
                _dir_check(
                    f"keypoint video root [{split.value}]",
                    runtime.assets.keypoint_video_dir(split).path,
                    required=False,
                ),
                _dir_check(
                    f"manifests root [{split.value}]",
                    runtime.manifests_root,
                    required=False,
                ),
                _dir_check(
                    f"samples root [{split.value}]",
                    runtime.samples_root,
                    required=False,
                ),
            )
        )
    errors = tuple(
        f"{check.label}: {check.message or 'missing or invalid'}"
        for check in checks
        if check.required and (not check.exists or not check.valid)
    )
    warnings = tuple(
        f"{check.label}: {check.message or 'missing or invalid'}"
        for check in checks
        if not check.required and (not check.exists or not check.valid)
    )
    return DebugRuntimeVerification(checks=tuple(checks), errors=errors, warnings=warnings)


def _file_check(label: str, path, *, required: bool, parser=None) -> DebugRuntimeCheck:
    if not path.exists():
        return DebugRuntimeCheck(label, path, required, False, False, "file does not exist")
    if not path.is_file():
        return DebugRuntimeCheck(label, path, required, True, False, "path is not a file")
    if parser is not None:
        try:
            parser(path)
        except (OSError, TypeError, ValueError) as exc:
            return DebugRuntimeCheck(label, path, required, True, False, str(exc))
    return DebugRuntimeCheck(label, path, required, True, True)


def _dir_check(
    label: str,
    path,
    *,
    required: bool,
    create: bool = False,
) -> DebugRuntimeCheck:
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return DebugRuntimeCheck(label, path, required, path.exists(), False, str(exc))
    if not path.exists():
        return DebugRuntimeCheck(label, path, required, False, False, "directory does not exist")
    if not path.is_dir():
        return DebugRuntimeCheck(label, path, required, True, False, "path is not a directory")
    return DebugRuntimeCheck(label, path, required, True, True)


__all__ = ["verify_debug_runtime"]
