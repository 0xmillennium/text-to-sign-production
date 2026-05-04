"""Validation for the gates package contracts."""

from __future__ import annotations

from numbers import Real

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import (
    GateResult,
    GateStage,
    GateStatus,
    GateValidationIssue,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS


def validate_gates_config(config: GatesConfig) -> list[GateValidationIssue]:
    """Validate that the structural gates configuration is semantically valid."""
    issues: list[GateValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(GateValidationIssue(code=code, message=message))

    if isinstance(config.min_valid_frames, bool) or not isinstance(config.min_valid_frames, int):
        add("invalid_min_valid_frames_type", "min_valid_frames must be an integer.")
    elif config.min_valid_frames < 1:
        add("invalid_min_valid_frames", "min_valid_frames must be positive.")

    if isinstance(config.max_out_of_bounds_ratio, bool) or not isinstance(
        config.max_out_of_bounds_ratio, Real
    ):
        add("invalid_max_out_of_bounds_ratio_type", "max_out_of_bounds_ratio must be numeric.")
    elif not (0.0 <= config.max_out_of_bounds_ratio <= 1.0):
        add("invalid_max_out_of_bounds_ratio", "max_out_of_bounds_ratio must be within [0, 1].")

    for channel in CANONICAL_POSE_CHANNELS:
        if channel not in config.channel_configs:
            add("missing_canonical_channel_config", f"Missing channel config for {channel}.")

    for channel, ch_config in config.channel_configs.items():
        if channel not in CANONICAL_POSE_CHANNELS:
            add("unknown_channel_config", f"Unknown channel config {channel!r}.")
        if isinstance(ch_config.min_nonzero_frames, bool) or not isinstance(
            ch_config.min_nonzero_frames, int
        ):
            add("invalid_min_nonzero_frames_type", f"{channel} min_nonzero_frames is invalid.")
            continue
        if ch_config.min_nonzero_frames < 0:
            add("invalid_min_nonzero_frames", f"{channel} min_nonzero_frames is negative.")
        if ch_config.min_nonzero_frames > config.min_valid_frames:
            add(
                "channel_requires_more_frames_than_min_valid",
                f"{channel} min_nonzero_frames exceeds min_valid_frames.",
            )

    return issues


def validate_gate_result(result: GateResult) -> list[GateValidationIssue]:
    """Validate a single gate result."""
    issues: list[GateValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(GateValidationIssue(code=code, message=message))

    if not isinstance(result.status, GateStatus):
        add("invalid_gate_status", f"Invalid gate status: {result.status!r}.")
    if not result.passed and not result.reasons:
        add("dropped_missing_reasons", "Dropped gate results must include reasons.")
    if result.passed and result.reasons:
        add("passed_with_reasons", "Passed gate results must not include reasons.")
    return issues


def validate_decision(decision: ProcessingDecision) -> list[GateValidationIssue]:
    """Validate a final processing decision structure."""
    issues: list[GateValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(GateValidationIssue(code=code, message=message))

    if decision.status == ProcessingStatus.DROPPED:
        if decision.drop_stage is None:
            add("dropped_missing_stage", "Dropped decisions must include a drop stage.")
        if not decision.drop_reasons:
            add("dropped_missing_reasons", "Dropped decisions must include drop reasons.")

        if decision.drop_stage and decision.drop_stage not in decision.gate_results:
            add(
                "missing_gate_result_for_stage",
                f"Missing gate result for stage {_stage_label(decision.drop_stage)}.",
            )

        for stage, result in decision.gate_results.items():
            issues.extend(
                GateValidationIssue(
                    code=f"gate_{_stage_label(stage)}:{issue.code}",
                    message=issue.message,
                )
                for issue in validate_gate_result(result)
            )
            if stage == decision.drop_stage and result.passed:
                add("drop_stage_gate_passed", f"Drop stage gate passed: {_stage_label(stage)}.")

    elif decision.status == ProcessingStatus.PROCESSED:
        if decision.drop_stage is not None:
            add("processed_with_drop_stage", "Processed decisions must not include a drop stage.")
        if decision.drop_reasons:
            add("processed_with_drop_reasons", "Processed decisions must not include reasons.")
        if not decision.can_materialize_debug:
            add("processed_but_not_materializable", "Processed decisions must be materializable.")

        for stage, result in decision.gate_results.items():
            issues.extend(
                GateValidationIssue(
                    code=f"gate_{_stage_label(stage)}:{issue.code}",
                    message=issue.message,
                )
                for issue in validate_gate_result(result)
            )
            if not result.passed:
                add(
                    "processed_with_dropped_gate",
                    f"Processed decision has dropped {_stage_label(stage)}.",
                )

    for stage in decision.gate_results:
        if not isinstance(stage, GateStage):
            add("invalid_gate_stage", f"Invalid gate stage key: {stage!r}.")

    return issues


def _stage_label(stage: object) -> str:
    if isinstance(stage, GateStage):
        return stage.value
    return str(stage)
