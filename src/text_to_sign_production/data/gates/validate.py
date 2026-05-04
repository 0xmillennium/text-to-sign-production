"""Validation for the gates package contracts."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import (
    GateResult,
    ProcessingDecision,
    ProcessingStatus,
)
from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS


def validate_gates_config(config: GatesConfig) -> list[str]:
    """Validate that the structural gates configuration is semantically valid."""
    issues: list[str] = []

    if config.min_valid_frames < 1:
        issues.append(f"invalid_min_valid_frames:{config.min_valid_frames}")

    if not (0.0 <= config.max_out_of_bounds_ratio <= 1.0):
        issues.append(f"invalid_max_out_of_bounds_ratio:{config.max_out_of_bounds_ratio}")

    for channel in CANONICAL_POSE_CHANNELS:
        if channel not in config.channel_configs:
            issues.append(f"missing_canonical_channel_config:{channel}")

    for channel, ch_config in config.channel_configs.items():
        if channel not in CANONICAL_POSE_CHANNELS:
            issues.append(f"unknown_channel_config:{channel}")
        if ch_config.min_nonzero_frames < 0:
            issues.append(
                f"invalid_min_nonzero_frames_for_{channel}:{ch_config.min_nonzero_frames}"
            )
        if ch_config.min_nonzero_frames > config.min_valid_frames:
            issues.append(
                f"channel_{channel}_requires_more_frames_than_min_valid:"
                f"{ch_config.min_nonzero_frames}>{config.min_valid_frames}"
            )

    return issues


def validate_gate_result(result: GateResult) -> list[str]:
    """Validate a single gate result."""
    issues: list[str] = []
    if result.status.value not in {"passed", "dropped"}:
        issues.append(f"invalid_gate_status:{result.status}")
    if not result.passed and not result.reasons:
        issues.append("dropped_missing_reasons")
    if result.passed and result.reasons:
        issues.append("passed_with_reasons")
    return issues


def validate_decision(decision: ProcessingDecision) -> list[str]:
    """Validate a final processing decision structure."""
    issues: list[str] = []

    if decision.status == ProcessingStatus.DROPPED:
        if decision.drop_stage is None:
            issues.append("dropped_missing_stage")
        if not decision.drop_reasons:
            issues.append("dropped_missing_reasons")

        if decision.drop_stage and decision.drop_stage not in decision.gate_results:
            issues.append(f"missing_gate_result_for_stage:{decision.drop_stage}")

        for stage, result in decision.gate_results.items():
            issues.extend([f"gate_{stage}:{issue}" for issue in validate_gate_result(result)])
            if stage == decision.drop_stage and result.passed:
                issues.append(f"drop_stage_gate_passed:{stage}")

    elif decision.status == ProcessingStatus.PROCESSED:
        if decision.drop_stage is not None:
            issues.append("processed_with_drop_stage")
        if decision.drop_reasons:
            issues.append("processed_with_drop_reasons")
        if not decision.can_materialize_debug:
            issues.append("processed_but_not_materializable")

        for stage, result in decision.gate_results.items():
            issues.extend([f"gate_{stage}:{issue}" for issue in validate_gate_result(result)])
            if not result.passed:
                issues.append(f"processed_with_dropped_gate:{stage}")

    return issues
