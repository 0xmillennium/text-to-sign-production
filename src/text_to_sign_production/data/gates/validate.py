"""Validation for the gates package contracts."""

from __future__ import annotations

from text_to_sign_production.data._shared.validate import (
    is_non_bool_int,
    is_positive_finite_number,
    is_unit_interval,
)
from text_to_sign_production.data.gates.types import (
    GateResult,
    GatesConfig,
    GateStage,
    GateStatus,
    GateValidationIssue,
    ProcessingDecision,
    ProcessingStatus,
)

_CHANNEL_GATE_CONFIG_CHANNELS = ("body", "face")


def validate_gates_config(config: GatesConfig) -> list[GateValidationIssue]:
    """Validate that the structural gates configuration is semantically valid."""
    issues: list[GateValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(GateValidationIssue(code=code, message=message))

    integrity = config.integrity
    tracking = config.tracking_integrity
    channel_presence = config.channel_presence
    text_sanity = config.text_sanity

    if not is_non_bool_int(integrity.min_valid_frames):
        add("invalid_min_valid_frames_type", "integrity.min_valid_frames must be an integer.")
    elif integrity.min_valid_frames < 1:
        add("invalid_min_valid_frames", "integrity.min_valid_frames must be positive.")

    if not is_unit_interval(integrity.max_out_of_bounds_ratio):
        add(
            "invalid_max_out_of_bounds_ratio",
            "integrity.max_out_of_bounds_ratio must be within [0, 1].",
        )

    if not is_non_bool_int(integrity.min_num_frames):
        add("invalid_min_num_frames_type", "integrity.min_num_frames must be an integer.")
    elif integrity.min_num_frames < 1:
        add("invalid_min_num_frames", "integrity.min_num_frames must be positive.")

    if not is_positive_finite_number(integrity.min_duration_seconds):
        add(
            "invalid_min_duration_seconds",
            "integrity.min_duration_seconds must be positive and finite.",
        )

    for field_name in (
        "max_tracked_target_missing_frame_ratio",
        "max_zeroed_canonical_joint_frame_ratio",
        "max_person_tracking_continuity_break_ratio",
        "max_person_tracking_reanchor_ratio",
    ):
        if not is_unit_interval(getattr(tracking, field_name)):
            add(
                f"invalid_{field_name}",
                f"tracking_integrity.{field_name} must be within [0, 1].",
            )

    if not is_non_bool_int(channel_presence.min_any_hand_nonzero_frames):
        add(
            "invalid_min_any_hand_nonzero_frames_type",
            "channel_presence.min_any_hand_nonzero_frames is invalid.",
        )
    elif channel_presence.min_any_hand_nonzero_frames < 0:
        add(
            "invalid_min_any_hand_nonzero_frames",
            "channel_presence.min_any_hand_nonzero_frames is negative.",
        )
    elif channel_presence.min_any_hand_nonzero_frames > integrity.min_valid_frames:
        add(
            "any_hand_requires_more_frames_than_min_valid",
            "channel_presence.min_any_hand_nonzero_frames exceeds integrity.min_valid_frames.",
        )

    for channel in _CHANNEL_GATE_CONFIG_CHANNELS:
        if channel not in channel_presence.channels:
            add("missing_canonical_channel_config", f"Missing channel config for {channel}.")

    for channel, ch_config in channel_presence.channels.items():
        if channel not in _CHANNEL_GATE_CONFIG_CHANNELS:
            add("unknown_channel_config", f"Unknown channel config {channel!r}.")
        if not is_non_bool_int(ch_config.min_nonzero_frames):
            add("invalid_min_nonzero_frames_type", f"{channel} min_nonzero_frames is invalid.")
            continue
        if ch_config.min_nonzero_frames < 0:
            add("invalid_min_nonzero_frames", f"{channel} min_nonzero_frames is negative.")
        if ch_config.min_nonzero_frames > integrity.min_valid_frames:
            add(
                "channel_requires_more_frames_than_min_valid",
                f"{channel} min_nonzero_frames exceeds integrity.min_valid_frames.",
            )

    if not is_non_bool_int(text_sanity.min_character_count):
        add(
            "invalid_min_character_count_type",
            "text_sanity.min_character_count must be an integer.",
        )
    elif text_sanity.min_character_count < 0:
        add("invalid_min_character_count", "text_sanity.min_character_count is negative.")

    if not is_non_bool_int(text_sanity.min_token_count):
        add("invalid_min_token_count_type", "text_sanity.min_token_count must be an integer.")
    elif text_sanity.min_token_count < 0:
        add("invalid_min_token_count", "text_sanity.min_token_count is negative.")

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
