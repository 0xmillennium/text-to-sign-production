"""Pose schema and parse structure gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.schema import (
    CANONICAL_POSE_CHANNELS,
    POSE_CHANNEL_JOINT_COUNTS,
    POSE_COORDINATE_DIMENSIONS,
)
from text_to_sign_production.data.pose.types import PoseBuildOutput


def _canonical_coordinate_slots_per_frame() -> int:
    return (
        sum(POSE_CHANNEL_JOINT_COUNTS[channel] for channel in CANONICAL_POSE_CHANNELS)
        * POSE_COORDINATE_DIMENSIONS
    )


def evaluate_schema_gate(pose_output: PoseBuildOutput, config: GatesConfig) -> GateResult:
    """Evaluate if the parsed pose is structurally coherent."""
    reasons = []

    if pose_output.diagnostics.unrecoverable_error:
        reasons.append(f"unrecoverable_pose_error:{pose_output.diagnostics.unrecoverable_error}")

    quality = pose_output.frame_quality
    total_frames = quality.valid_frame_count + quality.invalid_frame_count

    if total_frames == 0:
        reasons.append("no_frames_processed")
    elif quality.valid_frame_count < config.min_valid_frames:
        reasons.append(
            f"insufficient_valid_frames:{quality.valid_frame_count}<{config.min_valid_frames}"
        )

    if total_frames > 0:
        total_coordinate_slots = total_frames * _canonical_coordinate_slots_per_frame()
        if total_coordinate_slots > 0:
            ratio = quality.out_of_bounds_coordinate_count / total_coordinate_slots
            if ratio > config.max_out_of_bounds_ratio:
                reasons.append(
                    f"excessive_out_of_bounds_ratio:{ratio:.2f}>{config.max_out_of_bounds_ratio}"
                )

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=tuple(reasons))

    return GateResult(status=GateStatus.PASSED)
