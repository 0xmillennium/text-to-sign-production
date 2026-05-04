"""Pose schema and parse structure gate."""

from __future__ import annotations

from text_to_sign_production.data.gates.config import GatesConfig
from text_to_sign_production.data.gates.types import GateResult, GateStatus
from text_to_sign_production.data.pose.types import PoseBuildOutput


def evaluate_schema_gate(pose_output: PoseBuildOutput, config: GatesConfig) -> GateResult:
    """Evaluate if the parsed pose is structurally coherent."""
    reasons: list[str] = []

    if pose_output.diagnostics.parse_error:
        reasons.append(f"parse_error:{pose_output.diagnostics.parse_error}")

    quality = pose_output.frame_quality
    total_frames = quality.valid_frame_count + quality.invalid_frame_count

    if total_frames == 0:
        reasons.append("no_frames_processed")
    elif quality.valid_frame_count < config.min_valid_frames:
        reasons.append(
            f"insufficient_valid_frames:{quality.valid_frame_count}<{config.min_valid_frames}"
        )

    if total_frames > 0:
        expected_coordinates = total_frames * 137  # Approx 137 points across BFH
        if expected_coordinates > 0:
            ratio = quality.out_of_bounds_coordinate_count / expected_coordinates
            if ratio > config.max_out_of_bounds_ratio:
                reasons.append(
                    f"excessive_out_of_bounds_ratio:{ratio:.2f}>{config.max_out_of_bounds_ratio}"
                )

    if reasons:
        return GateResult(status=GateStatus.DROPPED, reasons=reasons)

    return GateResult(status=GateStatus.PASSED)
