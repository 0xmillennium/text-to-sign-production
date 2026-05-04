"""Pose-level validation helpers."""

from __future__ import annotations

from text_to_sign_production.data.pose.schema import OPENPOSE_CHANNEL_SPECS
from text_to_sign_production.data.pose.types import PoseBuildOutput


def validate_pose_build(output: PoseBuildOutput) -> list[str]:
    """Validate a constructed pose output.

    Checks canonical BFH channel presence, tensor/result-level shape sanity,
    and consistency between selected-person facts and frame-quality facts.
    """
    issues: list[str] = []

    if output.diagnostics.parse_error is not None:
        issues.append(f"parse_error:{output.diagnostics.parse_error}")

    pose = output.pose
    channels = {
        "body": pose.body,
        "left_hand": pose.left_hand,
        "right_hand": pose.right_hand,
        "face": pose.face,
    }

    expected_frames = (
        output.frame_quality.valid_frame_count + output.frame_quality.invalid_frame_count
    )

    for name, channel in channels.items():
        if channel.coordinates is None or channel.confidence is None:
            issues.append(f"missing_channel_data:{name}")
            continue

        coords_shape = channel.coordinates.shape
        conf_shape = channel.confidence.shape

        _, expected_points = OPENPOSE_CHANNEL_SPECS[name]

        if len(coords_shape) != 3 or coords_shape[1] != expected_points or coords_shape[2] != 2:
            issues.append(f"invalid_coordinates_shape:{name}")

        if len(conf_shape) != 2 or conf_shape[1] != expected_points:
            issues.append(f"invalid_confidence_shape:{name}")

        if coords_shape[0] != expected_frames or conf_shape[0] != expected_frames:
            issues.append(f"frame_count_mismatch:{name}")

    if output.selected_person.multi_person_frame_count > expected_frames:
        issues.append("invalid_multi_person_count")

    return issues
