"""Pose-level validation helpers."""

from __future__ import annotations

from text_to_sign_production.data.pose.schema import OPENPOSE_CHANNEL_SPECS
from text_to_sign_production.data.pose.types import PoseBuildOutput, PoseValidationIssue


def validate_pose_build(output: PoseBuildOutput) -> list[PoseValidationIssue]:
    """Validate a constructed pose output.

    Checks canonical BFH channel presence, tensor/result-level shape sanity,
    and consistency between selected-person facts and frame-quality facts.
    """
    issues: list[PoseValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(PoseValidationIssue(code=code, message=message))

    if output.diagnostics.unrecoverable_error is not None:
        add(
            "unrecoverable_pose_error",
            f"Pose build had an unrecoverable error: {output.diagnostics.unrecoverable_error}.",
        )

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
            add("missing_channel_data", f"Pose channel {name!r} has missing data.")
            continue

        coords_shape = channel.coordinates.shape
        conf_shape = channel.confidence.shape

        _, expected_points = OPENPOSE_CHANNEL_SPECS[name]

        if len(coords_shape) != 3 or coords_shape[1] != expected_points or coords_shape[2] != 2:
            add("invalid_coordinates_shape", f"Pose channel {name!r} coordinates shape is invalid.")

        if len(conf_shape) != 2 or conf_shape[1] != expected_points:
            add("invalid_confidence_shape", f"Pose channel {name!r} confidence shape is invalid.")

        if coords_shape[0] != expected_frames or conf_shape[0] != expected_frames:
            add("frame_count_mismatch", f"Pose channel {name!r} frame count mismatches quality.")

    if output.selected_person.multi_person_frame_count > expected_frames:
        add("invalid_multi_person_count", "Selected-person multi-person count exceeds frames.")

    return issues
