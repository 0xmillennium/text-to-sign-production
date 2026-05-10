"""Pose-domain schema constants and structural validation."""

from __future__ import annotations

from typing import Final

from text_to_sign_production.data.gate.pose.types import (
    CoordinateSpace,
    ParsedFrame,
    PersonTrackingResult,
    PoseChannel,
    PoseTensorOutput,
    PoseValidationIssue,
    PoseValidationIssueCode,
)

EXPECTED_OPENPOSE_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset({"people", "version"})
EXPECTED_OPENPOSE_PERSON_KEYS: Final[frozenset[str]] = frozenset(
    {
        "person_id",
        "pose_keypoints_2d",
        "face_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
        "pose_keypoints_3d",
        "face_keypoints_3d",
        "hand_left_keypoints_3d",
        "hand_right_keypoints_3d",
    }
)

OPENPOSE_CHANNEL_SPECS: Final[dict[PoseChannel, tuple[str, int]]] = {
    PoseChannel.BODY: ("pose_keypoints_2d", 25),
    PoseChannel.FACE: ("face_keypoints_2d", 70),
    PoseChannel.LEFT_HAND: ("hand_left_keypoints_2d", 21),
    PoseChannel.RIGHT_HAND: ("hand_right_keypoints_2d", 21),
}
CANONICAL_POSE_CHANNELS: Final[tuple[PoseChannel, ...]] = (
    PoseChannel.BODY,
    PoseChannel.LEFT_HAND,
    PoseChannel.RIGHT_HAND,
    PoseChannel.FACE,
)
POSE_CHANNEL_JOINT_COUNTS: Final[dict[PoseChannel, int]] = {
    channel: point_count for channel, (_, point_count) in OPENPOSE_CHANNEL_SPECS.items()
}
POSE_COORDINATE_DIMENSIONS: Final[int] = 2
DEFAULT_CANVAS_WIDTH: Final[int] = 1280
DEFAULT_CANVAS_HEIGHT: Final[int] = 720


def _issue(code: PoseValidationIssueCode, message: str) -> PoseValidationIssue:
    return PoseValidationIssue(code=code, message=message)


def validate_parsed_frame_schema(frame: ParsedFrame) -> tuple[PoseValidationIssue, ...]:
    """Validate structural parsed-frame shape."""
    issues: list[PoseValidationIssue] = []
    if frame.frame_index < 0:
        issues.append(
            _issue(
                PoseValidationIssueCode.INVALID_FRAME_INDEX,
                "Parsed frame index cannot be negative.",
            )
        )
    for person_index, person in enumerate(frame.people):
        for channel in CANONICAL_POSE_CHANNELS:
            parsed_channel = person.channels.get(channel)
            if parsed_channel is None:
                issues.append(
                    _issue(
                        PoseValidationIssueCode.MISSING_POSE_CHANNEL,
                        f"Person {person_index} missing {channel}.",
                    )
                )
                continue
            expected_points = POSE_CHANNEL_JOINT_COUNTS[channel]
            if parsed_channel.coordinates.shape != (expected_points, POSE_COORDINATE_DIMENSIONS):
                issues.append(
                    _issue(
                        PoseValidationIssueCode.INVALID_CHANNEL_COORDINATES,
                        f"{channel} coordinates shape invalid.",
                    )
                )
            if parsed_channel.confidences.shape != (expected_points,):
                issues.append(
                    _issue(
                        PoseValidationIssueCode.INVALID_CHANNEL_CONFIDENCES,
                        f"{channel} confidences shape invalid.",
                    )
                )
            if parsed_channel.coordinate_space is not CoordinateSpace.NORMALIZED_IMAGE:
                issues.append(
                    _issue(
                        PoseValidationIssueCode.INVALID_COORDINATE_SPACE,
                        f"{channel} coordinate space invalid.",
                    )
                )
    return tuple(issues)


def validate_tracking_schema(
    tracking: PersonTrackingResult,
    *,
    expected_frame_count: int | None = None,
) -> tuple[PoseValidationIssue, ...]:
    """Validate structural tracking-result shape."""
    issues: list[PoseValidationIssue] = []
    if tracking.anchor.anchor_person_index < 0:
        issues.append(
            _issue(PoseValidationIssueCode.INVALID_ANCHOR_INDEX, "Anchor index cannot be negative.")
        )
    if expected_frame_count is not None and len(tracking.frame_selections) != expected_frame_count:
        issues.append(
            _issue(
                PoseValidationIssueCode.TRACKING_FRAME_COUNT_MISMATCH,
                "Tracking selections mismatch frame count.",
            )
        )
    for selection in tracking.frame_selections:
        if selection.frame_index < 0:
            issues.append(
                _issue(
                    PoseValidationIssueCode.INVALID_TRACKING_FRAME_INDEX,
                    "Tracking frame index invalid.",
                )
            )
        if selection.selected_person_index is not None and selection.selected_person_index < 0:
            issues.append(
                _issue(
                    PoseValidationIssueCode.INVALID_SELECTED_PERSON_INDEX,
                    "Selected person index invalid.",
                )
            )
    return tuple(issues)


def validate_tensor_schema(output: PoseTensorOutput) -> tuple[PoseValidationIssue, ...]:
    """Validate structural tensor-output shape."""
    issues: list[PoseValidationIssue] = []
    frame_count = len(output.frame_valid_mask)
    if output.people_per_frame.shape != (frame_count,):
        issues.append(
            _issue(
                PoseValidationIssueCode.PEOPLE_PER_FRAME_SHAPE, "people_per_frame shape mismatch."
            )
        )
    if len(output.selected_person_indices) != frame_count:
        issues.append(
            _issue(
                PoseValidationIssueCode.SELECTED_PERSON_INDICES_SHAPE,
                "selected_person_indices length mismatch.",
            )
        )
    for channel in CANONICAL_POSE_CHANNELS:
        tensor = output.channels.get(channel)
        if tensor is None:
            issues.append(
                _issue(
                    PoseValidationIssueCode.MISSING_TENSOR_CHANNEL,
                    f"Missing tensor channel {channel}.",
                )
            )
            continue
        expected_points = POSE_CHANNEL_JOINT_COUNTS[channel]
        if tensor.coordinates.shape != (frame_count, expected_points, POSE_COORDINATE_DIMENSIONS):
            issues.append(
                _issue(
                    PoseValidationIssueCode.INVALID_TENSOR_COORDINATES,
                    f"{channel} tensor coordinates shape invalid.",
                )
            )
        if tensor.confidences.shape != (frame_count, expected_points):
            issues.append(
                _issue(
                    PoseValidationIssueCode.INVALID_TENSOR_CONFIDENCES,
                    f"{channel} tensor confidences shape invalid.",
                )
            )
        if tensor.coordinate_space is not CoordinateSpace.NORMALIZED_IMAGE:
            issues.append(
                _issue(
                    PoseValidationIssueCode.INVALID_TENSOR_COORDINATE_SPACE,
                    f"{channel} tensor coordinate space invalid.",
                )
            )
    return tuple(issues)


__all__ = [
    "CANONICAL_POSE_CHANNELS",
    "DEFAULT_CANVAS_HEIGHT",
    "DEFAULT_CANVAS_WIDTH",
    "EXPECTED_OPENPOSE_PERSON_KEYS",
    "EXPECTED_OPENPOSE_TOP_LEVEL_KEYS",
    "OPENPOSE_CHANNEL_SPECS",
    "POSE_CHANNEL_JOINT_COUNTS",
    "POSE_COORDINATE_DIMENSIONS",
    "validate_parsed_frame_schema",
    "validate_tensor_schema",
    "validate_tracking_schema",
]
