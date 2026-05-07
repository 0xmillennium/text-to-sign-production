"""Reusable typed component validators for sample-domain contracts."""

from __future__ import annotations

from text_to_sign_production.data._shared.types import JsonValue
from text_to_sign_production.data._shared.validate import is_unit_interval
from text_to_sign_production.data.pose.schema import (
    CANONICAL_POSE_CHANNELS,
    POSE_CHANNEL_JOINT_COUNTS,
    POSE_COORDINATE_DIMENSIONS,
)
from text_to_sign_production.data.samples._shared.validation_primitives import (
    add_issue,
    validate_array_shape,
    validate_count_mapping_values,
    validate_json_mapping_values,
    validate_non_empty_text,
    validate_non_negative_count,
)
from text_to_sign_production.data.samples.types import (
    BfhPosePayload,
    FrameQualitySummary,
    PoseChannelPayload,
    SampleValidationIssue,
    SelectedPersonMetadata,
)


def validate_selected_person_metadata(
    issues: list[SampleValidationIssue],
    selected_person: SelectedPersonMetadata,
    *,
    num_frames: int | None,
    label: str,
) -> None:
    """Validate selected-person semantic invariants."""
    index = validate_non_negative_count(issues, selected_person.index, label=f"{label}.index")
    multi_person_frame_count = validate_non_negative_count(
        issues,
        selected_person.multi_person_frame_count,
        label=f"{label}.multi_person_frame_count",
    )
    max_people_per_frame = validate_non_negative_count(
        issues,
        selected_person.max_people_per_frame,
        label=f"{label}.max_people_per_frame",
    )

    if max_people_per_frame is not None and max_people_per_frame < 1:
        add_issue(
            issues,
            "invalid_selected_person",
            f"{label}.max_people_per_frame must be >= 1.",
        )
    if (
        index is not None
        and max_people_per_frame is not None
        and max_people_per_frame >= 1
        and index >= max_people_per_frame
    ):
        add_issue(
            issues,
            "invalid_selected_person",
            f"{label}.index must be less than max_people_per_frame.",
        )
    if (
        multi_person_frame_count is not None
        and num_frames is not None
        and multi_person_frame_count > num_frames
    ):
        add_issue(
            issues,
            "selected_person_frame_count_mismatch",
            f"{label}.multi_person_frame_count cannot exceed num_frames.",
        )


def validate_frame_quality_summary(
    issues: list[SampleValidationIssue],
    frame_quality: FrameQualitySummary,
    *,
    num_frames: int | None,
    label: str,
) -> None:
    """Validate frame-quality semantic invariants."""
    valid_frame_count = validate_non_negative_count(
        issues,
        frame_quality.valid_frame_count,
        label=f"{label}.valid_frame_count",
    )
    invalid_frame_count = validate_non_negative_count(
        issues,
        frame_quality.invalid_frame_count,
        label=f"{label}.invalid_frame_count",
    )
    face_missing_frame_count = validate_non_negative_count(
        issues,
        frame_quality.face_missing_frame_count,
        label=f"{label}.face_missing_frame_count",
    )
    validate_non_negative_count(
        issues,
        frame_quality.out_of_bounds_coordinate_count,
        label=f"{label}.out_of_bounds_coordinate_count",
    )
    zeroed_count = validate_non_negative_count(
        issues,
        frame_quality.frames_with_any_zeroed_canonical_joint,
        label=f"{label}.frames_with_any_zeroed_canonical_joint",
    )
    tracked_missing_count = validate_non_negative_count(
        issues,
        frame_quality.tracked_target_missing_frame_count,
        label=f"{label}.tracked_target_missing_frame_count",
    )
    continuity_break_count = validate_non_negative_count(
        issues,
        frame_quality.person_tracking_continuity_break_count,
        label=f"{label}.person_tracking_continuity_break_count",
    )
    reanchor_count = validate_non_negative_count(
        issues,
        frame_quality.person_tracking_reanchor_count,
        label=f"{label}.person_tracking_reanchor_count",
    )

    if (
        valid_frame_count is not None
        and invalid_frame_count is not None
        and num_frames is not None
        and valid_frame_count + invalid_frame_count != num_frames
    ):
        add_issue(
            issues,
            "frame_quality_count_mismatch",
            f"{label} valid_frame_count + invalid_frame_count must equal num_frames.",
        )

    for field_name, count in (
        ("face_missing_frame_count", face_missing_frame_count),
        ("frames_with_any_zeroed_canonical_joint", zeroed_count),
        ("tracked_target_missing_frame_count", tracked_missing_count),
        ("person_tracking_continuity_break_count", continuity_break_count),
        ("person_tracking_reanchor_count", reanchor_count),
    ):
        if count is not None and num_frames is not None and count > num_frames:
            add_issue(
                issues,
                "frame_quality_count_mismatch",
                f"{label}.{field_name} cannot exceed num_frames.",
            )

    for field_name, ratio in (
        ("tracked_target_missing_frame_ratio", frame_quality.tracked_target_missing_frame_ratio),
        (
            "person_tracking_continuity_break_ratio",
            frame_quality.person_tracking_continuity_break_ratio,
        ),
        ("person_tracking_reanchor_ratio", frame_quality.person_tracking_reanchor_ratio),
    ):
        if not is_unit_interval(ratio):
            add_issue(
                issues,
                "invalid_frame_quality_ratio",
                f"{label}.{field_name} must be finite and within [0, 1].",
            )

    validate_count_mapping_values(
        issues,
        frame_quality.frame_issue_counts,
        required_keys=None,
        num_frames=None,
        label=f"{label}.frame_issue_counts",
    )
    validate_count_mapping_values(
        issues,
        frame_quality.channel_nonzero_frames,
        required_keys=CANONICAL_POSE_CHANNELS,
        num_frames=num_frames,
        label=f"{label}.channel_nonzero_frames",
    )


def validate_pose_payload(
    issues: list[SampleValidationIssue],
    pose: BfhPosePayload,
    *,
    num_frames: int | None,
    label: str,
) -> None:
    """Validate canonical body, face, and hands pose payload semantics."""
    for channel_name in CANONICAL_POSE_CHANNELS:
        validate_pose_channel_payload(
            issues,
            getattr(pose, channel_name),
            channel_name=channel_name,
            num_frames=num_frames,
            label=f"{label}.{channel_name}",
        )


def validate_pose_channel_payload(
    issues: list[SampleValidationIssue],
    channel: PoseChannelPayload,
    *,
    channel_name: str,
    num_frames: int | None,
    label: str,
) -> None:
    """Validate one canonical pose channel's array shapes."""
    expected_joint_count = POSE_CHANNEL_JOINT_COUNTS[channel_name]
    validate_array_shape(
        issues,
        channel.coordinates,
        missing_code="missing_pose_coordinates",
        shape_code="invalid_pose_coordinates_shape",
        expected_shape=(
            (num_frames, expected_joint_count, POSE_COORDINATE_DIMENSIONS)
            if num_frames is not None
            else None
        ),
        label=f"{label}.coordinates",
    )
    validate_array_shape(
        issues,
        channel.confidence,
        missing_code="missing_pose_confidence",
        shape_code="invalid_pose_confidence_shape",
        expected_shape=((num_frames, expected_joint_count) if num_frames is not None else None),
        label=f"{label}.confidence",
    )


def validate_drop_reasons(
    issues: list[SampleValidationIssue],
    drop_reasons: tuple[str, ...],
) -> None:
    """Validate dropped manifest reason semantics."""
    if not drop_reasons:
        add_issue(
            issues,
            "invalid_drop_reasons",
            "Dropped manifest drop_reasons must contain at least one reason.",
        )
        return

    for reason in drop_reasons:
        validate_non_empty_text(
            issues,
            reason,
            label="Dropped manifest drop_reasons values",
            code="invalid_drop_reasons",
        )

    if len(set(drop_reasons)) != len(drop_reasons):
        add_issue(
            issues,
            "duplicate_drop_reasons",
            "Dropped manifest drop_reasons must not contain duplicates.",
        )


def validate_drop_details(
    issues: list[SampleValidationIssue],
    drop_details: dict[str, JsonValue],
) -> None:
    """Validate dropped manifest detail semantics."""
    validate_json_mapping_values(
        issues,
        drop_details,
        label="Dropped manifest drop_details",
    )
