"""Invariant validation for pose-domain truths."""

from __future__ import annotations

from text_to_sign_production.data.gate.pose.schema import (
    validate_parsed_frame_schema,
    validate_tensor_schema,
    validate_tracking_schema,
)
from text_to_sign_production.data.gate.pose.types import (
    FrameFileListing,
    ParsedFrame,
    PersonTrackingResult,
    PoseBuildOutput,
    PoseTensorOutput,
    PoseValidationIssue,
    PoseValidationIssueCode,
)


def _issue(code: PoseValidationIssueCode, message: str) -> PoseValidationIssue:
    return PoseValidationIssue(code=code, message=message)


def validate_frame_listing_invariants(
    listing: FrameFileListing,
) -> tuple[PoseValidationIssue, ...]:
    """Validate frame-file access invariants."""
    issues: list[PoseValidationIssue] = []
    if listing.candidate.keypoints_dir != listing.directory:
        issues.append(
            _issue(
                PoseValidationIssueCode.LISTING_CANDIDATE_DIRECTORY_MISMATCH,
                "Frame listing directory must come from SourceCandidate keypoints_dir.",
            )
        )
    if listing.missing and listing.files:
        issues.append(
            _issue(
                PoseValidationIssueCode.MISSING_LISTING_HAS_FILES,
                "Missing frame listing cannot have files.",
            )
        )
    return tuple(issues)


def validate_parsed_frames_invariants(
    frames: tuple[ParsedFrame, ...],
) -> tuple[PoseValidationIssue, ...]:
    """Validate parsed-frame collection invariants."""
    issues: list[PoseValidationIssue] = []
    expected_indices = tuple(range(len(frames)))
    actual_indices = tuple(frame.frame_index for frame in frames)
    if actual_indices != expected_indices:
        issues.append(
            _issue(
                PoseValidationIssueCode.PARSED_FRAME_INDICES_NOT_CONTIGUOUS,
                "Parsed frame indices must be contiguous.",
            )
        )
    for frame in frames:
        issues.extend(validate_parsed_frame_schema(frame))
    return tuple(issues)


def validate_tracking_invariants(
    tracking: PersonTrackingResult,
    frames: tuple[ParsedFrame, ...],
) -> tuple[PoseValidationIssue, ...]:
    """Validate tracking invariants against parsed frames."""
    issues: list[PoseValidationIssue] = list(
        validate_tracking_schema(tracking, expected_frame_count=len(frames))
    )
    frames_by_index = {frame.frame_index: frame for frame in frames}
    for selection in tracking.frame_selections:
        frame = frames_by_index.get(selection.frame_index)
        if frame is None:
            issues.append(
                _issue(
                    PoseValidationIssueCode.TRACKING_FRAME_MISSING,
                    "Tracking references an unknown frame index.",
                )
            )
            continue
        if selection.selected_person_index is not None and selection.selected_person_index >= len(
            frame.people
        ):
            issues.append(
                _issue(
                    PoseValidationIssueCode.TRACKING_SELECTED_PERSON_OUT_OF_RANGE,
                    "Tracking selected person index exceeds frame people count.",
                )
            )
    missing_count = sum(
        1 for selection in tracking.frame_selections if selection.selected_person_index is None
    )
    if missing_count != tracking.target_missing_frame_count:
        issues.append(
            _issue(
                PoseValidationIssueCode.TRACKING_MISSING_COUNT_MISMATCH,
                "Tracking missing count does not match frame selections.",
            )
        )
    return tuple(issues)


def validate_tensor_invariants(
    output: PoseTensorOutput,
    tracking: PersonTrackingResult | None = None,
) -> tuple[PoseValidationIssue, ...]:
    """Validate tensor output invariants."""
    issues: list[PoseValidationIssue] = list(validate_tensor_schema(output))
    frame_count = len(output.frame_valid_mask)
    if output.candidate.frame_count != frame_count:
        issues.append(
            _issue(
                PoseValidationIssueCode.TENSOR_CANDIDATE_FRAME_COUNT_MISMATCH,
                "Tensor frame count mismatches candidate.",
            )
        )
    if tracking is not None and output.selected_person_indices != tracking.selected_person_indices:
        issues.append(
            _issue(
                PoseValidationIssueCode.TENSOR_TRACKING_SELECTION_MISMATCH,
                "Tensor selected person indices must match tracking output.",
            )
        )
    for channel, count in output.channel_nonzero_frame_counts.items():
        if count < 0 or count > frame_count:
            issues.append(
                _issue(
                    PoseValidationIssueCode.CHANNEL_NONZERO_COUNT_INVALID,
                    f"Channel {channel.value} nonzero count is outside frame range.",
                )
            )
    return tuple(issues)


def validate_pose_build_output(output: PoseBuildOutput) -> tuple[PoseValidationIssue, ...]:
    """Validate final pose build output invariants."""
    return validate_tensor_invariants(output.tensors, output.tracking)


__all__ = [
    "validate_frame_listing_invariants",
    "validate_parsed_frames_invariants",
    "validate_pose_build_output",
    "validate_tensor_invariants",
    "validate_tracking_invariants",
]
