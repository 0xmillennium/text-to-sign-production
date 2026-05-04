"""Validation for the metrics package contracts."""

from __future__ import annotations

import math

from text_to_sign_production.data.metrics.types import MetricBundle, MetricValidationIssue


def validate_metric_bundle(bundle: MetricBundle) -> list[MetricValidationIssue]:
    """Validate that the metric bundle structure is semantically valid."""
    issues: list[MetricValidationIssue] = []

    def _add(code: str, msg: str) -> None:
        issues.append(MetricValidationIssue(code=code, message=msg))

    def _check_ratio(name: str, val: float) -> None:
        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
            _add(f"invalid_{name}", f"{name} must be in [0, 1], got {val}")

    def _check_count(name: str, val: int) -> None:
        if val < 0:
            _add(f"invalid_{name}", f"{name} must be >= 0, got {val}")

    # OOB
    _check_count("analysis_window.start_frame_index", bundle.analysis_window.start_frame_index)
    _check_count(
        "analysis_window.end_frame_index_exclusive",
        bundle.analysis_window.end_frame_index_exclusive,
    )
    _check_count("analysis_window.frame_count", bundle.analysis_window.frame_count)
    _check_ratio("analysis_window.frame_ratio", bundle.analysis_window.frame_ratio)
    _check_count(
        "analysis_window.pre_sign_excluded_frame_count",
        bundle.analysis_window.pre_sign_excluded_frame_count,
    )
    _check_ratio(
        "analysis_window.pre_sign_excluded_frame_ratio",
        bundle.analysis_window.pre_sign_excluded_frame_ratio,
    )
    _check_count(
        "analysis_window.post_sign_excluded_frame_count",
        bundle.analysis_window.post_sign_excluded_frame_count,
    )
    _check_ratio(
        "analysis_window.post_sign_excluded_frame_ratio",
        bundle.analysis_window.post_sign_excluded_frame_ratio,
    )
    _check_count(
        "analysis_window.sustained_hand_evidence_frame_count",
        bundle.analysis_window.sustained_hand_evidence_frame_count,
    )
    _check_ratio(
        "analysis_window.sustained_hand_evidence_frame_ratio",
        bundle.analysis_window.sustained_hand_evidence_frame_ratio,
    )
    if bundle.analysis_window.end_frame_index_exclusive > bundle.length.num_frames:
        _add("invalid_analysis_window", "analysis window end exceeds num_frames")
    if (
        bundle.analysis_window.start_frame_index
        >= bundle.analysis_window.end_frame_index_exclusive
    ):
        _add("invalid_analysis_window", "analysis window start must be before end")
    if (
        bundle.analysis_window.end_frame_index_exclusive
        - bundle.analysis_window.start_frame_index
        != bundle.analysis_window.frame_count
    ):
        _add("invalid_analysis_window", "analysis window frame_count mismatches bounds")
    if (
        bundle.analysis_window.pre_sign_excluded_frame_count
        + bundle.analysis_window.frame_count
        + bundle.analysis_window.post_sign_excluded_frame_count
        != bundle.length.num_frames
    ):
        _add("invalid_analysis_window", "analysis window partitions must equal num_frames")
    if (
        bundle.analysis_window.sustained_hand_evidence_frame_count
        > bundle.analysis_window.frame_count
    ):
        _add("invalid_analysis_window", "sustained hand evidence exceeds analysis-window frames")

    # OOB
    _check_count("oob.out_of_bounds_coordinate_count", bundle.oob.out_of_bounds_coordinate_count)
    _check_count("oob.total_coordinate_slots", bundle.oob.total_coordinate_slots)
    _check_ratio("oob.out_of_bounds_ratio", bundle.oob.out_of_bounds_ratio)

    # Coverage
    for name in (
        "signing_relevant_body_landmark_coverage_ratio",
        "full_body_landmark_coverage_ratio",
        "left_hand_landmark_coverage_ratio",
        "right_hand_landmark_coverage_ratio",
        "any_hand_landmark_coverage_ratio",
        "face_landmark_coverage_ratio",
    ):
        _check_ratio(f"coverage.{name}", getattr(bundle.coverage, name))

    # Hand
    _check_count(
        "hand.whole_clip_left_hand_available_frame_count",
        bundle.hand.whole_clip_left_hand_available_frame_count,
    )
    _check_count(
        "hand.whole_clip_right_hand_available_frame_count",
        bundle.hand.whole_clip_right_hand_available_frame_count,
    )
    _check_count(
        "hand.whole_clip_any_hand_available_frame_count",
        bundle.hand.whole_clip_any_hand_available_frame_count,
    )
    _check_count(
        "hand.active_window_any_hand_available_frame_count",
        bundle.hand.active_window_any_hand_available_frame_count,
    )
    _check_ratio(
        "hand.whole_clip_left_hand_available_frame_ratio",
        bundle.hand.whole_clip_left_hand_available_frame_ratio,
    )
    _check_ratio(
        "hand.whole_clip_right_hand_available_frame_ratio",
        bundle.hand.whole_clip_right_hand_available_frame_ratio,
    )
    _check_ratio(
        "hand.whole_clip_any_hand_available_frame_ratio",
        bundle.hand.whole_clip_any_hand_available_frame_ratio,
    )
    _check_ratio(
        "hand.active_window_any_hand_available_frame_ratio",
        bundle.hand.active_window_any_hand_available_frame_ratio,
    )
    _check_count(
        "hand.max_active_window_any_hand_unavailable_run_count",
        bundle.hand.max_active_window_any_hand_unavailable_run_count,
    )
    _check_ratio(
        "hand.max_active_window_any_hand_unavailable_run_ratio",
        bundle.hand.max_active_window_any_hand_unavailable_run_ratio,
    )

    if bundle.hand.whole_clip_any_hand_available_frame_count < max(
        bundle.hand.whole_clip_left_hand_available_frame_count,
        bundle.hand.whole_clip_right_hand_available_frame_count,
    ):
        _add(
            "invalid_any_hand_availability",
            "whole_clip_any_hand_available_frame_count must be >= max of left/right counts",
        )
    if bundle.hand.whole_clip_any_hand_available_frame_count > bundle.length.num_frames:
        _add(
            "invalid_any_hand_availability",
            "whole_clip_any_hand_available_frame_count exceeds num_frames",
        )
    if (
        bundle.hand.active_window_any_hand_available_frame_count
        > bundle.analysis_window.frame_count
    ):
        _add(
            "invalid_any_hand_availability",
            "active_window_any_hand_available_frame_count exceeds analysis-window frames",
        )
    if (
        bundle.hand.max_active_window_any_hand_unavailable_run_count
        > bundle.analysis_window.frame_count
    ):
        _add(
            "invalid_any_hand_availability",
            "max_active_window_any_hand_unavailable_run_count exceeds analysis-window frames",
        )

    # Face
    _check_count("face.face_available_frame_count", bundle.face.face_available_frame_count)
    _check_count("face.face_unavailable_frame_count", bundle.face.face_unavailable_frame_count)
    _check_ratio("face.face_available_frame_ratio", bundle.face.face_available_frame_ratio)
    _check_ratio("face.face_unavailable_frame_ratio", bundle.face.face_unavailable_frame_ratio)

    if (
        bundle.face.face_available_frame_count + bundle.face.face_unavailable_frame_count
        != bundle.length.num_frames
    ):
        _add("face_frame_mismatch", "face available + unavailable must equal num_frames")

    # Valid
    _check_count("valid.valid_frame_count", bundle.valid.valid_frame_count)
    _check_count("valid.invalid_frame_count", bundle.valid.invalid_frame_count)
    _check_ratio("valid.valid_frame_ratio", bundle.valid.valid_frame_ratio)
    _check_ratio("valid.invalid_frame_ratio", bundle.valid.invalid_frame_ratio)
    _check_count(
        "valid.zeroed_canonical_joint_frame_count", bundle.valid.zeroed_canonical_joint_frame_count
    )
    _check_ratio(
        "valid.zeroed_canonical_joint_frame_ratio", bundle.valid.zeroed_canonical_joint_frame_ratio
    )

    if (
        bundle.valid.valid_frame_count + bundle.valid.invalid_frame_count
        != bundle.length.num_frames
    ):
        _add("valid_frame_mismatch", "valid + invalid must equal num_frames")

    if bundle.valid.zeroed_canonical_joint_frame_count > bundle.length.num_frames:
        _add("zeroed_frame_exceeds_total", "zeroed canonical joint frame count exceeds num_frames")

    # Confidence
    for name in [
        "body_available_mean_confidence",
        "active_window_left_hand_available_mean_confidence",
        "active_window_right_hand_available_mean_confidence",
        "active_window_any_hand_available_mean_confidence",
        "face_available_mean_confidence",
        "overall_available_mean_confidence",
        "body_nonzero_confidence_ratio",
        "left_hand_nonzero_confidence_ratio",
        "right_hand_nonzero_confidence_ratio",
        "face_nonzero_confidence_ratio",
        "overall_nonzero_confidence_ratio",
    ]:
        val = getattr(bundle.confidence, name)
        _check_ratio(f"confidence.{name}", val)

    # Text
    if bundle.text.normalized_text != " ".join(bundle.text.normalized_text.split()):
        _add("text_not_normalized", "normalized_text contains unnormalized whitespace")
    _check_count("text.character_count", bundle.text.character_count)
    _check_count("text.token_count", bundle.text.token_count)

    # Length
    _check_count("length.num_frames", bundle.length.num_frames)

    if bundle.length.duration_seconds is not None:
        if not math.isfinite(bundle.length.duration_seconds) or bundle.length.duration_seconds <= 0:
            _add("invalid_duration", "duration_seconds must be positive and finite")

    if bundle.length.frames_per_token is not None:
        if not math.isfinite(bundle.length.frames_per_token) or bundle.length.frames_per_token <= 0:
            _add("invalid_frames_per_token", "frames_per_token must be positive and finite")

    if bundle.length.frames_per_character is not None:
        if (
            not math.isfinite(bundle.length.frames_per_character)
            or bundle.length.frames_per_character <= 0
        ):
            _add("invalid_frames_per_character", "frames_per_character must be positive and finite")

    return issues
