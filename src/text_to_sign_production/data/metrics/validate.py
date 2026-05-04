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
    _check_count("oob.out_of_bounds_coordinate_count", bundle.oob.out_of_bounds_coordinate_count)
    _check_count("oob.total_coordinate_slots", bundle.oob.total_coordinate_slots)
    _check_ratio("oob.out_of_bounds_ratio", bundle.oob.out_of_bounds_ratio)

    # Hand
    _check_count("hand.left_hand_nonzero_frame_count", bundle.hand.left_hand_nonzero_frame_count)
    _check_count("hand.right_hand_nonzero_frame_count", bundle.hand.right_hand_nonzero_frame_count)
    _check_count("hand.any_hand_nonzero_frame_count", bundle.hand.any_hand_nonzero_frame_count)
    _check_ratio("hand.left_hand_nonzero_frame_ratio", bundle.hand.left_hand_nonzero_frame_ratio)
    _check_ratio("hand.right_hand_nonzero_frame_ratio", bundle.hand.right_hand_nonzero_frame_ratio)
    _check_ratio("hand.any_hand_nonzero_frame_ratio", bundle.hand.any_hand_nonzero_frame_ratio)

    if bundle.hand.any_hand_nonzero_frame_count < max(
        bundle.hand.left_hand_nonzero_frame_count, bundle.hand.right_hand_nonzero_frame_count
    ):
        _add(
            "invalid_any_hand_nonzero",
            "any_hand_nonzero_frame_count must be >= max of left/right counts",
        )
    if bundle.hand.any_hand_nonzero_frame_count > bundle.length.num_frames:
        _add("invalid_any_hand_nonzero", "any_hand_nonzero_frame_count exceeds num_frames")

    # Face
    _check_count("face.face_nonzero_frame_count", bundle.face.face_nonzero_frame_count)
    _check_count("face.face_missing_frame_count", bundle.face.face_missing_frame_count)
    _check_ratio("face.face_nonzero_frame_ratio", bundle.face.face_nonzero_frame_ratio)
    _check_ratio("face.face_missing_frame_ratio", bundle.face.face_missing_frame_ratio)

    if (
        bundle.face.face_nonzero_frame_count + bundle.face.face_missing_frame_count
        != bundle.length.num_frames
    ):
        _add("face_frame_mismatch", "face nonzero + missing must equal num_frames")

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
        "body_mean_confidence",
        "left_hand_mean_confidence",
        "right_hand_mean_confidence",
        "face_mean_confidence",
        "overall_mean_confidence",
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
