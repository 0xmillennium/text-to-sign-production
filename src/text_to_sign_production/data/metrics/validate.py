"""Semantic validation for composed metric bundles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from text_to_sign_production.data._shared.validate import (
    count_within_total,
    counts_sum_to_total,
    is_positive_finite_number,
    is_unit_interval,
)
from text_to_sign_production.data.metrics.types import MetricBundle, MetricValidationIssue

FieldKind = Literal["count", "ratio", "optional_positive"]
Invariant = Callable[[list[MetricValidationIssue], MetricBundle], None]


@dataclass(frozen=True, slots=True)
class FieldRule:
    """Declarative validation rule for one metric field."""

    section_name: str
    field_name: str
    kind: FieldKind
    code: str | None = None
    message: str | None = None


COUNT_FIELDS: tuple[FieldRule, ...] = tuple(
    FieldRule(section, field, "count")
    for section, fields in {
        "active_signing_span": (
            "start_frame_index",
            "end_frame_index_exclusive",
            "frame_count",
            "trimmed_prefix_frame_count",
            "trimmed_suffix_frame_count",
            "sustained_any_hand_evidence_frame_count",
            "sustained_upper_body_evidence_frame_count",
            "sustained_motion_evidence_frame_count",
        ),
        "oob": ("out_of_bounds_coordinate_count", "total_coordinate_slots"),
        "manual_visibility": (
            "whole_clip_left_hand_available_frame_count",
            "whole_clip_right_hand_available_frame_count",
            "whole_clip_any_hand_available_frame_count",
            "active_span_any_hand_available_frame_count",
            "max_active_span_any_hand_unavailable_run_count",
        ),
        "non_manual_visibility": (
            "face_available_frame_count",
            "face_unavailable_frame_count",
            "active_span_face_available_frame_count",
            "max_active_span_face_unavailable_run_count",
        ),
        "valid": (
            "valid_frame_count",
            "invalid_frame_count",
            "zeroed_canonical_joint_frame_count",
        ),
        "text": ("character_count", "token_count"),
        "length": ("num_frames",),
    }.items()
    for field in fields
)

RATIO_FIELDS: tuple[FieldRule, ...] = tuple(
    FieldRule(section, field, "ratio")
    for section, fields in {
        "active_signing_span": (
            "frame_ratio",
            "trimmed_prefix_frame_ratio",
            "trimmed_suffix_frame_ratio",
            "sustained_any_hand_evidence_frame_ratio",
            "sustained_upper_body_evidence_frame_ratio",
            "sustained_motion_evidence_frame_ratio",
        ),
        "oob": ("out_of_bounds_ratio",),
        "upper_body_support": (
            "active_span_upper_body_support_landmark_coverage_ratio",
            "full_clip_upper_body_support_landmark_coverage_ratio",
        ),
        "coverage": (
            "full_body_landmark_coverage_ratio",
            "left_hand_landmark_coverage_ratio",
            "right_hand_landmark_coverage_ratio",
            "any_hand_landmark_coverage_ratio",
            "face_landmark_coverage_ratio",
        ),
        "manual_visibility": (
            "whole_clip_left_hand_available_frame_ratio",
            "whole_clip_right_hand_available_frame_ratio",
            "whole_clip_any_hand_available_frame_ratio",
            "active_span_any_hand_available_frame_ratio",
            "max_active_span_any_hand_unavailable_run_ratio",
        ),
        "non_manual_visibility": (
            "face_available_frame_ratio",
            "face_unavailable_frame_ratio",
            "active_span_face_available_frame_ratio",
            "max_active_span_face_unavailable_run_ratio",
        ),
        "valid": (
            "valid_frame_ratio",
            "invalid_frame_ratio",
            "zeroed_canonical_joint_frame_ratio",
        ),
        "confidence": (
            "active_span_body_available_mean_confidence",
            "full_clip_body_available_mean_confidence",
            "active_span_left_hand_available_mean_confidence",
            "active_span_right_hand_available_mean_confidence",
            "active_span_any_hand_available_mean_confidence",
            "face_available_mean_confidence",
            "overall_available_mean_confidence",
            "body_nonzero_confidence_ratio",
            "left_hand_nonzero_confidence_ratio",
            "right_hand_nonzero_confidence_ratio",
            "face_nonzero_confidence_ratio",
            "overall_nonzero_confidence_ratio",
        ),
        "kinematic_naturalness": (
            "active_span_abrupt_motion_frame_ratio",
            "active_span_discontinuity_frame_ratio",
            "max_active_span_frozen_run_ratio",
        ),
        "tracking_quality": (
            "tracked_target_missing_frame_ratio",
            "person_tracking_continuity_break_ratio",
            "person_tracking_reanchor_ratio",
        ),
        "manual_detail": (
            "active_span_representative_hand_landmark_coverage_ratio",
            "active_span_representative_hand_fingertip_coverage_ratio",
            "active_span_representative_hand_distal_chain_coverage_ratio",
            "max_active_span_representative_hand_detail_dropout_run_ratio",
        ),
        "non_manual_quality": (
            "active_span_face_landmark_coverage_ratio",
            "active_span_upper_face_landmark_coverage_ratio",
            "active_span_lower_face_landmark_coverage_ratio",
            "max_active_span_face_detail_dropout_run_ratio",
            "active_span_manual_face_overlap_frame_ratio",
        ),
        "geometry": (
            "active_span_upper_body_bone_length_outlier_frame_ratio",
            "active_span_representative_hand_bone_length_outlier_frame_ratio",
            "active_span_cross_channel_scale_outlier_frame_ratio",
        ),
    }.items()
    for field in fields
)

OPTIONAL_POSITIVE_FIELDS: tuple[FieldRule, ...] = (
    FieldRule(
        "length",
        "duration_seconds",
        "optional_positive",
        "invalid_duration",
        "duration_seconds must be positive and finite",
    ),
    FieldRule(
        "length",
        "frames_per_token",
        "optional_positive",
        "invalid_frames_per_token",
        "frames_per_token must be positive and finite",
    ),
    FieldRule(
        "length",
        "frames_per_character",
        "optional_positive",
        "invalid_frames_per_character",
        "frames_per_character must be positive and finite",
    ),
)

FIELD_RULES: tuple[FieldRule, ...] = COUNT_FIELDS + RATIO_FIELDS + OPTIONAL_POSITIVE_FIELDS


def validate_metric_bundle(bundle: MetricBundle) -> list[MetricValidationIssue]:
    """Validate that a metric bundle is semantically coherent."""
    issues: list[MetricValidationIssue] = []
    _run_field_rules(issues, bundle)
    _run_invariant_rules(issues, bundle)
    return issues


def _run_field_rules(issues: list[MetricValidationIssue], bundle: MetricBundle) -> None:
    for rule in FIELD_RULES:
        section = getattr(bundle, rule.section_name)
        value = getattr(section, rule.field_name)
        path = f"{rule.section_name}.{rule.field_name}"
        if rule.kind == "count":
            _check_count(issues, path, value)
        elif rule.kind == "ratio":
            _check_ratio(issues, path, value)
        elif rule.kind == "optional_positive":
            _check_optional_positive(issues, value, rule)


def _run_invariant_rules(issues: list[MetricValidationIssue], bundle: MetricBundle) -> None:
    for rule in INVARIANT_RULES:
        rule(issues, bundle)


def _validate_active_signing_span_invariants(
    issues: list[MetricValidationIssue],
    bundle: MetricBundle,
) -> None:
    span = bundle.active_signing_span
    if span.end_frame_index_exclusive > bundle.length.num_frames:
        _add(issues, "invalid_active_signing_span", "active signing span end exceeds num_frames")
    if span.start_frame_index >= span.end_frame_index_exclusive:
        _add(issues, "invalid_active_signing_span", "active signing span start must be before end")
    if span.end_frame_index_exclusive - span.start_frame_index != span.frame_count:
        _add(
            issues,
            "invalid_active_signing_span",
            "active signing span frame_count mismatches bounds",
        )
    if (
        span.trimmed_prefix_frame_count + span.frame_count + span.trimmed_suffix_frame_count
        != bundle.length.num_frames
    ):
        _add(
            issues,
            "invalid_active_signing_span",
            "active signing span partitions must equal num_frames",
        )

    for count_name, message in (
        ("sustained_any_hand_evidence_frame_count", "any-hand evidence exceeds active-span frames"),
        (
            "sustained_upper_body_evidence_frame_count",
            "upper-body evidence exceeds active-span frames",
        ),
        ("sustained_motion_evidence_frame_count", "motion evidence exceeds active-span frames"),
    ):
        if not count_within_total(getattr(span, count_name), span.frame_count):
            _add(issues, "invalid_active_signing_span", message)


def _validate_hand_invariants(
    issues: list[MetricValidationIssue],
    bundle: MetricBundle,
) -> None:
    hand = bundle.manual_visibility
    span = bundle.active_signing_span
    if hand.whole_clip_any_hand_available_frame_count < max(
        hand.whole_clip_left_hand_available_frame_count,
        hand.whole_clip_right_hand_available_frame_count,
    ):
        _add(
            issues,
            "invalid_any_hand_availability",
            "whole_clip_any_hand_available_frame_count must be >= max of left/right counts",
        )
    if hand.whole_clip_any_hand_available_frame_count > bundle.length.num_frames:
        _add(
            issues,
            "invalid_any_hand_availability",
            "whole_clip_any_hand_available_frame_count exceeds num_frames",
        )
    if hand.active_span_any_hand_available_frame_count > span.frame_count:
        _add(
            issues,
            "invalid_any_hand_availability",
            "active_span_any_hand_available_frame_count exceeds active signing span frames",
        )
    if hand.max_active_span_any_hand_unavailable_run_count > span.frame_count:
        _add(
            issues,
            "invalid_any_hand_availability",
            "max_active_span_any_hand_unavailable_run_count exceeds active signing span frames",
        )


def _validate_face_invariants(
    issues: list[MetricValidationIssue],
    bundle: MetricBundle,
) -> None:
    face = bundle.non_manual_visibility
    span = bundle.active_signing_span
    if not counts_sum_to_total(
        face.face_available_frame_count,
        face.face_unavailable_frame_count,
        bundle.length.num_frames,
    ):
        _add(issues, "face_frame_mismatch", "face available + unavailable must equal num_frames")
    if face.active_span_face_available_frame_count > span.frame_count:
        _add(
            issues,
            "invalid_active_span_face_availability",
            "active_span_face_available_frame_count exceeds active signing span frames",
        )
    if face.max_active_span_face_unavailable_run_count > span.frame_count:
        _add(
            issues,
            "invalid_active_span_face_availability",
            "max_active_span_face_unavailable_run_count exceeds active signing span frames",
        )


def _validate_valid_frame_invariants(
    issues: list[MetricValidationIssue],
    bundle: MetricBundle,
) -> None:
    valid = bundle.valid
    if not counts_sum_to_total(
        valid.valid_frame_count,
        valid.invalid_frame_count,
        bundle.length.num_frames,
    ):
        _add(issues, "valid_frame_mismatch", "valid + invalid must equal num_frames")
    if valid.zeroed_canonical_joint_frame_count > bundle.length.num_frames:
        _add(
            issues,
            "zeroed_frame_exceeds_total",
            "zeroed canonical joint frame count exceeds num_frames",
        )


def _validate_text_invariants(
    issues: list[MetricValidationIssue],
    bundle: MetricBundle,
) -> None:
    if bundle.text.normalized_text != " ".join(bundle.text.normalized_text.split()):
        _add(issues, "text_not_normalized", "normalized_text contains unnormalized whitespace")


INVARIANT_RULES: tuple[Invariant, ...] = (
    _validate_active_signing_span_invariants,
    _validate_hand_invariants,
    _validate_face_invariants,
    _validate_valid_frame_invariants,
    _validate_text_invariants,
)


def _check_ratio(issues: list[MetricValidationIssue], name: str, value: float) -> None:
    if not is_unit_interval(value):
        _add(issues, f"invalid_{name}", f"{name} must be in [0, 1], got {value}")


def _check_count(issues: list[MetricValidationIssue], name: str, value: int) -> None:
    if value < 0:
        _add(issues, f"invalid_{name}", f"{name} must be >= 0, got {value}")


def _check_optional_positive(
    issues: list[MetricValidationIssue],
    value: float | None,
    rule: FieldRule,
) -> None:
    if value is not None and not is_positive_finite_number(value):
        _add(issues, str(rule.code), str(rule.message))


def _add(issues: list[MetricValidationIssue], code: str, message: str) -> None:
    issues.append(MetricValidationIssue(code=code, message=message))
