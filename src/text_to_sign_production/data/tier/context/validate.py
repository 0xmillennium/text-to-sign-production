"""Validation for shared quality context."""

from __future__ import annotations

from text_to_sign_production.data.tier.context.types import (
    ActiveSpanContext,
    ContextValidationCode,
    ContextValidationIssue,
    FaceRegionContext,
    GeometryReferenceContext,
    QualityContext,
    RepresentativeArticulatorContext,
    RepresentativeHandContext,
)


def _issue(code: ContextValidationCode, message: str) -> ContextValidationIssue:
    return ContextValidationIssue(code=code, message=message)


def validate_active_span_context(context: ActiveSpanContext) -> tuple[ContextValidationIssue, ...]:
    """Validate active-span context structure and invariants."""
    issues: list[ContextValidationIssue] = []
    if len(context.active_frame_mask) != context.frame_count:
        issues.append(
            _issue(
                ContextValidationCode.ACTIVE_MASK_LENGTH_MISMATCH,
                "Active frame mask length must match frame count.",
            )
        )
    for name, mask in (
        ("raw evidence", context.raw_evidence_mask),
        ("stabilized evidence", context.stabilized_evidence_mask),
        ("bridged evidence", context.bridged_evidence_mask),
        ("padded active", context.padded_active_mask),
    ):
        if len(mask) != context.frame_count:
            issues.append(
                _issue(
                    ContextValidationCode.ACTIVE_MASK_LENGTH_MISMATCH,
                    f"{name} mask length must match frame count.",
                )
            )
    if context.active_frame_mask != context.padded_active_mask:
        issues.append(
            _issue(
                ContextValidationCode.ACTIVE_FINAL_MASK_MISMATCH,
                "Active frame mask must equal the final padded active mask.",
            )
        )
    expected_transition_count = max(0, context.frame_count - 1)
    if len(context.active_transition_mask) != expected_transition_count:
        issues.append(
            _issue(
                ContextValidationCode.ACTIVE_MASK_LENGTH_MISMATCH,
                "Active transition mask length must be frame_count - 1.",
            )
        )
    elif context.active_transition_mask != tuple(
        left and right
        for left, right in zip(
            context.active_frame_mask[:-1],
            context.active_frame_mask[1:],
            strict=True,
        )
    ):
        issues.append(
            _issue(
                ContextValidationCode.ACTIVE_TRANSITION_MASK_MISMATCH,
                "Active transition mask must be derived from adjacent active-frame truth.",
            )
        )
    if (
        not 0
        <= context.start_frame_index
        <= context.end_frame_index_exclusive
        <= context.frame_count
    ):
        issues.append(
            _issue(
                ContextValidationCode.ACTIVE_BOUNDS_INVALID,
                "Active span bounds must be inside frame range.",
            )
        )
    else:
        active_indices = tuple(
            index for index, active in enumerate(context.active_frame_mask) if active
        )
        expected_start = active_indices[0] if active_indices else 0
        expected_end = active_indices[-1] + 1 if active_indices else 0
        if (
            context.start_frame_index != expected_start
            or context.end_frame_index_exclusive != expected_end
        ):
            issues.append(
                _issue(
                    ContextValidationCode.ACTIVE_BOUNDS_INVALID,
                    "Active span bounds must be derived from the final active-frame mask.",
                )
            )
    if context.active_frame_count != sum(1 for value in context.active_frame_mask if value):
        issues.append(
            _issue(
                ContextValidationCode.ACTIVE_COUNT_MISMATCH,
                "Active frame count must equal active mask population.",
            )
        )
    return tuple(issues)


def validate_representative_articulator_context(
    context: RepresentativeArticulatorContext,
    *,
    frame_count: int,
) -> tuple[ContextValidationIssue, ...]:
    """Validate representative-articulator context shape."""
    issues: list[ContextValidationIssue] = []
    if len(context.frames) != frame_count:
        issues.append(
            _issue(
                ContextValidationCode.FRAME_CONTEXT_LENGTH_MISMATCH,
                "Representative articulator frame count mismatch.",
            )
        )
    expected_transition_count = max(0, frame_count - 1)
    for name, mask in (
        ("comparable", context.comparable_transition_mask),
        ("source_switch", context.source_switch_transition_mask),
        ("unavailable", context.unavailable_transition_mask),
    ):
        if len(mask) != expected_transition_count:
            issues.append(
                _issue(
                    ContextValidationCode.FRAME_CONTEXT_LENGTH_MISMATCH,
                    f"{name} transition mask length mismatch.",
                )
            )
    return tuple(issues)


def validate_representative_hand_context(
    context: RepresentativeHandContext,
    *,
    frame_count: int,
) -> tuple[ContextValidationIssue, ...]:
    """Validate representative-hand context shape."""
    if len(context.frames) != frame_count:
        return (
            _issue(
                ContextValidationCode.FRAME_CONTEXT_LENGTH_MISMATCH,
                "Representative hand frame count mismatch.",
            ),
        )
    return ()


def validate_face_region_context(
    context: FaceRegionContext,
    *,
    frame_count: int,
) -> tuple[ContextValidationIssue, ...]:
    """Validate face-region context shape."""
    issues: list[ContextValidationIssue] = []
    if len(context.frames) != frame_count:
        issues.append(
            _issue(
                ContextValidationCode.FRAME_CONTEXT_LENGTH_MISMATCH,
                "Face-region frame count mismatch.",
            ),
        )
    for name, mask in (
        ("active face", context.active_face_available_mask),
        ("active upper face", context.active_upper_face_supported_mask),
        ("active lower face", context.active_lower_face_supported_mask),
        ("manual face overlap", context.manual_face_overlap_ready_mask),
    ):
        if len(mask) != frame_count:
            issues.append(
                _issue(
                    ContextValidationCode.FRAME_CONTEXT_LENGTH_MISMATCH,
                    f"{name} mask length mismatch.",
                )
            )
    return tuple(issues)


def validate_geometry_reference_context(
    context: GeometryReferenceContext,
    *,
    frame_count: int,
) -> tuple[ContextValidationIssue, ...]:
    """Validate geometry-reference context shape."""
    issues: list[ContextValidationIssue] = []
    if len(context.frames) != frame_count:
        issues.append(
            _issue(
                ContextValidationCode.FRAME_CONTEXT_LENGTH_MISMATCH,
                "Geometry frame context count mismatch.",
            )
        )
    if not context.upper_body_segment_references:
        issues.append(
            _issue(
                ContextValidationCode.GEOMETRY_REFERENCE_EMPTY,
                "Upper-body geometry references must be present.",
            )
        )
    if not context.representative_hand_segment_references:
        issues.append(
            _issue(
                ContextValidationCode.GEOMETRY_REFERENCE_EMPTY,
                "Representative-hand geometry references must be present.",
            )
        )
    return tuple(issues)


def validate_quality_context(context: QualityContext) -> tuple[ContextValidationIssue, ...]:
    """Validate composed quality context structure and invariants."""
    issues: list[ContextValidationIssue] = []
    frame_count = context.active_span.frame_count
    issues.extend(validate_active_span_context(context.active_span))
    issues.extend(
        validate_representative_articulator_context(
            context.representative_articulator,
            frame_count=frame_count,
        )
    )
    issues.extend(
        validate_representative_hand_context(
            context.representative_hand,
            frame_count=frame_count,
        )
    )
    issues.extend(validate_face_region_context(context.face, frame_count=frame_count))
    issues.extend(
        validate_geometry_reference_context(
            context.geometry,
            frame_count=frame_count,
        )
    )
    return tuple(issues)


__all__ = [
    "validate_active_span_context",
    "validate_face_region_context",
    "validate_geometry_reference_context",
    "validate_quality_context",
    "validate_representative_articulator_context",
    "validate_representative_hand_context",
]
