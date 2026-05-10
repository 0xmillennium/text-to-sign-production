"""Type system for PreparedSample-derived quality context."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ContextValidationCode(enum.StrEnum):
    """Controlled validation codes for quality context."""

    ACTIVE_MASK_LENGTH_MISMATCH = "active_mask_length_mismatch"
    ACTIVE_BOUNDS_INVALID = "active_bounds_invalid"
    ACTIVE_COUNT_MISMATCH = "active_count_mismatch"
    ACTIVE_FINAL_MASK_MISMATCH = "active_final_mask_mismatch"
    ACTIVE_TRANSITION_MASK_MISMATCH = "active_transition_mask_mismatch"
    FRAME_CONTEXT_LENGTH_MISMATCH = "frame_context_length_mismatch"
    GEOMETRY_REFERENCE_EMPTY = "geometry_reference_empty"


class ArticulatorSource(enum.StrEnum):
    """Representative articulator source labels."""

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    BODY = "body"


class HandLabel(enum.StrEnum):
    """Representative hand labels."""

    LEFT = "left_hand"
    RIGHT = "right_hand"


class TransitionKind(enum.StrEnum):
    """Adjacent representative transition classification."""

    UNAVAILABLE = "unavailable"
    SOURCE_SWITCH = "source_switch"
    COMPARABLE = "comparable"


@dataclass(frozen=True, slots=True)
class ContextValidationIssue:
    """Structured context validation issue."""

    code: ContextValidationCode
    message: str


@dataclass(frozen=True, slots=True)
class ActiveSpanContext:
    """Active signing span bounds and masks."""

    frame_count: int
    start_frame_index: int
    end_frame_index_exclusive: int
    raw_evidence_mask: tuple[bool, ...]
    stabilized_evidence_mask: tuple[bool, ...]
    bridged_evidence_mask: tuple[bool, ...]
    padded_active_mask: tuple[bool, ...]
    active_frame_mask: tuple[bool, ...]
    active_transition_mask: tuple[bool, ...]
    active_frame_count: int


@dataclass(frozen=True, slots=True)
class RepresentativeArticulatorFrame:
    """Representative articulator context for one frame."""

    frame_index: int
    source: ArticulatorSource | None
    available: bool
    point: tuple[float, float] | None
    transition_from_previous: TransitionKind | None


@dataclass(frozen=True, slots=True)
class RepresentativeArticulatorContext:
    """Stable representative articulator sequence over the sample."""

    frames: tuple[RepresentativeArticulatorFrame, ...]
    comparable_transition_mask: tuple[bool, ...]
    source_switch_transition_mask: tuple[bool, ...]
    unavailable_transition_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class RepresentativeHandFrame:
    """Representative hand context for one frame."""

    frame_index: int
    selected_hand: HandLabel | None
    available: bool
    landmark_support_count: int
    fingertip_support_count: int
    distal_chain_support_count: int
    detail_supported: bool
    dropout_from_previous: bool


@dataclass(frozen=True, slots=True)
class RepresentativeHandContext:
    """Stable representative hand sequence and support context."""

    frames: tuple[RepresentativeHandFrame, ...]


@dataclass(frozen=True, slots=True)
class FaceRegionFrameContext:
    """Face-region support context for one frame."""

    frame_index: int
    active: bool
    whole_face_present_count: int
    whole_face_coverage_fraction: float
    upper_face_present_count: int
    upper_face_coverage_fraction: float
    lower_face_present_count: int
    lower_face_coverage_fraction: float
    whole_face_available: bool
    upper_face_supported: bool
    lower_face_supported: bool
    manual_available: bool
    manual_face_overlap_ready: bool


@dataclass(frozen=True, slots=True)
class FaceRegionContext:
    """Shared face/non-manual region context."""

    frames: tuple[FaceRegionFrameContext, ...]
    active_face_available_mask: tuple[bool, ...]
    active_upper_face_supported_mask: tuple[bool, ...]
    active_lower_face_supported_mask: tuple[bool, ...]
    manual_face_overlap_ready_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class GeometryFrameContext:
    """Per-frame observed geometry support."""

    frame_index: int
    upper_body_segment_lengths: dict[tuple[int, int], float | None]
    representative_hand_segment_lengths: dict[tuple[int, int], float | None]
    body_scale_reference_length: float | None
    hand_scale_reference_length: float | None


@dataclass(frozen=True, slots=True)
class GeometryReferenceContext:
    """Sample-internal robust geometry references."""

    upper_body_segment_references: dict[tuple[int, int], float | None]
    representative_hand_segment_references: dict[tuple[int, int], float | None]
    cross_channel_scale_reference: float | None
    frames: tuple[GeometryFrameContext, ...]


@dataclass(frozen=True, slots=True)
class QualityContext:
    """Composed shared derived context for later quality families."""

    active_span: ActiveSpanContext
    representative_articulator: RepresentativeArticulatorContext
    representative_hand: RepresentativeHandContext
    face: FaceRegionContext
    geometry: GeometryReferenceContext


__all__ = [
    "ActiveSpanContext",
    "ArticulatorSource",
    "ContextValidationCode",
    "ContextValidationIssue",
    "FaceRegionContext",
    "FaceRegionFrameContext",
    "GeometryFrameContext",
    "GeometryReferenceContext",
    "HandLabel",
    "QualityContext",
    "RepresentativeArticulatorContext",
    "RepresentativeArticulatorFrame",
    "RepresentativeHandContext",
    "RepresentativeHandFrame",
    "TransitionKind",
]
