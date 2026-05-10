"""Core pose-domain contracts for parser, tracking, tensors, and diagnostics."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.gate.sources import SourceCandidate

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.integer[Any]]


class PoseChannel(enum.StrEnum):
    """Canonical OpenPose channels owned by the pose bounded context."""

    BODY = "body"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    FACE = "face"


class CoordinateSpace(enum.StrEnum):
    """Coordinate-space contract for parsed pose truth."""

    NORMALIZED_IMAGE = "normalized_image"


class PoseDiagnosticSeverity(enum.StrEnum):
    """Severity for structured pose diagnostics."""

    ERROR = "error"
    WARNING = "warning"


class PoseDiagnosticCode(enum.StrEnum):
    """Controlled pose diagnostic codes."""

    FRAME_FILE_DIRECTORY_MISSING = "frame_file_directory_missing"
    FRAME_FILE_DIRECTORY_INVALID = "frame_file_directory_invalid"
    FRAME_FILE_COUNT_MISMATCH = "frame_file_count_mismatch"
    FRAME_READ_ERROR = "frame_read_error"
    JSON_DECODE_ERROR = "json_decode_error"
    UNICODE_DECODE_ERROR = "unicode_decode_error"
    TOP_LEVEL_NOT_OBJECT = "top_level_not_object"
    MISSING_TOP_LEVEL_KEY = "missing_top_level_key"
    PEOPLE_NOT_LIST = "people_not_list"
    PEOPLE_EMPTY = "people_empty"
    PERSON_NOT_OBJECT = "person_not_object"
    MISSING_PERSON_KEY = "missing_person_key"
    CHANNEL_NOT_LIST = "channel_not_list"
    UNEXPECTED_CHANNEL_LENGTH = "unexpected_channel_length"
    CHANNEL_NON_NUMERIC = "channel_non_numeric"
    OUT_OF_BOUNDS_COORDINATE = "out_of_bounds_coordinate"
    ZEROED_LANDMARK = "zeroed_landmark"
    FACE_MISSING = "face_missing"
    NO_PEOPLE_DETECTED = "no_people_detected"
    TARGET_PERSON_MISSING = "target_person_missing"
    TRACKING_CONTINUITY_BREAK = "tracking_continuity_break"
    TRACKING_REANCHOR = "tracking_reanchor"
    INVALID_PARSED_FRAME = "invalid_parsed_frame"
    INVALID_TRACKING_STATE = "invalid_tracking_state"
    INVALID_TENSOR_SHAPE = "invalid_tensor_shape"


class PersonSelectionPolicy(enum.StrEnum):
    """Deterministic anchor-person selection policy."""

    OPENPOSE_PRIMARY = "openpose_primary"
    HIGHEST_CANONICAL_SIGNAL = "highest_canonical_signal"


class TrackingSelectionReason(enum.StrEnum):
    """Reason a frame-wise person selection was made or left missing."""

    ANCHOR = "anchor"
    CONTINUITY = "continuity"
    REANCHOR = "reanchor"
    MISSING = "missing"


class PoseValidationIssueCode(enum.StrEnum):
    """Controlled pose validation issue codes."""

    INVALID_FRAME_INDEX = "invalid_frame_index"
    MISSING_POSE_CHANNEL = "missing_pose_channel"
    INVALID_CHANNEL_COORDINATES = "invalid_channel_coordinates"
    INVALID_CHANNEL_CONFIDENCES = "invalid_channel_confidences"
    INVALID_COORDINATE_SPACE = "invalid_coordinate_space"
    INVALID_ANCHOR_INDEX = "invalid_anchor_index"
    TRACKING_FRAME_COUNT_MISMATCH = "tracking_frame_count_mismatch"
    INVALID_TRACKING_FRAME_INDEX = "invalid_tracking_frame_index"
    INVALID_SELECTED_PERSON_INDEX = "invalid_selected_person_index"
    PEOPLE_PER_FRAME_SHAPE = "people_per_frame_shape"
    SELECTED_PERSON_INDICES_SHAPE = "selected_person_indices_shape"
    MISSING_TENSOR_CHANNEL = "missing_tensor_channel"
    INVALID_TENSOR_COORDINATES = "invalid_tensor_coordinates"
    INVALID_TENSOR_CONFIDENCES = "invalid_tensor_confidences"
    INVALID_TENSOR_COORDINATE_SPACE = "invalid_tensor_coordinate_space"
    LISTING_CANDIDATE_DIRECTORY_MISMATCH = "listing_candidate_directory_mismatch"
    MISSING_LISTING_HAS_FILES = "missing_listing_has_files"
    PARSED_FRAME_INDICES_NOT_CONTIGUOUS = "parsed_frame_indices_not_contiguous"
    TRACKING_FRAME_MISSING = "tracking_frame_missing"
    TRACKING_SELECTED_PERSON_OUT_OF_RANGE = "tracking_selected_person_out_of_range"
    TRACKING_MISSING_COUNT_MISMATCH = "tracking_missing_count_mismatch"
    TENSOR_CANDIDATE_FRAME_COUNT_MISMATCH = "tensor_candidate_frame_count_mismatch"
    TENSOR_TRACKING_SELECTION_MISMATCH = "tensor_tracking_selection_mismatch"
    CHANNEL_NONZERO_COUNT_INVALID = "channel_nonzero_count_invalid"


@dataclass(frozen=True, slots=True)
class PoseDiagnostic:
    """Structured pose diagnostic emitted during pose construction."""

    code: PoseDiagnosticCode
    severity: PoseDiagnosticSeverity
    message: str
    frame_index: int | None = None
    person_index: int | None = None
    channel: PoseChannel | None = None


@dataclass(frozen=True, slots=True)
class PoseValidationIssue:
    """Pose-domain validation issue."""

    code: PoseValidationIssueCode
    message: str


@dataclass(frozen=True, slots=True)
class NormalizedLandmarkObservation:
    """One landmark observation in normalized image coordinate space."""

    x: float
    y: float
    confidence: float
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED_IMAGE


@dataclass(frozen=True, slots=True)
class ParsedPoseChannel:
    """Parsed channel truth in normalized image coordinate space."""

    channel: PoseChannel
    coordinates: FloatArray
    confidences: FloatArray
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED_IMAGE


@dataclass(frozen=True, slots=True)
class ParsedPerson:
    """Parsed person truth for one OpenPose frame."""

    channels: dict[PoseChannel, ParsedPoseChannel]
    person_valid: bool
    face_missing: bool
    out_of_bounds_coordinate_count: int
    has_any_zeroed_landmark: bool
    diagnostics: tuple[PoseDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedFrame:
    """Parsed frame truth, with all coordinates normalized to image space."""

    frame_index: int
    source_path: Path | None
    people: tuple[ParsedPerson, ...]
    frame_valid: bool
    diagnostics: tuple[PoseDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class FrameFileListing:
    """Deterministic pose frame file listing resolved from a source candidate."""

    candidate: SourceCandidate
    directory: Path
    files: tuple[Path, ...]
    missing: bool
    diagnostics: tuple[PoseDiagnostic, ...] = ()

    @property
    def frame_count(self) -> int:
        """Number of discovered frame files."""
        return len(self.files)


@dataclass(frozen=True, slots=True)
class PersonSelectionScore:
    """Inspectable anchor selection score for one frame-local person index."""

    person_index: int
    aggregate_canonical_signal: float
    aggregate_body_signal: float
    valid_frame_presence_count: int


@dataclass(frozen=True, slots=True)
class AnchorSelection:
    """Global anchor selection used as the start of frame-wise tracking."""

    anchor_person_index: int
    policy: PersonSelectionPolicy
    fallback_used: bool
    fallback_reason: PoseDiagnosticCode | None
    candidate_scores: tuple[PersonSelectionScore, ...]


@dataclass(frozen=True, slots=True)
class FrameTrackingSelection:
    """Frame-wise selected person truth."""

    frame_index: int
    selected_person_index: int | None
    reason: TrackingSelectionReason
    diagnostics: tuple[PoseDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonTrackingResult:
    """Tracking truth: anchor selection plus frame-wise person resolution."""

    anchor: AnchorSelection
    frame_selections: tuple[FrameTrackingSelection, ...]
    continuity_break_count: int
    reanchor_count: int
    target_missing_frame_count: int
    diagnostics: tuple[PoseDiagnostic, ...] = ()

    @property
    def selected_person_indices(self) -> tuple[int | None, ...]:
        """Frame-wise selected person indices."""
        return tuple(selection.selected_person_index for selection in self.frame_selections)


@dataclass(frozen=True, slots=True)
class PoseBuildInput:
    """Pose build input sourced from data.sources.SourceCandidate."""

    candidate: SourceCandidate
    frame_listing: FrameFileListing
    parsed_frames: tuple[ParsedFrame, ...]
    tracking: PersonTrackingResult


@dataclass(frozen=True, slots=True)
class PoseChannelTensor:
    """Tensorized channel truth."""

    channel: PoseChannel
    coordinates: FloatArray
    confidences: FloatArray
    coordinate_space: CoordinateSpace = CoordinateSpace.NORMALIZED_IMAGE


@dataclass(frozen=True, slots=True)
class PoseTensorOutput:
    """Tensorized pose truth produced from parsed frames and tracking."""

    candidate: SourceCandidate
    channels: dict[PoseChannel, PoseChannelTensor]
    people_per_frame: IntArray
    frame_valid_mask: BoolArray
    selected_person_indices: tuple[int | None, ...]
    channel_nonzero_frame_counts: dict[PoseChannel, int]
    diagnostics: tuple[PoseDiagnostic, ...] = ()


@dataclass(frozen=True, slots=True)
class PoseBuildDiagnostics:
    """Build-level diagnostic summary for pose truth construction."""

    diagnostics: tuple[PoseDiagnostic, ...] = ()
    frame_diagnostic_count: int = 0
    tracking_diagnostic_count: int = 0
    tensor_diagnostic_count: int = 0


@dataclass(frozen=True, slots=True)
class PoseBuildOutput:
    """Final pose bounded-context output."""

    tensors: PoseTensorOutput
    tracking: PersonTrackingResult
    diagnostics: PoseBuildDiagnostics


@dataclass(frozen=True, slots=True)
class ParsedFrameSummary:
    """Lightweight parsed-frame summary for pose inspection."""

    frame_count: int
    valid_frame_count: int
    people_count_total: int
    max_people_per_frame: int


@dataclass(frozen=True, slots=True)
class TrackingSummary:
    """Lightweight tracking summary for pose inspection."""

    frame_count: int
    missing_frame_count: int
    continuity_break_count: int
    reanchor_count: int


@dataclass(frozen=True, slots=True)
class TensorAvailabilitySummary:
    """Lightweight tensor availability summary for pose inspection."""

    frame_count: int
    valid_frame_count: int
    channel_nonzero_frame_counts: dict[PoseChannel, int]


__all__ = [
    "AnchorSelection",
    "BoolArray",
    "CoordinateSpace",
    "FloatArray",
    "FrameFileListing",
    "FrameTrackingSelection",
    "IntArray",
    "NormalizedLandmarkObservation",
    "ParsedFrame",
    "ParsedFrameSummary",
    "ParsedPerson",
    "ParsedPoseChannel",
    "PersonSelectionPolicy",
    "PersonSelectionScore",
    "PersonTrackingResult",
    "PoseBuildDiagnostics",
    "PoseBuildInput",
    "PoseBuildOutput",
    "PoseChannel",
    "PoseChannelTensor",
    "PoseDiagnostic",
    "PoseDiagnosticCode",
    "PoseDiagnosticSeverity",
    "PoseTensorOutput",
    "PoseValidationIssue",
    "PoseValidationIssueCode",
    "TensorAvailabilitySummary",
    "TrackingSelectionReason",
    "TrackingSummary",
]
