"""Typed models for sample payloads and manifest entries."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from text_to_sign_production.core.ids import SampleSplit as _SampleSplit
from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.legacy_data._shared.types import (
    JsonValue,
    SevereValidationIssue,
)
from text_to_sign_production.legacy_data.samples.schema import PROCESSED_SCHEMA_VERSION


class DroppedDebugMaterializationOutcome(enum.StrEnum):
    """Attempt outcome for optional dropped-sample debug payload materialization."""

    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


SampleValidationIssue = SevereValidationIssue


@runtime_checkable
class ArrayLike(Protocol):
    """Structural type for array-like pose data with an inspectable shape.

    Any object exposing a ``shape`` attribute (e.g. numpy arrays, PyTorch
    tensors, or custom wrappers) satisfies this protocol.  This keeps the
    contract explicit without coupling to a specific array library.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...


@dataclass(slots=True)
class PoseChannelPayload:
    """A single canonical pose channel carrying coordinates and confidence data.

    This record models per-sample data only.  Schema-level metadata such as
    expected joint counts and coordinate dimensionality are owned by
    ``pose.schema`` and enforced by validation, not embedded in every payload
    record.
    """

    coordinates: ArrayLike
    confidence: ArrayLike

    def to_record(self) -> dict[str, object]:
        """Convert the channel to a plain payload mapping."""
        return {
            "coordinates": self.coordinates,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class BfhPosePayload:
    """Canonical body, face, and hands payload structure.

    All four channels are canonical, first-class members of a processed sample.
    No channel is modeled as optional at the contract level.
    """

    body: PoseChannelPayload
    left_hand: PoseChannelPayload
    right_hand: PoseChannelPayload
    face: PoseChannelPayload

    def to_record(self) -> dict[str, dict[str, object]]:
        """Convert the pose payload to a plain channel mapping."""
        return {
            "body": self.body.to_record(),
            "left_hand": self.left_hand.to_record(),
            "right_hand": self.right_hand.to_record(),
            "face": self.face.to_record(),
        }


@dataclass(slots=True)
class SelectedPersonMetadata:
    """Metadata about the selected signer/person in a processed sample."""

    index: int
    multi_person_frame_count: int
    max_people_per_frame: int

    def to_record(self) -> dict[str, int]:
        """Convert selected-person metadata to a plain mapping."""
        return asdict(self)


@dataclass(slots=True)
class FrameQualitySummary:
    """Frame-level quality facts used by gates, metrics, and tiers."""

    valid_frame_count: int
    invalid_frame_count: int
    face_missing_frame_count: int
    out_of_bounds_coordinate_count: int
    frames_with_any_zeroed_canonical_joint: int
    tracked_target_missing_frame_count: int
    tracked_target_missing_frame_ratio: float
    person_tracking_continuity_break_count: int
    person_tracking_continuity_break_ratio: float
    person_tracking_reanchor_count: int
    person_tracking_reanchor_ratio: float
    frame_issue_counts: dict[str, int] = field(default_factory=dict)
    channel_nonzero_frames: dict[str, int] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Convert frame-quality facts to a plain mapping."""
        return asdict(self)


@dataclass(slots=True)
class ProcessedSamplePayload:
    """Canonical processed sample payload."""

    sample_id: str
    text: str
    split: _SampleSplit
    num_frames: int
    fps: float | None
    selected_person: SelectedPersonMetadata
    frame_quality: FrameQualitySummary
    pose: BfhPosePayload
    people_per_frame: ArrayLike | None = None
    frame_valid_mask: ArrayLike | None = None
    schema_version: str = PROCESSED_SCHEMA_VERSION

    def to_record(self) -> dict[str, Any]:
        """Convert the payload to a plain mapping."""
        return {
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "text": self.text,
            "split": self.split.value,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "selected_person": self.selected_person.to_record(),
            "frame_quality": self.frame_quality.to_record(),
            "pose": self.pose.to_record(),
        }


@dataclass(slots=True, kw_only=True)
class PassedManifestEntry:
    """A manifest entry for a sample that has passed all structural gates."""

    sample_id: str
    text: str
    split: _SampleSplit
    num_frames: int
    fps: float | None
    sample_path: str

    # Source tracking facts
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str

    # Payload-derived facts needed by downstream gates/metrics/tiers.
    selected_person: SelectedPersonMetadata
    frame_quality: FrameQualitySummary

    schema_version: str = PROCESSED_SCHEMA_VERSION
    status: SampleStatus = SampleStatus.PASSED

    def to_record(self) -> dict[str, Any]:
        """Convert entry to a serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "text": self.text,
            "split": self.split.value,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "sample_path": self.sample_path,
            "source_video_id": self.source_video_id,
            "source_sentence_id": self.source_sentence_id,
            "source_sentence_name": self.source_sentence_name,
            "selected_person": self.selected_person.to_record(),
            "frame_quality": self.frame_quality.to_record(),
        }


@dataclass(slots=True, kw_only=True)
class DroppedMaterializationLifecycle:
    """Lifecycle facts for optional dropped-sample debug payload materialization."""

    debug_materialization_eligible: bool
    debug_materialization_attempted: bool
    debug_materialization_outcome: DroppedDebugMaterializationOutcome
    payload_path: str | None = None
    payload_exists: bool = False
    archive_publishable: bool = False
    failure_reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Convert lifecycle facts to a serializable dictionary."""
        return {
            "debug_materialization_eligible": self.debug_materialization_eligible,
            "debug_materialization_attempted": self.debug_materialization_attempted,
            "debug_materialization_outcome": self.debug_materialization_outcome.value,
            "payload_path": self.payload_path,
            "payload_exists": self.payload_exists,
            "archive_publishable": self.archive_publishable,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True, kw_only=True)
class DroppedManifestEntry:
    """A manifest entry for a sample that was rejected."""

    sample_id: str
    split: _SampleSplit

    # Why it was dropped.
    drop_stage: str
    drop_reasons: tuple[str, ...]

    # Debug payload materialization is explicit and separate from dropped status.
    materialization: DroppedMaterializationLifecycle
    drop_details: dict[str, JsonValue] = field(default_factory=dict)

    # Partial sample facts are present only when meaningful.
    text: str | None = None
    num_frames: int | None = None
    fps: float | None = None
    selected_person: SelectedPersonMetadata | None = None
    frame_quality: FrameQualitySummary | None = None

    schema_version: str = PROCESSED_SCHEMA_VERSION
    status: SampleStatus = SampleStatus.DROPPED

    def to_record(self) -> dict[str, Any]:
        """Convert entry to a serializable dictionary."""
        record: dict[str, Any] = {
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "split": self.split.value,
            "drop_stage": self.drop_stage,
            "drop_reasons": list(self.drop_reasons),
            "materialization": self.materialization.to_record(),
            "drop_details": dict(self.drop_details),
        }
        if self.text is not None:
            record["text"] = self.text
        if self.num_frames is not None:
            record["num_frames"] = self.num_frames
        if self.fps is not None:
            record["fps"] = self.fps
        if self.selected_person is not None:
            record["selected_person"] = self.selected_person.to_record()
        if self.frame_quality is not None:
            record["frame_quality"] = self.frame_quality.to_record()
        return record


ManifestEntry: TypeAlias = PassedManifestEntry | DroppedManifestEntry


@dataclass(frozen=True, slots=True)
class SampleManifestStatusCountRecord:
    """Count of manifest entries by sample status."""

    status: SampleStatus
    count: int


@dataclass(frozen=True, slots=True)
class SampleSplitStatusCountRecord:
    """Split-aware count of manifest entries by sample status."""

    split: _SampleSplit
    status: SampleStatus
    count: int


@dataclass(frozen=True, slots=True)
class SampleProcessingSummaryRecord:
    """Split-level passed/dropped processing counts."""

    split: _SampleSplit
    passed_count: int
    dropped_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class SampleSplitCountRecord:
    """Count of manifest entries by split."""

    split: _SampleSplit
    count: int


@dataclass(frozen=True, slots=True)
class SampleDroppedMaterializationSummaryRecord:
    """Split-level dropped-sample materialization lifecycle counts."""

    split: _SampleSplit
    dropped_count: int
    debug_materialization_eligible_count: int
    debug_materialization_not_eligible_count: int
    debug_materialization_attempted_count: int
    debug_materialization_not_attempted_count: int
    debug_materialization_succeeded_count: int
    debug_materialization_failed_count: int
    dropped_payload_exists_count: int
    archive_publishable_count: int


@dataclass(frozen=True, slots=True)
class SampleDroppedMaterializationOutcomeCountRecord:
    """Count of dropped manifest entries by split and materialization outcome."""

    split: _SampleSplit
    outcome: DroppedDebugMaterializationOutcome
    count: int


@dataclass(frozen=True, slots=True)
class SampleDroppedArchivePublishableCountRecord:
    """Split-level archive-publishable count for materialized dropped payloads."""

    split: _SampleSplit
    dropped_count: int
    archive_publishable_count: int


@dataclass(frozen=True, slots=True)
class SamplePayloadCompletenessRecord:
    """Count of payloads by optional array completeness."""

    has_people_per_frame: bool
    has_frame_valid_mask: bool
    count: int


@dataclass(frozen=True, slots=True)
class SampleNumericDistributionRecord:
    """Numeric distribution for a sample-owned field."""

    surface: str
    field_name: str
    sample_count: int
    missing_count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None
