"""Typed models for sample payloads and manifest entries."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from text_to_sign_production.data.samples.schema import PROCESSED_SCHEMA_VERSION

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class SampleStatus(enum.StrEnum):
    """The outcome status of a sample after processing and gating."""

    PASSED = "passed"
    DROPPED = "dropped"


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    """A specific issue found during sample validation."""

    severity: str
    code: str
    message: str


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
    expected joint counts, coordinate dimensionality, and required-vs-optional
    policy are owned by ``schema.py`` and enforced by validation, not embedded
    in every payload record.
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
    frames_with_any_zeroed_required_joint: int
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
    split: str
    num_frames: int
    fps: float | None
    selected_person: SelectedPersonMetadata
    frame_quality: FrameQualitySummary
    pose: BfhPosePayload
    schema_version: str = PROCESSED_SCHEMA_VERSION

    def to_record(self) -> dict[str, Any]:
        """Convert the payload to a plain mapping."""
        return {
            "sample_id": self.sample_id,
            "schema_version": self.schema_version,
            "text": self.text,
            "split": self.split,
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
    split: str
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
            "split": self.split,
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
class DroppedManifestEntry:
    """A manifest entry for a sample that was rejected."""

    sample_id: str
    split: str

    # Why it was dropped.
    drop_stage: str
    drop_reasons: tuple[str, ...]

    # Materialized dropped samples are debug artifacts, never passed samples.
    debug_only: bool = False
    sample_path: str | None = None
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
            "split": self.split,
            "drop_stage": self.drop_stage,
            "drop_reasons": list(self.drop_reasons),
            "debug_only": self.debug_only,
            "drop_details": dict(self.drop_details),
        }
        if self.sample_path is not None:
            record["sample_path"] = self.sample_path
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
