"""Dataset-layer secondary contracts for production, validation, and analysis."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    DroppedSample,
    PassedManifestEntry,
    PreparedSample,
)

ManifestEntry: TypeAlias = PassedManifestEntry | DroppedManifestEntry


@dataclass(frozen=True, slots=True)
class DatasetPayloadProduction:
    """Dataset-owned prepared sample payload write plan/result."""

    sample: PreparedSample
    path: Path
    payload_ref: str
    status: SampleStatus


@dataclass(frozen=True, slots=True)
class DatasetDroppedSampleProduction:
    """Dataset-owned DroppedSample JSON payload write plan/result."""

    sample: DroppedSample
    path: Path
    payload_ref: str


class DatasetValidationIssueCode(enum.StrEnum):
    """Stable dataset-layer validation issue codes."""

    EMPTY_SAMPLE_ID = "empty_sample_id"
    EMPTY_TEXT = "empty_text"
    INVALID_SCHEMA_VERSION = "invalid_schema_version"
    INVALID_FRAME_COUNT = "invalid_frame_count"
    INVALID_FPS = "invalid_fps"
    MISSING_SOURCE_TRUTH = "missing_source_truth"
    INVALID_POSE_ARRAY_SHAPE = "invalid_pose_array_shape"
    INVALID_POSE_ARRAY_DTYPE = "invalid_pose_array_dtype"
    INVALID_COORDINATE_SPACE = "invalid_coordinate_space"
    INVALID_FRAME_MASK = "invalid_frame_mask"
    INVALID_TRACKING_SUMMARY = "invalid_tracking_summary"
    INVALID_CHANNEL_COUNTS = "invalid_channel_counts"
    INVALID_MANIFEST_ENTRY = "invalid_manifest_entry"
    MANIFEST_PAYLOAD_MISMATCH = "manifest_payload_mismatch"
    DUPLICATE_ISSUE_CODE = "duplicate_issue_code"


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    """Structured dataset-layer validation issue."""

    code: DatasetValidationIssueCode
    message: str
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedSampleSummary:
    """Compact observation of a PreparedSample payload."""

    sample_id: str
    split: str
    frame_count: int
    valid_frame_count: int
    source_complete: bool
    pose_complete: bool
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class PassedManifestSummary:
    """Compact observation of passed manifest rows."""

    entry_count: int
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class DroppedManifestSummary:
    """Compact observation of dropped manifest rows."""

    entry_count: int
    entries_with_issue_codes_count: int
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class CheckpointHandoffSummary:
    """Compact payload/manifest handoff integrity observation."""

    payload_count: int
    passed_count: int
    dropped_count: int
    coherent_passed_count: int
    coherence_issue_count: int


__all__ = [
    "CheckpointHandoffSummary",
    "DatasetPayloadProduction",
    "DatasetDroppedSampleProduction",
    "DatasetValidationIssue",
    "DatasetValidationIssueCode",
    "DroppedManifestSummary",
    "ManifestEntry",
    "PassedManifestSummary",
    "PreparedSampleSummary",
]
