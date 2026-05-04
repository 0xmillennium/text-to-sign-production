"""Sample payload and manifest contracts.

This package owns the semantic definitions of what constitutes a valid
processed sample and its corresponding passed/dropped manifest entries.
It does NOT own layout, paths, workflow staging, or rendering generation.
"""

from text_to_sign_production.data.samples.manifests import (
    build_dropped_entry,
    build_passed_entry,
    manifest_entry_from_record,
)
from text_to_sign_production.data.samples.payloads import (
    build_payload,
    payload_from_record,
)
from text_to_sign_production.data.samples.schema import (
    CANONICAL_POSE_CHANNELS,
    POSE_CHANNEL_JOINT_COUNTS,
    POSE_COORDINATE_DIMENSIONS,
    PROCESSED_SCHEMA_VERSION,
    REQUIRED_DROPPED_MANIFEST_KEYS,
    REQUIRED_PASSED_MANIFEST_KEYS,
    REQUIRED_PAYLOAD_KEYS,
)
from text_to_sign_production.data.samples.types import (
    ArrayLike,
    BfhPosePayload,
    DroppedManifestEntry,
    FrameQualitySummary,
    ManifestEntry,
    PassedManifestEntry,
    PoseChannelPayload,
    ProcessedSamplePayload,
    SampleStatus,
    SelectedPersonMetadata,
    ValidationIssue,
)
from text_to_sign_production.data.samples.validate import (
    validate_dropped_entry,
    validate_manifest_record,
    validate_passed_entry,
    validate_payload,
    validate_payload_record,
)

__all__ = (
    "CANONICAL_POSE_CHANNELS",
    "POSE_CHANNEL_JOINT_COUNTS",
    "POSE_COORDINATE_DIMENSIONS",
    "PROCESSED_SCHEMA_VERSION",
    "REQUIRED_DROPPED_MANIFEST_KEYS",
    "REQUIRED_PASSED_MANIFEST_KEYS",
    "REQUIRED_PAYLOAD_KEYS",
    "ArrayLike",
    "BfhPosePayload",
    "DroppedManifestEntry",
    "FrameQualitySummary",
    "ManifestEntry",
    "PassedManifestEntry",
    "PoseChannelPayload",
    "ProcessedSamplePayload",
    "SampleStatus",
    "SelectedPersonMetadata",
    "ValidationIssue",
    "build_dropped_entry",
    "build_passed_entry",
    "build_payload",
    "manifest_entry_from_record",
    "payload_from_record",
    "validate_dropped_entry",
    "validate_manifest_record",
    "validate_passed_entry",
    "validate_payload",
    "validate_payload_record",
)
