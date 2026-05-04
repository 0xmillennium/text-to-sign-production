"""Sample payload and manifest contracts.

This package owns the semantic definitions of what constitutes a valid
processed sample and its corresponding passed/dropped manifest entries.
It does NOT own layout, paths, workflow staging, or rendering generation.
"""

from text_to_sign_production.data.samples.manifests import (
    build_dropped_entry,
    build_passed_entry,
    manifest_entry_from_record,
    write_manifest_jsonl,
)
from text_to_sign_production.data.samples.payloads import (
    build_payload,
    load_processed_sample_payload,
    payload_from_record,
    write_processed_sample_payload,
)
from text_to_sign_production.data.samples.schema import (
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
    SampleValidationIssue,
    SelectedPersonMetadata,
)
from text_to_sign_production.data.samples.validate import (
    validate_dropped_entry,
    validate_manifest_record,
    validate_passed_entry,
    validate_payload,
    validate_payload_record,
)

__all__ = (
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
    "SampleValidationIssue",
    "build_dropped_entry",
    "build_passed_entry",
    "build_payload",
    "load_processed_sample_payload",
    "manifest_entry_from_record",
    "payload_from_record",
    "validate_dropped_entry",
    "validate_manifest_record",
    "validate_passed_entry",
    "validate_payload",
    "validate_payload_record",
    "write_processed_sample_payload",
    "write_manifest_jsonl",
)
