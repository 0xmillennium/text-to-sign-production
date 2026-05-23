"""Generated-pose artifact contracts and IO."""

from __future__ import annotations

from text_to_sign_production.modeling.artifacts.generated_pose import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    GeneratedPoseSample,
)
from text_to_sign_production.modeling.artifacts.generated_pose_io import (
    DiagnosticGeneratedPosePaths,
    GeneratedPoseSplitWriteResult,
    GeneratedPoseStreamWriteResult,
    diagnostic_generated_pose_paths,
    load_generated_pose_payload,
    read_generated_pose_manifest_jsonl,
    validate_diagnostic_generated_pose_paths,
    write_generated_pose_manifest_jsonl,
    write_generated_pose_payload,
    write_generated_pose_split,
    write_generated_pose_split_to_explicit_root,
)
from text_to_sign_production.modeling.artifacts.generated_pose_archive import (
    GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_NAME,
    GENERATED_POSE_SAMPLE_ARCHIVE_SCHEMA_VERSION,
    GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME,
    GeneratedPoseSampleArchive,
    GeneratedPoseSampleArchiveError,
    build_generated_pose_samples_archive,
)
from text_to_sign_production.modeling.artifacts.generated_pose_manifest import (
    GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
    GeneratedPoseManifestEntry,
    generated_manifest_entry_from_record,
    generated_manifest_entry_from_sample,
    generated_manifest_entry_to_record,
)
from text_to_sign_production.modeling.artifacts.run_metadata import (
    GeneratedPoseConfidencePolicy,
    GeneratedPoseGenerationMode,
    GeneratedPoseLengthPolicy,
    GeneratedPoseProducerType,
    GeneratedPoseRunMetadata,
)
from text_to_sign_production.modeling.artifacts.validation import (
    GeneratedPoseArtifactError,
    GeneratedPoseValidationIssue,
    validate_generated_pose_manifest_entries,
    validate_generated_pose_manifest_entry,
    validate_generated_pose_sample,
)

__all__ = [
    "GENERATED_POSE_PAYLOAD_SCHEMA_VERSION",
    "GENERATED_POSE_MANIFEST_SCHEMA_VERSION",
    "GENERATED_POSE_CHANNEL_POLICY",
    "GENERATED_POSE_SAMPLE_ARCHIVE_MANIFEST_NAME",
    "GENERATED_POSE_SAMPLE_ARCHIVE_NAME",
    "GENERATED_POSE_SAMPLE_ARCHIVE_SCHEMA_VERSION",
    "GENERATED_POSE_SAMPLE_ARCHIVE_SHA256_NAME",
    "GeneratedPoseSample",
    "GeneratedPoseSampleArchive",
    "GeneratedPoseSampleArchiveError",
    "GeneratedPoseManifestEntry",
    "GeneratedPoseProducerType",
    "GeneratedPoseGenerationMode",
    "GeneratedPoseLengthPolicy",
    "GeneratedPoseConfidencePolicy",
    "GeneratedPoseRunMetadata",
    "DiagnosticGeneratedPosePaths",
    "GeneratedPoseSplitWriteResult",
    "GeneratedPoseStreamWriteResult",
    "GeneratedPoseValidationIssue",
    "GeneratedPoseArtifactError",
    "diagnostic_generated_pose_paths",
    "generated_manifest_entry_from_sample",
    "generated_manifest_entry_to_record",
    "generated_manifest_entry_from_record",
    "write_generated_pose_payload",
    "load_generated_pose_payload",
    "write_generated_pose_manifest_jsonl",
    "read_generated_pose_manifest_jsonl",
    "write_generated_pose_split",
    "write_generated_pose_split_to_explicit_root",
    "build_generated_pose_samples_archive",
    "validate_diagnostic_generated_pose_paths",
    "validate_generated_pose_sample",
    "validate_generated_pose_manifest_entry",
    "validate_generated_pose_manifest_entries",
]
