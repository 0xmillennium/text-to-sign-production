"""Sample payload and manifest contracts.

This package owns the semantic definitions of what constitutes a valid
processed sample and its corresponding passed/dropped manifest entries.
It does NOT own layout, paths, workflow staging, or rendering generation.
"""

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.legacy_data.samples.analysis import (
    build_manifest_status_count_records,
    build_payload_completeness_records,
    build_sample_dropped_archive_publishable_count_records,
    build_sample_dropped_materialization_outcome_count_records,
    build_sample_dropped_materialization_summary_records,
    build_sample_numeric_distribution_records,
    build_sample_split_count_records,
)
from text_to_sign_production.legacy_data.samples.manifests import (
    ManifestWriteProgressEvent,
    ManifestWriteProgressSink,
    build_dropped_entry,
    build_dropped_materialization_lifecycle,
    build_passed_entry,
    manifest_entry_from_record,
    read_manifest_jsonl,
    write_manifest_jsonl,
)
from text_to_sign_production.legacy_data.samples.payloads import (
    build_payload,
    load_processed_sample_payload,
    payload_from_record,
    write_processed_sample_payload,
)
from text_to_sign_production.legacy_data.samples.schema import (
    PROCESSED_SCHEMA_VERSION,
    REQUIRED_DROPPED_MANIFEST_KEYS,
    REQUIRED_DROPPED_MATERIALIZATION_KEYS,
    REQUIRED_PASSED_MANIFEST_KEYS,
    REQUIRED_PAYLOAD_KEYS,
)
from text_to_sign_production.legacy_data.samples.types import (
    ArrayLike,
    BfhPosePayload,
    DroppedDebugMaterializationOutcome,
    DroppedManifestEntry,
    DroppedMaterializationLifecycle,
    FrameQualitySummary,
    ManifestEntry,
    PassedManifestEntry,
    PoseChannelPayload,
    ProcessedSamplePayload,
    SampleDroppedArchivePublishableCountRecord,
    SampleDroppedMaterializationOutcomeCountRecord,
    SampleDroppedMaterializationSummaryRecord,
    SampleManifestStatusCountRecord,
    SampleNumericDistributionRecord,
    SamplePayloadCompletenessRecord,
    SampleSplitCountRecord,
    SampleValidationIssue,
    SelectedPersonMetadata,
)
from text_to_sign_production.legacy_data.samples.validate import (
    validate_manifest_entry,
    validate_payload,
)

__all__ = (
    "PROCESSED_SCHEMA_VERSION",
    "REQUIRED_DROPPED_MANIFEST_KEYS",
    "REQUIRED_DROPPED_MATERIALIZATION_KEYS",
    "REQUIRED_PASSED_MANIFEST_KEYS",
    "REQUIRED_PAYLOAD_KEYS",
    "ArrayLike",
    "BfhPosePayload",
    "DroppedDebugMaterializationOutcome",
    "DroppedManifestEntry",
    "DroppedMaterializationLifecycle",
    "FrameQualitySummary",
    "ManifestEntry",
    "ManifestWriteProgressEvent",
    "ManifestWriteProgressSink",
    "PassedManifestEntry",
    "PoseChannelPayload",
    "ProcessedSamplePayload",
    "SampleDroppedArchivePublishableCountRecord",
    "SampleDroppedMaterializationOutcomeCountRecord",
    "SampleDroppedMaterializationSummaryRecord",
    "SampleManifestStatusCountRecord",
    "SampleNumericDistributionRecord",
    "SamplePayloadCompletenessRecord",
    "SampleSplitCountRecord",
    "SampleStatus",
    "SelectedPersonMetadata",
    "SampleValidationIssue",
    "build_dropped_entry",
    "build_dropped_materialization_lifecycle",
    "build_manifest_status_count_records",
    "build_passed_entry",
    "build_payload_completeness_records",
    "build_payload",
    "build_sample_dropped_archive_publishable_count_records",
    "build_sample_dropped_materialization_outcome_count_records",
    "build_sample_dropped_materialization_summary_records",
    "build_sample_numeric_distribution_records",
    "build_sample_split_count_records",
    "load_processed_sample_payload",
    "manifest_entry_from_record",
    "payload_from_record",
    "read_manifest_jsonl",
    "validate_manifest_entry",
    "validate_payload",
    "write_processed_sample_payload",
    "write_manifest_jsonl",
)
