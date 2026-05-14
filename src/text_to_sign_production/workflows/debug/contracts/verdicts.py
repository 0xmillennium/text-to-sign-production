from __future__ import annotations

import enum


class TargetResolutionStatus(enum.StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    INVALID_REQUEST = "invalid_request"


class DebugRestoreOperationStatus(enum.StrEnum):
    SUCCEEDED = "succeeded"
    FAILED_REQUIRED = "failed_required"
    FAILED_OPTIONAL = "failed_optional"
    SKIPPED_OPTIONAL = "skipped_optional"


class DebugPublishOperationStatus(enum.StrEnum):
    PUBLISHED = "published"
    SKIPPED_MISSING_SOURCE = "skipped_missing_source"
    FAILED = "failed"


class DebugPublishVerificationStatus(enum.StrEnum):
    VERIFIED = "verified"
    VERIFIED_WITH_WARNINGS = "verified_with_warnings"
    FAILED = "failed"


class DebugLeakageContextMode(enum.StrEnum):
    SINGLE_SPLIT_CONTEXT = "single_split_context"
    MULTI_SPLIT_CONTEXT = "multi_split_context"
    GLOBAL_SPLIT_CONTEXT = "global_split_context"


class DebugQualityMetricStatus(enum.StrEnum):
    OBSERVED = "observed"


class ExistingGateStatus(enum.StrEnum):
    PASSED = "passed"
    DROPPED = "dropped"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    CORRUPT = "corrupt"


class ExistingTierMembershipState(enum.StrEnum):
    INCLUDED = "included"
    EXCLUDED = "excluded"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"


class DebugGateVerdict(enum.StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    COMPUTE_FAILED = "compute_failed"
    EXISTING_MISSING = "existing_missing"
    EXISTING_AMBIGUOUS = "existing_ambiguous"


class DebugTierVerdict(enum.StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    CONTEXT_LIMITED_DIFFERENCE = "context_limited_difference"
    NOT_APPLICABLE = "not_applicable"
    COMPUTE_FAILED = "compute_failed"
    EXISTING_MISSING = "existing_missing"
    EXISTING_AMBIGUOUS = "existing_ambiguous"


class DebugVisualizationVerdict(enum.StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class DebugFinalVerdict(enum.StrEnum):
    PASS = "pass"
    PASS_WITH_CONTEXT_WARNING = "pass_with_context_warning"
    PASS_WITH_VISUALIZATION_WARNING = "pass_with_visualization_warning"
    PASS_WITH_REPORT_WARNING = "pass_with_report_warning"
    PASS_WITH_PUBLISH_WARNING = "pass_with_publish_warning"
    FAIL_TARGET_NOT_FOUND = "fail_target_not_found"
    FAIL_TARGET_AMBIGUOUS = "fail_target_ambiguous"
    FAIL_RUNTIME_RESTORE = "fail_runtime_restore"
    FAIL_GATE_MISMATCH = "fail_gate_mismatch"
    FAIL_TIER_MISMATCH = "fail_tier_mismatch"
    PARTIAL_DEBUG = "partial_debug"


__all__ = [
    "DebugFinalVerdict",
    "DebugGateVerdict",
    "DebugLeakageContextMode",
    "DebugPublishOperationStatus",
    "DebugPublishVerificationStatus",
    "DebugQualityMetricStatus",
    "DebugRestoreOperationStatus",
    "DebugTierVerdict",
    "DebugVisualizationVerdict",
    "ExistingGateStatus",
    "ExistingTierMembershipState",
    "TargetResolutionStatus",
]
