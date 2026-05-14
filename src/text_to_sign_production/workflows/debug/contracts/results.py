from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleSplit, SampleStatus, TierName
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
    TierDecisionBundle,
    TierStatus,
)
from text_to_sign_production.data.gate.processing.outputs import GateSourceEvaluation
from text_to_sign_production.data.tier.leakages import LeakageSampleSummary
from text_to_sign_production.workflows.debug.contracts.request import DebugSampleRequest
from text_to_sign_production.workflows.debug.contracts.verdicts import (
    DebugFinalVerdict,
    DebugGateVerdict,
    DebugLeakageContextMode,
    DebugPublishOperationStatus,
    DebugPublishVerificationStatus,
    DebugQualityMetricStatus,
    DebugRestoreOperationStatus,
    DebugTierVerdict,
    DebugVisualizationVerdict,
    ExistingGateStatus,
    ExistingTierMembershipState,
    TargetResolutionStatus,
)
from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)


@dataclass(frozen=True, slots=True)
class DebugTranslationMatch:
    split: SampleSplit
    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    start_realigned: float
    end_realigned: float
    sentence: str
    translation_path: Path
    row_index: int


@dataclass(frozen=True, slots=True)
class DebugTargetResolution:
    request: DebugSampleRequest
    status: TargetResolutionStatus
    match: DebugTranslationMatch | None
    matches: tuple[DebugTranslationMatch, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugRestoreOperation:
    group: str
    split: SampleSplit | None
    source: Path
    target: Path
    required: bool
    description: str
    workflow_operation: WorkflowOperation


@dataclass(frozen=True, slots=True)
class DebugRestorePlan:
    debug_splits: tuple[SampleSplit, ...]
    operations: tuple[DebugRestoreOperation, ...]
    required_groups: tuple[str, ...]
    optional_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugIssue:
    code: str
    message: str
    path: Path | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class DebugRestorePlanValidation:
    plan: DebugRestorePlan
    valid: bool
    blocking_errors: tuple[DebugIssue, ...]
    warnings: tuple[DebugIssue, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.blocking_errors)

    @property
    def succeeded(self) -> bool:
        return self.valid


@dataclass(frozen=True, slots=True)
class DebugRestoreOperationResult:
    operation: DebugRestoreOperation
    status: DebugRestoreOperationStatus
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {
            DebugRestoreOperationStatus.SUCCEEDED,
            DebugRestoreOperationStatus.SKIPPED_OPTIONAL,
            DebugRestoreOperationStatus.FAILED_OPTIONAL,
        }


@dataclass(frozen=True, slots=True)
class DebugRestoreResult:
    plan: DebugRestorePlan
    operation_results: tuple[DebugRestoreOperationResult, ...]
    execution: OperationBatchExecutionResult | None

    @property
    def succeeded(self) -> bool:
        return all(result.succeeded for result in self.operation_results)


@dataclass(frozen=True, slots=True)
class DebugRuntimeCheck:
    label: str
    path: Path
    required: bool
    exists: bool
    valid: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class DebugRuntimeVerification:
    checks: tuple[DebugRuntimeCheck, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return not self.errors and all(
            check.exists and check.valid for check in self.checks if check.required
        )


@dataclass(frozen=True, slots=True)
class DebugSourceEvidence:
    keypoint_json_dir: Path
    keypoint_json_dir_exists: bool
    keypoint_frame_json_count: int | None
    raw_video_path: Path
    raw_video_exists: bool
    source_issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugExistingGateEvidence:
    status: ExistingGateStatus
    passed_manifest_path: Path
    dropped_manifest_path: Path
    passed_entry: PassedManifestEntry | None
    dropped_entry: DroppedManifestEntry | None
    passed_payload_path: Path | None
    passed_payload_exists: bool
    dropped_record_path: Path | None
    dropped_record_exists: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugExistingTierEvidence:
    tier: TierName
    state: ExistingTierMembershipState
    included_manifest_path: Path
    excluded_manifest_path: Path
    included_entry: PassedManifestEntry | None
    excluded_entry: PassedManifestEntry | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugSampleDossier:
    resolution: DebugTargetResolution
    translation: DebugTranslationMatch
    source: DebugSourceEvidence
    existing_gate: DebugExistingGateEvidence
    existing_tiers: tuple[DebugExistingTierEvidence, ...]
    consistency_notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugGatePreflight:
    can_compute: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugGateComparison:
    compared: bool
    status_matches: bool | None
    differences: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugGateFrameMetrics:
    frame_count: int | None
    valid_frame_count: int | None
    body_nonzero_frame_count: int | None
    face_nonzero_frame_count: int | None
    left_hand_nonzero_frame_count: int | None
    right_hand_nonzero_frame_count: int | None


@dataclass(frozen=True, slots=True)
class DebugGateResult:
    preflight: DebugGatePreflight
    existing_gate: DebugExistingGateEvidence
    existing_status: ExistingGateStatus
    evaluation: GateSourceEvaluation | None
    computed_status: SampleStatus | None
    computed_gate: GateDecisionBundle | None
    computed_sample: PreparedSample | None
    computed_passed_entry: PassedManifestEntry | None
    computed_dropped_entry: DroppedManifestEntry | None
    frame_metrics: DebugGateFrameMetrics
    comparison: DebugGateComparison
    verdict: DebugGateVerdict
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugLeakageContext:
    debug_splits: tuple[SampleSplit, ...]
    mode: DebugLeakageContextMode
    production_like: bool
    warning: str | None


@dataclass(frozen=True, slots=True)
class DebugTierRecomputeSource:
    payload_path: Path | None
    payload_exists: bool
    manifest_entry: PassedManifestEntry | None
    can_compute: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class DebugTierComputedMembership:
    tier: TierName
    computed_state: ExistingTierMembershipState
    existing_state: ExistingTierMembershipState
    matches: bool


@dataclass(frozen=True, slots=True)
class DebugExistingTierStateSummary:
    tier: TierName
    state: ExistingTierMembershipState
    included_manifest_path: Path
    excluded_manifest_path: Path
    included_entry: PassedManifestEntry | None
    excluded_entry: PassedManifestEntry | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugTierFamilyDecisionSummary:
    family: str
    status: TierStatus
    supported_tiers: tuple[TierName, ...]
    best_supported_tier: TierName | None
    issue_codes: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugTierQualityMetricSummary:
    metric_name: str
    value: str | float | int | bool | None
    status: DebugQualityMetricStatus
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugTierComparison:
    compared: bool
    memberships: tuple[DebugTierComputedMembership, ...]
    differences: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugTierResult:
    applicable: bool
    applicability_reason: str
    leakage_context: DebugLeakageContext
    recompute_source: DebugTierRecomputeSource
    existing_tier_states: tuple[DebugExistingTierStateSummary, ...]
    leakage_summary: LeakageSampleSummary | None
    computed_decision: TierDecisionBundle | None
    family_decisions: tuple[DebugTierFamilyDecisionSummary, ...]
    quality_metrics: tuple[DebugTierQualityMetricSummary, ...]
    comparison: DebugTierComparison
    verdict: DebugTierVerdict
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugVisualArtifact:
    label: str
    path: Path
    created: bool
    metadata_json: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class DebugVisualizationResult:
    output_root: Path
    artifacts: tuple[DebugVisualArtifact, ...]
    verdict: DebugVisualizationVerdict
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugReportResult:
    output_root: Path
    run_id: str
    files: tuple[Path, ...]
    succeeded: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DebugPublishOperation:
    source: Path
    target: Path
    required: bool
    description: str
    workflow_operation: WorkflowOperation


@dataclass(frozen=True, slots=True)
class DebugPublishPlan:
    runtime_output_root: Path
    drive_output_root: Path
    operations: tuple[DebugPublishOperation, ...]
    expected_file_count: int
    blocking_errors: tuple[DebugIssue, ...]
    warnings: tuple[DebugIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.blocking_errors


@dataclass(frozen=True, slots=True)
class DebugPublishOperationResult:
    source: Path
    target: Path
    status: DebugPublishOperationStatus
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is DebugPublishOperationStatus.PUBLISHED


@dataclass(frozen=True, slots=True)
class DebugPublishResult:
    plan: DebugPublishPlan
    operation_results: tuple[DebugPublishOperationResult, ...]
    execution: OperationBatchExecutionResult | None
    published_file_count: int
    blocking_errors: tuple[DebugIssue, ...]
    warnings: tuple[DebugIssue, ...]

    @property
    def succeeded(self) -> bool:
        return not self.blocking_errors and all(
            result.succeeded for result in self.operation_results
        )


@dataclass(frozen=True, slots=True)
class DebugPublishVerification:
    drive_output_root: Path
    verified_file_count: int
    missing_targets: tuple[Path, ...]
    status: DebugPublishVerificationStatus
    warnings: tuple[DebugIssue, ...]

    @property
    def succeeded(self) -> bool:
        return self.status in {
            DebugPublishVerificationStatus.VERIFIED,
            DebugPublishVerificationStatus.VERIFIED_WITH_WARNINGS,
        }


@dataclass(frozen=True, slots=True)
class DebugFinalTierExistingSummary:
    tier: TierName
    state: ExistingTierMembershipState


@dataclass(frozen=True, slots=True)
class DebugFinalResult:
    target_sentence_name: str
    resolved_split: SampleSplit | None
    debug_splits: tuple[SampleSplit, ...]
    source_translation_found: bool
    source_keypoints_found: bool
    source_video_found: bool
    gate_computed: SampleStatus | None
    gate_existing: ExistingGateStatus
    gate_verdict: DebugGateVerdict
    tier_context_mode: DebugLeakageContextMode
    tier_context_production_like: bool
    tier_computed_status: TierStatus | None
    tier_selected_tier: TierName | None
    tier_existing: tuple[DebugFinalTierExistingSummary, ...]
    tier_verdict: DebugTierVerdict
    visualization_verdict: DebugVisualizationVerdict
    visual_artifact_count: int
    report_output_path: Path | None
    runtime_output_root: Path | None
    drive_output_root: Path | None
    publish_status: DebugPublishVerificationStatus | None
    published_file_count: int
    final_verdict: DebugFinalVerdict
    next_actions: tuple[str, ...]
    warnings: tuple[str, ...]


__all__ = [
    "DebugExistingGateEvidence",
    "DebugExistingTierEvidence",
    "DebugExistingTierStateSummary",
    "DebugFinalResult",
    "DebugFinalTierExistingSummary",
    "DebugGateComparison",
    "DebugGateFrameMetrics",
    "DebugGatePreflight",
    "DebugGateResult",
    "DebugIssue",
    "DebugLeakageContext",
    "DebugPublishOperation",
    "DebugPublishOperationResult",
    "DebugPublishPlan",
    "DebugPublishResult",
    "DebugPublishVerification",
    "DebugReportResult",
    "DebugRestoreOperation",
    "DebugRestoreOperationResult",
    "DebugRestorePlan",
    "DebugRestorePlanValidation",
    "DebugRestoreResult",
    "DebugRuntimeCheck",
    "DebugRuntimeVerification",
    "DebugSampleDossier",
    "DebugSourceEvidence",
    "DebugTargetResolution",
    "DebugTierComparison",
    "DebugTierComputedMembership",
    "DebugTierFamilyDecisionSummary",
    "DebugTierQualityMetricSummary",
    "DebugTierRecomputeSource",
    "DebugTierResult",
    "DebugTranslationMatch",
    "DebugVisualArtifact",
    "DebugVisualizationResult",
]
