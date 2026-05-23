"""Typed result bundles for the single-sample test-model workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.modeling.candidates import ModelSingleSampleInferenceResult
from text_to_sign_production.modeling.data import ModelingManifestFamily, ModelingManifestSample
from text_to_sign_production.workflows.foundation.execution import (
    OperationBatchExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import WrittenFileReceipt
from text_to_sign_production.workflows.test_model.contracts.request import CheckpointPolicy


@dataclass(frozen=True, slots=True)
class TestModelIssue:
    code: str
    message: str
    path: Path | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class TestModelRestoreOperation:
    group: str
    source: Path
    target: Path
    required: bool
    description: str
    workflow_operation: WorkflowOperation
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class TestModelRestorePlan:
    model_run_name: str
    model_run_metadata_source: Path | None
    operations: tuple[TestModelRestoreOperation, ...]
    blocking_errors: tuple[TestModelIssue, ...]
    warnings: tuple[TestModelIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.blocking_errors


@dataclass(frozen=True, slots=True)
class TestModelRestorePlanValidation:
    plan: TestModelRestorePlan
    valid: bool
    blocking_errors: tuple[TestModelIssue, ...]
    warnings: tuple[TestModelIssue, ...]

    @property
    def succeeded(self) -> bool:
        return self.valid


@dataclass(frozen=True, slots=True)
class TestModelRestoreOperationResult:
    operation: TestModelRestoreOperation
    status: str
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {"succeeded", "skipped_optional", "failed_optional"}


@dataclass(frozen=True, slots=True)
class TestModelRestoreResult:
    plan: TestModelRestorePlan
    operation_results: tuple[TestModelRestoreOperationResult, ...]
    execution: OperationBatchExecutionResult | None

    @property
    def succeeded(self) -> bool:
        return all(result.succeeded for result in self.operation_results)


@dataclass(frozen=True, slots=True)
class TestModelRuntimeCheck:
    label: str
    path: Path
    required: bool
    exists: bool
    valid: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class TestModelRuntimeVerification:
    checks: tuple[TestModelRuntimeCheck, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class TestModelRunResolution:
    model_run_name: str
    model_key: str
    manifest_family: ModelingManifestFamily | None
    train_split: SampleSplit
    validation_split: SampleSplit
    test_split: SampleSplit
    run_mode: str
    run_metadata_path: Path
    run_metadata: dict[str, Any]
    status: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class TestModelCheckpointSelection:
    policy: CheckpointPolicy
    checkpoint_role: str
    checkpoint_path: Path
    checkpoint_exists: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.checkpoint_exists and not self.errors


@dataclass(frozen=True, slots=True)
class TestModelTargetResolution:
    status: str
    target_sentence_name: str
    resolved_sample_id: str | None
    source_sentence_name: str | None
    manifest_family: ModelingManifestFamily | None
    split: SampleSplit
    manifest_path: Path
    manifest_entry: PassedManifestEntry | None
    manifest_sample: ModelingManifestSample | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.status == "found" and self.manifest_sample is not None


@dataclass(frozen=True, slots=True)
class TestModelSampleEvidence:
    target: TestModelTargetResolution
    source_video_path: Path
    source_video_exists: bool
    prepared_payload_path: Path | None
    prepared_payload_exists: bool
    model_context: dict[str, Any]
    checkpoint_context: dict[str, Any]
    consistency_notes: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestModelInferenceResult:
    provider_result: ModelSingleSampleInferenceResult | None
    metadata_path: Path | None
    receipts: tuple[WrittenFileReceipt, ...]
    status: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.provider_result is not None and not self.errors


@dataclass(frozen=True, slots=True)
class TestModelComparisonMetric:
    metric_key: str
    value: float | None
    frame_count: int
    joint_count: int
    valid_value_count: int
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestModelChannelComparisonMetric:
    metric_key: str
    channel: str
    value: float | None
    frame_count: int
    joint_count: int
    valid_value_count: int
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestModelReferenceComparisonResult:
    schema_version: str
    status: str
    split: SampleSplit
    reference_sample_id: str | None
    generated_sample_id: str | None
    reference_frame_count: int | None
    generated_frame_count: int | None
    aligned_frame_count: int | None
    metrics: tuple[TestModelComparisonMetric, ...]
    channel_metrics: tuple[TestModelChannelComparisonMetric, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and not self.errors


@dataclass(frozen=True, slots=True)
class TestModelVisualArtifact:
    label: str
    path: Path
    created: bool
    metadata_json: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class TestModelVisualizationResult:
    output_root: Path
    artifacts: tuple[TestModelVisualArtifact, ...]
    receipts: tuple[WrittenFileReceipt, ...]
    status: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestModelReportResult:
    output_root: Path
    files: tuple[Path, ...]
    receipts: tuple[WrittenFileReceipt, ...]
    succeeded: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TestModelPublishTarget:
    label: str
    source_path: Path
    target_path: Path
    source_sha256: str


@dataclass(frozen=True, slots=True)
class TestModelPublishPlan:
    runtime_output_root: Path
    drive_output_root: Path
    targets: tuple[TestModelPublishTarget, ...]
    operations: tuple[WorkflowOperation, ...]
    blocking_errors: tuple[TestModelIssue, ...]
    warnings: tuple[TestModelIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.blocking_errors


@dataclass(frozen=True, slots=True)
class TestModelPublishExecution:
    plan: TestModelPublishPlan
    execution: OperationBatchExecutionResult | None
    published_file_count: int
    errors: tuple[TestModelIssue, ...]
    warnings: tuple[TestModelIssue, ...]

    @property
    def succeeded(self) -> bool:
        return not self.errors and self.published_file_count == len(self.plan.targets)


@dataclass(frozen=True, slots=True)
class TestModelPublishVerification:
    checks: tuple[tuple[TestModelPublishTarget, bool, str | None], ...]

    @property
    def succeeded(self) -> bool:
        return all(match for _, match, _ in self.checks)


@dataclass(frozen=True, slots=True)
class TestModelPublishResult:
    plan: TestModelPublishPlan
    execution: TestModelPublishExecution
    verification: TestModelPublishVerification


@dataclass(frozen=True, slots=True)
class TestModelFinalResult:
    model_run: TestModelRunResolution | None
    checkpoint: TestModelCheckpointSelection | None
    target: TestModelTargetResolution | None
    evidence: TestModelSampleEvidence | None
    inference: TestModelInferenceResult | None
    comparison: TestModelReferenceComparisonResult | None
    visualization: TestModelVisualizationResult | None
    reports: TestModelReportResult | None
    publish: TestModelPublishResult | None

    @property
    def succeeded(self) -> bool:
        return bool(
            self.model_run is not None
            and self.model_run.succeeded
            and self.checkpoint is not None
            and self.checkpoint.succeeded
            and self.target is not None
            and self.target.succeeded
            and self.inference is not None
            and self.inference.succeeded
            and self.comparison is not None
            and self.comparison.succeeded
            and self.visualization is not None
            and self.visualization.status == "completed"
            and self.reports is not None
            and self.reports.succeeded
            and (self.publish is None or self.publish.verification.succeeded)
        )


__all__ = [name for name in globals() if name.startswith("TestModel")]
