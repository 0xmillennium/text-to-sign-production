"""Processing and final-result contracts for the model workflow."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from text_to_sign_production.modeling.candidates import (
    ModelExecutionResult,
    ModelProvider,
    ModelProviderLoadedConfig,
    ModelRunRequest,
    ModelStageArtifactRef,
    ModelStagePlan,
)
from text_to_sign_production.modeling.research import (
    ModelKey,
    ModelSpec,
    ObjectiveKey,
    ObjectiveSpec,
    ResearchTraceIssue,
)
from text_to_sign_production.workflows.foundation.provenance import WrittenFileReceipt
from text_to_sign_production.workflows.model.contracts.config import (
    ModelWorkflowConfig,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.contracts.runtime import ModelRuntimeVerification

if TYPE_CHECKING:
    from text_to_sign_production.workflows.model.contracts.publish import ModelPublishResult


@dataclass(frozen=True, slots=True)
class ModelResearchResolution:
    request: ModelRunRequest
    model_spec: ModelSpec
    objective_specs: tuple[ObjectiveSpec, ...]
    trace_issues: tuple[ResearchTraceIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRunRequest):
            raise ModelWorkflowInvariantError("research.request must be ModelRunRequest")
        if not isinstance(self.model_spec, ModelSpec):
            raise ModelWorkflowInvariantError("research.model_spec must be ModelSpec")
        if self.model_spec.key is not self.request.model_key:
            raise ModelWorkflowInvariantError("research model key must match request model key")
        object.__setattr__(self, "objective_specs", tuple(self.objective_specs))
        object.__setattr__(self, "trace_issues", tuple(self.trace_issues))


@dataclass(frozen=True, slots=True)
class ModelProviderResolution:
    request: ModelRunRequest
    provider_available: bool
    provider_key: ModelKey
    provider: ModelProvider | None
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRunRequest):
            raise ModelWorkflowInvariantError("provider request must be ModelRunRequest")
        object.__setattr__(self, "provider_key", ModelKey(self.provider_key))
        if self.provider_key is not self.request.model_key:
            raise ModelWorkflowInvariantError("provider key must match request model key")
        if self.provider_available and self.provider is None:
            raise ModelWorkflowInvariantError("provider_available requires provider")
        if self.provider is not None and self.provider.spec.key is not self.provider_key:
            raise ModelWorkflowInvariantError("provider spec key must match provider_key")


@dataclass(frozen=True, slots=True)
class ModelProviderConfigResult:
    request: ModelRunRequest
    provider: ModelProvider
    loaded_config: ModelProviderLoadedConfig

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelRunRequest):
            raise ModelWorkflowInvariantError(
                "provider_config.request must be ModelRunRequest"
            )
        if not isinstance(self.loaded_config, ModelProviderLoadedConfig):
            raise ModelWorkflowInvariantError(
                "provider_config.loaded_config must be ModelProviderLoadedConfig"
            )
        if self.provider.spec.key is not self.request.model_key:
            raise ModelWorkflowInvariantError(
                "provider config request model key must match provider spec key"
            )
        if self.loaded_config.model_key is not self.request.model_key:
            raise ModelWorkflowInvariantError(
                "provider config loaded_config model key must match request model key"
            )
        if self.provider.spec.key is not self.loaded_config.model_key:
            raise ModelWorkflowInvariantError(
                "provider config model key must match provider spec key"
            )


@dataclass(frozen=True, slots=True)
class ModelStagePlanningResult:
    provider: ModelProvider
    loaded_config: ModelProviderLoadedConfig
    stage_plan: ModelStagePlan

    def __post_init__(self) -> None:
        if not isinstance(self.loaded_config, ModelProviderLoadedConfig):
            raise ModelWorkflowInvariantError(
                "stage planning loaded_config must be ModelProviderLoadedConfig"
            )
        if not isinstance(self.stage_plan, ModelStagePlan):
            raise ModelWorkflowInvariantError("stage planning stage_plan must be ModelStagePlan")
        if self.provider.spec.key is not self.loaded_config.model_key:
            raise ModelWorkflowInvariantError(
                "stage planning loaded_config does not match provider key"
            )
        if self.stage_plan.model.key is not self.provider.spec.key:
            raise ModelWorkflowInvariantError("stage plan model key does not match provider key")


@dataclass(frozen=True, slots=True)
class ModelStageExecutionWorkflowResult:
    provider: ModelProvider
    loaded_config: ModelProviderLoadedConfig
    stage_plan: ModelStagePlan
    execution: ModelExecutionResult
    pre_calibration: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.loaded_config, ModelProviderLoadedConfig):
            raise ModelWorkflowInvariantError(
                "stage execution loaded_config must be ModelProviderLoadedConfig"
            )
        if not isinstance(self.stage_plan, ModelStagePlan):
            raise ModelWorkflowInvariantError("stage execution stage_plan must be ModelStagePlan")
        if not isinstance(self.execution, ModelExecutionResult):
            raise ModelWorkflowInvariantError(
                "stage execution execution must be ModelExecutionResult"
            )
        if self.provider.spec.key is not self.loaded_config.model_key:
            raise ModelWorkflowInvariantError(
                "stage execution loaded_config does not match provider key"
            )
        if self.execution.model_key is not self.stage_plan.model.key:
            raise ModelWorkflowInvariantError("stage execution model key does not match plan")


@dataclass(frozen=True, slots=True)
class ModelRunMetadataArtifacts:
    execution_id: str
    effective_config_path: Path
    research_spec_path: Path
    run_metadata_path: Path
    runtime_support_manifest_path: Path
    effective_config: WrittenFileReceipt
    research_spec: WrittenFileReceipt
    run_metadata: WrittenFileReceipt
    runtime_support_manifest: WrittenFileReceipt

    def __post_init__(self) -> None:
        _require_text("execution_id", self.execution_id)
        object.__setattr__(self, "effective_config_path", Path(self.effective_config_path))
        object.__setattr__(self, "research_spec_path", Path(self.research_spec_path))
        object.__setattr__(self, "run_metadata_path", Path(self.run_metadata_path))
        object.__setattr__(
            self,
            "runtime_support_manifest_path",
            Path(self.runtime_support_manifest_path),
        )
        _require_receipt(
            self.effective_config,
            path=self.effective_config_path,
            kind="model_effective_config",
            label="metadata.effective_config",
        )
        _require_receipt(
            self.research_spec,
            path=self.research_spec_path,
            kind="model_research_spec",
            label="metadata.research_spec",
        )
        _require_receipt(
            self.run_metadata,
            path=self.run_metadata_path,
            kind="model_run_metadata",
            label="metadata.run_metadata",
        )
        _require_receipt(
            self.runtime_support_manifest,
            path=self.runtime_support_manifest_path,
            kind="model_runtime_support_manifest",
            label="metadata.runtime_support_manifest",
        )
        if any(
            receipt.execution_id != self.execution_id
            for receipt in (
                self.effective_config,
                self.research_spec,
                self.run_metadata,
                self.runtime_support_manifest,
            )
        ):
            raise ModelWorkflowInvariantError(
                "metadata receipts must match metadata execution_id"
            )


@dataclass(frozen=True, slots=True)
class ModelValidationArtifactResult:
    pairing_manifest_path: Path
    metric_results_path: Path
    channel_metric_results_path: Path
    aggregate_metrics_path: Path
    channel_aggregate_metrics_path: Path
    limitations_path: Path
    summary_markdown_path: Path
    paired_count: int
    missing_generated_count: int
    missing_reference_count: int
    failed_generated_count: int
    identity_mismatch_count: int
    receipts: tuple[WrittenFileReceipt, ...]

    def __post_init__(self) -> None:
        path_kinds = (
            ("pairing_manifest_path", "model_validation_pairing_manifest"),
            ("metric_results_path", "model_validation_metric_results"),
            ("channel_metric_results_path", "model_validation_channel_metric_results"),
            ("aggregate_metrics_path", "model_validation_aggregate_metrics"),
            ("channel_aggregate_metrics_path", "model_validation_channel_aggregate_metrics"),
            ("limitations_path", "model_validation_limitations"),
            ("summary_markdown_path", "model_validation_summary_report"),
        )
        for name, _ in path_kinds:
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "paired_count",
            "missing_generated_count",
            "missing_reference_count",
            "failed_generated_count",
            "identity_mismatch_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelWorkflowInvariantError(f"validation {name} must be non-negative")
        receipts = tuple(self.receipts)
        if len(receipts) != len(path_kinds):
            raise ModelWorkflowInvariantError("validation artifacts must include seven receipts")
        for receipt, (path_name, kind) in zip(receipts, path_kinds, strict=True):
            _require_receipt(
                receipt,
                path=getattr(self, path_name),
                kind=kind,
                label=f"validation.{path_name}",
            )
        execution_ids = {receipt.execution_id for receipt in receipts}
        if len(execution_ids) != 1:
            raise ModelWorkflowInvariantError("validation artifact receipts must share execution_id")
        object.__setattr__(self, "receipts", receipts)


@dataclass(frozen=True, slots=True)
class ModelObjectiveArtifactResult:
    """Workflow-visible materialization result for one attached auxiliary objective."""

    objective_key: ObjectiveKey
    artifact_paths: Mapping[str, Path]
    artifact_refs: tuple[ModelStageArtifactRef, ...]
    records_count: int
    skipped_count: int
    receipts: tuple[WrittenFileReceipt, ...]
    ablation_readiness_path: Path | None = None
    candidate_policy: str | None = None
    generated_pose_manifest_path: Path | None = None
    config_snapshot_path: Path | None = None
    config_snapshot_sha256: str | None = None
    ready_for_comparison: bool | None = None
    required_baseline_missing: bool | None = None
    readiness_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "objective_key", ObjectiveKey(self.objective_key))
        except (TypeError, ValueError) as exc:
            raise ModelWorkflowInvariantError("objective artifact result has unknown objective_key") from exc
        if self.objective_key is not ObjectiveKey.SEMANTIC_CONSISTENCY:
            raise ModelWorkflowInvariantError(
                "only semantic_consistency objective artifacts are integrated in this stage"
            )
        for name in (
            "ablation_readiness_path",
            "generated_pose_manifest_path",
            "config_snapshot_path",
        ):
            value = getattr(self, name)
            if value is None:
                raise ModelWorkflowInvariantError(f"semantic objective {name} is required")
            object.__setattr__(self, name, Path(value))
        if not isinstance(self.config_snapshot_sha256, str) or not self.config_snapshot_sha256.strip():
            raise ModelWorkflowInvariantError(
                "semantic objective config_snapshot_sha256 is required"
            )
        if self.candidate_policy != "single_candidate_only":
            raise ModelWorkflowInvariantError(
                "semantic objective candidate_policy must be single_candidate_only"
            )
        if not isinstance(self.ready_for_comparison, bool):
            raise ModelWorkflowInvariantError("semantic objective ready_for_comparison is required")
        if not isinstance(self.required_baseline_missing, bool):
            raise ModelWorkflowInvariantError(
                "semantic objective required_baseline_missing is required"
            )
        issues = tuple(self.readiness_issues)
        if not self.ready_for_comparison and not issues:
            raise ModelWorkflowInvariantError(
                "not-ready semantic objective result requires readiness_issues"
            )
        if self.ready_for_comparison and issues:
            raise ModelWorkflowInvariantError(
                "ready semantic objective result cannot contain readiness_issues"
            )
        if any(not isinstance(issue, str) or not issue.strip() for issue in issues):
            raise ModelWorkflowInvariantError(
                "semantic objective readiness_issues must be non-empty strings"
            )
        object.__setattr__(self, "readiness_issues", issues)
        if not isinstance(self.artifact_paths, Mapping):
            raise ModelWorkflowInvariantError("objective artifact_paths must be a mapping")
        paths: dict[str, Path] = {}
        for key, path in self.artifact_paths.items():
            if not isinstance(key, str) or not key.strip():
                raise ModelWorkflowInvariantError("objective artifact_paths keys must be non-empty")
            paths[key] = Path(path)
        refs = tuple(self.artifact_refs)
        receipts = tuple(self.receipts)
        if len(refs) != len(receipts) or any(
            not isinstance(ref, ModelStageArtifactRef) for ref in refs
        ):
            raise ModelWorkflowInvariantError(
                "objective artifact refs and receipts must describe every materialized file"
            )
        for value, label in (
            (self.records_count, "records_count"),
            (self.skipped_count, "skipped_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelWorkflowInvariantError(f"objective {label} must be non-negative")
        if self.skipped_count > self.records_count:
            raise ModelWorkflowInvariantError("objective skipped_count exceeds records_count")
        for ref, receipt in zip(refs, receipts, strict=True):
            _require_receipt(receipt, path=ref.path, kind=ref.kind, label="objective.receipt")
        object.__setattr__(self, "artifact_paths", MappingProxyType(paths))
        object.__setattr__(self, "artifact_refs", refs)
        object.__setattr__(self, "receipts", receipts)


@dataclass(frozen=True, slots=True)
class ModelReportArtifacts:
    execution_id: str
    artifacts: tuple[WrittenFileReceipt, ...]
    provider_artifacts: tuple[ModelStageArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _require_text("execution_id", self.execution_id)
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ModelWorkflowInvariantError("report artifacts must be non-empty")
        allowed_report_kinds = {
            "model_report",
            "compute_calibration",
            "compute_calibration_report",
            "selected_overrides",
            "calibrated_effective_config",
            "calibration_summary",
        }
        for receipt in artifacts:
            if not isinstance(receipt, WrittenFileReceipt):
                raise ModelWorkflowInvariantError(
                    "report_artifacts.artifacts must be WrittenFileReceipt"
                )
            if receipt.kind not in allowed_report_kinds:
                raise ModelWorkflowInvariantError(
                    f"report artifact kind is unsupported: {receipt.kind}"
                )
            if receipt.execution_id != self.execution_id:
                raise ModelWorkflowInvariantError(
                    "report artifact receipts must match report execution_id"
                )
        provider_artifacts = tuple(self.provider_artifacts)
        if any(
            not isinstance(artifact, ModelStageArtifactRef)
            for artifact in provider_artifacts
        ):
            raise ModelWorkflowInvariantError(
                "provider_artifacts must contain ModelStageArtifactRef values"
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "provider_artifacts", provider_artifacts)


@dataclass(frozen=True, slots=True)
class ModelStageArtifactSkip:
    role: str
    kind: str
    path: Path | None
    reason: str

    def __post_init__(self) -> None:
        _require_text("role", self.role)
        _require_text("kind", self.kind)
        _require_text("reason", self.reason)
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class ModelStageArtifactReceiptResult:
    execution_id: str
    receipts: tuple[WrittenFileReceipt, ...]
    skipped_artifacts: tuple[ModelStageArtifactSkip, ...]
    index_path: Path
    index_receipt: WrittenFileReceipt

    def __post_init__(self) -> None:
        _require_text("execution_id", self.execution_id)
        receipts = tuple(self.receipts)
        if any(not isinstance(receipt, WrittenFileReceipt) for receipt in receipts):
            raise ModelWorkflowInvariantError(
                "stage artifact receipts must contain WrittenFileReceipt values"
            )
        if any(receipt.execution_id != self.execution_id for receipt in receipts):
            raise ModelWorkflowInvariantError(
                "stage artifact receipts must match result execution_id"
            )
        skipped_artifacts = tuple(self.skipped_artifacts)
        if any(
            not isinstance(skipped, ModelStageArtifactSkip)
            for skipped in skipped_artifacts
        ):
            raise ModelWorkflowInvariantError(
                "skipped_artifacts must contain ModelStageArtifactSkip values"
            )
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "skipped_artifacts", skipped_artifacts)
        object.__setattr__(self, "index_path", Path(self.index_path))
        _require_receipt(
            self.index_receipt,
            path=self.index_path,
            kind="model_stage_artifacts_index",
            label="stage_artifact_receipts.index_receipt",
        )
        if self.index_receipt.execution_id != self.execution_id:
            raise ModelWorkflowInvariantError(
                "stage artifact index receipt must match result execution_id"
            )


@dataclass(frozen=True, slots=True)
class ModelWorkflowFinalResult:
    config: ModelWorkflowConfig
    request: ModelRunRequest
    research: ModelResearchResolution | None
    provider_available: bool
    runtime_verification: ModelRuntimeVerification | None
    stage_execution: ModelStageExecutionWorkflowResult | None
    metadata_artifacts: ModelRunMetadataArtifacts | None
    validation_artifacts: ModelValidationArtifactResult | None
    objective_artifacts: tuple[ModelObjectiveArtifactResult, ...]
    report_artifacts: ModelReportArtifacts | None
    publish_result: ModelPublishResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.config, ModelWorkflowConfig):
            raise ModelWorkflowInvariantError("final_result.config must be ModelWorkflowConfig")
        if not isinstance(self.request, ModelRunRequest):
            raise ModelWorkflowInvariantError("final_result.request must be ModelRunRequest")
        if self.request.model_key is not self.config.model_key:
            raise ModelWorkflowInvariantError("final_result request model key must match config")
        if self.request.run_name != self.config.run_name:
            raise ModelWorkflowInvariantError("final_result request run_name must match config")
        if self.research is not None:
            if not hasattr(self.research, "request"):
                raise ModelWorkflowInvariantError("final_result research must expose request")
            if self.research.request != self.request:
                raise ModelWorkflowInvariantError("final_result research request must match request")
        if (
            self.stage_execution is not None
            and self.stage_execution.stage_plan.request != self.request
        ):
            raise ModelWorkflowInvariantError(
                "final_result stage execution request must match request"
            )
        object.__setattr__(self, "objective_artifacts", tuple(self.objective_artifacts))
        if any(
            not isinstance(artifact, ModelObjectiveArtifactResult)
            for artifact in self.objective_artifacts
        ):
            raise ModelWorkflowInvariantError(
                "final_result objective_artifacts must contain ModelObjectiveArtifactResult values"
            )

    @property
    def succeeded(self) -> bool:
        return bool(
            self.provider_available
            and self.research is not None
            and self.runtime_verification is not None
            and self.runtime_verification.succeeded
            and self.stage_execution is not None
            and self.stage_execution.execution.completed
            and self.metadata_artifacts is not None
            and self.validation_artifacts is not None
            and (
                not self.request.auxiliary_objectives
                or bool(self.objective_artifacts)
            )
            and self.report_artifacts is not None
            and bool(self.report_artifacts.artifacts)
            and self._publish_succeeded
        )

    @property
    def _publish_succeeded(self) -> bool:
        if self.publish_result is None:
            return True
        return bool(
            self.publish_result.execution.execution.succeeded
            and self.publish_result.verification.succeeded
        )

    @property
    def limitations(self) -> tuple[str, ...]:
        limitations: list[str] = []
        if not self.provider_available:
            limitations.append(
                "No provider is registered for this model key yet. "
                "This is expected before the model candidate implementation stage."
            )
        if self.runtime_verification is not None:
            limitations.extend(self.runtime_verification.limitations)
        return tuple(limitations)


def _require_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInvariantError(f"{field_name} must be non-empty")


def _require_receipt(
    receipt: WrittenFileReceipt,
    *,
    kind: str,
    label: str,
    path: Path | None = None,
) -> None:
    if not isinstance(receipt, WrittenFileReceipt):
        raise ModelWorkflowInvariantError(f"{label} must be WrittenFileReceipt")
    if receipt.kind != kind:
        raise ModelWorkflowInvariantError(
            f"{label} kind mismatch: expected {kind}, observed {receipt.kind}"
        )
    if path is not None and receipt.path.resolve(strict=False) != path.resolve(strict=False):
        raise ModelWorkflowInvariantError(
            f"{label} path mismatch: expected {path}, observed {receipt.path}"
        )


__all__ = [
    "ModelProviderConfigResult",
    "ModelProviderResolution",
    "ModelReportArtifacts",
    "ModelResearchResolution",
    "ModelRunMetadataArtifacts",
    "ModelValidationArtifactResult",
    "ModelObjectiveArtifactResult",
    "ModelStageArtifactReceiptResult",
    "ModelStageArtifactSkip",
    "ModelStageExecutionWorkflowResult",
    "ModelStagePlanningResult",
    "ModelWorkflowFinalResult",
]
