"""Facade for provider-neutral model workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistry,
    ModelStageExecutionContext,
)
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.model.contracts import (
    ModelProviderConfigResult,
    ModelProviderResolution,
    ModelPublishExecution,
    ModelPublishPlan,
    ModelPublishResult,
    ModelPublishSourceBundle,
    ModelPublishVerification,
    ModelPreflightResult,
    ModelReportArtifacts,
    ModelObjectiveArtifactResult,
    ModelResearchResolution,
    ModelRunMetadataArtifacts,
    ModelSmokeExecutionProtocol,
    ModelValidationArtifactResult,
    ModelRuntimePlan,
    ModelRuntimeRestoreResult,
    ModelRuntimeVerification,
    ModelStageArtifactReceiptResult,
    ModelStageExecutionWorkflowResult,
    ModelStagePlanningResult,
    ModelWorkflowConfig,
    ModelWorkflowExecutionInputs,
    ModelWorkflowFinalResult,
    ModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.model.layout import ModelLayout, build_model_layout
from text_to_sign_production.workflows.model.processing import (
    build_model_final_result,
    execute_model_stage_plan,
    load_model_provider_config,
    materialize_model_stage_artifact_receipts,
    plan_model_stages,
    resolve_model_provider,
    resolve_model_research,
    run_model_preflight,
    build_model_smoke_execution_protocol,
    write_model_reports,
    run_compute_calibration as run_model_compute_calibration,
    write_model_objective_artifacts,
    write_model_run_metadata_artifacts,
    write_model_validation_artifacts,
)
from text_to_sign_production.workflows.model.publish import (
    build_model_publish_plan,
    execute_model_publish,
    verify_model_publish,
)
from text_to_sign_production.workflows.model.review import (
    review_final_operator_summary as build_final_operator_summary_review,
    review_model_reports as build_model_reports_review,
    review_model_validation_artifacts as build_model_validation_artifacts_review,
    review_model_objective_artifacts as build_model_objective_artifacts_review,
    review_model_run_metadata_artifacts as build_metadata_artifacts_review,
    review_preflight as build_preflight_review,
    review_stage_artifact_receipts as build_stage_artifact_receipts_review,
    review_provider_config as build_provider_config_review,
    review_provider_effective_config as build_provider_effective_config_review,
    review_provider_resolution as build_provider_resolution_review,
    review_publish_execution as build_publish_execution_review,
    review_publish_plan as build_publish_plan_review,
    review_publish_result as build_publish_result_review,
    review_publish_verification as build_publish_verification_review,
    review_research_resolution as build_research_resolution_review,
    review_runtime_plan as build_runtime_plan_review,
    review_runtime_restore as build_runtime_restore_review,
    review_runtime_verification as build_runtime_verification_review,
    review_smoke_execution_protocol as build_smoke_execution_protocol_review,
    review_stage_execution as build_stage_execution_review,
    review_stage_plan as build_stage_plan_review,
)
from text_to_sign_production.workflows.model.runtime import (
    build_model_runtime_plan,
    restore_model_runtime,
    validate_model_runtime_plan,
    verify_model_runtime,
)
from text_to_sign_production.workflows.model.request import (
    build_model_run_request_from_layout,
)


@dataclass(slots=True)
class ModelWorkflow:
    """Notebook-friendly shell around future concrete model providers."""

    config: ModelWorkflowConfig
    layout: ModelLayout | None = None
    executor: WorkflowExecutor | None = None
    provider_registry: ModelProviderRegistry | None = None
    execution_id: str = field(default_factory=lambda: _new_execution_id())

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_model_layout(self.config)
        elif self.layout.config != self.config:
            raise ModelWorkflowInvariantError(
                "Model workflow layout config does not match workflow config."
            )
        if self.executor is None:
            self.executor = NotebookShellExecutor()
        if self.provider_registry is None:
            self.provider_registry = DEFAULT_MODEL_PROVIDER_REGISTRY

    def plan_runtime(self) -> ModelRuntimePlan:
        return build_model_runtime_plan(self.config, self._layout())

    def review_runtime_plan(self, plan: ModelRuntimePlan) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_review(plan)

    def validate_runtime_plan(self, plan: ModelRuntimePlan) -> None:
        validate_model_runtime_plan(self.config, self._layout(), plan)

    def run_preflight(self, runtime_plan: ModelRuntimePlan) -> ModelPreflightResult:
        return run_model_preflight(
            config=self.config,
            layout=self._layout(),
            runtime_plan=runtime_plan,
        )

    def review_preflight(
        self,
        result: ModelPreflightResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_preflight_review(result)

    def build_smoke_execution_protocol(
        self,
        preflight: ModelPreflightResult,
    ) -> ModelSmokeExecutionProtocol:
        return build_model_smoke_execution_protocol(
            config=self.config,
            preflight=preflight,
        )

    def review_smoke_execution_protocol(
        self,
        protocol: ModelSmokeExecutionProtocol,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_smoke_execution_protocol_review(protocol)

    def restore_runtime(
        self,
        plan: ModelRuntimePlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> ModelRuntimeRestoreResult:
        return restore_model_runtime(
            plan,
            executor=self._executor(executor),
            progress_session=progress_session,
        )

    def review_runtime_restore(
        self,
        result: ModelRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_review(result)

    def verify_runtime(self, plan: ModelRuntimePlan) -> ModelRuntimeVerification:
        return verify_model_runtime(plan)

    def review_runtime_verification(
        self,
        verification: ModelRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_review(verification)

    def resolve_research(
        self,
        execution_inputs: ModelWorkflowExecutionInputs,
    ) -> ModelResearchResolution:
        return resolve_model_research(self.config, execution_inputs)

    def review_research_resolution(
        self,
        resolution: ModelResearchResolution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_research_resolution_review(resolution)

    def resolve_provider(self, request) -> ModelProviderResolution:
        return resolve_model_provider(
            request,
            provider_registry=self.provider_registry,
        )

    def review_provider_resolution(
        self,
        resolution: ModelProviderResolution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_provider_resolution_review(resolution)

    def load_provider_config(
        self,
        provider_resolution: ModelProviderResolution,
        request,
    ) -> ModelProviderConfigResult:
        if provider_resolution.request != request:
            raise ModelWorkflowInvariantError(
                "provider resolution request does not match config load request"
            )
        if not provider_resolution.provider_available or provider_resolution.provider is None:
            raise ModelWorkflowInvariantError(
                provider_resolution.message
                or "No provider is registered for this model key yet."
            )
        return load_model_provider_config(provider_resolution.provider, request)

    def review_provider_config(
        self,
        result: ModelProviderConfigResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_provider_config_review(result)

    def review_provider_effective_config(
        self,
        result: ModelProviderConfigResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_provider_effective_config_review(result)

    def plan_model_stages(
        self,
        provider_config: ModelProviderConfigResult,
    ) -> ModelStagePlanningResult:
        if provider_config.request != self._request():
            raise ModelWorkflowInvariantError(
                "provider config request does not match workflow request"
            )
        return plan_model_stages(
            provider_config.provider,
            provider_config.request,
            provider_config.loaded_config,
        )

    def review_stage_plan(
        self,
        result: ModelStagePlanningResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_stage_plan_review(result)

    def execute_model_stages(
        self,
        stage_plan_result: ModelStagePlanningResult,
        *,
        progress_session: ProgressSession | None = None,
    ) -> ModelStageExecutionWorkflowResult:
        from text_to_sign_production.workflows.model.contracts.compute_profiles import (
            apply_torch_runtime_settings,
        )

        apply_torch_runtime_settings(self.config.resolved_compute_profile)
        return execute_model_stage_plan(
            provider=stage_plan_result.provider,
            request=stage_plan_result.stage_plan.request,
            loaded_config=stage_plan_result.loaded_config,
            topology=self._layout().stores.runtime,
            working_dir=self._layout().outputs.model_run_root,
            stage_plan=stage_plan_result.stage_plan,
            calibration_root=self._layout().reports.root,
            progress_session=progress_session,
        )

    def review_stage_execution(
        self,
        result: ModelStageExecutionWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_stage_execution_review(result)

    def run_compute_calibration(
        self,
        provider_config: ModelProviderConfigResult,
        stage_plan: ModelStagePlanningResult,
        *,
        measurements=None,
        progress_session: ProgressSession | None = None,
    ):
        return run_model_compute_calibration(
            provider_config=provider_config.loaded_config,
            stage_plan=stage_plan.stage_plan,
            output_root=self._layout().reports.root,
            measurements=measurements,
            max_samples=None,
            progress_session=progress_session,
        )

    def write_model_run_metadata_artifacts(
        self,
        research: ModelResearchResolution,
        provider_config: ModelProviderConfigResult,
        stage_plan: ModelStagePlanningResult,
        stage_execution: ModelStageExecutionWorkflowResult | None = None,
        objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
        *,
        execution_id: str | None = None,
    ) -> ModelRunMetadataArtifacts:
        return write_model_run_metadata_artifacts(
            layout=self._layout(),
            research=research,
            provider=provider_config.provider,
            loaded_config=provider_config.loaded_config,
            stage_plan=stage_plan.stage_plan,
            stage_execution=stage_execution,
            objective_artifacts=objective_artifacts,
            execution_id=_execution_id(
                self.execution_id if execution_id is None else execution_id
            ),
        )

    def review_model_run_metadata_artifacts(
        self,
        artifacts: ModelRunMetadataArtifacts,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_metadata_artifacts_review(artifacts)

    def write_model_validation_artifacts(
        self,
        stage_execution: ModelStageExecutionWorkflowResult,
        *,
        execution_id: str | None = None,
    ) -> ModelValidationArtifactResult:
        return write_model_validation_artifacts(
            layout=self._layout(),
            stage_execution=stage_execution,
            execution_id=_execution_id(
                self.execution_id if execution_id is None else execution_id
            ),
        )

    def review_model_validation_artifacts(
        self,
        artifacts: ModelValidationArtifactResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_model_validation_artifacts_review(artifacts)

    def write_model_objective_artifacts(
        self,
        stage_execution: ModelStageExecutionWorkflowResult,
        *,
        execution_id: str | None = None,
    ) -> tuple[ModelObjectiveArtifactResult, ...]:
        return write_model_objective_artifacts(
            layout=self._layout(),
            stage_execution=stage_execution,
            execution_id=_execution_id(
                self.execution_id if execution_id is None else execution_id
            ),
        )

    def review_model_objective_artifacts(
        self,
        artifacts: tuple[ModelObjectiveArtifactResult, ...],
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_model_objective_artifacts_review(artifacts)

    def materialize_stage_artifact_receipts(
        self,
        stage_execution: ModelStageExecutionWorkflowResult,
        *,
        execution_id: str | None = None,
    ) -> ModelStageArtifactReceiptResult:
        return materialize_model_stage_artifact_receipts(
            layout=self._layout(),
            stage_execution=stage_execution,
            execution_id=_execution_id(
                self.execution_id if execution_id is None else execution_id
            ),
        )

    def review_stage_artifact_receipts(
        self,
        result: ModelStageArtifactReceiptResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_stage_artifact_receipts_review(result)

    def write_model_reports(
        self,
        stage_execution: ModelStageExecutionWorkflowResult,
        validation_artifacts: ModelValidationArtifactResult,
        objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
        *,
        execution_id: str | None = None,
    ) -> ModelReportArtifacts:
        context = ModelStageExecutionContext(
            request=stage_execution.stage_plan.request,
            loaded_config=stage_execution.loaded_config,
            topology=self._layout().stores.runtime,
            working_dir=self._layout().outputs.model_run_root,
            stage_results=stage_execution.execution.stages,
        )
        return write_model_reports(
            provider=stage_execution.provider,
            context=context,
            results=stage_execution.execution,
            layout=self._layout(),
            validation_artifacts=validation_artifacts,
            objective_artifacts=objective_artifacts,
            execution_id=_execution_id(
                self.execution_id if execution_id is None else execution_id
            ),
        )

    def review_model_reports(
        self,
        artifacts: ModelReportArtifacts,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_model_reports_review(artifacts)

    def build_publish_plan(
        self,
        *,
        metadata_artifacts: ModelRunMetadataArtifacts | None = None,
        validation_artifacts: ModelValidationArtifactResult | None = None,
        report_artifacts: ModelReportArtifacts | None = None,
        objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
        stage_artifact_receipts: ModelStageArtifactReceiptResult | None = None,
        stage_execution: ModelStageExecutionWorkflowResult | None = None,
    ) -> ModelPublishPlan:
        if (
            stage_execution is not None
            and stage_execution.stage_plan.request.auxiliary_objectives
            and not objective_artifacts
        ):
            raise ModelWorkflowInvariantError(
                "requested auxiliary objectives require objective artifacts in the publish plan."
            )
        return build_model_publish_plan(
            layout=self._layout(),
            sources=ModelPublishSourceBundle(
                metadata_artifacts=metadata_artifacts,
                validation_artifacts=validation_artifacts,
                report_artifacts=report_artifacts,
                objective_artifacts=objective_artifacts,
                stage_artifact_receipts=stage_artifact_receipts,
                stage_execution=stage_execution,
            ),
        )

    def review_publish_plan(self, plan: ModelPublishPlan) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_review(plan)

    def execute_publish(
        self,
        plan: ModelPublishPlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> ModelPublishExecution:
        return execute_model_publish(
            plan,
            executor=self._executor(executor),
            progress_session=progress_session,
        )

    def review_publish_execution(
        self,
        execution: ModelPublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_review(execution)

    def verify_publish(self, execution: ModelPublishExecution) -> ModelPublishVerification:
        return verify_model_publish(execution)

    def review_publish_verification(
        self,
        verification: ModelPublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_review(verification)

    def build_publish_result(
        self,
        plan: ModelPublishPlan,
        execution: ModelPublishExecution,
        verification: ModelPublishVerification,
    ) -> ModelPublishResult:
        if execution.plan != plan:
            raise ModelWorkflowInvariantError("publish execution does not match publish plan")
        return ModelPublishResult(
            plan=plan,
            execution=execution,
            verification=verification,
        )

    def review_publish_result(
        self,
        result: ModelPublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_review(result)

    def build_final_result(
        self,
        *,
        research: ModelResearchResolution | None = None,
        provider_available: bool = False,
        runtime_verification: ModelRuntimeVerification | None = None,
        stage_execution: ModelStageExecutionWorkflowResult | None = None,
        metadata_artifacts: ModelRunMetadataArtifacts | None = None,
        validation_artifacts: ModelValidationArtifactResult | None = None,
        objective_artifacts: tuple[ModelObjectiveArtifactResult, ...] = (),
        report_artifacts: ModelReportArtifacts | None = None,
        publish_result: ModelPublishResult | None = None,
    ) -> ModelWorkflowFinalResult:
        request = (
            stage_execution.stage_plan.request
            if stage_execution is not None
            else self._request()
        )
        if stage_execution is not None and request != self._request():
            raise ModelWorkflowInvariantError(
                "stage execution request does not match workflow request"
            )
        return build_model_final_result(
            config=self.config,
            request=request,
            research=research,
            provider_available=provider_available,
            runtime_verification=runtime_verification,
            stage_execution=stage_execution,
            metadata_artifacts=metadata_artifacts,
            validation_artifacts=validation_artifacts,
            objective_artifacts=objective_artifacts,
            report_artifacts=report_artifacts,
            publish_result=publish_result,
        )

    def review_final_operator_summary(
        self,
        result: ModelWorkflowFinalResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_operator_summary_review(result)

    def _layout(self) -> ModelLayout:
        if self.layout is None:
            raise ModelWorkflowInvariantError("Model workflow layout is not configured.")
        return self.layout

    def _executor(self, executor: WorkflowExecutor | None) -> WorkflowExecutor:
        if executor is not None:
            return executor
        if self.executor is None:
            raise ModelWorkflowInvariantError("Model workflow executor is not configured.")
        return self.executor

    def _request(self):
        return build_model_run_request_from_layout(
            self.config,
            self._layout(),
        )


def _execution_id(value: str | None) -> str:
    if value is None or not value.strip():
        raise ModelWorkflowInvariantError("execution_id must be non-empty")
    return value


def _new_execution_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


__all__ = ["ModelWorkflow"]
