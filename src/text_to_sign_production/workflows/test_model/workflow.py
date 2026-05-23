"""Facade for single-sample test-model orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.modeling.candidates import (
    DEFAULT_MODEL_PROVIDER_REGISTRY,
    ModelProviderRegistry,
)
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.test_model.contracts import (
    CheckpointPolicy,
    TestModelCheckpointSelection,
    TestModelFinalResult,
    TestModelInferenceResult,
    TestModelPreflightResult,
    TestModelPublishExecution,
    TestModelPublishPlan,
    TestModelPublishResult,
    TestModelPublishVerification,
    TestModelReferenceComparisonResult,
    TestModelReportResult,
    TestModelRequest,
    TestModelRestorePlan,
    TestModelRestorePlanValidation,
    TestModelRestoreResult,
    TestModelRunResolution,
    TestModelRuntimeVerification,
    TestModelSampleEvidence,
    TestModelSmokeExecutionProtocol,
    TestModelTargetResolution,
    TestModelVisualizationResult,
    TestModelWorkflowConfig,
    TestModelWorkflowInvariantError,
)
from text_to_sign_production.workflows.test_model.layout import (
    TestModelLayout,
    build_test_model_layout,
)
from text_to_sign_production.workflows.test_model.processing import (
    build_test_model_final_result,
    collect_sample_evidence,
    compare_reference_and_generated as run_reference_comparison,
    render_test_model_visualization,
    resolve_model_run,
    resolve_target_sample,
    run_test_model_preflight,
    build_test_model_smoke_execution_protocol,
    run_test_model_sample_inference,
    select_checkpoint,
    write_test_model_reports,
)
from text_to_sign_production.workflows.test_model.progress import (
    visible_test_model_progress_session,
)
from text_to_sign_production.workflows.test_model.publish import (
    build_test_model_publish_plan,
    execute_test_model_publish,
    verify_test_model_publish,
)
from text_to_sign_production.workflows.test_model.review import (
    review_checkpoint_selection as build_checkpoint_selection_review,
    review_final_operator_summary as build_final_operator_summary_review,
    review_inference_result as build_inference_result_review,
    review_model_run_resolution as build_model_run_resolution_review,
    review_preflight as build_preflight_review,
    review_publish_execution as build_publish_execution_review,
    review_publish_plan as build_publish_plan_review,
    review_publish_result as build_publish_result_review,
    review_publish_verification as build_publish_verification_review,
    review_reference_comparison_result as build_reference_comparison_review,
    review_report_outputs as build_report_outputs_review,
    review_request as build_request_review,
    review_restore_plan as build_restore_plan_review,
    review_restore_plan_validation as build_restore_plan_validation_review,
    review_restore_result as build_restore_result_review,
    review_runtime_verification as build_runtime_verification_review,
    review_sample_evidence as build_sample_evidence_review,
    review_smoke_execution_protocol as build_smoke_execution_protocol_review,
    review_target_resolution as build_target_resolution_review,
    review_visualization_result as build_visualization_result_review,
)
from text_to_sign_production.workflows.test_model.runtime import (
    build_test_model_restore_plan,
    restore_test_model_runtime,
    validate_test_model_restore_plan,
    verify_test_model_runtime,
)


@dataclass(slots=True)
class TestModelWorkflow:
    config: TestModelWorkflowConfig
    layout: TestModelLayout | None = None
    executor: WorkflowExecutor | None = None
    provider_registry: ModelProviderRegistry | None = None
    execution_id: str = field(default_factory=lambda: _new_execution_id())

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_test_model_layout(self.config)
        elif self.layout.config != self.config:
            raise TestModelWorkflowInvariantError(
                "test_model workflow layout config does not match workflow config"
            )
        if self.executor is None:
            self.executor = NotebookShellExecutor()
        if self.provider_registry is None:
            self.provider_registry = DEFAULT_MODEL_PROVIDER_REGISTRY

    def review_request(self, request: TestModelRequest) -> tuple[WorkflowReviewSection, ...]:
        return build_request_review(request)

    def run_preflight(self, request: TestModelRequest) -> TestModelPreflightResult:
        return run_test_model_preflight(
            config=self.config,
            request=request,
            execution_id=self.execution_id,
        )

    def review_preflight(
        self,
        result: TestModelPreflightResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_preflight_review(result)

    def build_smoke_execution_protocol(
        self,
        request: TestModelRequest,
        preflight: TestModelPreflightResult,
    ) -> TestModelSmokeExecutionProtocol:
        return build_test_model_smoke_execution_protocol(
            request=request,
            preflight=preflight,
        )

    def review_smoke_execution_protocol(
        self,
        protocol: TestModelSmokeExecutionProtocol,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_smoke_execution_protocol_review(protocol)

    def build_restore_plan(self, request: TestModelRequest) -> TestModelRestorePlan:
        return build_test_model_restore_plan(self.config, request, layout=self._layout())

    def review_restore_plan(self, plan) -> tuple[WorkflowReviewSection, ...]:
        return build_restore_plan_review(plan)

    def validate_restore_plan(self, plan) -> TestModelRestorePlanValidation:
        return validate_test_model_restore_plan(self._layout(), plan)

    def review_restore_plan_validation(self, validation) -> tuple[WorkflowReviewSection, ...]:
        return build_restore_plan_validation_review(validation)

    def restore_runtime(
        self,
        plan,
        *,
        executor: WorkflowExecutor | None = None,
        progress_session: ProgressSession | None = None,
    ) -> TestModelRestoreResult:
        progress = visible_test_model_progress_session(progress_session)
        return restore_test_model_runtime(
            plan,
            executor=self._executor(executor),
            progress_session=progress,
        )

    def review_restore_result(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_restore_result_review(result)

    def verify_runtime(self, plan) -> TestModelRuntimeVerification:
        return verify_test_model_runtime(plan, layout=self._layout())

    def review_runtime_verification(self, verification) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_review(verification)

    def resolve_model_run(self, plan) -> TestModelRunResolution:
        return resolve_model_run(plan)

    def review_model_run_resolution(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_model_run_resolution_review(result)

    def select_checkpoint(
        self,
        model_run: TestModelRunResolution,
        policy: CheckpointPolicy,
    ) -> TestModelCheckpointSelection:
        return select_checkpoint(self._layout(), model_run, policy)

    def review_checkpoint_selection(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_checkpoint_selection_review(result)

    def resolve_target_sample(
        self,
        request: TestModelRequest,
        model_run: TestModelRunResolution,
        *,
        progress_session: ProgressSession | None = None,
    ) -> TestModelTargetResolution:
        progress = visible_test_model_progress_session(progress_session)
        return resolve_target_sample(
            self._layout(),
            request,
            model_run,
            progress_session=progress,
        )

    def review_target_resolution(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_target_resolution_review(result)

    def collect_sample_evidence(
        self,
        *,
        model_run: TestModelRunResolution,
        checkpoint: TestModelCheckpointSelection,
        target: TestModelTargetResolution,
        progress_session: ProgressSession | None = None,
    ) -> TestModelSampleEvidence:
        progress = visible_test_model_progress_session(progress_session)
        return collect_sample_evidence(
            self._layout(),
            model_run=model_run,
            checkpoint=checkpoint,
            target=target,
            progress_session=progress,
        )

    def review_sample_evidence(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_sample_evidence_review(result)

    def run_sample_inference(
        self,
        *,
        model_run: TestModelRunResolution,
        checkpoint: TestModelCheckpointSelection,
        target: TestModelTargetResolution,
        progress_session: ProgressSession | None = None,
    ) -> TestModelInferenceResult:
        progress = visible_test_model_progress_session(progress_session)
        return run_test_model_sample_inference(
            self._layout(),
            model_run=model_run,
            checkpoint=checkpoint,
            target=target,
            provider_registry=self.provider_registry,
            execution_id=self.execution_id,
            progress_session=progress,
        )

    def review_inference_result(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_inference_result_review(result)

    def compare_reference_and_generated(
        self,
        *,
        target: TestModelTargetResolution,
        inference: TestModelInferenceResult,
        progress_session: ProgressSession | None = None,
    ) -> TestModelReferenceComparisonResult:
        progress = visible_test_model_progress_session(progress_session)
        return run_reference_comparison(
            self._layout(),
            target=target,
            inference=inference,
            progress_session=progress,
        )

    def review_reference_comparison_result(
        self,
        result: TestModelReferenceComparisonResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_reference_comparison_review(result)

    def render_visualization(
        self,
        *,
        evidence: TestModelSampleEvidence,
        inference: TestModelInferenceResult,
        progress_session: ProgressSession | None = None,
    ) -> TestModelVisualizationResult:
        progress = visible_test_model_progress_session(progress_session)
        return render_test_model_visualization(
            self._layout(),
            evidence=evidence,
            inference=inference,
            execution_id=self.execution_id,
            progress_session=progress,
        )

    def review_visualization_result(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_visualization_result_review(result)

    def write_reports(
        self,
        *,
        model_run,
        checkpoint,
        target,
        evidence,
        inference,
        comparison,
        visualization,
        progress_session: ProgressSession | None = None,
    ) -> TestModelReportResult:
        progress = visible_test_model_progress_session(progress_session)
        return write_test_model_reports(
            self._layout(),
            model_run=model_run,
            checkpoint=checkpoint,
            target=target,
            evidence=evidence,
            inference=inference,
            comparison=comparison,
            visualization=visualization,
            execution_id=self.execution_id,
            progress_session=progress,
        )

    def review_report_outputs(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_report_outputs_review(result)

    def build_publish_plan(
        self,
        *,
        report_result,
        inference_result,
        visualization_result,
    ) -> TestModelPublishPlan:
        return build_test_model_publish_plan(
            self._layout(),
            report_result=report_result,
            inference_result=inference_result,
            visualization_result=visualization_result,
        )

    def review_publish_plan(self, plan) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_review(plan)

    def execute_publish(
        self,
        plan,
        *,
        executor: WorkflowExecutor | None = None,
        progress_session: ProgressSession | None = None,
    ) -> TestModelPublishExecution:
        progress = visible_test_model_progress_session(progress_session)
        return execute_test_model_publish(
            plan,
            executor=self._executor(executor),
            progress_session=progress,
        )

    def review_publish_execution(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_review(result)

    def verify_publish(self, execution) -> TestModelPublishVerification:
        return verify_test_model_publish(execution)

    def review_publish_verification(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_review(result)

    def build_publish_result(self, plan, execution, verification) -> TestModelPublishResult:
        return TestModelPublishResult(plan=plan, execution=execution, verification=verification)

    def review_publish_result(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_review(result)

    def build_final_result(self, **kwargs) -> TestModelFinalResult:
        return build_test_model_final_result(**kwargs)

    def review_final_operator_summary(self, result) -> tuple[WorkflowReviewSection, ...]:
        return build_final_operator_summary_review(result)

    def _layout(self) -> TestModelLayout:
        if self.layout is None:
            raise TestModelWorkflowInvariantError("test_model layout is not configured")
        return self.layout

    def _executor(self, executor: WorkflowExecutor | None) -> WorkflowExecutor:
        if executor is not None:
            return executor
        if self.executor is None:
            raise TestModelWorkflowInvariantError("test_model executor is not configured")
        return self.executor


def _new_execution_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


__all__ = ["TestModelWorkflow"]
