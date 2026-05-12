from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.gate.contracts import (
    GatePublishExecution,
    GatePublishPlan,
    GatePublishResult,
    GatePublishVerification,
    GateRuntimePlan,
    GateRuntimeRestoreResult,
    GateRuntimeVerification,
    GateWorkflowConfig,
    GateWorkflowInvariantError,
    GateWorkflowResult,
    GateWrittenReportArtifacts,
)
from text_to_sign_production.workflows.gate.layout import (
    GateLayout,
    build_gate_layout,
)
from text_to_sign_production.workflows.gate.processing import (
    GateExecutionBundle,
    execute_gate_processing,
)
from text_to_sign_production.workflows.gate.publish import (
    build_gate_publish_plan,
    execute_gate_publish,
    verify_gate_publish,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_execution as build_publish_execution_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_execution_detail as build_publish_execution_detail_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_plan as build_publish_plan_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_plan_detail as build_publish_plan_detail_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_result as build_publish_result_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_result_detail as build_publish_result_detail_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_verification as build_publish_verification_review,
)
from text_to_sign_production.workflows.gate.publish import (
    review_publish_verification_detail as build_publish_verification_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_final as build_final_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_final_operator_summary as build_final_operator_summary_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_outputs as build_outputs_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_outputs_detail as build_outputs_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_processing as build_processing_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_processing_detail as build_processing_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_runtime_plan as build_runtime_plan_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_runtime_plan_detail as build_runtime_plan_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_runtime_restore as build_runtime_restore_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_runtime_restore_detail as build_runtime_restore_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_runtime_verification as build_runtime_verification_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_runtime_verification_detail as build_runtime_verification_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_written_artifacts as build_written_artifacts_review,
)
from text_to_sign_production.workflows.gate.review import (
    review_written_artifacts_detail as build_written_artifacts_detail_review,
)
from text_to_sign_production.workflows.gate.review import (
    write_gate_reports,
)
from text_to_sign_production.workflows.gate.runtime import (
    build_gate_runtime_plan,
    restore_gate_runtime,
    validate_gate_runtime_plan,
    verify_gate_runtime,
)


@dataclass(slots=True)
class GateWorkflow:
    config: GateWorkflowConfig
    layout: GateLayout | None = None
    executor: WorkflowExecutor | None = None

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_gate_layout(self.config)
        elif self.layout.config != self.config:
            raise GateWorkflowInvariantError(
                "Gate workflow layout config does not match workflow config."
            )

        if self.executor is None:
            self.executor = NotebookShellExecutor()

    def _resolved_executor(
        self,
        executor: WorkflowExecutor | None,
    ) -> WorkflowExecutor:
        if executor is not None:
            return executor
        if self.executor is None:
            raise GateWorkflowInvariantError("Gate workflow executor is not configured.")
        return self.executor

    def plan_runtime(self) -> GateRuntimePlan:
        return build_gate_runtime_plan(self.config, cast(GateLayout, self.layout))

    def review_runtime_plan(
        self,
        plan: GateRuntimePlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_review(plan)

    def review_runtime_plan_detail(
        self,
        plan: GateRuntimePlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_detail_review(plan)

    def restore_runtime(
        self,
        plan: GateRuntimePlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> GateRuntimeRestoreResult:
        return restore_gate_runtime(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_runtime_restore(
        self,
        result: GateRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_review(result)

    def review_runtime_restore_detail(
        self,
        result: GateRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_detail_review(result)

    def validate_runtime_plan(
        self,
        plan: GateRuntimePlan,
    ) -> None:
        validate_gate_runtime_plan(self.config, cast(GateLayout, self.layout), plan)

    def verify_runtime(
        self,
        plan: GateRuntimePlan,
    ) -> GateRuntimeVerification:
        return verify_gate_runtime(plan)

    def review_runtime_verification(
        self,
        verification: GateRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_review(verification)

    def review_runtime_verification_detail(
        self,
        verification: GateRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_detail_review(verification)

    def execute_processing(
        self,
        plan: GateRuntimePlan,
        runtime_verification: GateRuntimeVerification,
        *,
        progress_session: ProgressSession | None = None,
    ) -> GateExecutionBundle:
        return execute_gate_processing(
            config=self.config,
            layout=cast(GateLayout, self.layout),
            execution_inputs=plan.execution_inputs,
            runtime_verification=runtime_verification,
            progress_session=progress_session,
        )

    def review_processing(
        self,
        bundle: GateExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_processing_review(bundle)

    def review_processing_detail(
        self,
        bundle: GateExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_processing_detail_review(bundle)

    def review_outputs(
        self,
        result: GateWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_outputs_review(result)

    def review_outputs_detail(
        self,
        result: GateWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_outputs_detail_review(result)

    def review_written_artifacts(
        self,
        bundle: GateExecutionBundle,
        report_artifacts: GateWrittenReportArtifacts | None = None,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_written_artifacts_review(bundle, report_artifacts=report_artifacts)

    def review_written_artifacts_detail(
        self,
        bundle: GateExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_written_artifacts_detail_review(bundle)

    def review_final(
        self,
        bundle: GateExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_review(bundle)

    def review_final_operator_summary(
        self,
        bundle: GateExecutionBundle,
        report_artifacts: GateWrittenReportArtifacts | None = None,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_operator_summary_review(
            bundle,
            report_artifacts=report_artifacts,
        )

    def write_reports(
        self,
        bundle: GateExecutionBundle,
        *,
        progress_session: ProgressSession | None = None,
    ) -> GateWrittenReportArtifacts:
        return write_gate_reports(
            bundle=bundle,
            progress_session=progress_session,
        )

    def build_publish_plan(
        self,
        bundle: GateExecutionBundle,
        report_artifacts: GateWrittenReportArtifacts,
    ) -> GatePublishPlan:
        return build_gate_publish_plan(
            bundle=bundle,
            report_artifacts=report_artifacts,
            layout=cast(GateLayout, self.layout),
        )

    def review_publish_plan(
        self,
        plan: GatePublishPlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_review(plan)

    def review_publish_plan_detail(
        self,
        plan: GatePublishPlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_detail_review(plan)

    def execute_publish(
        self,
        plan: GatePublishPlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> GatePublishExecution:
        return execute_gate_publish(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_publish_execution(
        self,
        execution: GatePublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_review(execution)

    def review_publish_execution_detail(
        self,
        execution: GatePublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_detail_review(execution)

    def verify_publish(
        self,
        plan: GatePublishPlan,
    ) -> GatePublishVerification:
        return verify_gate_publish(plan)

    def review_publish_verification(
        self,
        verification: GatePublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_review(verification)

    def review_publish_verification_detail(
        self,
        verification: GatePublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_detail_review(verification)

    def build_publish_result(
        self,
        plan: GatePublishPlan,
        execution: GatePublishExecution,
        verification: GatePublishVerification,
    ) -> GatePublishResult:
        return GatePublishResult(
            plan=plan,
            execution=execution,
            verification=verification,
        )

    def review_publish_result(
        self,
        result: GatePublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_review(result)

    def review_publish_result_detail(
        self,
        result: GatePublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_detail_review(result)
