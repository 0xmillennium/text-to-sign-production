from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.tier.contracts import (
    TierPublishExecution,
    TierPublishPlan,
    TierPublishResult,
    TierPublishVerification,
    TierRuntimePlan,
    TierRuntimeRestoreResult,
    TierRuntimeVerification,
    TierWorkflowConfig,
    TierWorkflowExecutionInputs,
    TierWorkflowInvariantError,
    TierWorkflowResult,
    TierWrittenReportArtifacts,
)
from text_to_sign_production.workflows.tier.layout import (
    TierLayout,
    build_tier_layout,
)
from text_to_sign_production.workflows.tier.processing import (
    TierExecutionBundle,
    execute_tier_processing,
)
from text_to_sign_production.workflows.tier.publish import (
    build_tier_publish_plan,
    execute_tier_publish,
    verify_tier_publish,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_execution as build_publish_execution_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_execution_detail as build_publish_execution_detail_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_plan as build_publish_plan_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_plan_detail as build_publish_plan_detail_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_result as build_publish_result_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_result_detail as build_publish_result_detail_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_verification as build_publish_verification_review,
)
from text_to_sign_production.workflows.tier.publish import (
    review_publish_verification_detail as build_publish_verification_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_calibration as build_calibration_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_calibration_detail as build_calibration_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_final as build_final_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_final_operator_summary as build_final_operator_summary_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_outputs as build_outputs_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_outputs_detail as build_outputs_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_processing as build_processing_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_processing_detail as build_processing_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_runtime_plan as build_runtime_plan_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_runtime_plan_detail as build_runtime_plan_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_runtime_restore as build_runtime_restore_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_runtime_restore_detail as build_runtime_restore_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_runtime_verification as build_runtime_verification_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_runtime_verification_detail as build_runtime_verification_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_written_reports as build_written_reports_review,
)
from text_to_sign_production.workflows.tier.review import (
    review_written_reports_detail as build_written_reports_detail_review,
)
from text_to_sign_production.workflows.tier.review import (
    write_tier_reports,
)
from text_to_sign_production.workflows.tier.runtime import (
    build_tier_runtime_plan,
    restore_tier_runtime,
    validate_tier_runtime_plan,
    verify_tier_runtime,
)


@dataclass(slots=True)
class TierWorkflow:
    config: TierWorkflowConfig
    layout: TierLayout | None = None
    executor: WorkflowExecutor | None = None

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_tier_layout(self.config)
        elif self.layout.config != self.config:
            raise TierWorkflowInvariantError("workflow layout config must match workflow config")
        if self.executor is None:
            self.executor = NotebookShellExecutor()

    def plan_runtime(self) -> TierRuntimePlan:
        return build_tier_runtime_plan(self.config, cast(TierLayout, self.layout))

    def review_runtime_plan(
        self,
        plan: TierRuntimePlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_review(plan)

    def review_runtime_plan_detail(
        self,
        plan: TierRuntimePlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_detail_review(plan)

    def restore_runtime(
        self,
        plan: TierRuntimePlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> TierRuntimeRestoreResult:
        return restore_tier_runtime(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_runtime_restore(
        self,
        result: TierRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_review(result)

    def review_runtime_restore_detail(
        self,
        result: TierRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_detail_review(result)

    def validate_runtime_plan(
        self,
        plan: TierRuntimePlan,
    ) -> None:
        validate_tier_runtime_plan(self.config, cast(TierLayout, self.layout), plan)

    def verify_runtime(
        self,
        plan: TierRuntimePlan,
    ) -> TierRuntimeVerification:
        return verify_tier_runtime(plan)

    def review_runtime_verification(
        self,
        verification: TierRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_review(verification)

    def review_runtime_verification_detail(
        self,
        verification: TierRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_detail_review(verification)

    def execute_processing(
        self,
        execution_inputs: TierWorkflowExecutionInputs,
        runtime_verification: TierRuntimeVerification,
        *,
        progress_session: ProgressSession | None = None,
    ) -> TierExecutionBundle:
        return execute_tier_processing(
            config=self.config,
            layout=cast(TierLayout, self.layout),
            execution_inputs=execution_inputs,
            runtime_verification=runtime_verification,
            progress_session=progress_session,
        )

    def review_processing(
        self,
        bundle: TierExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_processing_review(bundle)

    def review_processing_detail(
        self,
        bundle: TierExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_processing_detail_review(bundle)

    def review_calibration(
        self,
        bundle: TierExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_calibration_review(bundle)

    def review_calibration_detail(
        self,
        bundle: TierExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_calibration_detail_review(bundle)

    def review_outputs(
        self,
        result: TierWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_outputs_review(result)

    def review_outputs_detail(
        self,
        result: TierWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_outputs_detail_review(result)

    def review_written_reports(
        self,
        artifacts: TierWrittenReportArtifacts,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_written_reports_review(artifacts)

    def review_written_reports_detail(
        self,
        artifacts: TierWrittenReportArtifacts,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_written_reports_detail_review(artifacts)

    def review_final(
        self,
        bundle: TierExecutionBundle,
        report_artifacts: TierWrittenReportArtifacts | None = None,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_review(bundle, report_artifacts=report_artifacts)

    def review_final_operator_summary(
        self,
        bundle: TierExecutionBundle,
        report_artifacts: TierWrittenReportArtifacts | None = None,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_operator_summary_review(
            bundle,
            report_artifacts=report_artifacts,
        )

    def write_reports(
        self,
        bundle: TierExecutionBundle,
        *,
        progress_session: ProgressSession | None = None,
    ) -> TierWrittenReportArtifacts:
        return write_tier_reports(
            bundle=bundle,
            progress_session=progress_session,
        )

    def build_publish_plan(
        self,
        bundle: TierExecutionBundle,
        report_artifacts: TierWrittenReportArtifacts,
    ) -> TierPublishPlan:
        return build_tier_publish_plan(
            bundle=bundle,
            report_artifacts=report_artifacts,
            layout=cast(TierLayout, self.layout),
        )

    def review_publish_plan(
        self,
        plan: TierPublishPlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_review(plan)

    def review_publish_plan_detail(
        self,
        plan: TierPublishPlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_detail_review(plan)

    def execute_publish(
        self,
        plan: TierPublishPlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> TierPublishExecution:
        return execute_tier_publish(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_publish_execution(
        self,
        execution: TierPublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_review(execution)

    def review_publish_execution_detail(
        self,
        execution: TierPublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_detail_review(execution)

    def verify_publish(
        self,
        plan: TierPublishPlan,
    ) -> TierPublishVerification:
        return verify_tier_publish(plan)

    def review_publish_verification(
        self,
        verification: TierPublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_review(verification)

    def review_publish_verification_detail(
        self,
        verification: TierPublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_detail_review(verification)

    def build_publish_result(
        self,
        plan: TierPublishPlan,
        execution: TierPublishExecution,
        verification: TierPublishVerification,
    ) -> TierPublishResult:
        return TierPublishResult(
            plan=plan,
            execution=execution,
            verification=verification,
        )

    def review_publish_result(
        self,
        result: TierPublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_review(result)

    def review_publish_result_detail(
        self,
        result: TierPublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_detail_review(result)

    def _resolved_executor(self, executor: WorkflowExecutor | None) -> WorkflowExecutor:
        if executor is not None:
            return executor
        return cast(WorkflowExecutor, self.executor)
