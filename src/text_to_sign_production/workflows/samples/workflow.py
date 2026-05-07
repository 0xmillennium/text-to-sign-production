from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.samples.contracts import (
    SamplesPublishExecution,
    SamplesPublishPlan,
    SamplesPublishResult,
    SamplesPublishVerification,
    SamplesReportArtifacts,
    SamplesRuntimePlan,
    SamplesRuntimeRestoreResult,
    SamplesRuntimeVerification,
    SamplesWorkflowConfig,
    SamplesWorkflowInvariantError,
    SamplesWorkflowResult,
)
from text_to_sign_production.workflows.samples.layout import (
    SamplesLayout,
    build_samples_layout,
)
from text_to_sign_production.workflows.samples.processing import (
    SamplesExecutionBundle,
    execute_samples_processing,
)
from text_to_sign_production.workflows.samples.publish import (
    build_samples_publish_plan,
    execute_samples_publish,
    verify_samples_publish,
)
from text_to_sign_production.workflows.samples.publish import (
    review_publish_execution as build_publish_execution_review,
)
from text_to_sign_production.workflows.samples.publish import (
    review_publish_plan as build_publish_plan_review,
)
from text_to_sign_production.workflows.samples.publish import (
    review_publish_result as build_publish_result_review,
)
from text_to_sign_production.workflows.samples.publish import (
    review_publish_verification as build_publish_verification_review,
)
from text_to_sign_production.workflows.samples.review import (
    review_final as build_final_review,
)
from text_to_sign_production.workflows.samples.review import (
    review_outputs as build_outputs_review,
)
from text_to_sign_production.workflows.samples.review import (
    review_processing as build_processing_review,
)
from text_to_sign_production.workflows.samples.review import (
    review_runtime_plan as build_runtime_plan_review,
)
from text_to_sign_production.workflows.samples.review import (
    review_runtime_restore as build_runtime_restore_review,
)
from text_to_sign_production.workflows.samples.review import (
    review_runtime_verification as build_runtime_verification_review,
)
from text_to_sign_production.workflows.samples.review import (
    write_samples_reports,
)
from text_to_sign_production.workflows.samples.runtime import (
    build_samples_runtime_plan,
    restore_samples_runtime,
    validate_samples_runtime_plan,
    verify_samples_runtime,
)


@dataclass(slots=True)
class SamplesWorkflow:
    config: SamplesWorkflowConfig
    layout: SamplesLayout | None = None
    executor: WorkflowExecutor | None = None

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_samples_layout(self.config)
        elif self.layout.config != self.config:
            raise SamplesWorkflowInvariantError(
                "Samples workflow layout config does not match workflow config."
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
            raise SamplesWorkflowInvariantError("Samples workflow executor is not configured.")
        return self.executor

    def plan_runtime(self) -> SamplesRuntimePlan:
        return build_samples_runtime_plan(self.config, cast(SamplesLayout, self.layout))

    def review_runtime_plan(
        self,
        plan: SamplesRuntimePlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_review(plan)

    def restore_runtime(
        self,
        plan: SamplesRuntimePlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> SamplesRuntimeRestoreResult:
        return restore_samples_runtime(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_runtime_restore(
        self,
        result: SamplesRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_review(result)

    def validate_runtime_plan(
        self,
        plan: SamplesRuntimePlan,
    ) -> None:
        validate_samples_runtime_plan(self.config, cast(SamplesLayout, self.layout), plan)

    def verify_runtime(
        self,
        plan: SamplesRuntimePlan,
    ) -> SamplesRuntimeVerification:
        return verify_samples_runtime(plan)

    def review_runtime_verification(
        self,
        verification: SamplesRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_review(verification)

    def execute_processing(
        self,
        plan: SamplesRuntimePlan,
        runtime_verification: SamplesRuntimeVerification,
        *,
        progress_session: ProgressSession | None = None,
    ) -> SamplesExecutionBundle:
        return execute_samples_processing(
            config=self.config,
            layout=cast(SamplesLayout, self.layout),
            execution_inputs=plan.execution_inputs,
            runtime_verification=runtime_verification,
            progress_session=progress_session,
        )

    def review_processing(
        self,
        bundle: SamplesExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_processing_review(bundle)

    def review_outputs(
        self,
        result: SamplesWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_outputs_review(result)

    def review_final(
        self,
        bundle: SamplesExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_review(bundle)

    def write_reports(
        self,
        bundle: SamplesExecutionBundle,
        *,
        progress_session: ProgressSession | None = None,
    ) -> SamplesReportArtifacts:
        return write_samples_reports(
            bundle=bundle,
            progress_session=progress_session,
        )

    def build_publish_plan(
        self,
        bundle: SamplesExecutionBundle,
    ) -> SamplesPublishPlan:
        return build_samples_publish_plan(bundle=bundle, layout=cast(SamplesLayout, self.layout))

    def review_publish_plan(
        self,
        plan: SamplesPublishPlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_review(plan)

    def execute_publish(
        self,
        plan: SamplesPublishPlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> SamplesPublishExecution:
        return execute_samples_publish(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_publish_execution(
        self,
        execution: SamplesPublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_review(execution)

    def verify_publish(
        self,
        plan: SamplesPublishPlan,
    ) -> SamplesPublishVerification:
        return verify_samples_publish(plan)

    def review_publish_verification(
        self,
        verification: SamplesPublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_review(verification)

    def build_publish_result(
        self,
        plan: SamplesPublishPlan,
        execution: SamplesPublishExecution,
        verification: SamplesPublishVerification,
    ) -> SamplesPublishResult:
        return SamplesPublishResult(
            plan=plan,
            execution=execution,
            verification=verification,
        )

    def review_publish_result(
        self,
        result: SamplesPublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_review(result)
