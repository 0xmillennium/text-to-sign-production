from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.tiers.contracts import (
    TiersPublishExecution,
    TiersPublishPlan,
    TiersPublishResult,
    TiersPublishVerification,
    TiersReportArtifacts,
    TiersRuntimePlan,
    TiersRuntimeRestoreResult,
    TiersRuntimeVerification,
    TiersWorkflowConfig,
    TiersWorkflowExecutionInputs,
    TiersWorkflowInvariantError,
    TiersWorkflowResult,
)
from text_to_sign_production.workflows.tiers.layout import (
    TiersLayout,
    build_tiers_layout,
)
from text_to_sign_production.workflows.tiers.processing import (
    TiersExecutionBundle,
    execute_tiers_processing,
)
from text_to_sign_production.workflows.tiers.publish import (
    build_tiers_publish_plan,
    execute_tiers_publish,
    verify_tiers_publish,
)
from text_to_sign_production.workflows.tiers.publish import (
    review_publish_execution as build_publish_execution_review,
)
from text_to_sign_production.workflows.tiers.publish import (
    review_publish_plan as build_publish_plan_review,
)
from text_to_sign_production.workflows.tiers.publish import (
    review_publish_result as build_publish_result_review,
)
from text_to_sign_production.workflows.tiers.publish import (
    review_publish_verification as build_publish_verification_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_calibration as build_calibration_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_final as build_final_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_outputs as build_outputs_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_processing as build_processing_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_runtime_plan as build_runtime_plan_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_runtime_restore as build_runtime_restore_review,
)
from text_to_sign_production.workflows.tiers.review import (
    review_runtime_verification as build_runtime_verification_review,
)
from text_to_sign_production.workflows.tiers.review import (
    write_tiers_reports,
)
from text_to_sign_production.workflows.tiers.runtime import (
    build_tiers_runtime_plan,
    restore_tiers_runtime,
    validate_tiers_runtime_plan,
    verify_tiers_runtime,
)


@dataclass(slots=True)
class TiersWorkflow:
    config: TiersWorkflowConfig
    layout: TiersLayout | None = None
    executor: WorkflowExecutor | None = None

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_tiers_layout(self.config)
        elif self.layout.config != self.config:
            raise TiersWorkflowInvariantError("workflow layout config must match workflow config")
        if self.executor is None:
            self.executor = NotebookShellExecutor()

    def plan_runtime(self) -> TiersRuntimePlan:
        return build_tiers_runtime_plan(self.config, cast(TiersLayout, self.layout))

    def review_runtime_plan(
        self,
        plan: TiersRuntimePlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_plan_review(plan)

    def restore_runtime(
        self,
        plan: TiersRuntimePlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> TiersRuntimeRestoreResult:
        return restore_tiers_runtime(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_runtime_restore(
        self,
        result: TiersRuntimeRestoreResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_restore_review(result)

    def validate_runtime_plan(
        self,
        plan: TiersRuntimePlan,
    ) -> None:
        validate_tiers_runtime_plan(self.config, cast(TiersLayout, self.layout), plan)

    def verify_runtime(
        self,
        plan: TiersRuntimePlan,
    ) -> TiersRuntimeVerification:
        return verify_tiers_runtime(plan)

    def review_runtime_verification(
        self,
        verification: TiersRuntimeVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_runtime_verification_review(verification)

    def execute_processing(
        self,
        execution_inputs: TiersWorkflowExecutionInputs,
        runtime_verification: TiersRuntimeVerification,
        *,
        progress_session: ProgressSession | None = None,
    ) -> TiersExecutionBundle:
        return execute_tiers_processing(
            config=self.config,
            layout=cast(TiersLayout, self.layout),
            execution_inputs=execution_inputs,
            runtime_verification=runtime_verification,
            progress_session=progress_session,
        )

    def review_processing(
        self,
        bundle: TiersExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_processing_review(bundle)

    def review_calibration(
        self,
        bundle: TiersExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_calibration_review(bundle)

    def review_outputs(
        self,
        result: TiersWorkflowResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_outputs_review(result)

    def review_final(
        self,
        bundle: TiersExecutionBundle,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_final_review(bundle)

    def write_reports(
        self,
        bundle: TiersExecutionBundle,
        *,
        progress_session: ProgressSession | None = None,
    ) -> TiersReportArtifacts:
        return write_tiers_reports(
            bundle=bundle,
            progress_session=progress_session,
        )

    def build_publish_plan(
        self,
        bundle: TiersExecutionBundle,
    ) -> TiersPublishPlan:
        return build_tiers_publish_plan(bundle=bundle, layout=cast(TiersLayout, self.layout))

    def review_publish_plan(
        self,
        plan: TiersPublishPlan,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_plan_review(plan)

    def execute_publish(
        self,
        plan: TiersPublishPlan,
        *,
        progress_session: ProgressSession | None = None,
        executor: WorkflowExecutor | None = None,
    ) -> TiersPublishExecution:
        return execute_tiers_publish(
            plan,
            executor=self._resolved_executor(executor),
            progress_session=progress_session,
        )

    def review_publish_execution(
        self,
        execution: TiersPublishExecution,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_execution_review(execution)

    def verify_publish(
        self,
        plan: TiersPublishPlan,
    ) -> TiersPublishVerification:
        return verify_tiers_publish(plan)

    def review_publish_verification(
        self,
        verification: TiersPublishVerification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_verification_review(verification)

    def build_publish_result(
        self,
        plan: TiersPublishPlan,
        execution: TiersPublishExecution,
        verification: TiersPublishVerification,
    ) -> TiersPublishResult:
        return TiersPublishResult(
            plan=plan,
            execution=execution,
            verification=verification,
        )

    def review_publish_result(
        self,
        result: TiersPublishResult,
    ) -> tuple[WorkflowReviewSection, ...]:
        return build_publish_result_review(result)

    def _resolved_executor(self, executor: WorkflowExecutor | None) -> WorkflowExecutor:
        if executor is not None:
            return executor
        return cast(WorkflowExecutor, self.executor)
