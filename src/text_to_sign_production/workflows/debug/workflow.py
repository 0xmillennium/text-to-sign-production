from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from text_to_sign_production.workflows.debug.contracts import (
    DebugFinalResult,
    DebugGateResult,
    DebugPublishPlan,
    DebugPublishResult,
    DebugPublishVerification,
    DebugReportResult,
    DebugRestorePlan,
    DebugRestorePlanValidation,
    DebugRestoreResult,
    DebugRuntimeVerification,
    DebugSampleDossier,
    DebugSampleRequest,
    DebugTargetResolution,
    DebugTierResult,
    DebugVisualizationResult,
    DebugWorkflowConfig,
)
from text_to_sign_production.workflows.debug.contracts.request import normalize_debug_splits
from text_to_sign_production.workflows.debug.layout import (
    DebugLayout,
    build_debug_layout,
)
from text_to_sign_production.workflows.debug.processing import (
    build_final_result,
    collect_sample_evidence,
    debug_gate,
    debug_tier,
    debug_visualization,
    resolve_target_sample,
    write_debug_reports,
)
from text_to_sign_production.workflows.debug.processing import (
    print_final_result as print_debug_final_result,
)
from text_to_sign_production.workflows.debug.progress import visible_debug_progress_session
from text_to_sign_production.workflows.debug.review import (
    review_gate_debug,
    review_publish_plan,
    review_publish_result,
    review_publish_verification,
    review_report_outputs,
    review_request,
    review_restore_plan,
    review_restore_plan_validation,
    review_restore_result,
    review_runtime_verification,
    review_sample_dossier,
    review_target_resolution,
    review_tier_debug,
    review_visualization_debug,
)
from text_to_sign_production.workflows.debug.runtime import (
    build_debug_publish_plan,
    build_debug_restore_plan,
    execute_debug_restore,
    publish_debug_outputs as execute_debug_publish_outputs,
    validate_debug_restore_plan,
    verify_debug_publish,
    verify_debug_runtime,
)
from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.workflows.foundation.execution import (
    NotebookShellExecutor,
    WorkflowExecutor,
)
from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection


@dataclass(slots=True)
class SampleDebugWorkflow:
    config: DebugWorkflowConfig
    layout: DebugLayout | None = None
    executor: WorkflowExecutor | None = None
    run_id: str = field(default_factory=lambda: _new_run_id())

    def __post_init__(self) -> None:
        if self.layout is None:
            self.layout = build_debug_layout(self.config)
        elif self.layout.config != self.config:
            raise ValueError("Debug workflow layout config does not match workflow config.")
        if self.executor is None:
            self.executor = NotebookShellExecutor()

    def build_restore_plan(self, debug_splits) -> DebugRestorePlan:
        splits = normalize_debug_splits(debug_splits)
        return build_debug_restore_plan(self.config, splits, layout=self.layout)

    def review_request(self, request: DebugSampleRequest) -> tuple[WorkflowReviewSection, ...]:
        return review_request(request)

    def review_restore_plan(self, plan) -> tuple[WorkflowReviewSection, ...]:
        return review_restore_plan(plan)

    def validate_restore_plan(self, plan) -> DebugRestorePlanValidation:
        return validate_debug_restore_plan(plan)

    def review_restore_plan_validation(
        self,
        validation,
    ) -> tuple[WorkflowReviewSection, ...]:
        return review_restore_plan_validation(validation)

    def execute_restore(
        self,
        plan,
        *,
        executor: WorkflowExecutor | None = None,
        progress_session: ProgressSession | None = None,
    ) -> DebugRestoreResult:
        resolved_executor = executor if executor is not None else self.executor
        if resolved_executor is None:
            raise RuntimeError("Debug workflow executor is not configured.")
        return execute_debug_restore(
            plan,
            executor=resolved_executor,
            progress_session=progress_session,
        )

    def review_restore_result(self, result) -> tuple[WorkflowReviewSection, ...]:
        return review_restore_result(result)

    def verify_restored_runtime(self, plan) -> DebugRuntimeVerification:
        return verify_debug_runtime(self.config, plan, layout=self.layout)

    def review_runtime_verification(self, verification) -> tuple[WorkflowReviewSection, ...]:
        return review_runtime_verification(verification)

    def resolve_target_sample(
        self,
        request,
        *,
        progress_session: ProgressSession | None = None,
    ) -> DebugTargetResolution:
        return resolve_target_sample(
            self.layout,
            request,
            progress_session=visible_debug_progress_session(progress_session),
        )

    def review_target_resolution(self, resolution) -> tuple[WorkflowReviewSection, ...]:
        return review_target_resolution(resolution)

    def collect_sample_evidence(
        self,
        resolution,
        *,
        progress_session: ProgressSession | None = None,
    ) -> DebugSampleDossier:
        return collect_sample_evidence(
            self.layout,
            resolution,
            progress_session=visible_debug_progress_session(progress_session),
        )

    def review_sample_dossier(self, dossier) -> tuple[WorkflowReviewSection, ...]:
        return review_sample_dossier(dossier)

    def debug_gate(
        self,
        dossier,
        *,
        progress_session: ProgressSession | None = None,
    ) -> DebugGateResult:
        return debug_gate(
            self.config,
            self.layout,
            dossier,
            progress_session=visible_debug_progress_session(progress_session),
        )

    def review_gate_debug(self, result) -> tuple[WorkflowReviewSection, ...]:
        return review_gate_debug(result)

    def debug_tier(
        self,
        dossier,
        gate_result,
        *,
        progress_session: ProgressSession | None = None,
    ) -> DebugTierResult:
        return debug_tier(
            self.config,
            self.layout,
            dossier,
            gate_result,
            progress_session=visible_debug_progress_session(progress_session),
        )

    def review_tier_debug(self, result) -> tuple[WorkflowReviewSection, ...]:
        return review_tier_debug(result)

    def debug_visualization(
        self,
        dossier,
        gate_result,
        tier_result,
        *,
        progress_session: ProgressSession | None = None,
    ) -> DebugVisualizationResult:
        return debug_visualization(
            self.layout,
            run_id=self.run_id,
            dossier=dossier,
            gate_result=gate_result,
            tier_result=tier_result,
            progress_session=visible_debug_progress_session(progress_session),
        )

    def review_visualization_debug(self, result) -> tuple[WorkflowReviewSection, ...]:
        return review_visualization_debug(result)

    def write_debug_reports(
        self,
        *,
        dossier,
        gate_result,
        tier_result,
        visual_result,
        progress_session: ProgressSession | None = None,
    ) -> DebugReportResult:
        return write_debug_reports(
            self.layout,
            run_id=self.run_id,
            dossier=dossier,
            gate_result=gate_result,
            tier_result=tier_result,
            visual_result=visual_result,
            progress_session=visible_debug_progress_session(progress_session),
        )

    def review_report_outputs(self, result) -> tuple[WorkflowReviewSection, ...]:
        return review_report_outputs(result)

    def build_publish_plan(
        self,
        *,
        report_result: DebugReportResult,
        visual_result: DebugVisualizationResult,
    ) -> DebugPublishPlan:
        return build_debug_publish_plan(
            layout=self.layout,
            report_result=report_result,
            visual_result=visual_result,
        )

    def review_publish_plan(self, plan) -> tuple[WorkflowReviewSection, ...]:
        return review_publish_plan(plan)

    def publish_debug_outputs(
        self,
        plan,
        *,
        executor: WorkflowExecutor | None = None,
        progress_session: ProgressSession | None = None,
    ) -> DebugPublishResult:
        resolved_executor = executor if executor is not None else self.executor
        if resolved_executor is None:
            raise RuntimeError("Debug workflow executor is not configured.")
        return execute_debug_publish_outputs(
            plan,
            executor=resolved_executor,
            progress_session=progress_session,
        )

    def review_publish_result(self, result) -> tuple[WorkflowReviewSection, ...]:
        return review_publish_result(result)

    def verify_published_outputs(self, result) -> DebugPublishVerification:
        return verify_debug_publish(result)

    def review_publish_verification(
        self,
        verification,
    ) -> tuple[WorkflowReviewSection, ...]:
        return review_publish_verification(verification)

    def build_final_result(
        self,
        *,
        dossier,
        gate_result,
        tier_result,
        visual_result,
        report_result,
        publish_result=None,
        publish_verification=None,
    ) -> DebugFinalResult:
        return build_final_result(
            dossier=dossier,
            gate_result=gate_result,
            tier_result=tier_result,
            visual_result=visual_result,
            report_result=report_result,
            publish_result=publish_result,
            publish_verification=publish_verification,
        )

    def print_final_result(self, result) -> None:
        print_debug_final_result(result)


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


__all__ = ["SampleDebugWorkflow"]
