from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.gate.contracts import (
    GateRuntimePlan,
    GateRuntimeRestoreResult,
    GateRuntimeVerification,
    GateWorkflowResult,
    GateWrittenReportArtifacts,
)
from text_to_sign_production.workflows.gate.processing import GateExecutionBundle
from text_to_sign_production.workflows.gate.review.reports import write_gate_reports
from text_to_sign_production.workflows.gate.review.sections import (
    build_final_operator_summary_sections,
    build_output_detail_sections,
    build_processing_detail_sections,
    build_runtime_plan_detail_sections,
    build_runtime_restore_detail_sections,
    build_runtime_verification_detail_sections,
    build_written_artifact_detail_sections,
    build_written_artifact_summary_sections,
)
from text_to_sign_production.workflows.gate.review.sections import (
    build_output_summary_sections as review_outputs,
)
from text_to_sign_production.workflows.gate.review.sections import (
    build_processing_summary_sections as review_processing,
)
from text_to_sign_production.workflows.gate.review.sections import (
    build_runtime_plan_sections as review_runtime_plan,
)
from text_to_sign_production.workflows.gate.review.sections import (
    build_runtime_restore_sections as review_runtime_restore,
)
from text_to_sign_production.workflows.gate.review.sections import (
    build_runtime_verification_sections as review_runtime_verification,
)


def review_final(bundle: GateExecutionBundle) -> tuple[WorkflowReviewSection, ...]:
    return build_final_operator_summary_sections(bundle)


def review_final_operator_summary(
    bundle: GateExecutionBundle,
    report_artifacts: GateWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    return build_final_operator_summary_sections(bundle, report_artifacts=report_artifacts)


def review_written_artifacts(
    bundle: GateExecutionBundle,
    report_artifacts: GateWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    return build_written_artifact_summary_sections(bundle, report_artifacts=report_artifacts)


def review_runtime_plan_detail(plan: GateRuntimePlan) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_plan_detail_sections(plan)


def review_runtime_restore_detail(
    result: GateRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_restore_detail_sections(result)


def review_runtime_verification_detail(
    verification: GateRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_verification_detail_sections(verification)


def review_processing_detail(bundle: GateExecutionBundle) -> tuple[WorkflowReviewSection, ...]:
    return build_processing_detail_sections(bundle)


def review_outputs_detail(result: GateWorkflowResult) -> tuple[WorkflowReviewSection, ...]:
    return build_output_detail_sections(result)


def review_written_artifacts_detail(
    bundle: GateExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_written_artifact_detail_sections(bundle)

__all__ = [
    "review_runtime_plan",
    "review_runtime_restore",
    "review_runtime_verification",
    "review_processing",
    "review_outputs",
    "review_outputs_detail",
    "review_final",
    "review_final_operator_summary",
    "review_processing_detail",
    "review_runtime_plan_detail",
    "review_runtime_restore_detail",
    "review_runtime_verification_detail",
    "review_written_artifacts",
    "review_written_artifacts_detail",
    "write_gate_reports",
]
