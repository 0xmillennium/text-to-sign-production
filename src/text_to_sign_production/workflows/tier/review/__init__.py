from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.tier.contracts import (
    TierRuntimePlan,
    TierRuntimeRestoreResult,
    TierRuntimeVerification,
    TierWorkflowResult,
    TierWrittenReportArtifacts,
)
from text_to_sign_production.workflows.tier.processing import TierExecutionBundle
from text_to_sign_production.workflows.tier.review.reports import write_tier_reports
from text_to_sign_production.workflows.tier.review.sections import (
    build_calibration_detail_sections,
    build_final_operator_summary_sections,
    build_output_detail_sections,
    build_processing_detail_sections,
    build_runtime_plan_detail_sections,
    build_runtime_restore_detail_sections,
    build_runtime_verification_detail_sections,
    build_written_report_artifact_detail_sections,
    build_written_report_artifact_summary_sections,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_calibration_sections as review_calibration,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_final_review_sections as review_final,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_output_summary_sections as review_outputs,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_processing_summary_sections as review_processing,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_runtime_plan_sections as review_runtime_plan,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_runtime_restore_sections as review_runtime_restore,
)
from text_to_sign_production.workflows.tier.review.sections import (
    build_runtime_verification_sections as review_runtime_verification,
)


def review_runtime_plan_detail(plan: TierRuntimePlan) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_plan_detail_sections(plan)


def review_runtime_restore_detail(
    result: TierRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_restore_detail_sections(result)


def review_runtime_verification_detail(
    verification: TierRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_verification_detail_sections(verification)


def review_processing_detail(bundle: TierExecutionBundle) -> tuple[WorkflowReviewSection, ...]:
    return build_processing_detail_sections(bundle)


def review_calibration_detail(bundle: TierExecutionBundle) -> tuple[WorkflowReviewSection, ...]:
    return build_calibration_detail_sections(bundle)


def review_outputs_detail(result: TierWorkflowResult) -> tuple[WorkflowReviewSection, ...]:
    return build_output_detail_sections(result)


def review_written_reports(
    artifacts: TierWrittenReportArtifacts,
) -> tuple[WorkflowReviewSection, ...]:
    return build_written_report_artifact_summary_sections(artifacts)


def review_written_reports_detail(
    artifacts: TierWrittenReportArtifacts,
) -> tuple[WorkflowReviewSection, ...]:
    return build_written_report_artifact_detail_sections(artifacts)


def review_final_operator_summary(
    bundle: TierExecutionBundle,
    report_artifacts: TierWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    return build_final_operator_summary_sections(bundle, report_artifacts=report_artifacts)


__all__ = [
    "review_runtime_plan",
    "review_runtime_plan_detail",
    "review_runtime_restore",
    "review_runtime_restore_detail",
    "review_runtime_verification",
    "review_runtime_verification_detail",
    "review_processing",
    "review_processing_detail",
    "review_calibration",
    "review_calibration_detail",
    "review_outputs",
    "review_outputs_detail",
    "review_written_reports",
    "review_written_reports_detail",
    "review_final",
    "review_final_operator_summary",
    "write_tier_reports",
]
