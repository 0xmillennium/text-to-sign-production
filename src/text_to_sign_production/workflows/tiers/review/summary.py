from __future__ import annotations

from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.tiers.contracts import (
    TiersRuntimePlan,
    TiersRuntimeRestoreResult,
    TiersRuntimeVerification,
    TiersWorkflowResult,
)
from text_to_sign_production.workflows.tiers.processing import TiersExecutionBundle
from text_to_sign_production.workflows.tiers.review.sections import (
    build_calibration_sections,
    build_final_review_sections,
    build_output_summary_sections,
    build_processing_summary_sections,
    build_runtime_plan_sections,
    build_runtime_restore_sections,
    build_runtime_verification_sections,
)


def review_runtime_plan(
    plan: TiersRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_plan_sections(plan)


def review_runtime_restore(
    result: TiersRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_restore_sections(result)


def review_runtime_verification(
    verification: TiersRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_verification_sections(verification)


def review_processing(
    bundle: TiersExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_processing_summary_sections(bundle)


def review_calibration(
    bundle: TiersExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_calibration_sections(bundle)


def review_outputs(
    result: TiersWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_output_summary_sections(result)


def review_final(
    bundle: TiersExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_final_review_sections(bundle)
