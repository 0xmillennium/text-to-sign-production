from __future__ import annotations

from text_to_sign_production.workflows.foundation.review import WorkflowReviewSection
from text_to_sign_production.workflows.samples.contracts import (
    SamplesRuntimePlan,
    SamplesRuntimeRestoreResult,
    SamplesRuntimeVerification,
    SamplesWorkflowResult,
)
from text_to_sign_production.workflows.samples.processing import SamplesExecutionBundle
from text_to_sign_production.workflows.samples.review.sections import (
    build_final_review_sections,
    build_output_summary_sections,
    build_processing_summary_sections,
    build_runtime_plan_sections,
    build_runtime_restore_sections,
    build_runtime_verification_sections,
)


def review_runtime_plan(
    plan: SamplesRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_plan_sections(plan)


def review_runtime_restore(
    result: SamplesRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_restore_sections(result)


def review_runtime_verification(
    verification: SamplesRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_verification_sections(verification)


def review_processing(
    bundle: SamplesExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_processing_summary_sections(bundle)


def review_outputs(
    result: SamplesWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_output_summary_sections(result)


def review_final(
    bundle: SamplesExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_final_review_sections(bundle)
