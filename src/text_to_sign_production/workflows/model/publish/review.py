"""Operator-facing publish review sections for model workflows."""

from __future__ import annotations

from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.model.contracts import (
    ModelPublishExecution,
    ModelPublishPlan,
    ModelPublishResult,
    ModelPublishVerification,
)


def review_publish_plan(plan: ModelPublishPlan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish plan summary",
            (
                review_item(
                    "model",
                    (
                        ("planned target count", len(plan.targets)),
                        ("skipped source count", len(plan.skipped_sources)),
                        ("operation count", len(plan.operations)),
                    ),
                ),
                *(
                    review_item(
                        f"skipped: {source.label}",
                        (("path", source.path), ("reason", source.reason)),
                    )
                    for source in plan.skipped_sources
                ),
            ),
        ),
    )


def review_publish_plan_detail(plan: ModelPublishPlan) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish plan target detail",
            tuple(
                review_item(
                    target.label,
                    (
                        ("kind", target.kind),
                        ("source path", target.source_path),
                        ("target path", target.target_path),
                        ("source sha256", target.source_sha256),
                    ),
                )
                for target in plan.targets
            ),
        ),
        review_section(
            "Publish plan skipped source detail",
            tuple(
                review_item(
                    source.label,
                    (("path", source.path), ("reason", source.reason)),
                )
                for source in plan.skipped_sources
            ),
        ),
    )


def review_publish_execution(
    execution: ModelPublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish execution summary",
            (
                review_item(
                    "model",
                    (
                        ("operation count", len(execution.execution.results)),
                        ("successful operation count", len(execution.execution.successful_results())),
                        ("failed operation count", len(execution.execution.failed_results())),
                    ),
                ),
            ),
        ),
    )


def review_publish_verification(
    verification: ModelPublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish verification summary",
            (
                review_item(
                    "model",
                    (
                        ("check count", len(verification.checks)),
                        ("missing target count", len(verification.missing_targets)),
                        ("succeeded", verification.succeeded),
                    ),
                ),
                *(
                    review_item(
                        f"failed: {check.target.label}",
                        (
                            ("target_path", check.target.target_path),
                            ("message", check.message),
                        ),
                    )
                    for check in verification.checks
                    if not check.exists or not check.sha256_matches
                ),
            ),
        ),
    )


def review_publish_result(result: ModelPublishResult) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish result summary",
            (
                review_item(
                    "model",
                    (
                        ("planned target count", len(result.plan.targets)),
                        ("execution succeeded", result.execution.execution.succeeded),
                        ("verification succeeded", result.verification.succeeded),
                    ),
                ),
            ),
        ),
    )


__all__ = [
    "review_publish_execution",
    "review_publish_plan",
    "review_publish_plan_detail",
    "review_publish_result",
    "review_publish_verification",
]
