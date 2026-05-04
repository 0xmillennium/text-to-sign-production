"""Thin summary helpers for the tiers workflow."""

from __future__ import annotations

from text_to_sign_production.workflows._shared.review import (
    WorkflowReviewField,
    WorkflowReviewItem,
    WorkflowReviewSection,
)
from text_to_sign_production.workflows.tiers.types import (
    TiersPublishPlan,
    TiersPublishPlanSummary,
    TiersRuntimePlan,
    TiersWorkflowOutputSummary,
    TiersWorkflowResult,
)


def summarize_tiers_workflow_outputs(
    result: TiersWorkflowResult,
) -> TiersWorkflowOutputSummary:
    return result.output_summary


def summarize_tiers_runtime_plan_review(
    plan: TiersRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        WorkflowReviewSection(
            title="Untiered passed manifest restore specs",
            items=tuple(
                _review_item(
                    spec.label,
                    (
                        ("source", spec.source_path),
                        ("target", spec.target_path),
                        ("expected bytes", spec.expected_input_bytes),
                    ),
                )
                for spec in plan.restore_file_transfers
            ),
        ),
        WorkflowReviewSection(
            title="Passed sample archive extract specs",
            items=tuple(
                _review_item(
                    spec.label,
                    (
                        ("archive", spec.archive_path),
                        ("extract to", spec.extraction_root),
                        ("expected bytes", spec.expected_input_bytes),
                    ),
                )
                for spec in plan.restore_archive_extracts
            ),
        ),
    )


def summarize_tiers_publish_plan(plan: TiersPublishPlan) -> TiersPublishPlanSummary:
    target_paths = tuple(spec.target_path for spec in plan.file_transfers)
    return TiersPublishPlanSummary(
        planned_file_count=len(plan.file_transfers),
        target_paths=target_paths,
    )


def summarize_tiers_publish_plan_review(
    plan: TiersPublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        WorkflowReviewSection(
            title="File publish specs",
            items=tuple(
                _review_item(
                    spec.label,
                    (
                        ("source", spec.source_path),
                        ("target", spec.target_path),
                        ("expected bytes", spec.expected_input_bytes),
                    ),
                )
                for spec in plan.file_transfers
            ),
        ),
    )


def _review_item(
    label: str,
    fields: tuple[tuple[str, object], ...],
) -> WorkflowReviewItem:
    return WorkflowReviewItem(
        label=label,
        fields=tuple(WorkflowReviewField(label=key, value=value) for key, value in fields),
    )


__all__ = [
    "summarize_tiers_publish_plan",
    "summarize_tiers_publish_plan_review",
    "summarize_tiers_runtime_plan_review",
    "summarize_tiers_workflow_outputs",
]
