"""Thin summary helpers for the samples workflow."""

from __future__ import annotations

from text_to_sign_production.workflows._shared.review import (
    WorkflowReviewField,
    WorkflowReviewItem,
    WorkflowReviewSection,
)
from text_to_sign_production.workflows.samples.types import (
    SamplesPublishPlan,
    SamplesPublishPlanSummary,
    SamplesRuntimePlan,
    SamplesWorkflowOutputSummary,
    SamplesWorkflowResult,
)


def summarize_samples_workflow_outputs(
    result: SamplesWorkflowResult,
) -> SamplesWorkflowOutputSummary:
    return result.output_summary


def summarize_samples_runtime_plan_review(
    plan: SamplesRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        WorkflowReviewSection(
            title="Translation restore specs",
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
            title="Keypoint archive extract specs",
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


def summarize_samples_publish_plan(plan: SamplesPublishPlan) -> SamplesPublishPlanSummary:
    target_paths = tuple(
        spec.target_path for spec in plan.file_transfers
    ) + tuple(spec.archive_path for spec in plan.archive_creates)
    return SamplesPublishPlanSummary(
        planned_file_count=len(plan.file_transfers),
        planned_archive_count=len(plan.archive_creates),
        target_paths=target_paths,
    )


def summarize_samples_publish_plan_review(
    plan: SamplesPublishPlan,
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
        WorkflowReviewSection(
            title="Archive metadata specs",
            items=tuple(
                _review_item(
                    spec.label,
                    (
                        ("member list", spec.member_list_path),
                        ("members", len(spec.members)),
                    ),
                )
                for spec in plan.archive_member_lists
            ),
        ),
        WorkflowReviewSection(
            title="Archive create specs",
            items=tuple(
                _review_item(
                    spec.label,
                    (
                        ("archive", spec.archive_path),
                        ("source root", spec.source_root),
                        ("member list", spec.member_list_path),
                        ("members", len(spec.members)),
                    ),
                )
                for spec in plan.archive_creates
            ),
        ),
        WorkflowReviewSection(
            title="Archive verify specs",
            items=tuple(
                _review_item(
                    spec.label,
                    (
                        ("archive", spec.archive_path),
                        ("expected members", spec.expected_member_list_path),
                        ("observed members", spec.observed_member_list_path),
                    ),
                )
                for spec in plan.archive_verifies
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
    "summarize_samples_publish_plan",
    "summarize_samples_publish_plan_review",
    "summarize_samples_runtime_plan_review",
    "summarize_samples_workflow_outputs",
]
