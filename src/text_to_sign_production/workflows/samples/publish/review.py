from __future__ import annotations

from typing import Any

from text_to_sign_production.workflows.foundation.execution import operation_kind
from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.samples.contracts import (
    SamplesPublishExecution,
    SamplesPublishPlan,
    SamplesPublishResult,
    SamplesPublishTarget,
    SamplesPublishTargetRow,
    SamplesPublishVerification,
)


def review_publish_plan(
    plan: SamplesPublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish targets",
            tuple(_publish_target_item(target) for target in plan.targets),
        ),
        review_section(
            "Publish operations",
            tuple(_publish_operation_item(operation) for operation in plan.operations),
        ),
    )


def review_publish_execution(
    execution: SamplesPublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish execution",
            tuple(_publish_execution_item(result) for result in execution.execution.results),
        ),
    )


def review_publish_verification(
    verification: SamplesPublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish verification",
            tuple(
                review_item(
                    check.label,
                    (
                        ("target_path", check.target_path),
                        ("exists", check.exists),
                    ),
                )
                for check in verification.checks
            ),
        ),
    )


def review_publish_result(
    result: SamplesPublishResult,
) -> tuple[WorkflowReviewSection, ...]:
    successful_results = result.execution.execution.successful_results()
    failed_results = result.execution.execution.failed_results()
    verified_targets = tuple(check for check in result.verification.checks if check.exists)
    missing_targets = result.verification.missing_targets()
    return (
        review_section(
            "Publish result",
            (
                review_item(
                    "publish",
                    (
                        ("planned target count", len(result.plan.targets)),
                        ("successful execution count", len(successful_results)),
                        ("failed execution count", len(failed_results)),
                        ("verified target count", len(verified_targets)),
                        ("missing target count", len(missing_targets)),
                    ),
                ),
            ),
        ),
    )


def _publish_target_item(target: SamplesPublishTarget) -> WorkflowReviewItem:
    row = SamplesPublishTargetRow(
        label=target.label,
        kind=target.kind,
        source_path=target.source_path,
        target_path=target.target_path,
    )
    return review_item(
        row.label,
        (
            ("kind", row.kind),
            ("source_path", row.source_path),
            ("target_path", row.target_path),
        ),
    )


def _publish_operation_item(operation: Any) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation kind", operation_kind(operation)),
        ("label", getattr(operation, "label", "")),
    ]
    for field_name in (
        "source_path",
        "target_path",
        "archive_path",
        "source_root",
    ):
        if hasattr(operation, field_name):
            value = getattr(operation, field_name)
            if value is not None:
                fields.append((field_name, value))
    if hasattr(operation, "members"):
        fields.append(("expected member count", len(operation.members)))
    elif hasattr(operation, "expected_members"):
        fields.append(("expected member count", len(operation.expected_members)))
    item_label = operation.label if hasattr(operation, "label") else operation_kind(operation)
    return review_item(str(item_label), fields)


def _publish_execution_item(result: Any) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation_kind", result.operation_kind),
        ("succeeded", result.succeeded),
        ("returncode", result.returncode),
        ("execution_mode", result.execution_mode),
    ]
    if result.observed_outputs:
        fields.append(("observed_outputs", result.observed_outputs))
    return review_item(result.label, fields)
