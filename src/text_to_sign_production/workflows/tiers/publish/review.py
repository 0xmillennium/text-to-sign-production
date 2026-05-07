from __future__ import annotations

from typing import Any

from text_to_sign_production.workflows.foundation.execution import operation_kind
from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersPublishExecution,
    TiersPublishPlan,
    TiersPublishResult,
    TiersPublishTarget,
    TiersPublishTargetRow,
    TiersPublishVerification,
)


def review_publish_plan(
    plan: TiersPublishPlan,
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
    execution: TiersPublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish execution",
            tuple(_publish_execution_item(result) for result in execution.execution.results),
        ),
    )


def review_publish_verification(
    verification: TiersPublishVerification,
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
    result: TiersPublishResult,
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


def _publish_target_item(target: TiersPublishTarget) -> WorkflowReviewItem:
    row = TiersPublishTargetRow(
        label=target.label,
        kind=target.kind,
        source_path=target.source_path,
        target_path=target.target_path,
        tier=target.tier,
        membership=target.membership,
        split=target.split,
    )
    fields: list[tuple[object, object]] = [
        ("kind", row.kind),
        ("source_path", row.source_path),
        ("target_path", row.target_path),
    ]
    for field_name in ("tier", "membership", "split"):
        value = getattr(row, field_name)
        if value is not None:
            fields.append((field_name, value))
    return review_item(row.label, fields)


def _publish_operation_item(operation: Any) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation kind", operation_kind(operation)),
        ("label", operation.label),
        ("source_path", operation.source_path),
        ("target_path", operation.target_path),
    ]
    if operation.expected_input_bytes is not None:
        fields.append(("expected_input_bytes", operation.expected_input_bytes))
    return review_item(str(operation.label), fields)


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
