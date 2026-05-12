from __future__ import annotations

from text_to_sign_production.workflows.foundation.execution import (
    FileCopyOperation,
    OperationExecutionResult,
    WorkflowOperation,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.review import (
    RenderableValue,
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.tier.contracts import (
    TierPublishCheck,
    TierPublishedArtifactRow,
    TierPublishExecution,
    TierPublishPlan,
    TierPublishResult,
    TierPublishTarget,
    TierPublishTargetRow,
    TierPublishVerification,
)


def review_publish_plan(
    plan: TierPublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_plan_summary(plan)


def review_publish_plan_summary(
    plan: TierPublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    report_count = sum(1 for target in plan.targets if target.kind == "report_file")
    manifest_count = sum(1 for target in plan.targets if target.kind == "tiered_manifest_file")
    return (
        review_section(
            "Publish plan summary",
            (
                review_item(
                    "publish",
                    (
                        ("planned target count", len(plan.targets)),
                        ("tiered manifest file count", manifest_count),
                        ("report file count", report_count),
                        ("operation count", len(plan.operations)),
                    ),
                ),
            ),
        ),
    )


def review_publish_plan_detail(
    plan: TierPublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish target details",
            tuple(_publish_target_item(target) for target in plan.targets),
        ),
        review_section(
            "Publish operation details",
            tuple(_publish_operation_item(operation) for operation in plan.operations),
        ),
    )


def review_publish_execution(
    execution: TierPublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_execution_summary(execution)


def review_publish_execution_summary(
    execution: TierPublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    successful_results = execution.execution.successful_results()
    failed_results = execution.execution.failed_results()
    return (
        review_section(
            "Publish execution summary",
            (
                review_item(
                    "publish",
                    (
                        ("operation count", len(execution.execution.results)),
                        ("successful execution count", len(successful_results)),
                        ("failed execution count", len(failed_results)),
                    ),
                ),
                *_publish_failure_items(failed_results),
            ),
        ),
    )


def review_publish_execution_detail(
    execution: TierPublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish execution details",
            tuple(_publish_execution_item(result) for result in execution.execution.results),
        ),
    )


def review_publish_verification(
    verification: TierPublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_verification_summary(verification)


def review_publish_verification_summary(
    verification: TierPublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    failed_checks = tuple(
        check
        for check in verification.checks
        if not (
            check.source_exists
            and check.target_exists
            and check.coherent
            and check.digest_match is not False
        )
    )
    return (
        review_section(
            "Publish verification summary",
            (
                review_item(
                    "publish",
                    (
                        ("checked target count", len(verification.checks)),
                        ("verified target count", len(verification.checks) - len(failed_checks)),
                        ("failed target count", len(failed_checks)),
                        ("missing target count", len(verification.missing_targets())),
                        ("succeeded", verification.succeeded),
                    ),
                ),
                *_publish_check_failure_items(failed_checks),
            ),
        ),
    )


def review_publish_verification_detail(
    verification: TierPublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish verification details",
            tuple(
                review_item(
                    check.label,
                    (
                        ("kind", check.kind),
                        ("source_path", check.source_path),
                        ("source_exists", check.source_exists),
                        ("source_sha256", check.source_sha256),
                        ("source_execution_id", check.source_execution_id),
                        ("target_path", check.target_path),
                        ("target_exists", check.target_exists),
                        ("target_sha256", check.target_sha256),
                        ("digest_match", check.digest_match),
                        ("coherent", check.coherent),
                        ("message", check.message),
                    ),
                )
                for check in verification.checks
            ),
        ),
    )


def review_publish_result(
    result: TierPublishResult,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_result_summary(result)


def review_publish_result_summary(
    result: TierPublishResult,
) -> tuple[WorkflowReviewSection, ...]:
    successful_results = result.execution.execution.successful_results()
    failed_results = result.execution.execution.failed_results()
    verified_targets = tuple(
        check
        for check in result.verification.checks
        if check.target_exists and check.coherent and check.digest_match is not False
    )
    missing_targets = result.verification.missing_targets()
    return (
        review_section(
            "Publish result summary",
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


def review_publish_result_detail(
    result: TierPublishResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        *review_publish_result_summary(result),
        review_section(
            "Published artifact details",
            tuple(
                _published_artifact_item(_published_artifact_row(check))
                for check in result.verification.checks
                if check.target_exists
            ),
        ),
    )


def _publish_failure_items(
    results: tuple[OperationExecutionResult, ...],
    *,
    limit: int = 5,
) -> tuple[WorkflowReviewItem, ...]:
    return tuple(
        review_item(
            f"failed: {result.label}",
            (
                ("operation_kind", result.operation_kind),
                ("returncode", result.returncode),
                ("execution_mode", result.execution_mode),
            ),
        )
        for result in results[:limit]
    )


def _publish_check_failure_items(
    checks: tuple[TierPublishCheck, ...],
    *,
    limit: int = 5,
) -> tuple[WorkflowReviewItem, ...]:
    return tuple(
        review_item(
            f"failed: {check.label}",
            (
                ("kind", check.kind),
                ("source_exists", check.source_exists),
                ("target_exists", check.target_exists),
                ("digest_match", check.digest_match),
                ("coherent", check.coherent),
                ("message", check.message),
            ),
        )
        for check in checks[:limit]
    )


def _publish_target_item(target: TierPublishTarget) -> WorkflowReviewItem:
    row = TierPublishTargetRow(
        label=target.label,
        kind=target.kind,
        source_path=target.source_path,
        target_path=target.target_path,
        tier=target.tier,
        membership=target.membership,
        split=target.split,
    )
    fields: list[tuple[str, RenderableValue]] = [
        ("kind", row.kind),
        ("source_path", row.source_path),
        ("target_path", row.target_path),
        ("source_sha256", target.source_sha256),
        ("source_execution_id", target.source_execution_id),
    ]
    if row.tier is not None:
        fields.append(("tier", row.tier))
    if row.membership is not None:
        fields.append(("membership", row.membership))
    if row.split is not None:
        fields.append(("split", row.split))
    return review_item(row.label, fields)


def _publish_operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    if not isinstance(operation, FileCopyOperation):
        return review_item(
            operation_kind(operation),
            (("operation kind", operation_kind(operation)),),
        )
    fields: list[tuple[str, RenderableValue]] = [
        ("operation kind", operation_kind(operation)),
        ("label", operation.label),
        ("source_path", operation.source_path),
        ("target_path", operation.target_path),
    ]
    if operation.expected_input_bytes is not None:
        fields.append(("expected_input_bytes", operation.expected_input_bytes))
    return review_item(str(operation.label), fields)


def _publish_execution_item(result: OperationExecutionResult) -> WorkflowReviewItem:
    fields: list[tuple[str, RenderableValue]] = [
        ("operation_kind", result.operation_kind),
        ("succeeded", result.succeeded),
        ("returncode", result.returncode),
        ("execution_mode", result.execution_mode),
    ]
    if result.observed_outputs:
        fields.append(("observed_outputs", result.observed_outputs))
    return review_item(result.label, fields)


def _published_artifact_row(check: TierPublishCheck) -> TierPublishedArtifactRow:
    return TierPublishedArtifactRow(
        label=check.label,
        kind=check.kind,
        target_path=check.target_path,
        target_sha256=check.target_sha256,
        source_file_sha256=check.source_sha256,
        file_digest_match=check.digest_match,
        coherent=check.coherent,
    )


def _published_artifact_item(row: TierPublishedArtifactRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("kind", row.kind),
            ("target_path", row.target_path),
            ("target_sha256", row.target_sha256),
            ("source_file_sha256", row.source_file_sha256),
            ("file_digest_match", row.file_digest_match),
            ("coherent", row.coherent),
        ),
    )
