from __future__ import annotations

from text_to_sign_production.workflows.foundation.execution import (
    ArchiveCreateOperation,
    ArchiveVerifyOperation,
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
from text_to_sign_production.workflows.gate.contracts import (
    GatePublishCheck,
    GatePublishedArtifactRow,
    GatePublishExecution,
    GatePublishPlan,
    GatePublishResult,
    GatePublishTarget,
    GatePublishTargetRow,
    GatePublishVerification,
    GateSplitArchivePublishPlan,
)


def review_publish_plan(
    plan: GatePublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_plan_summary(plan)


def review_publish_plan_summary(
    plan: GatePublishPlan,
) -> tuple[WorkflowReviewSection, ...]:
    report_count = sum(1 for target in plan.targets if target.kind == "report_file")
    manifest_count = sum(1 for target in plan.targets if target.kind == "manifest_file")
    archive_count = sum(1 for target in plan.targets if target.kind == "archive_file")
    archive_member_total = sum(
        target.expected_member_count or 0
        for target in plan.targets
        if target.kind == "archive_file"
    )
    return (
        review_section(
            "Publish plan summary",
            (
                review_item(
                    "publish",
                    (
                        ("planned target count", len(plan.targets)),
                        ("report file count", report_count),
                        ("manifest file count", manifest_count),
                        ("archive target count", archive_count),
                        ("archive member total", archive_member_total),
                        ("operation count", len(plan.operations)),
                    ),
                ),
            ),
        ),
        review_section(
            "Publish split archive plan",
            tuple(_split_archive_plan_item(row) for row in plan.split_archive_plans),
        ),
    )


def review_publish_plan_detail(
    plan: GatePublishPlan,
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
    execution: GatePublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_execution_summary(execution)


def review_publish_execution_summary(
    execution: GatePublishExecution,
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
    execution: GatePublishExecution,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Publish execution details",
            tuple(_publish_execution_item(result) for result in execution.execution.results),
        ),
    )


def review_publish_verification(
    verification: GatePublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_verification_summary(verification)


def review_publish_verification_summary(
    verification: GatePublishVerification,
) -> tuple[WorkflowReviewSection, ...]:
    failed_checks = tuple(
        check for check in verification.checks if not _publish_check_succeeded(check)
    )
    archive_checks = tuple(check for check in verification.checks if check.kind == "archive_file")
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
                        ("archive target count", len(archive_checks)),
                        (
                            "archive target member total",
                            sum(check.target_member_count or 0 for check in archive_checks),
                        ),
                        ("succeeded", verification.succeeded),
                    ),
                ),
                *_publish_check_failure_items(failed_checks),
            ),
        ),
    )


def review_publish_verification_detail(
    verification: GatePublishVerification,
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
                        ("source_member_count", check.source_member_count),
                        (
                            "source_member_listing_digest",
                            check.source_member_listing_digest,
                        ),
                        ("source_member_tree_digest", check.source_member_tree_digest),
                        (
                            "observed_source_member_count",
                            check.observed_source_member_count,
                        ),
                        (
                            "observed_source_member_listing_digest",
                            check.observed_source_member_listing_digest,
                        ),
                        (
                            "observed_source_member_tree_digest",
                            check.observed_source_member_tree_digest,
                        ),
                        ("target_path", check.target_path),
                        ("target_exists", check.target_exists),
                        ("target_sha256", check.target_sha256),
                        ("file_digest_match", check.digest_match),
                        ("expected_member_count", check.expected_member_count),
                        ("target_member_count", check.target_member_count),
                        (
                            "expected_member_listing_digest",
                            check.expected_member_listing_digest,
                        ),
                        (
                            "target_member_listing_digest",
                            check.target_member_listing_digest,
                        ),
                        ("source_members_preview", check.source_members_preview),
                        ("target_members_preview", check.target_members_preview),
                        ("expected_absent", check.expected_absent),
                        ("coherent", check.coherent),
                        ("message", check.message),
                    ),
                )
                for check in verification.checks
            ),
        ),
    )


def review_publish_result(
    result: GatePublishResult,
) -> tuple[WorkflowReviewSection, ...]:
    return review_publish_result_summary(result)


def review_publish_result_summary(
    result: GatePublishResult,
) -> tuple[WorkflowReviewSection, ...]:
    successful_results = result.execution.execution.successful_results()
    failed_results = result.execution.execution.failed_results()
    verified_targets = tuple(
        check for check in result.verification.checks if _publish_check_succeeded(check)
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
    result: GatePublishResult,
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
    checks: tuple[GatePublishCheck, ...],
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
                ("file_digest_match", check.digest_match),
                ("expected_absent", check.expected_absent),
                ("coherent", check.coherent),
                ("message", check.message),
            ),
        )
        for check in checks[:limit]
    )


def _publish_check_succeeded(check: GatePublishCheck) -> bool:
    if check.expected_absent:
        return not check.target_exists and check.coherent
    return (
        check.source_exists
        and check.target_exists
        and check.coherent
        and check.digest_match is not False
    )


def _split_archive_plan_item(row: GateSplitArchivePublishPlan) -> WorkflowReviewItem:
    return review_item(
        row.split,
        (
            ("passed archive planned", row.passed_archive_planned),
            ("passed archive path", row.passed_archive_path),
            ("passed archive member count", row.passed_archive_member_count),
            ("passed archive member type", row.passed_archive_member_type),
            ("dropped archive planned", row.dropped_archive_planned),
            ("dropped archive path", row.dropped_archive_path),
            ("dropped archive member count", row.dropped_archive_member_count),
            ("dropped archive member type", row.dropped_archive_member_type),
            (
                "dropped archive not planned reason",
                row.dropped_archive_not_planned_reason,
            ),
        ),
    )


def _publish_target_item(target: GatePublishTarget) -> WorkflowReviewItem:
    row = GatePublishTargetRow(
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
            ("source_sha256", target.source_sha256),
            ("source_execution_id", target.source_execution_id),
            (
                "source_member_tree_digest",
                (
                    None
                    if target.source_member_tree is None
                    else target.source_member_tree.member_tree_digest
                ),
            ),
            ("expected_member_count", target.expected_member_count),
            (
                "expected_member_listing_digest",
                target.expected_member_listing_digest,
            ),
            ("members_preview", target.members_preview),
        ),
    )


def _publish_operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    fields: list[tuple[str, RenderableValue]] = [
        ("operation kind", operation_kind(operation)),
    ]
    item_label: str
    if isinstance(operation, FileCopyOperation):
        item_label = operation.label
        fields.extend(
            (
                ("source_path", operation.source_path),
                ("target_path", operation.target_path),
            )
        )
    elif isinstance(operation, ArchiveCreateOperation):
        item_label = operation.label
        fields.extend(
            (
                ("archive_path", operation.archive_path),
                ("source_root", operation.source_root),
                ("expected member count", len(operation.members)),
            )
        )
    elif isinstance(operation, ArchiveVerifyOperation):
        item_label = operation.label
        fields.append(("expected member count", len(operation.expected_members)))
        fields.append(("archive_path", operation.archive_path))
    else:
        item_label = operation_kind(operation)
    return review_item(str(item_label), fields)


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


def _published_artifact_row(check: GatePublishCheck) -> GatePublishedArtifactRow:
    return GatePublishedArtifactRow(
        label=check.label,
        kind=check.kind,
        target_path=check.target_path,
        target_sha256=check.target_sha256,
        source_file_sha256=check.source_sha256,
        file_digest_match=check.digest_match,
        source_member_count=check.source_member_count,
        source_member_listing_digest=check.source_member_listing_digest,
        source_member_tree_digest=check.source_member_tree_digest,
        target_member_count=check.target_member_count,
        target_member_listing_digest=check.target_member_listing_digest,
        members_preview=check.target_members_preview,
        coherent=check.coherent,
    )


def _published_artifact_item(row: GatePublishedArtifactRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("kind", row.kind),
            ("target_path", row.target_path),
            ("target_sha256", row.target_sha256),
            ("source_file_sha256", row.source_file_sha256),
            ("file_digest_match", row.file_digest_match),
            ("source_member_count", row.source_member_count),
            ("source_member_listing_digest", row.source_member_listing_digest),
            ("source_member_tree_digest", row.source_member_tree_digest),
            ("target_member_count", row.target_member_count),
            ("target_member_listing_digest", row.target_member_listing_digest),
            ("members_preview", row.members_preview),
            ("coherent", row.coherent),
        ),
    )
