from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.integrity import sha256_file
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveVerifyOperation,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import (
    SourceMemberTreeReceipt,
    source_member_tree_receipt,
)
from text_to_sign_production.workflows.gate.contracts import (
    GatePublishCheck,
    GatePublishPlan,
    GatePublishTarget,
    GatePublishVerification,
    GateSplitArchivePublishPlan,
)


def verify_gate_publish(
    plan: GatePublishPlan,
) -> GatePublishVerification:
    return GatePublishVerification(
        checks=(
            *tuple(_publish_check(target, plan.operations) for target in plan.targets),
            *tuple(
                _expected_absent_dropped_archive_check(split_plan)
                for split_plan in plan.split_archive_plans
                if not split_plan.dropped_archive_planned
            ),
        )
    )


def _publish_check(
    target: GatePublishTarget,
    operations: tuple[WorkflowOperation, ...],
) -> GatePublishCheck:
    source_exists = target.source_path.exists()
    target_exists = target.target_path.exists()
    source_sha256 = target.source_sha256
    target_sha256 = (
        _file_sha256(target.target_path) if target_exists and target.target_path.is_file() else None
    )
    digest_match = None
    coherent = True
    message = None
    if target.kind in {"report_file", "manifest_file"}:
        observed_source_sha256 = (
            _file_sha256(target.source_path)
            if source_exists and target.source_path.is_file()
            else None
        )
        if observed_source_sha256 != source_sha256:
            coherent = False
            message = (
                "source digest no longer matches current-execution receipt: "
                f"expected {source_sha256}, observed {observed_source_sha256}"
            )
        digest_match = (
            source_sha256 is not None
            and target_sha256 is not None
            and source_sha256 == target_sha256
        )
    elif target.kind == "archive_file":
        planned_source_tree = target.source_member_tree
        observed_source_tree = _source_member_tree_receipt(target)
        archive_verification_planned = _archive_verification_planned(target, operations)
        target_member_count = target.expected_member_count if target_exists else None
        target_member_listing_digest = (
            target.expected_member_listing_digest if target_exists else None
        )
        coherent = (
            planned_source_tree is not None
            and observed_source_tree is not None
            and archive_verification_planned
            and target_sha256 is not None
            and observed_source_tree.member_count == planned_source_tree.member_count
            and (
                observed_source_tree.member_listing_digest
                == planned_source_tree.member_listing_digest
            )
            and observed_source_tree.member_tree_digest == planned_source_tree.member_tree_digest
        )
        if not coherent:
            message = (
                "archive source member tree or publish verification plan is incoherent: "
                f"planned_source_tree={planned_source_tree}, "
                f"observed_source_tree={observed_source_tree}, "
                f"archive_verification_planned={archive_verification_planned}, "
                f"target_member_count={target_member_count}, "
                f"target_member_listing_digest={target_member_listing_digest}"
            )
        else:
            message = (
                "archive member count and listing digest are projected from the publish plan; "
                "target archive membership is enforced by the planned ArchiveVerifyOperation"
            )
        return GatePublishCheck(
            label=target.label,
            kind=target.kind,
            source_path=target.source_path,
            source_exists=source_exists,
            target_path=target.target_path,
            target_exists=target_exists,
            source_sha256=source_sha256,
            target_sha256=target_sha256,
            digest_match=None,
            source_execution_id=target.source_execution_id,
            source_member_count=(
                None if planned_source_tree is None else planned_source_tree.member_count
            ),
            source_member_listing_digest=(
                None if planned_source_tree is None else planned_source_tree.member_listing_digest
            ),
            source_member_tree_digest=(
                None if planned_source_tree is None else planned_source_tree.member_tree_digest
            ),
            observed_source_member_count=(
                None if observed_source_tree is None else observed_source_tree.member_count
            ),
            observed_source_member_listing_digest=(
                None if observed_source_tree is None else observed_source_tree.member_listing_digest
            ),
            observed_source_member_tree_digest=(
                None if observed_source_tree is None else observed_source_tree.member_tree_digest
            ),
            expected_member_count=target.expected_member_count,
            target_member_count=target_member_count,
            expected_member_listing_digest=target.expected_member_listing_digest,
            target_member_listing_digest=target_member_listing_digest,
            source_members_preview=(
                () if planned_source_tree is None else planned_source_tree.members_preview
            ),
            target_members_preview=target.members_preview if target_exists else (),
            coherent=coherent,
            message=message,
        )
    return GatePublishCheck(
        label=target.label,
        kind=target.kind,
        source_path=target.source_path,
        source_exists=source_exists,
        target_path=target.target_path,
        target_exists=target_exists,
        source_sha256=source_sha256,
        target_sha256=target_sha256,
        digest_match=digest_match,
        source_execution_id=target.source_execution_id,
        expected_member_count=target.expected_member_count,
        target_member_count=None,
        expected_member_listing_digest=target.expected_member_listing_digest,
        target_member_listing_digest=None,
        coherent=coherent,
        message=message,
    )


def _expected_absent_dropped_archive_check(
    split_plan: GateSplitArchivePublishPlan,
) -> GatePublishCheck:
    return GatePublishCheck(
        label=f"dropped sample archive absent as expected [{split_plan.split}]",
        kind="archive_file",
        source_path=split_plan.dropped_archive_path,
        source_exists=False,
        target_path=split_plan.dropped_archive_path,
        target_exists=split_plan.dropped_archive_path.exists(),
        expected_member_count=split_plan.dropped_archive_member_count,
        target_member_count=None,
        expected_absent=True,
        coherent=not split_plan.dropped_archive_planned,
        message=(
            "dropped sample archive absent as expected: "
            f"{split_plan.dropped_archive_not_planned_reason}"
        ),
    )


def _file_sha256(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None


def _source_member_tree_receipt(target: GatePublishTarget) -> SourceMemberTreeReceipt | None:
    try:
        return source_member_tree_receipt(
            f"observed {target.label}",
            target.source_path,
            target.expected_members,
            execution_id=target.source_execution_id or "",
        )
    except OSError:
        return None


def _archive_verification_planned(
    target: GatePublishTarget,
    operations: tuple[WorkflowOperation, ...],
) -> bool:
    return any(
        isinstance(operation, ArchiveVerifyOperation)
        and operation.archive_path == target.target_path
        and operation.expected_members == target.expected_members
        for operation in operations
    )
