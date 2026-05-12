from __future__ import annotations

from pathlib import Path

from text_to_sign_production.artifacts.store import SamplesTopology, build_artifact_topology
from text_to_sign_production.core import build_repo_roots
from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveCreateOperation,
    ArchiveVerifyOperation,
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import (
    WrittenFileReceipt,
    source_member_tree_receipt,
)
from text_to_sign_production.workflows.gate.constants import (
    GATE_STAGE_PUBLISH_EXECUTE,
    GATE_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.gate.contracts import (
    GatePublishPlan,
    GatePublishTarget,
    GateSplitArchivePublishPlan,
    GateWorkflowInvariantError,
    GateWrittenReportArtifacts,
)
from text_to_sign_production.workflows.gate.layout import (
    GateLayout,
    GatePublishSplitLayout,
)
from text_to_sign_production.workflows.gate.processing import (
    GateExecutionBundle,
    GateSplitProcessingResult,
)


def build_gate_publish_plan(
    *,
    bundle: GateExecutionBundle,
    report_artifacts: GateWrittenReportArtifacts,
    layout: GateLayout,
) -> GatePublishPlan:
    _validate_publish_alignment(bundle=bundle, report_artifacts=report_artifacts, layout=layout)
    targets = (
        *_build_report_targets(report_artifacts=report_artifacts, layout=layout),
        *_build_manifest_targets(bundle=bundle, layout=layout),
        *_build_archive_targets(bundle=bundle, layout=layout),
    )
    operations = (
        *_build_report_copy_operations(report_artifacts=report_artifacts, layout=layout),
        *_build_manifest_copy_operations(bundle=bundle, layout=layout),
        *_build_archive_operations(bundle=bundle, layout=layout),
    )
    return GatePublishPlan(
        targets=targets,
        operations=operations,
        split_archive_plans=_build_split_archive_plans(bundle=bundle, layout=layout),
    )


def _validate_publish_alignment(
    *,
    bundle: GateExecutionBundle,
    report_artifacts: GateWrittenReportArtifacts,
    layout: GateLayout,
) -> None:
    result = bundle.workflow_result
    output_summary = result.output_summary
    if result.config != layout.config:
        raise GateWorkflowInvariantError("publish layout config must match workflow result config")
    if any(
        receipt.execution_id != result.execution_id
        for receipt in _report_artifact_receipts(report_artifacts)
    ):
        raise GateWorkflowInvariantError("report receipts do not belong to this gate execution")
    expected_manifests = tuple(
        (manifest.partition, manifest.split, manifest.path)
        for manifest in layout.outputs.manifest_outputs
    )
    observed_manifests = tuple(
        (manifest.partition, manifest.split, manifest.path)
        for manifest in bundle.written_manifest_artifacts
    )
    if observed_manifests != expected_manifests:
        raise GateWorkflowInvariantError("manifest output paths are misaligned")
    if any(
        manifest.execution_id != result.execution_id
        for manifest in bundle.written_manifest_artifacts
    ):
        raise GateWorkflowInvariantError("manifest receipts do not belong to this gate execution")
    if any(
        payload.execution_id != result.execution_id
        for payload in bundle.written_payload_artifacts
    ):
        raise GateWorkflowInvariantError("payload receipts do not belong to this gate execution")
    _validate_payload_receipts(bundle)
    if output_summary.passed_samples_root != layout.outputs.passed_samples_root:
        raise GateWorkflowInvariantError("passed samples source root is misaligned")
    if output_summary.dropped_samples_root != layout.outputs.dropped_samples_root:
        raise GateWorkflowInvariantError("dropped samples source root is misaligned")
    if report_artifacts.summary_markdown_path != layout.reports.summary_markdown_path:
        raise GateWorkflowInvariantError("summary report source path is misaligned")
    if (
        report_artifacts.processing_summary_jsonl_path
        != layout.reports.processing_summary_jsonl_path
    ):
        raise GateWorkflowInvariantError("processing summary source path is misaligned")
    if report_artifacts.processing_detail_jsonl_path != layout.reports.processing_detail_jsonl_path:
        raise GateWorkflowInvariantError("processing detail source path is misaligned")
    if report_artifacts.gate_summary_jsonl_path != layout.reports.gate_summary_jsonl_path:
        raise GateWorkflowInvariantError("gate summary source path is misaligned")
    if report_artifacts.gate_detail_jsonl_path != layout.reports.gate_detail_jsonl_path:
        raise GateWorkflowInvariantError("gate detail source path is misaligned")
    if (
        report_artifacts.source_issue_summary_jsonl_path
        != layout.reports.source_issue_summary_jsonl_path
    ):
        raise GateWorkflowInvariantError("source issue summary source path is misaligned")
    if (
        report_artifacts.source_issue_detail_jsonl_path
        != layout.reports.source_issue_detail_jsonl_path
    ):
        raise GateWorkflowInvariantError("source issue detail source path is misaligned")
    if report_artifacts.index_json_path != layout.reports.index_json_path:
        raise GateWorkflowInvariantError("report index source path is misaligned")
    for path in _report_artifact_paths(report_artifacts):
        if not path.is_file():
            raise GateWorkflowInvariantError(f"Report artifact is not materialized: {path}")
    for manifest in bundle.written_manifest_artifacts:
        if not manifest.path.is_file():
            raise GateWorkflowInvariantError(
                f"Manifest artifact is not materialized: {manifest.path}"
            )
    for payload in bundle.written_payload_artifacts:
        if not payload.path.is_file():
            raise GateWorkflowInvariantError(
                f"Payload artifact is not materialized: {payload.path}"
            )


def _build_report_targets(
    *,
    report_artifacts: GateWrittenReportArtifacts,
    layout: GateLayout,
) -> tuple[GatePublishTarget, ...]:
    artifacts = report_artifacts
    return (
        GatePublishTarget(
            label="publish summary markdown",
            kind="report_file",
            source_path=artifacts.summary_markdown_path,
            target_path=layout.publish.summary_markdown_target_path,
            source_sha256=artifacts.summary_markdown.sha256,
            source_execution_id=artifacts.summary_markdown.execution_id,
        ),
        GatePublishTarget(
            label="publish processing summary",
            kind="report_file",
            source_path=artifacts.processing_summary_jsonl_path,
            target_path=layout.publish.processing_summary_jsonl_target_path,
            source_sha256=artifacts.processing_summary_jsonl.sha256,
            source_execution_id=artifacts.processing_summary_jsonl.execution_id,
        ),
        GatePublishTarget(
            label="publish processing detail",
            kind="report_file",
            source_path=artifacts.processing_detail_jsonl_path,
            target_path=layout.publish.processing_detail_jsonl_target_path,
            source_sha256=artifacts.processing_detail_jsonl.sha256,
            source_execution_id=artifacts.processing_detail_jsonl.execution_id,
        ),
        GatePublishTarget(
            label="publish gate summary",
            kind="report_file",
            source_path=artifacts.gate_summary_jsonl_path,
            target_path=layout.publish.gate_summary_jsonl_target_path,
            source_sha256=artifacts.gate_summary_jsonl.sha256,
            source_execution_id=artifacts.gate_summary_jsonl.execution_id,
        ),
        GatePublishTarget(
            label="publish gate detail",
            kind="report_file",
            source_path=artifacts.gate_detail_jsonl_path,
            target_path=layout.publish.gate_detail_jsonl_target_path,
            source_sha256=artifacts.gate_detail_jsonl.sha256,
            source_execution_id=artifacts.gate_detail_jsonl.execution_id,
        ),
        GatePublishTarget(
            label="publish source issue summary",
            kind="report_file",
            source_path=artifacts.source_issue_summary_jsonl_path,
            target_path=layout.publish.source_issue_summary_jsonl_target_path,
            source_sha256=artifacts.source_issue_summary_jsonl.sha256,
            source_execution_id=artifacts.source_issue_summary_jsonl.execution_id,
        ),
        GatePublishTarget(
            label="publish source issue detail",
            kind="report_file",
            source_path=artifacts.source_issue_detail_jsonl_path,
            target_path=layout.publish.source_issue_detail_jsonl_target_path,
            source_sha256=artifacts.source_issue_detail_jsonl.sha256,
            source_execution_id=artifacts.source_issue_detail_jsonl.execution_id,
        ),
        GatePublishTarget(
            label="publish report index",
            kind="report_file",
            source_path=artifacts.index_json_path,
            target_path=layout.publish.index_json_target_path,
            source_sha256=artifacts.index_json.sha256,
            source_execution_id=artifacts.index_json.execution_id,
        ),
    )


def _report_artifact_paths(artifacts: GateWrittenReportArtifacts) -> tuple[Path, ...]:
    return (
        artifacts.summary_markdown_path,
        artifacts.processing_summary_jsonl_path,
        artifacts.processing_detail_jsonl_path,
        artifacts.gate_summary_jsonl_path,
        artifacts.gate_detail_jsonl_path,
        artifacts.source_issue_summary_jsonl_path,
        artifacts.source_issue_detail_jsonl_path,
        artifacts.index_json_path,
    )


def _report_artifact_receipts(
    artifacts: GateWrittenReportArtifacts,
) -> tuple[WrittenFileReceipt, ...]:
    return (
        artifacts.summary_markdown,
        artifacts.processing_summary_jsonl,
        artifacts.processing_detail_jsonl,
        artifacts.gate_summary_jsonl,
        artifacts.gate_detail_jsonl,
        artifacts.source_issue_summary_jsonl,
        artifacts.source_issue_detail_jsonl,
        artifacts.index_json,
    )


def _build_manifest_targets(
    *,
    bundle: GateExecutionBundle,
    layout: GateLayout,
) -> tuple[GatePublishTarget, ...]:
    targets: list[GatePublishTarget] = []
    for manifest in bundle.written_manifest_artifacts:
        publish_split = _publish_split_layout(layout, manifest.split)
        if manifest.partition == "passed":
            target_path = publish_split.passed_manifest_target_path
        elif manifest.partition == "dropped":
            target_path = publish_split.dropped_manifest_target_path
        else:
            raise GateWorkflowInvariantError(
                f"Unsupported manifest partition: {manifest.partition}"
            )
        targets.append(
            GatePublishTarget(
                label=f"publish {manifest.partition} manifest [{manifest.split}]",
                kind="manifest_file",
                source_path=manifest.path,
                target_path=target_path,
                source_sha256=manifest.sha256,
                source_execution_id=manifest.execution_id,
            )
        )
    return tuple(targets)


def _build_archive_targets(
    *,
    bundle: GateExecutionBundle,
    layout: GateLayout,
) -> tuple[GatePublishTarget, ...]:
    targets: list[GatePublishTarget] = []
    for split_result in _split_results_by_config_order(bundle):
        publish_split = _publish_split_layout(layout, split_result.split)
        passed_members = _passed_archive_members(bundle, split_result.split)
        if passed_members:
            targets.append(
                GatePublishTarget(
                    label=f"publish passed samples [{split_result.split}]",
                    kind="archive_file",
                    source_path=bundle.workflow_result.output_summary.passed_samples_root,
                    target_path=publish_split.passed_archive_path,
                    source_member_tree=source_member_tree_receipt(
                        f"source member tree passed samples [{split_result.split}]",
                        bundle.workflow_result.output_summary.passed_samples_root,
                        passed_members,
                        execution_id=bundle.workflow_result.execution_id,
                        source_lineage=("gate_passed_samples", split_result.split),
                    ),
                    source_execution_id=bundle.workflow_result.execution_id,
                    expected_members=passed_members,
                )
            )
        dropped_members = _dropped_archive_members(bundle, split_result.split)
        if dropped_members:
            targets.append(
                GatePublishTarget(
                    label=f"publish dropped samples [{split_result.split}]",
                    kind="archive_file",
                    source_path=bundle.workflow_result.output_summary.dropped_samples_root,
                    target_path=publish_split.dropped_archive_path,
                    source_member_tree=source_member_tree_receipt(
                        f"source member tree dropped samples [{split_result.split}]",
                        bundle.workflow_result.output_summary.dropped_samples_root,
                        dropped_members,
                        execution_id=bundle.workflow_result.execution_id,
                        source_lineage=("gate_dropped_samples", split_result.split),
                    ),
                    source_execution_id=bundle.workflow_result.execution_id,
                    expected_members=dropped_members,
                )
            )
    return tuple(targets)


def _build_report_copy_operations(
    *,
    report_artifacts: GateWrittenReportArtifacts,
    layout: GateLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_report_targets(report_artifacts=report_artifacts, layout=layout)
    )


def _build_manifest_copy_operations(
    *,
    bundle: GateExecutionBundle,
    layout: GateLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_manifest_targets(bundle=bundle, layout=layout)
    )


def _build_split_archive_plans(
    *,
    bundle: GateExecutionBundle,
    layout: GateLayout,
) -> tuple[GateSplitArchivePublishPlan, ...]:
    rows: list[GateSplitArchivePublishPlan] = []
    for split_result in _split_results_by_config_order(bundle):
        publish_split = _publish_split_layout(layout, split_result.split)
        passed_members = _passed_archive_members(bundle, split_result.split)
        dropped_members = _dropped_archive_members(bundle, split_result.split)
        rows.append(
            GateSplitArchivePublishPlan(
                split=split_result.split,
                passed_archive_path=publish_split.passed_archive_path,
                passed_archive_planned=bool(passed_members),
                passed_archive_member_count=len(passed_members),
                dropped_archive_path=publish_split.dropped_archive_path,
                dropped_archive_planned=bool(dropped_members),
                dropped_archive_member_count=len(dropped_members),
                dropped_archive_not_planned_reason=(
                    None
                    if dropped_members
                    else _dropped_archive_not_planned_reason(bundle, split_result)
                ),
            )
        )
    return tuple(rows)


def _build_archive_operations(
    *,
    bundle: GateExecutionBundle,
    layout: GateLayout,
) -> tuple[WorkflowOperation, ...]:
    operations: list[WorkflowOperation] = []
    for split_result in _split_results_by_config_order(bundle):
        publish_split = _publish_split_layout(layout, split_result.split)
        passed_members = _passed_archive_members(bundle, split_result.split)
        if passed_members:
            operations.extend(
                _archive_create_verify_operations(
                    label=f"publish passed samples [{split_result.split}]",
                    source_root=bundle.workflow_result.output_summary.passed_samples_root,
                    archive_path=publish_split.passed_archive_path,
                    members=passed_members,
                )
            )
        dropped_members = _dropped_archive_members(bundle, split_result.split)
        if dropped_members:
            operations.extend(
                _archive_create_verify_operations(
                    label=f"publish dropped samples [{split_result.split}]",
                    source_root=bundle.workflow_result.output_summary.dropped_samples_root,
                    archive_path=publish_split.dropped_archive_path,
                    members=dropped_members,
                )
            )
    return tuple(operations)


def _dropped_archive_not_planned_reason(
    bundle: GateExecutionBundle,
    split_result: GateSplitProcessingResult,
) -> str:
    if not split_result.dropped_entries:
        return "no dropped manifest entries for split"
    if not bundle.workflow_result.config.materialize_dropped_debug_payloads:
        return "dropped debug payload materialization is disabled"
    return "no dropped debug payload refs were materialized"


def _file_copy_operation(target: GatePublishTarget) -> FileCopyOperation:
    expected_input_bytes = _maybe_input_bytes(target.source_path)
    return FileCopyOperation(
        label=target.label,
        source_path=target.source_path,
        target_path=target.target_path,
        failure_message=f"Failed to {target.label}",
        overwrite_policy="atomic_replace",
        expected_input_bytes=expected_input_bytes,
        progress=_publish_progress_spec(expected_input_bytes),
    )


def _archive_create_verify_operations(
    *,
    label: str,
    source_root: Path,
    archive_path: Path,
    members: tuple[str, ...],
) -> tuple[WorkflowOperation, ...]:
    return (
        ArchiveCreateOperation(
            label=label,
            archive_path=archive_path,
            source_root=source_root,
            members=members,
            failure_message=f"Failed to create {label}",
            overwrite_policy="atomic_replace",
            progress=_publish_archive_progress_spec(
                label=label,
                operation_kind="archive_create",
                expected_member_count=len(members),
                total_semantics="archive members to create",
            ),
        ),
        ArchiveVerifyOperation(
            label=f"verify {label}",
            archive_path=archive_path,
            expected_members=members,
            failure_message=f"Failed to verify {label}",
            progress=_publish_archive_progress_spec(
                label=f"verify {label}",
                operation_kind="archive_verify",
                expected_member_count=len(members),
                total_semantics="expected archive members to verify",
            ),
        ),
    )


def _passed_archive_members(bundle: GateExecutionBundle, split: str) -> tuple[str, ...]:
    samples_topology = _drive_samples_topology(bundle)
    return tuple(
        samples_topology.archive_member(entry.split, entry.sample_id).path.as_posix()
        for split_result in bundle.split_results
        if split_result.split == split
        for entry in split_result.passed_entries
    )


def _dropped_archive_members(bundle: GateExecutionBundle, split: str) -> tuple[str, ...]:
    samples_topology = _drive_samples_topology(bundle)
    return tuple(
        samples_topology.archive_member(entry.split, entry.sample_id).path.as_posix()
        for split_result in bundle.split_results
        if split_result.split == split
        for entry in split_result.dropped_entries
        if entry.debug_ref is not None
    )


def _validate_payload_receipts(bundle: GateExecutionBundle) -> None:
    expected = {
        payload.payload_ref
        for split_result in bundle.split_results
        for payload in (*split_result.passed_payloads, *split_result.dropped_debug_payloads)
    }
    observed = {payload.payload_ref for payload in bundle.written_payload_artifacts}
    if observed != expected:
        raise GateWorkflowInvariantError(
            f"payload receipt set does not match planned payloads: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if len(observed) != len(bundle.written_payload_artifacts):
        raise GateWorkflowInvariantError("payload receipts must not contain duplicate refs")
    for payload in bundle.written_payload_artifacts:
        receipt_identity = payload.receipt.physical_sample
        if receipt_identity is None:
            raise GateWorkflowInvariantError("payload receipt is missing physical sample identity")
        if (
            receipt_identity.split != payload.split
            or receipt_identity.sample_id != payload.sample_id
        ):
            raise GateWorkflowInvariantError(
                "payload receipt physical identity does not match payload artifact"
            )


def _split_results_by_config_order(
    bundle: GateExecutionBundle,
) -> tuple[GateSplitProcessingResult, ...]:
    by_split = {result.split: result for result in bundle.split_results}
    ordered_results = []
    for split in bundle.workflow_result.config.splits:
        result = by_split.get(split)
        if result is None:
            raise GateWorkflowInvariantError(f"Missing split processing result: {split}")
        ordered_results.append(result)
    if len(by_split) != len(bundle.split_results):
        raise GateWorkflowInvariantError("Duplicate split processing results are not allowed")
    return tuple(ordered_results)


def _publish_split_layout(layout: GateLayout, split: str) -> GatePublishSplitLayout:
    for publish_split in layout.publish.splits:
        if publish_split.split == split:
            return publish_split
    raise GateWorkflowInvariantError(f"Missing publish layout for split: {split}")


def _drive_samples_topology(bundle: GateExecutionBundle) -> SamplesTopology:
    drive_topology = build_artifact_topology(
        build_repo_roots(bundle.workflow_result.config.drive_project_root)
    )
    return drive_topology.samples


def _publish_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=GATE_WORKFLOW_NAME,
            stage_id=GATE_STAGE_PUBLISH_EXECUTE,
            label="Publish gate workflow artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.gate.publish",
            split_behavior="global",
            operation_kind="publish",
            total_semantics="publish input bytes when known",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )


def _publish_archive_progress_spec(
    *,
    label: str,
    operation_kind: str,
    expected_member_count: int,
    total_semantics: str,
) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=GATE_WORKFLOW_NAME,
            stage_id=GATE_STAGE_PUBLISH_EXECUTE,
            label=label,
            unit="member",
            owner_module="text_to_sign_production.workflows.gate.publish",
            split_behavior="global",
            operation_kind=operation_kind,
            total_semantics=total_semantics,
            bar_eligible=False,
        ),
        expected_total=expected_member_count,
        live_owner="shell",
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None
