from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.workflows.foundation.execution import (
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.provenance import WrittenFileReceipt
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_PUBLISH_EXECUTE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import (
    TierPublishPlan,
    TierPublishTarget,
    TierWorkflowInvariantError,
    TierWrittenReportArtifacts,
)
from text_to_sign_production.workflows.tier.layout import TierLayout
from text_to_sign_production.workflows.tier.processing import TierExecutionBundle


def build_tier_publish_plan(
    *,
    bundle: TierExecutionBundle,
    report_artifacts: TierWrittenReportArtifacts,
    layout: TierLayout,
) -> TierPublishPlan:
    _validate_publish_alignment(bundle=bundle, report_artifacts=report_artifacts, layout=layout)
    targets = (
        *_build_report_targets(report_artifacts=report_artifacts, layout=layout),
        *_build_tiered_manifest_targets(bundle=bundle, layout=layout),
    )
    operations = (
        *_build_report_copy_operations(report_artifacts=report_artifacts, layout=layout),
        *_build_tiered_manifest_copy_operations(bundle=bundle, layout=layout),
    )
    return TierPublishPlan(targets=targets, operations=operations)


def _validate_publish_alignment(
    *,
    bundle: TierExecutionBundle,
    report_artifacts: TierWrittenReportArtifacts,
    layout: TierLayout,
) -> None:
    result = bundle.workflow_result
    if result.config != layout.config:
        raise TierWorkflowInvariantError("publish layout config must match workflow result config")
    if any(
        receipt.execution_id != result.execution_id
        for receipt in _report_artifact_receipts(report_artifacts)
    ):
        raise TierWorkflowInvariantError("report receipts do not belong to this tier execution")
    expected_report_targets = (
        layout.publish.summary_markdown_target_path,
        layout.publish.calibration_markdown_target_path,
        layout.publish.decision_detail_target_path,
        layout.publish.calibration_surfaces_target_path,
        layout.publish.calibration_detail_target_path,
        layout.publish.index_json_target_path,
    )
    if len(expected_report_targets) != 6:
        raise TierWorkflowInvariantError("tier publish layout report target set is invalid")
    if not (
        report_artifacts.summary_markdown_path
        and report_artifacts.calibration_markdown_path
        and report_artifacts.decision_detail_jsonl_path
        and report_artifacts.calibration_surfaces_json_path
        and report_artifacts.calibration_detail_json_path
        and report_artifacts.index_json_path
    ):
        raise TierWorkflowInvariantError("tier report artifact source set is invalid")
    for path in _report_artifact_paths(report_artifacts):
        if not path.is_file():
            raise TierWorkflowInvariantError(f"Tier report artifact is not materialized: {path}")
    for output in bundle.written_tiered_manifest_artifacts:
        if output.execution_id != result.execution_id:
            raise TierWorkflowInvariantError(
                "tiered manifest receipts do not belong to this tier execution"
            )
        if not output.path.is_file():
            raise TierWorkflowInvariantError(
                f"Tiered manifest artifact is not materialized: {output.path}"
            )
        expected_target = layout.stores.drive.manifests.tiered_manifest(
            output.tier,
            output.membership,
            output.split,
        ).path
        if expected_target == output.path:
            raise TierWorkflowInvariantError(
                "tiered manifest publish source and target paths must differ"
            )


def _build_report_targets(
    *,
    report_artifacts: TierWrittenReportArtifacts,
    layout: TierLayout,
) -> tuple[TierPublishTarget, ...]:
    artifacts = report_artifacts
    return (
        TierPublishTarget(
            label="publish report [summary]",
            kind="report_file",
            source_path=artifacts.summary_markdown_path,
            target_path=layout.publish.summary_markdown_target_path,
            source_sha256=artifacts.summary_markdown.sha256,
            source_execution_id=artifacts.summary_markdown.execution_id,
        ),
        TierPublishTarget(
            label="publish report [calibration]",
            kind="report_file",
            source_path=artifacts.calibration_markdown_path,
            target_path=layout.publish.calibration_markdown_target_path,
            source_sha256=artifacts.calibration_markdown.sha256,
            source_execution_id=artifacts.calibration_markdown.execution_id,
        ),
        TierPublishTarget(
            label="publish report [decision detail]",
            kind="report_file",
            source_path=artifacts.decision_detail_jsonl_path,
            target_path=layout.publish.decision_detail_target_path,
            source_sha256=artifacts.decision_detail_jsonl.sha256,
            source_execution_id=artifacts.decision_detail_jsonl.execution_id,
        ),
        TierPublishTarget(
            label="publish report [calibration surfaces]",
            kind="report_file",
            source_path=artifacts.calibration_surfaces_json_path,
            target_path=layout.publish.calibration_surfaces_target_path,
            source_sha256=artifacts.calibration_surfaces_json.sha256,
            source_execution_id=artifacts.calibration_surfaces_json.execution_id,
        ),
        TierPublishTarget(
            label="publish report [calibration detail]",
            kind="report_file",
            source_path=artifacts.calibration_detail_json_path,
            target_path=layout.publish.calibration_detail_target_path,
            source_sha256=artifacts.calibration_detail_json.sha256,
            source_execution_id=artifacts.calibration_detail_json.execution_id,
        ),
        TierPublishTarget(
            label="publish report [index]",
            kind="report_file",
            source_path=artifacts.index_json_path,
            target_path=layout.publish.index_json_target_path,
            source_sha256=artifacts.index_json.sha256,
            source_execution_id=artifacts.index_json.execution_id,
        ),
    )


def _report_artifact_paths(artifacts: TierWrittenReportArtifacts) -> tuple[Path, ...]:
    return (
        artifacts.summary_markdown_path,
        artifacts.calibration_markdown_path,
        artifacts.decision_detail_jsonl_path,
        artifacts.calibration_surfaces_json_path,
        artifacts.calibration_detail_json_path,
        artifacts.index_json_path,
    )


def _report_artifact_receipts(
    artifacts: TierWrittenReportArtifacts,
) -> tuple[WrittenFileReceipt, ...]:
    return (
        artifacts.summary_markdown,
        artifacts.calibration_markdown,
        artifacts.decision_detail_jsonl,
        artifacts.calibration_surfaces_json,
        artifacts.calibration_detail_json,
        artifacts.index_json,
    )


def _build_tiered_manifest_targets(
    *,
    bundle: TierExecutionBundle,
    layout: TierLayout,
) -> tuple[TierPublishTarget, ...]:
    return tuple(
        TierPublishTarget(
            label=(f"publish tiered manifest [{output.tier}/{output.membership}/{output.split}]"),
            kind="tiered_manifest_file",
            source_path=output.path,
            target_path=layout.stores.drive.manifests.tiered_manifest(
                output.tier,
                output.membership,
                output.split,
            ).path,
            source_sha256=output.sha256,
            source_execution_id=output.execution_id,
            tier=output.tier,
            membership=output.membership,
            split=output.split,
        )
        for output in bundle.written_tiered_manifest_artifacts
    )


def _build_report_copy_operations(
    *,
    report_artifacts: TierWrittenReportArtifacts,
    layout: TierLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_report_targets(report_artifacts=report_artifacts, layout=layout)
    )


def _build_tiered_manifest_copy_operations(
    *,
    bundle: TierExecutionBundle,
    layout: TierLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_tiered_manifest_targets(bundle=bundle, layout=layout)
    )


def _file_copy_operation(target: TierPublishTarget) -> FileCopyOperation:
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


def _publish_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=TIER_WORKFLOW_NAME,
            stage_id=TIER_STAGE_PUBLISH_EXECUTE,
            label="Publish tier workflow artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.tier.publish",
            split_behavior="global",
            operation_kind="publish",
            total_semantics="publish input bytes when known",
            bar_eligible=False,
        ),
        expected_total=expected_total,
        live_owner="shell",
    )


def _maybe_input_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None
