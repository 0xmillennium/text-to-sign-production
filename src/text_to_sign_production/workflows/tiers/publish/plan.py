from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import ProgressStageSpec
from text_to_sign_production.workflows.foundation.execution import (
    FileCopyOperation,
    OperationProgressSpec,
    WorkflowOperation,
)
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_PUBLISH_EXECUTE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import (
    TiersPublishPlan,
    TiersPublishTarget,
    TiersWorkflowInvariantError,
)
from text_to_sign_production.workflows.tiers.layout import TiersLayout
from text_to_sign_production.workflows.tiers.processing import TiersExecutionBundle


def build_tiers_publish_plan(
    *,
    bundle: TiersExecutionBundle,
    layout: TiersLayout,
) -> TiersPublishPlan:
    _validate_publish_alignment(bundle=bundle, layout=layout)
    targets = (
        *_build_report_targets(bundle=bundle, layout=layout),
        *_build_tiered_manifest_targets(bundle=bundle, layout=layout),
    )
    operations = (
        *_build_report_copy_operations(bundle=bundle, layout=layout),
        *_build_tiered_manifest_copy_operations(bundle=bundle, layout=layout),
    )
    return TiersPublishPlan(targets=targets, operations=operations)


def _validate_publish_alignment(
    *,
    bundle: TiersExecutionBundle,
    layout: TiersLayout,
) -> None:
    result = bundle.workflow_result
    if result.config != layout.config:
        raise TiersWorkflowInvariantError("publish layout config must match workflow result config")
    artifacts = result.output_summary.report_artifacts
    expected_report_targets = (
        layout.publish.summary_markdown_target_path,
        layout.publish.calibration_markdown_target_path,
        layout.publish.decision_detail_target_path,
        layout.publish.calibration_surfaces_target_path,
        layout.publish.calibration_detail_target_path,
        layout.publish.index_json_target_path,
    )
    if len(expected_report_targets) != 6:
        raise TiersWorkflowInvariantError("tiers publish layout report target set is invalid")
    if not (
        artifacts.summary_markdown_path
        and artifacts.calibration_markdown_path
        and artifacts.decision_detail_jsonl_path
        and artifacts.calibration_surfaces_json_path
        and artifacts.calibration_detail_json_path
        and artifacts.index_json_path
    ):
        raise TiersWorkflowInvariantError("tiers report artifact source set is invalid")
    for output in result.output_summary.tiered_manifest_outputs:
        expected_target = layout.stores.drive.manifests.tiered_manifest(
            output.tier,
            output.membership,
            output.split,
        ).path
        if expected_target == output.path:
            raise TiersWorkflowInvariantError(
                "tiered manifest publish source and target paths must differ"
            )


def _build_report_targets(
    *,
    bundle: TiersExecutionBundle,
    layout: TiersLayout,
) -> tuple[TiersPublishTarget, ...]:
    artifacts = bundle.workflow_result.output_summary.report_artifacts
    return (
        TiersPublishTarget(
            label="publish report [summary]",
            kind="report_file",
            source_path=artifacts.summary_markdown_path,
            target_path=layout.publish.summary_markdown_target_path,
        ),
        TiersPublishTarget(
            label="publish report [calibration]",
            kind="report_file",
            source_path=artifacts.calibration_markdown_path,
            target_path=layout.publish.calibration_markdown_target_path,
        ),
        TiersPublishTarget(
            label="publish report [decision detail]",
            kind="report_file",
            source_path=artifacts.decision_detail_jsonl_path,
            target_path=layout.publish.decision_detail_target_path,
        ),
        TiersPublishTarget(
            label="publish report [calibration surfaces]",
            kind="report_file",
            source_path=artifacts.calibration_surfaces_json_path,
            target_path=layout.publish.calibration_surfaces_target_path,
        ),
        TiersPublishTarget(
            label="publish report [calibration detail]",
            kind="report_file",
            source_path=artifacts.calibration_detail_json_path,
            target_path=layout.publish.calibration_detail_target_path,
        ),
        TiersPublishTarget(
            label="publish report [index]",
            kind="report_file",
            source_path=artifacts.index_json_path,
            target_path=layout.publish.index_json_target_path,
        ),
    )


def _build_tiered_manifest_targets(
    *,
    bundle: TiersExecutionBundle,
    layout: TiersLayout,
) -> tuple[TiersPublishTarget, ...]:
    return tuple(
        TiersPublishTarget(
            label=(f"publish tiered manifest [{output.tier}/{output.membership}/{output.split}]"),
            kind="tiered_manifest_file",
            source_path=output.path,
            target_path=layout.stores.drive.manifests.tiered_manifest(
                output.tier,
                output.membership,
                output.split,
            ).path,
            tier=output.tier,
            membership=output.membership,
            split=output.split,
        )
        for output in bundle.workflow_result.output_summary.tiered_manifest_outputs
    )


def _build_report_copy_operations(
    *,
    bundle: TiersExecutionBundle,
    layout: TiersLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_report_targets(bundle=bundle, layout=layout)
    )


def _build_tiered_manifest_copy_operations(
    *,
    bundle: TiersExecutionBundle,
    layout: TiersLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_tiered_manifest_targets(bundle=bundle, layout=layout)
    )


def _file_copy_operation(target: TiersPublishTarget) -> FileCopyOperation:
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
            workflow_id=TIERS_WORKFLOW_NAME,
            stage_id=TIERS_STAGE_PUBLISH_EXECUTE,
            label="Publish tiers workflow artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.tiers.publish",
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
