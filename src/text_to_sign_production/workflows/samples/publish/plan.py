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
from text_to_sign_production.workflows.samples.constants import (
    SAMPLES_STAGE_PUBLISH_EXECUTE,
    SAMPLES_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.samples.contracts import (
    SamplesPublishPlan,
    SamplesPublishTarget,
    SamplesWorkflowInvariantError,
)
from text_to_sign_production.workflows.samples.layout import (
    SamplesLayout,
    SamplesPublishSplitLayout,
)
from text_to_sign_production.workflows.samples.processing import (
    SamplesExecutionBundle,
    SamplesSplitProcessingResult,
)


def build_samples_publish_plan(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> SamplesPublishPlan:
    _validate_publish_alignment(bundle=bundle, layout=layout)
    targets = (
        *_build_report_targets(bundle=bundle, layout=layout),
        *_build_manifest_targets(bundle=bundle, layout=layout),
        *_build_archive_targets(bundle=bundle, layout=layout),
    )
    operations = (
        *_build_report_copy_operations(bundle=bundle, layout=layout),
        *_build_manifest_copy_operations(bundle=bundle, layout=layout),
        *_build_archive_operations(bundle=bundle, layout=layout),
    )
    return SamplesPublishPlan(targets=targets, operations=operations)


def _validate_publish_alignment(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> None:
    result = bundle.workflow_result
    output_summary = result.output_summary
    artifacts = output_summary.report_artifacts
    if result.config != layout.config:
        raise SamplesWorkflowInvariantError(
            "publish layout config must match workflow result config"
        )
    expected_manifests = tuple(
        (manifest.partition, manifest.split, manifest.path)
        for manifest in layout.outputs.manifest_outputs
    )
    observed_manifests = tuple(
        (manifest.partition, manifest.split, manifest.path)
        for manifest in output_summary.manifest_outputs
    )
    if observed_manifests != expected_manifests:
        raise SamplesWorkflowInvariantError("manifest output paths are misaligned")
    if output_summary.passed_samples_root != layout.outputs.passed_samples_root:
        raise SamplesWorkflowInvariantError("passed samples source root is misaligned")
    if output_summary.dropped_samples_root != layout.outputs.dropped_samples_root:
        raise SamplesWorkflowInvariantError("dropped samples source root is misaligned")
    if artifacts.summary_markdown_path != layout.reports.summary_markdown_path:
        raise SamplesWorkflowInvariantError("summary report source path is misaligned")
    if artifacts.processing_summary_jsonl_path != layout.reports.processing_summary_jsonl_path:
        raise SamplesWorkflowInvariantError("processing summary source path is misaligned")
    if artifacts.processing_detail_jsonl_path != layout.reports.processing_detail_jsonl_path:
        raise SamplesWorkflowInvariantError("processing detail source path is misaligned")
    if artifacts.gate_summary_jsonl_path != layout.reports.gate_summary_jsonl_path:
        raise SamplesWorkflowInvariantError("gate summary source path is misaligned")
    if artifacts.gate_detail_jsonl_path != layout.reports.gate_detail_jsonl_path:
        raise SamplesWorkflowInvariantError("gate detail source path is misaligned")
    if artifacts.source_issue_summary_jsonl_path != layout.reports.source_issue_summary_jsonl_path:
        raise SamplesWorkflowInvariantError("source issue summary source path is misaligned")
    if artifacts.source_issue_detail_jsonl_path != layout.reports.source_issue_detail_jsonl_path:
        raise SamplesWorkflowInvariantError("source issue detail source path is misaligned")
    if artifacts.index_json_path != layout.reports.index_json_path:
        raise SamplesWorkflowInvariantError("report index source path is misaligned")


def _build_report_targets(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> tuple[SamplesPublishTarget, ...]:
    artifacts = bundle.workflow_result.output_summary.report_artifacts
    return (
        SamplesPublishTarget(
            label="publish summary markdown",
            kind="report_file",
            source_path=artifacts.summary_markdown_path,
            target_path=layout.publish.summary_markdown_target_path,
        ),
        SamplesPublishTarget(
            label="publish processing summary",
            kind="report_file",
            source_path=artifacts.processing_summary_jsonl_path,
            target_path=layout.publish.processing_summary_jsonl_target_path,
        ),
        SamplesPublishTarget(
            label="publish processing detail",
            kind="report_file",
            source_path=artifacts.processing_detail_jsonl_path,
            target_path=layout.publish.processing_detail_jsonl_target_path,
        ),
        SamplesPublishTarget(
            label="publish gate summary",
            kind="report_file",
            source_path=artifacts.gate_summary_jsonl_path,
            target_path=layout.publish.gate_summary_jsonl_target_path,
        ),
        SamplesPublishTarget(
            label="publish gate detail",
            kind="report_file",
            source_path=artifacts.gate_detail_jsonl_path,
            target_path=layout.publish.gate_detail_jsonl_target_path,
        ),
        SamplesPublishTarget(
            label="publish source issue summary",
            kind="report_file",
            source_path=artifacts.source_issue_summary_jsonl_path,
            target_path=layout.publish.source_issue_summary_jsonl_target_path,
        ),
        SamplesPublishTarget(
            label="publish source issue detail",
            kind="report_file",
            source_path=artifacts.source_issue_detail_jsonl_path,
            target_path=layout.publish.source_issue_detail_jsonl_target_path,
        ),
        SamplesPublishTarget(
            label="publish report index",
            kind="report_file",
            source_path=artifacts.index_json_path,
            target_path=layout.publish.index_json_target_path,
        ),
    )


def _build_manifest_targets(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> tuple[SamplesPublishTarget, ...]:
    output_summary = bundle.workflow_result.output_summary
    targets: list[SamplesPublishTarget] = []
    for manifest in output_summary.manifest_outputs:
        publish_split = _publish_split_layout(layout, manifest.split)
        if manifest.partition == "passed":
            target_path = publish_split.passed_manifest_target_path
        elif manifest.partition == "dropped":
            target_path = publish_split.dropped_manifest_target_path
        else:
            raise SamplesWorkflowInvariantError(
                f"Unsupported manifest partition: {manifest.partition}"
            )
        targets.append(
            SamplesPublishTarget(
                label=f"publish {manifest.partition} manifest [{manifest.split}]",
                kind="manifest_file",
                source_path=manifest.path,
                target_path=target_path,
            )
        )
    return tuple(targets)


def _build_archive_targets(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> tuple[SamplesPublishTarget, ...]:
    targets: list[SamplesPublishTarget] = []
    for split_result in _split_results_by_config_order(bundle):
        publish_split = _publish_split_layout(layout, split_result.split)
        if _passed_archive_members(bundle, split_result.split):
            targets.append(
                SamplesPublishTarget(
                    label=f"publish passed samples [{split_result.split}]",
                    kind="archive_file",
                    source_path=bundle.workflow_result.output_summary.passed_samples_root,
                    target_path=publish_split.passed_archive_path,
                )
            )
        if _dropped_archive_members(bundle, split_result.split):
            targets.append(
                SamplesPublishTarget(
                    label=f"publish dropped samples [{split_result.split}]",
                    kind="archive_file",
                    source_path=bundle.workflow_result.output_summary.dropped_samples_root,
                    target_path=publish_split.dropped_archive_path,
                )
            )
    return tuple(targets)


def _build_report_copy_operations(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_report_targets(bundle=bundle, layout=layout)
    )


def _build_manifest_copy_operations(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
) -> tuple[WorkflowOperation, ...]:
    return tuple(
        _file_copy_operation(target)
        for target in _build_manifest_targets(bundle=bundle, layout=layout)
    )


def _build_archive_operations(
    *,
    bundle: SamplesExecutionBundle,
    layout: SamplesLayout,
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


def _file_copy_operation(target: SamplesPublishTarget) -> FileCopyOperation:
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


def _passed_archive_members(bundle: SamplesExecutionBundle, split: str) -> tuple[str, ...]:
    samples_topology = _drive_samples_topology(bundle)
    return tuple(
        samples_topology.archive_member(entry.split, entry.sample_id).path.as_posix()
        for split_result in bundle.split_results
        if split_result.split == split
        for entry in split_result.passed_entries
    )


def _dropped_archive_members(bundle: SamplesExecutionBundle, split: str) -> tuple[str, ...]:
    samples_topology = _drive_samples_topology(bundle)
    return tuple(
        samples_topology.archive_member(entry.split, entry.sample_id).path.as_posix()
        for split_result in bundle.split_results
        if split_result.split == split
        for entry in split_result.dropped_entries
        if entry.materialization.archive_publishable
    )


def _split_results_by_config_order(
    bundle: SamplesExecutionBundle,
) -> tuple[SamplesSplitProcessingResult, ...]:
    by_split = {result.split: result for result in bundle.split_results}
    ordered_results = []
    for split in bundle.workflow_result.config.splits:
        result = by_split.get(split)
        if result is None:
            raise SamplesWorkflowInvariantError(f"Missing split processing result: {split}")
        ordered_results.append(result)
    if len(by_split) != len(bundle.split_results):
        raise SamplesWorkflowInvariantError("Duplicate split processing results are not allowed")
    return tuple(ordered_results)


def _publish_split_layout(layout: SamplesLayout, split: str) -> SamplesPublishSplitLayout:
    for publish_split in layout.publish.splits:
        if publish_split.split == split:
            return publish_split
    raise SamplesWorkflowInvariantError(f"Missing publish layout for split: {split}")


def _drive_samples_topology(bundle: SamplesExecutionBundle) -> SamplesTopology:
    drive_topology = build_artifact_topology(
        build_repo_roots(bundle.workflow_result.config.drive_project_root)
    )
    return drive_topology.samples


def _publish_progress_spec(expected_total: int | None) -> OperationProgressSpec:
    return OperationProgressSpec(
        stage=ProgressStageSpec(
            workflow_id=SAMPLES_WORKFLOW_NAME,
            stage_id=SAMPLES_STAGE_PUBLISH_EXECUTE,
            label="Publish samples workflow artifacts",
            unit="bytes",
            owner_module="text_to_sign_production.workflows.samples.publish",
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
            workflow_id=SAMPLES_WORKFLOW_NAME,
            stage_id=SAMPLES_STAGE_PUBLISH_EXECUTE,
            label=label,
            unit="member",
            owner_module="text_to_sign_production.workflows.samples.publish",
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
