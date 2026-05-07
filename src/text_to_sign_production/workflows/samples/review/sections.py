from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from text_to_sign_production.core import SampleSplit
from text_to_sign_production.data.gates import (
    ProcessingDecision,
    build_drop_stage_count_records,
    build_gate_split_stage_summary_records,
    build_processing_status_count_records,
)
from text_to_sign_production.data.sources import (
    SourceMatchResult,
    build_source_availability_summary_record,
    build_source_split_unmatched_reason_count_records,
)
from text_to_sign_production.workflows.foundation.execution import (
    WorkflowOperation,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.review import (
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.samples.contracts import (
    SamplesManifestRow,
    SamplesReportArtifactRow,
    SamplesRuntimeAssetRow,
    SamplesRuntimePlan,
    SamplesRuntimeRestoreResult,
    SamplesRuntimeVerification,
    SamplesSplitCountRow,
    SamplesWorkflowResult,
)
from text_to_sign_production.workflows.samples.processing import SamplesExecutionBundle


def build_runtime_plan_sections(
    plan: SamplesRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore plan",
            tuple(_operation_item(operation) for operation in plan.restore_operations),
        ),
    )


def build_runtime_restore_sections(
    result: SamplesRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore execution",
            tuple(
                _execution_result_item(execution_result)
                for execution_result in result.execution.results
            ),
        ),
    )


def build_runtime_verification_sections(
    verification: SamplesRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime verification",
            tuple(_runtime_asset_item(row) for row in _runtime_asset_rows(verification)),
        ),
    )


def build_processing_summary_sections(
    bundle: SamplesExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Processing summary",
            tuple(_split_count_item(row) for row in _split_count_rows(bundle)),
        ),
    )


def build_output_summary_sections(
    result: SamplesWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    manifest_items = tuple(_manifest_item(row) for row in _manifest_rows(result))
    sample_root_items = (
        review_item(
            "passed samples root",
            (("path", result.output_summary.passed_samples_root),),
        ),
        review_item(
            "dropped samples root",
            (("path", result.output_summary.dropped_samples_root),),
        ),
    )
    report_items = tuple(_report_artifact_item(row) for row in _report_artifact_rows(result))
    return (
        review_section(
            "Output summary",
            (*manifest_items, *sample_root_items, *report_items),
        ),
    )


def build_final_review_sections(
    bundle: SamplesExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    result = bundle.workflow_result
    total_processed = sum(split_result.processed_count for split_result in bundle.split_results)
    total_passed = sum(split_result.passed_count for split_result in bundle.split_results)
    total_dropped = sum(split_result.dropped_count for split_result in bundle.split_results)
    return (
        review_section(
            "Final summary",
            (
                review_item(
                    "samples",
                    (
                        ("workflow", "samples"),
                        ("split count", len(result.config.splits)),
                        ("total processed", total_processed),
                        ("total passed", total_passed),
                        ("total dropped", total_dropped),
                        ("manifest output count", len(result.output_summary.manifest_outputs)),
                        (
                            "summary markdown path",
                            result.output_summary.report_artifacts.summary_markdown_path,
                        ),
                    ),
                ),
            ),
        ),
    )


def build_processing_detail_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for split_result in bundle.split_results:
        for match, decision in zip(
            split_result.source_matches,
            split_result.decisions,
            strict=True,
        ):
            records.append(
                {
                    "split": split_result.split,
                    "matched": match.matched,
                    "sample_id": match.translation.sentence_name,
                    "sentence_name": match.translation.sentence_name,
                    "final_status": _final_status(decision),
                    "unmatched_reason": match.unmatched_reason,
                }
            )
    return tuple(records)


def build_processing_summary_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "split": split_result.split,
            "processed_count": split_result.processed_count,
            "passed_count": split_result.passed_count,
            "dropped_count": split_result.dropped_count,
        }
        for split_result in bundle.split_results
    )


def build_gate_summary_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    gate_status_counts = _gate_status_counts_by_split(bundle)
    for split_result in bundle.split_results:
        processing_status_counts = {
            record.status.value: record.count
            for record in build_processing_status_count_records(split_result.decisions)
        }
        drop_stage_counts = {
            record.stage.value: record.count
            for record in build_drop_stage_count_records(split_result.decisions)
        }
        records.append(
            {
                "split": split_result.split,
                "processed_count": len(split_result.decisions),
                "processed_status_count": processing_status_counts.get("processed", 0),
                "dropped_status_count": processing_status_counts.get("dropped", 0),
                "drop_stage_counts": _sorted_counts(drop_stage_counts),
                "gate_status_counts": gate_status_counts.get(split_result.split, {}),
            }
        )
    return tuple(records)


def build_source_issue_summary_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    unmatched_reason_counts = _unmatched_reason_counts_by_split(bundle)
    for split_result in bundle.split_results:
        availability = build_source_availability_summary_record(split_result.source_matches)
        records.append(
            {
                "split": split_result.split,
                "total_rows": availability.match_count,
                "matched_count": availability.matched_count,
                "unmatched_count": availability.match_count - availability.matched_count,
                "unmatched_reason_counts": unmatched_reason_counts.get(split_result.split, {}),
            }
        )
    return tuple(records)


def build_gate_detail_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for split_result in bundle.split_results:
        for match, decision in zip(
            split_result.source_matches,
            split_result.decisions,
            strict=True,
        ):
            records.append(
                {
                    "split": split_result.split,
                    "sample_id": match.translation.sentence_name,
                    "processing_status": decision.status.value,
                    "drop_stage": (
                        decision.drop_stage.value if decision.drop_stage is not None else None
                    ),
                    "drop_reasons": decision.drop_reasons,
                    "gate_statuses": {
                        stage.value: gate_result.status.value
                        for stage, gate_result in decision.gate_results.items()
                    },
                }
            )
    return tuple(records)


def build_source_issue_detail_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for split_result in bundle.split_results:
        for match in split_result.source_matches:
            if match.matched and not match.source_issues:
                continue
            records.append(
                {
                    "split": split_result.split,
                    "matched": match.matched,
                    "unmatched_reason": match.unmatched_reason,
                    "sentence_name": match.translation.sentence_name,
                    "keypoint_directory": (
                        match.keypoints.directory if match.keypoints is not None else None
                    ),
                    "keypoint_exists": (
                        match.keypoints.exists if match.keypoints is not None else None
                    ),
                    "keypoint_frame_count": (
                        match.keypoints.frame_count if match.keypoints is not None else None
                    ),
                    "video_metadata_error": (
                        match.video_metadata.error if match.video_metadata is not None else None
                    ),
                    "source_issues": match.source_issues,
                }
            )
    return tuple(records)


def _runtime_asset_rows(
    verification: SamplesRuntimeVerification,
) -> tuple[SamplesRuntimeAssetRow, ...]:
    return tuple(
        SamplesRuntimeAssetRow(label=check.label, path=check.path, exists=check.exists)
        for check in verification.checks
    )


def _split_count_rows(
    bundle: SamplesExecutionBundle,
) -> tuple[SamplesSplitCountRow, ...]:
    return tuple(
        SamplesSplitCountRow(
            split=split_result.split,
            processed_count=split_result.processed_count,
            passed_count=split_result.passed_count,
            dropped_count=split_result.dropped_count,
        )
        for split_result in bundle.split_results
    )


def _report_artifact_rows(
    result: SamplesWorkflowResult,
) -> tuple[SamplesReportArtifactRow, ...]:
    artifacts = result.output_summary.report_artifacts
    return (
        SamplesReportArtifactRow("summary markdown", artifacts.summary_markdown_path),
        SamplesReportArtifactRow(
            "processing summary JSONL",
            artifacts.processing_summary_jsonl_path,
        ),
        SamplesReportArtifactRow("processing detail JSONL", artifacts.processing_detail_jsonl_path),
        SamplesReportArtifactRow("gate summary JSONL", artifacts.gate_summary_jsonl_path),
        SamplesReportArtifactRow("gate detail JSONL", artifacts.gate_detail_jsonl_path),
        SamplesReportArtifactRow(
            "source issue summary JSONL",
            artifacts.source_issue_summary_jsonl_path,
        ),
        SamplesReportArtifactRow(
            "source issue detail JSONL",
            artifacts.source_issue_detail_jsonl_path,
        ),
        SamplesReportArtifactRow("report index JSON", artifacts.index_json_path),
    )


def _manifest_rows(result: SamplesWorkflowResult) -> tuple[SamplesManifestRow, ...]:
    return tuple(
        SamplesManifestRow(
            label=f"{manifest.partition} manifest [{manifest.split}]",
            path=manifest.path,
            partition=manifest.partition,
            split=manifest.split,
        )
        for manifest in result.output_summary.manifest_outputs
    )


def _operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation kind", operation_kind(operation)),
        ("label", getattr(operation, "label", "")),
    ]
    for field_name in (
        "source_path",
        "target_path",
        "archive_path",
        "extraction_root",
        "expected_input_bytes",
    ):
        if hasattr(operation, field_name):
            value = getattr(operation, field_name)
            if value is not None:
                fields.append((field_name, value))
    progress = getattr(operation, "progress", None)
    if progress is not None:
        fields.append(("progress stage id", progress.stage.stage_id))
    return review_item(str(operation.label), fields)


def _execution_result_item(execution_result: Any) -> WorkflowReviewItem:
    fields: list[tuple[object, object]] = [
        ("operation_kind", execution_result.operation_kind),
        ("succeeded", execution_result.succeeded),
        ("returncode", execution_result.returncode),
        ("execution_mode", execution_result.execution_mode),
    ]
    observed_outputs = execution_result.observed_outputs
    if observed_outputs:
        fields.append(("observed_outputs", observed_outputs))
    return review_item(str(execution_result.label), fields)


def _runtime_asset_item(row: SamplesRuntimeAssetRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("path", row.path),
            ("exists", row.exists),
        ),
    )


def _split_count_item(row: SamplesSplitCountRow) -> WorkflowReviewItem:
    return review_item(
        row.split,
        (
            ("processed_count", row.processed_count),
            ("passed_count", row.passed_count),
            ("dropped_count", row.dropped_count),
        ),
    )


def _report_artifact_item(row: SamplesReportArtifactRow) -> WorkflowReviewItem:
    return review_item(row.label, (("path", row.path),))


def _manifest_item(row: SamplesManifestRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("path", row.path),
            ("partition", row.partition),
            ("split", row.split),
        ),
    )


def _decisions_by_split(
    bundle: SamplesExecutionBundle,
) -> dict[SampleSplit, tuple[ProcessingDecision, ...]]:
    return {
        SampleSplit(split_result.split): split_result.decisions
        for split_result in bundle.split_results
    }


def _matches_by_split(
    bundle: SamplesExecutionBundle,
) -> dict[SampleSplit, tuple[SourceMatchResult, ...]]:
    return {
        SampleSplit(split_result.split): split_result.source_matches
        for split_result in bundle.split_results
    }


def _gate_status_counts_by_split(
    bundle: SamplesExecutionBundle,
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for record in build_gate_split_stage_summary_records(_decisions_by_split(bundle)):
        counts.setdefault(record.split.value, {})[record.stage.value] = {
            "passed": record.passed_count,
            "dropped": record.dropped_count,
        }
    return counts


def _unmatched_reason_counts_by_split(
    bundle: SamplesExecutionBundle,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in build_source_split_unmatched_reason_count_records(_matches_by_split(bundle)):
        counts.setdefault(record.split.value, {})[record.reason] = record.count
    return counts


def _sorted_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: count for key, count in sorted(counter.items())}


def _final_status(decision: Any) -> str:
    status_value = decision.status.value
    if status_value == "processed":
        return "passed"
    return "dropped"
