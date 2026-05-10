from __future__ import annotations

from collections.abc import Mapping

from text_to_sign_production.data.gate.reports import (
    SamplesReportBundle,
    build_samples_report_bundle,
    samples_report_tables,
    summarize_samples_report,
)
from text_to_sign_production.workflows.foundation.execution import WorkflowOperation
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

REPORT_SCHEMA_VERSION = "samples.report.v1"


def build_samples_report(bundle: SamplesExecutionBundle) -> SamplesReportBundle:
    return build_samples_report_bundle(
        schema_version=REPORT_SCHEMA_VERSION,
        payloads=tuple(
            sample
            for split_result in bundle.split_results
            for sample in split_result.prepared_samples
        ),
        passed_entries=tuple(
            entry for split_result in bundle.split_results for entry in split_result.passed_entries
        ),
        dropped_entries=tuple(
            entry for split_result in bundle.split_results for entry in split_result.dropped_entries
        ),
        gate_bundles=tuple(
            gate for split_result in bundle.split_results for gate in split_result.gate_bundles
        ),
    )


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


def build_report_sections(bundle: SamplesExecutionBundle) -> tuple[WorkflowReviewSection, ...]:
    report = build_samples_report(bundle)
    summary = summarize_samples_report(report)
    return (
        review_section(
            "Samples report",
            (
                review_item(
                    "checkpoint",
                    (
                        ("schema version", report.schema_version),
                        ("total samples", summary.sample_count),
                        ("passed", summary.passed_count),
                        ("dropped", summary.dropped_count),
                        (
                            "canonical text missing",
                            report.checkpoint_integrity.canonical_normalized_text_missing_count,
                        ),
                    ),
                ),
            ),
        ),
    )


def build_output_summary_sections(
    result: SamplesWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    manifest_items = tuple(_manifest_item(row) for row in _manifest_rows(result))
    sample_root_items = (
        review_item(
            "passed PreparedSample payloads root",
            (("path", result.output_summary.passed_samples_root),),
        ),
        review_item(
            "dropped debug PreparedSample payloads root",
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
        passed_ids = {entry.sample_id for entry in split_result.passed_entries}
        dropped_by_id = {entry.sample_id: entry for entry in split_result.dropped_entries}
        for match in split_result.source_matches:
            sample_id = (
                match.candidate_identity.keypoint.sample_key.value
                if match.candidate_identity
                else match.translation.sentence_id
            )
            records.append(
                {
                    "split": split_result.split,
                    "sample_id": sample_id,
                    "match_status": match.status.value,
                    "passed": sample_id in passed_ids,
                    "drop_stage": (
                        dropped_by_id[sample_id].drop_stage.value
                        if sample_id in dropped_by_id
                        else None
                    ),
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
            "prepared_sample_count": len(split_result.prepared_samples),
            "passed_count": split_result.passed_count,
            "dropped_count": split_result.dropped_count,
        }
        for split_result in bundle.split_results
    )


def build_gate_summary_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "split": split_result.split,
            "evaluated_count": len(split_result.gate_bundles),
            "passed_count": sum(
                1 for gate in split_result.gate_bundles if gate.final_status.value == "passed"
            ),
            "dropped_count": sum(
                1 for gate in split_result.gate_bundles if gate.final_status.value == "dropped"
            ),
        }
        for split_result in bundle.split_results
    )


def build_gate_detail_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "split": split_result.split,
            "sample_id": gate.sample_id,
            "final_status": gate.final_status.value,
            "terminal_gate": None if gate.terminal_gate is None else gate.terminal_gate.value,
            "failed_gates": tuple(failed.value for failed in gate.failed_gates),
            "decisions": tuple(
                {
                    "gate": decision.gate.value,
                    "status": decision.status.value,
                    "issue_codes": tuple(code.value for code in decision.issue_codes),
                }
                for decision in gate.decisions
            ),
        }
        for split_result in bundle.split_results
        for gate in split_result.gate_bundles
    )


def build_source_issue_summary_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "split": split_result.split,
            "source_issue_count": sum(
                len(match.source_issues) for match in split_result.source_matches
            ),
            "unmatched_count": sum(1 for match in split_result.source_matches if not match.matched),
        }
        for split_result in bundle.split_results
    )


def build_source_issue_detail_records(
    bundle: SamplesExecutionBundle,
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for split_result in bundle.split_results:
        for match in split_result.source_matches:
            for issue in match.source_issues:
                records.append(
                    {
                        "split": split_result.split,
                        "sample_id": match.translation.sentence_id,
                        "issue_code": issue.code.value,
                        "detail": issue.detail,
                    }
                )
            if not match.matched and match.unmatched_reason is not None:
                records.append(
                    {
                        "split": split_result.split,
                        "sample_id": match.translation.sentence_id,
                        "issue_code": match.unmatched_reason.code.value,
                        "detail": match.unmatched_reason.detail,
                    }
                )
    return tuple(records)


def build_report_table_records(
    bundle: SamplesExecutionBundle,
) -> dict[str, tuple[dict[str, object], ...]]:
    return samples_report_tables(build_samples_report(bundle))


def _runtime_asset_rows(
    verification: SamplesRuntimeVerification,
) -> tuple[SamplesRuntimeAssetRow, ...]:
    return tuple(
        SamplesRuntimeAssetRow(
            label=check.label,
            path=check.path,
            exists=check.exists and check.valid,
        )
        for check in verification.checks
    )


def _split_count_rows(bundle: SamplesExecutionBundle) -> tuple[SamplesSplitCountRow, ...]:
    return tuple(
        SamplesSplitCountRow(
            split=split_result.split,
            processed_count=split_result.processed_count,
            passed_count=split_result.passed_count,
            dropped_count=split_result.dropped_count,
        )
        for split_result in bundle.split_results
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


def _report_artifact_rows(result: SamplesWorkflowResult) -> tuple[SamplesReportArtifactRow, ...]:
    artifacts = result.output_summary.report_artifacts
    return (
        SamplesReportArtifactRow("summary markdown", artifacts.summary_markdown_path),
        SamplesReportArtifactRow("processing summary", artifacts.processing_summary_jsonl_path),
        SamplesReportArtifactRow("processing detail", artifacts.processing_detail_jsonl_path),
        SamplesReportArtifactRow("gate summary", artifacts.gate_summary_jsonl_path),
        SamplesReportArtifactRow("gate detail", artifacts.gate_detail_jsonl_path),
        SamplesReportArtifactRow("source issue summary", artifacts.source_issue_summary_jsonl_path),
        SamplesReportArtifactRow("source issue detail", artifacts.source_issue_detail_jsonl_path),
        SamplesReportArtifactRow("report index", artifacts.index_json_path),
    )


def _operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    return review_item(
        operation.label,
        (
            ("kind", operation.__class__.__name__),
            ("live owner", operation.progress.live_owner if operation.progress else None),
        ),
    )


def _execution_result_item(result: object) -> WorkflowReviewItem:
    return review_item(str(result), ())


def _runtime_asset_item(row: SamplesRuntimeAssetRow) -> WorkflowReviewItem:
    return review_item(row.label, (("path", row.path), ("ready", row.exists)))


def _split_count_item(row: SamplesSplitCountRow) -> WorkflowReviewItem:
    return review_item(
        row.split,
        (
            ("processed", row.processed_count),
            ("passed", row.passed_count),
            ("dropped", row.dropped_count),
        ),
    )


def _manifest_item(row: SamplesManifestRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (("path", row.path), ("partition", row.partition), ("split", row.split)),
    )


def _report_artifact_item(row: SamplesReportArtifactRow) -> WorkflowReviewItem:
    return review_item(row.label, (("path", row.path),))


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_final_review_sections",
    "build_gate_detail_records",
    "build_gate_summary_records",
    "build_output_summary_sections",
    "build_processing_detail_records",
    "build_processing_summary_records",
    "build_processing_summary_sections",
    "build_report_sections",
    "build_report_table_records",
    "build_runtime_plan_sections",
    "build_runtime_restore_sections",
    "build_runtime_verification_sections",
    "build_samples_report",
    "build_source_issue_detail_records",
    "build_source_issue_summary_records",
]
