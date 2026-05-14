from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.data.gate.reports import (
    GateCheckpointIntegrityTableRow,
    GateDroppedSamplePayloadTableRow,
    GateFailedGateCountTableRow,
    GateManifestOutcomeTableRow,
    GateOutcomeTableRow,
    GateReportTables,
    GateSourceCoverageTableRow,
    GateSplitCountTableRow,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import (
    JsonValue,
    render_review_sections_markdown,
    write_json,
    write_markdown,
)
from text_to_sign_production.workflows.gate.constants import (
    GATE_STAGE_REPORT_GATE_DETAIL_WRITE,
    GATE_STAGE_REPORT_PROCESSING_DETAIL_WRITE,
    GATE_STAGE_REPORT_SOURCE_ISSUE_DETAIL_WRITE,
    GATE_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.gate.contracts import (
    GateDecisionReviewRecord,
    GateDetailRecord,
    GateProcessingDetailRecord,
    GateProcessingSummaryRecord,
    GateSourceIssueDetailRecord,
    GateSourceIssueSummaryRecord,
    GateSummaryRecord,
    GateWrittenReportArtifacts,
)
from text_to_sign_production.workflows.gate.processing import GateExecutionBundle
from text_to_sign_production.workflows.gate.review.sections import (
    build_final_review_sections,
    build_gate_detail_records,
    build_gate_report,
    build_output_summary_sections,
    build_processing_detail_records,
    build_processing_summary_sections,
    build_report_sections,
    build_report_table_records,
    build_runtime_verification_sections,
    build_source_issue_detail_records,
    build_source_issue_summary_sections,
)


def write_gate_reports(
    *,
    bundle: GateExecutionBundle,
    progress_session: ProgressSession | None = None,
) -> GateWrittenReportArtifacts:
    progress_session = _visible_progress_session(progress_session)
    output_summary = bundle.workflow_result.output_summary
    artifacts = output_summary.planned_report_outputs

    summary_sections = (
        *build_runtime_verification_sections(bundle.workflow_result.runtime_verification),
        *build_processing_summary_sections(bundle),
        *build_report_sections(bundle),
        *build_output_summary_sections(bundle.workflow_result),
        *build_final_review_sections(bundle),
    )
    write_markdown(
        artifacts.summary_markdown_path,
        render_review_sections_markdown(summary_sections),
    )
    write_markdown(
        artifacts.processing_summary_markdown_path,
        render_review_sections_markdown(build_processing_summary_sections(bundle)),
    )
    _write_json_report(
        artifacts.processing_detail_json_path,
        build_processing_detail_records(bundle),
        schema_version="gate.processing.detail.v1",
        report_kind="gate_processing_detail",
        stage_id=GATE_STAGE_REPORT_PROCESSING_DETAIL_WRITE,
        label="processing detail report write",
        artifact_role="processing_detail",
        progress_session=progress_session,
    )
    write_markdown(
        artifacts.gate_summary_markdown_path,
        render_review_sections_markdown(build_report_sections(bundle)),
    )
    _write_json_report(
        artifacts.gate_detail_json_path,
        build_gate_detail_records(bundle),
        schema_version="gate.detail.v1",
        report_kind="gate_detail",
        stage_id=GATE_STAGE_REPORT_GATE_DETAIL_WRITE,
        label="gate detail report write",
        artifact_role="gate_detail",
        progress_session=progress_session,
    )
    write_markdown(
        artifacts.source_issue_summary_markdown_path,
        render_review_sections_markdown(build_source_issue_summary_sections(bundle)),
    )
    _write_json_report(
        artifacts.source_issue_detail_json_path,
        build_source_issue_detail_records(bundle),
        schema_version="gate.source_issue.detail.v1",
        report_kind="gate_source_issue_detail",
        stage_id=GATE_STAGE_REPORT_SOURCE_ISSUE_DETAIL_WRITE,
        label="source issue detail report write",
        artifact_role="source_issue_detail",
        progress_session=progress_session,
    )
    write_json(
        artifacts.index_json_path,
        _gate_index_json_payload(bundle),
    )
    return GateWrittenReportArtifacts(
        summary_markdown=written_file_receipt(
            "gate summary markdown",
            artifacts.summary_markdown_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        processing_summary_markdown=written_file_receipt(
            "gate processing summary",
            artifacts.processing_summary_markdown_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        processing_detail_json=written_file_receipt(
            "gate processing detail",
            artifacts.processing_detail_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        gate_summary_markdown=written_file_receipt(
            "gate summary",
            artifacts.gate_summary_markdown_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        gate_detail_json=written_file_receipt(
            "gate detail",
            artifacts.gate_detail_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        source_issue_summary_markdown=written_file_receipt(
            "gate source issue summary",
            artifacts.source_issue_summary_markdown_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        source_issue_detail_json=written_file_receipt(
            "gate source issue detail",
            artifacts.source_issue_detail_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
        index_json=written_file_receipt(
            "gate report index",
            artifacts.index_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="gate_report",
        ),
    )


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=GATE_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def _write_json_report(
    path: Path,
    records: tuple[object, ...],
    *,
    schema_version: str,
    report_kind: str,
    stage_id: str,
    label: str,
    artifact_role: str,
    progress_session: ProgressSession | None,
) -> None:
    json_records = tuple(_gate_record_json(record) for record in records)
    payload = {
        "schema_version": schema_version,
        "report_kind": report_kind,
        "record_count": len(json_records),
        "records": json_records,
    }
    if progress_session is not None and records:
        with progress_session.task(
            _json_report_progress_spec(
                stage_id=stage_id,
                label=label,
                artifact_role=artifact_role,
            ),
            total=len(records),
        ) as progress_task:
            write_json(path, payload)
            for _ in records:
                progress_task.advance()
    else:
        write_json(path, payload)


def _gate_index_json_payload(bundle: GateExecutionBundle) -> JsonValue:
    artifacts = bundle.workflow_result.output_summary.planned_report_outputs
    output_summary = bundle.workflow_result.output_summary
    provenance = bundle.workflow_result.execution_inputs.gates_config_provenance
    report = build_gate_report(bundle)
    return {
        "workflow": "gate",
        "execution_id": bundle.workflow_result.execution_id,
        "config_provenance": {
            "gates_config": {
                "original_path": provenance.original_path,
                "execution_path": provenance.execution_path,
                "sha256": provenance.sha256,
            },
        },
        "gate_report": {
            "schema_version": report.schema_version,
            "source_coverage": {
                "prepared_sample_count": report.source_coverage.prepared_sample_count,
                "split_counts": report.source_coverage.split_counts,
                "prepared_samples_with_source_issues": (
                    report.source_coverage.prepared_samples_with_source_issues
                ),
                "source_complete_count": report.source_coverage.source_complete_count,
                "validation_issue_count": report.source_coverage.validation_issue_count,
            },
            "manifest_outcomes": {
                "passed_count": report.manifest_outcomes.passed_count,
                "dropped_count": report.manifest_outcomes.dropped_count,
                "passed_validation_issue_count": (
                    report.manifest_outcomes.passed_validation_issue_count
                ),
                "dropped_validation_issue_count": (
                    report.manifest_outcomes.dropped_validation_issue_count
                ),
            },
            "pose_health": {
                "prepared_payload_count": report.pose_health.prepared_payload_count,
                "total_frame_count": report.pose_health.total_frame_count,
                "total_valid_frame_count": report.pose_health.total_valid_frame_count,
                "pose_complete_count": report.pose_health.pose_complete_count,
            },
            "gate_outcomes": {
                "evaluated_count": report.gate_outcomes.evaluated_count,
                "passed_count": report.gate_outcomes.passed_count,
                "dropped_count": report.gate_outcomes.dropped_count,
                "failed_gate_counts": report.gate_outcomes.failed_gate_counts,
            },
            "checkpoint_integrity": {
                "payload_count": report.checkpoint_integrity.payload_count,
                "passed_manifest_count": report.checkpoint_integrity.passed_manifest_count,
                "dropped_manifest_count": report.checkpoint_integrity.dropped_manifest_count,
                "coherent_passed_count": report.checkpoint_integrity.coherent_passed_count,
                "coherence_issue_count": report.checkpoint_integrity.coherence_issue_count,
            },
            "dropped_sample_payloads": {
                "dropped_total_count": report.dropped_sample_payloads.dropped_total_count,
                "source_dropped_sample_count": (
                    report.dropped_sample_payloads.source_dropped_sample_count
                ),
                "pose_dropped_sample_count": (
                    report.dropped_sample_payloads.pose_dropped_sample_count
                ),
                "gate_dropped_sample_count": (
                    report.dropped_sample_payloads.gate_dropped_sample_count
                ),
                "dropped_sample_payload_written_count": (
                    report.dropped_sample_payloads.dropped_sample_payload_written_count
                ),
                "dropped_manifest_entries_with_payload_ref_count": (
                    report.dropped_sample_payloads.dropped_manifest_entries_with_payload_ref_count
                ),
                "dropped_manifest_entries_without_payload_ref_count": (
                    report.dropped_sample_payloads.dropped_manifest_entries_without_payload_ref_count
                ),
                "dropped_manifest_payload_ref_count_coherent": (
                    report.dropped_sample_payloads.dropped_manifest_payload_ref_count_coherent
                ),
                "dropped_manifest_payload_identity_coherent": (
                    report.dropped_sample_payloads.dropped_manifest_payload_identity_coherent
                ),
                "dropped_manifest_payload_coherence_issue_count": (
                    report.dropped_sample_payloads.dropped_manifest_payload_coherence_issue_count
                ),
            },
        },
        "gate_report_tables": _gate_report_tables_json(build_report_table_records(bundle)),
        "reports": {
            "summary_markdown_path": artifacts.summary_markdown_path,
            "processing_summary_markdown_path": artifacts.processing_summary_markdown_path,
            "processing_detail_json_path": artifacts.processing_detail_json_path,
            "gate_summary_markdown_path": artifacts.gate_summary_markdown_path,
            "gate_detail_json_path": artifacts.gate_detail_json_path,
            "source_issue_summary_markdown_path": (artifacts.source_issue_summary_markdown_path),
            "source_issue_detail_json_path": artifacts.source_issue_detail_json_path,
            "index_json_path": artifacts.index_json_path,
        },
        "outputs": {
            "planned_manifest_outputs": tuple(
                {
                    "partition": manifest.partition,
                    "split": manifest.split,
                    "path": manifest.path,
                }
                for manifest in output_summary.planned_manifest_outputs
            ),
            "passed_samples_root": output_summary.passed_samples_root,
            "dropped_samples_root": output_summary.dropped_samples_root,
        },
    }


def _gate_record_json(record: object) -> JsonValue:
    if isinstance(record, GateProcessingSummaryRecord):
        return {
            "split": record.split,
            "processed_count": record.processed_count,
            "prepared_sample_count": record.prepared_sample_count,
            "passed_count": record.passed_count,
            "dropped_count": record.dropped_count,
            "source_dropped_sample_count": record.source_dropped_sample_count,
            "pose_dropped_sample_count": record.pose_dropped_sample_count,
            "gate_dropped_sample_count": record.gate_dropped_sample_count,
            "dropped_sample_payload_written_count": (record.dropped_sample_payload_written_count),
            "dropped_manifest_entries_with_payload_ref_count": (
                record.dropped_manifest_entries_with_payload_ref_count
            ),
            "dropped_manifest_entries_without_payload_ref_count": (
                record.dropped_manifest_entries_without_payload_ref_count
            ),
            "dropped_manifest_payload_ref_count_coherent": (
                record.dropped_manifest_payload_ref_count_coherent
            ),
            "dropped_manifest_payload_identity_coherent": (
                record.dropped_manifest_payload_identity_coherent
            ),
            "dropped_manifest_payload_coherence_issue_count": (
                record.dropped_manifest_payload_coherence_issue_count
            ),
        }
    if isinstance(record, GateProcessingDetailRecord):
        return {
            "split": record.split,
            "sample_id": record.sample_id,
            "match_status": record.match_status,
            "passed": record.passed,
            "drop_stage": record.drop_stage,
        }
    if isinstance(record, GateSummaryRecord):
        return {
            "split": record.split,
            "evaluated_count": record.evaluated_count,
            "passed_count": record.passed_count,
            "dropped_count": record.dropped_count,
        }
    if isinstance(record, GateDetailRecord):
        return {
            "split": record.split,
            "sample_id": record.sample_id,
            "final_status": record.final_status,
            "terminal_gate": record.terminal_gate,
            "failed_gates": record.failed_gates,
            "decisions": tuple(_gate_decision_json(decision) for decision in record.decisions),
        }
    if isinstance(record, GateSourceIssueSummaryRecord):
        return {
            "split": record.split,
            "source_issue_count": record.source_issue_count,
            "unmatched_count": record.unmatched_count,
        }
    if isinstance(record, GateSourceIssueDetailRecord):
        return {
            "split": record.split,
            "sample_id": record.sample_id,
            "issue_code": record.issue_code,
            "detail": record.detail,
        }
    raise TypeError(f"Unsupported gate JSON record type: {type(record).__name__}")


def _gate_decision_json(record: GateDecisionReviewRecord) -> JsonValue:
    return {
        "gate": record.gate,
        "status": record.status,
        "issue_codes": record.issue_codes,
    }


def _gate_report_tables_json(tables: GateReportTables) -> JsonValue:
    return {
        "source_coverage": tuple(
            _gate_source_coverage_table_row_json(row) for row in tables.source_coverage
        ),
        "split_counts": tuple(_gate_split_count_table_row_json(row) for row in tables.split_counts),
        "gate_outcomes": tuple(_gate_outcome_table_row_json(row) for row in tables.gate_outcomes),
        "manifest_outcomes": tuple(
            _gate_manifest_outcome_table_row_json(row) for row in tables.manifest_outcomes
        ),
        "failed_gate_counts": tuple(
            _gate_failed_gate_count_table_row_json(row) for row in tables.failed_gate_counts
        ),
        "checkpoint_integrity": tuple(
            _gate_checkpoint_integrity_table_row_json(row) for row in tables.checkpoint_integrity
        ),
        "dropped_sample_payloads": tuple(
            _gate_dropped_sample_payload_table_row_json(row)
            for row in tables.dropped_sample_payloads
        ),
    }


def _gate_source_coverage_table_row_json(row: GateSourceCoverageTableRow) -> JsonValue:
    return {
        "prepared_sample_count": row.prepared_sample_count,
        "prepared_samples_with_source_issues": row.prepared_samples_with_source_issues,
        "source_complete_count": row.source_complete_count,
        "validation_issue_count": row.validation_issue_count,
    }


def _gate_split_count_table_row_json(row: GateSplitCountTableRow) -> JsonValue:
    return {
        "split": row.split,
        "count": row.count,
    }


def _gate_outcome_table_row_json(row: GateOutcomeTableRow) -> JsonValue:
    return {
        "evaluated_count": row.evaluated_count,
        "passed_count": row.passed_count,
        "dropped_count": row.dropped_count,
    }


def _gate_manifest_outcome_table_row_json(row: GateManifestOutcomeTableRow) -> JsonValue:
    return {
        "passed_count": row.passed_count,
        "dropped_count": row.dropped_count,
        "passed_validation_issue_count": row.passed_validation_issue_count,
        "dropped_validation_issue_count": row.dropped_validation_issue_count,
    }


def _gate_failed_gate_count_table_row_json(row: GateFailedGateCountTableRow) -> JsonValue:
    return {
        "gate": row.gate,
        "count": row.count,
    }


def _gate_checkpoint_integrity_table_row_json(
    row: GateCheckpointIntegrityTableRow,
) -> JsonValue:
    return {
        "payload_count": row.payload_count,
        "passed_manifest_count": row.passed_manifest_count,
        "dropped_manifest_count": row.dropped_manifest_count,
        "coherent_passed_count": row.coherent_passed_count,
        "coherence_issue_count": row.coherence_issue_count,
    }


def _gate_dropped_sample_payload_table_row_json(
    row: GateDroppedSamplePayloadTableRow,
) -> JsonValue:
    return {
        "dropped_total_count": row.dropped_total_count,
        "source_dropped_sample_count": row.source_dropped_sample_count,
        "pose_dropped_sample_count": row.pose_dropped_sample_count,
        "gate_dropped_sample_count": row.gate_dropped_sample_count,
        "dropped_sample_payload_written_count": row.dropped_sample_payload_written_count,
        "dropped_manifest_entries_with_payload_ref_count": (
            row.dropped_manifest_entries_with_payload_ref_count
        ),
        "dropped_manifest_entries_without_payload_ref_count": (
            row.dropped_manifest_entries_without_payload_ref_count
        ),
        "dropped_manifest_payload_ref_count_coherent": (
            row.dropped_manifest_payload_ref_count_coherent
        ),
        "dropped_manifest_payload_identity_coherent": (
            row.dropped_manifest_payload_identity_coherent
        ),
        "dropped_manifest_payload_coherence_issue_count": (
            row.dropped_manifest_payload_coherence_issue_count
        ),
    }


def _json_report_progress_spec(
    *,
    stage_id: str,
    label: str,
    artifact_role: str,
) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=GATE_WORKFLOW_NAME,
        stage_id=stage_id,
        label=label,
        unit="record",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="json_report_write",
        total_semantics="records written to one JSON report artifact",
        bar_eligible=True,
        artifact_role=artifact_role,
    )
