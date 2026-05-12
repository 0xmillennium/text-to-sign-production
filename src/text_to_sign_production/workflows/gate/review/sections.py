from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from text_to_sign_production.core.models import GateDropStage
from text_to_sign_production.data.dataset.analysis import summarize_checkpoint_handoff
from text_to_sign_production.data.gate.reports import (
    GateReportTables,
    GateReportBundle,
    build_gate_report_bundle,
    gate_report_tables,
    summarize_gate_report,
)
from text_to_sign_production.data.gate.sources import sample_id_from_translation
from text_to_sign_production.workflows.foundation.execution import (
    OperationExecutionResult,
    WorkflowOperation,
)
from text_to_sign_production.workflows.foundation.review import (
    RenderableValue,
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.gate.contracts import (
    GateDecisionReviewRecord,
    GateDetailRecord,
    GateIdentityInvariantRow,
    GateManifestRow,
    GateProcessingDetailRecord,
    GateProcessingSummaryRecord,
    GateReportArtifactRow,
    GateRuntimeAssetRow,
    GateRuntimePlan,
    GateRuntimeRestoreResult,
    GateRuntimeVerification,
    GateSourceIssueDetailRecord,
    GateSourceIssueSummaryRecord,
    GateSplitCountRow,
    GateSummaryRecord,
    GateViabilityDropRow,
    GateWorkflowResult,
    GateWrittenArtifactRow,
    GateWrittenReportArtifacts,
)
from text_to_sign_production.workflows.gate.processing import (
    GateExecutionBundle,
    GateSplitProcessingResult,
)

REPORT_SCHEMA_VERSION = "gate.report.v1"


def build_gate_report(bundle: GateExecutionBundle) -> GateReportBundle:
    return build_gate_report_bundle(
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
        materialize_dropped_debug_payloads=(
            bundle.workflow_result.config.materialize_dropped_debug_payloads
        ),
        dropped_debug_payload_written_count=sum(
            len(split_result.dropped_debug_payloads) for split_result in bundle.split_results
        ),
    )


def build_runtime_plan_sections(
    plan: GateRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_plan_summary_sections(plan)


def build_runtime_plan_summary_sections(
    plan: GateRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime plan summary",
            (
                review_item(
                    "gate",
                    (
                        ("restore operation count", len(plan.restore_operations)),
                        (
                            "gates config execution path",
                            plan.execution_inputs.gates_config_provenance.execution_path,
                        ),
                        (
                            "gates config sha256",
                            plan.execution_inputs.gates_config_provenance.sha256,
                        ),
                    ),
                ),
            ),
        ),
    )


def build_runtime_plan_detail_sections(
    plan: GateRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Config provenance",
            (
                review_item(
                    plan.execution_inputs.gates_config_provenance.label,
                    (
                        (
                            "original_path",
                            plan.execution_inputs.gates_config_provenance.original_path,
                        ),
                        (
                            "execution_path",
                            plan.execution_inputs.gates_config_provenance.execution_path,
                        ),
                        ("sha256", plan.execution_inputs.gates_config_provenance.sha256),
                    ),
                ),
            ),
        ),
        review_section(
            "Runtime restore plan",
            tuple(_operation_item(operation) for operation in plan.restore_operations),
        ),
    )


def build_runtime_restore_sections(
    result: GateRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_restore_summary_sections(result)


def build_runtime_restore_summary_sections(
    result: GateRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    successful_results = result.execution.successful_results()
    failed_results = result.execution.failed_results()
    return (
        review_section(
            "Runtime restore summary",
            (
                review_item(
                    "restore",
                    (
                        ("operation count", len(result.execution.results)),
                        ("successful operation count", len(successful_results)),
                        ("failed operation count", len(failed_results)),
                    ),
                ),
                *_operation_failure_items(failed_results),
            ),
        ),
    )


def build_runtime_restore_detail_sections(
    result: GateRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore execution details",
            tuple(
                _execution_result_item(execution_result)
                for execution_result in result.execution.results
            ),
        ),
    )


def build_runtime_verification_sections(
    verification: GateRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_verification_summary_sections(verification)


def build_runtime_verification_summary_sections(
    verification: GateRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    asset_rows = _runtime_asset_rows(verification, "asset")
    domain_rows = _runtime_asset_rows(verification, "domain")
    failed_rows = tuple(
        row for row in (*asset_rows, *domain_rows) if not row.exists or not row.valid
    )
    return (
        review_section(
            "Runtime verification summary",
            (
                review_item(
                    "readiness",
                    (
                        ("level", verification.readiness.readiness_level),
                        ("asset check count", len(asset_rows)),
                        ("domain check count", len(domain_rows)),
                        ("failed check count", len(failed_rows)),
                        ("checked_semantics", verification.checked_semantics),
                        ("limitations", verification.limitations),
                        ("succeeded", verification.succeeded),
                    ),
                ),
                *_runtime_check_failure_items(failed_rows),
            ),
        ),
    )


def build_runtime_verification_detail_sections(
    verification: GateRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime verification",
            (
                review_item(
                    "readiness",
                    (
                        ("level", verification.readiness.readiness_level),
                        ("checked_semantics", verification.checked_semantics),
                        ("limitations", verification.limitations),
                        ("succeeded", verification.succeeded),
                    ),
                ),
            ),
        ),
        review_section(
            "Runtime asset checks",
            tuple(_runtime_asset_item(row) for row in _runtime_asset_rows(verification, "asset")),
        ),
        review_section(
            "Runtime domain checks",
            tuple(_runtime_asset_item(row) for row in _runtime_asset_rows(verification, "domain")),
        ),
    )


def build_processing_summary_sections(
    bundle: GateExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    identity_rows = _identity_invariant_rows(bundle)
    failing_identity_rows = tuple(row for row in identity_rows if row.status != "passed")
    viability_rows = _viability_drop_rows(bundle)
    non_viable_rows = tuple(row for row in viability_rows if row.viability_status != "viable")
    dropped_rows = tuple(row for row in viability_rows if row.dropped)
    return (
        review_section(
            "Processing summary",
            tuple(_split_count_item(row) for row in _split_count_rows(bundle)),
        ),
        review_section(
            "Identity invariant summary",
            (
                review_item(
                    "identity",
                    (
                        ("checked sample count", len(identity_rows)),
                        ("passed identity count", len(identity_rows) - len(failing_identity_rows)),
                        ("failed identity count", len(failing_identity_rows)),
                    ),
                ),
                *_identity_failure_items(failing_identity_rows),
            ),
        ),
        review_section(
            "Viability and drop summary",
            (
                review_item(
                    "viability",
                    (
                        ("candidate count", len(viability_rows)),
                        ("viable count", len(viability_rows) - len(non_viable_rows)),
                        ("non-viable count", len(non_viable_rows)),
                        ("dropped count", len(dropped_rows)),
                        (
                            "drop stage breakdown",
                            _count_labels(
                                row.drop_stage or "not_dropped"
                                for row in viability_rows
                                if row.dropped
                            ),
                        ),
                        (
                            "issue code breakdown",
                            _count_labels(
                                code for row in non_viable_rows for code in row.issue_codes
                            ),
                        ),
                    ),
                ),
                *_viability_anomaly_items(non_viable_rows),
            ),
        ),
    )


def build_processing_detail_sections(
    bundle: GateExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Identity invariant details",
            tuple(_identity_invariant_item(row) for row in _identity_invariant_rows(bundle)),
        ),
        review_section(
            "Viability and drop details",
            tuple(_viability_drop_item(row) for row in _viability_drop_rows(bundle)),
        ),
    )


def build_report_sections(bundle: GateExecutionBundle) -> tuple[WorkflowReviewSection, ...]:
    report = build_gate_report(bundle)
    summary = summarize_gate_report(report)
    handoff = summarize_checkpoint_handoff(
        tuple(
            sample
            for split_result in bundle.split_results
            for sample in split_result.prepared_samples
        ),
        tuple(
            entry for split_result in bundle.split_results for entry in split_result.passed_entries
        ),
        tuple(
            entry for split_result in bundle.split_results for entry in split_result.dropped_entries
        ),
    )
    return (
        review_section(
            "Gate report",
            (
                review_item(
                    "checkpoint",
                    (
                        ("schema version", report.schema_version),
                        ("prepared samples", summary.prepared_sample_count),
                        ("passed", summary.passed_count),
                        ("dropped", summary.dropped_count),
                        ("coherent passed", handoff.coherent_passed_count),
                        ("coherence issues", handoff.coherence_issue_count),
                        (
                            "payload validation issues",
                            report.source_coverage.validation_issue_count,
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Dropped manifest and debug payload semantics",
            (
                review_item(
                    "dropped debug payloads",
                    (
                        (
                            "materialize_dropped_debug_payloads",
                            report.dropped_debug_payloads.materialize_dropped_debug_payloads,
                        ),
                        (
                            "dropped_total_count",
                            report.dropped_debug_payloads.dropped_total_count,
                        ),
                        (
                            "pose_or_source_dropped_without_prepared_payload_count",
                            report.dropped_debug_payloads
                            .pose_or_source_dropped_without_prepared_payload_count,
                        ),
                        (
                            "gate_dropped_prepared_sample_count",
                            report.dropped_debug_payloads.gate_dropped_prepared_sample_count,
                        ),
                        (
                            "dropped_debug_payload_written_count",
                            report.dropped_debug_payloads
                            .dropped_debug_payload_written_count,
                        ),
                        (
                            "dropped_manifest_entries_with_debug_ref_count",
                            report.dropped_debug_payloads
                            .dropped_manifest_entries_with_debug_ref_count,
                        ),
                        (
                            "dropped_manifest_entries_without_debug_ref_count",
                            report.dropped_debug_payloads
                            .dropped_manifest_entries_without_debug_ref_count,
                        ),
                        (
                            "manifest semantics",
                            "Dropped manifest entries are audit/trace records.",
                        ),
                        (
                            "archive semantics",
                            "Dropped sample archives are produced only when debug payloads are materialized.",
                        ),
                        (
                            "source_pose_drop_semantics",
                            "Source/pose-stage dropped examples may not have PreparedSample payloads.",
                        ),
                        (
                            "manifest_without_archive",
                            "A dropped manifest can exist without a dropped sample archive.",
                        ),
                    ),
                ),
            ),
        ),
    )


def build_output_summary_sections(
    result: GateWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    manifest_rows = _manifest_rows(result)
    report_rows = _report_artifact_rows(result)
    return (
        review_section(
            "Planned output summary",
            (
                review_item(
                    "outputs",
                    (
                        ("planned manifest output count", len(manifest_rows)),
                        ("planned report output count", len(report_rows)),
                        ("passed samples root", result.output_summary.passed_samples_root),
                        ("dropped samples root", result.output_summary.dropped_samples_root),
                        (
                            "report index path",
                            result.output_summary.planned_report_outputs.index_json_path,
                        ),
                    ),
                ),
            ),
        ),
    )


def build_output_detail_sections(
    result: GateWorkflowResult,
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
            "Planned outputs",
            (*manifest_items, *sample_root_items, *report_items),
        ),
    )


def build_final_review_sections(
    bundle: GateExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_final_operator_summary_sections(bundle)


def build_written_artifact_summary_sections(
    bundle: GateExecutionBundle,
    report_artifacts: GateWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    report_fields: tuple[tuple[str, RenderableValue], ...] = ()
    if report_artifacts is not None:
        report_fields = (
            ("written report artifact count", 8),
            ("report index path", report_artifacts.index_json_path),
            ("report execution id", report_artifacts.execution_id),
        )
    return (
        review_section(
            "Written artifact summary",
            (
                review_item(
                    "artifacts",
                    (
                        ("written payload artifact count", len(bundle.written_payload_artifacts)),
                        (
                            "written manifest artifact count",
                            len(bundle.written_manifest_artifacts),
                        ),
                        *report_fields,
                    ),
                ),
            ),
        ),
    )


def build_written_artifact_detail_sections(
    bundle: GateExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Written artifact details",
            tuple(_written_artifact_item(row) for row in _written_artifact_rows(bundle)),
        ),
    )


def build_final_operator_summary_sections(
    bundle: GateExecutionBundle,
    report_artifacts: GateWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    result = bundle.workflow_result
    total_processed = sum(split_result.processed_count for split_result in bundle.split_results)
    total_passed = sum(split_result.passed_count for split_result in bundle.split_results)
    total_dropped = sum(split_result.dropped_count for split_result in bundle.split_results)
    identity_rows = _identity_invariant_rows(bundle)
    identity_failure_count = sum(1 for row in identity_rows if row.status != "passed")
    viability_rows = _viability_drop_rows(bundle)
    non_viable_rows = tuple(row for row in viability_rows if row.viability_status != "viable")
    report_fields: tuple[tuple[str, RenderableValue], ...] = ()
    if report_artifacts is not None:
        report_fields = (("report index path", report_artifacts.index_json_path),)
    return (
        review_section(
            "Final operator summary",
            (
                review_item(
                    "gate",
                    (
                        ("workflow", "gate"),
                        ("split count", len(result.config.splits)),
                        ("total processed", total_processed),
                        ("total passed", total_passed),
                        ("total dropped", total_dropped),
                        ("identity failure count", identity_failure_count),
                        ("non-viable count", len(non_viable_rows)),
                        (
                            "drop stage breakdown",
                            _count_labels(
                                row.drop_stage or "not_dropped"
                                for row in viability_rows
                                if row.dropped
                            ),
                        ),
                        (
                            "planned manifest output count",
                            len(result.output_summary.planned_manifest_outputs),
                        ),
                        (
                            "written payload artifact count",
                            len(bundle.written_payload_artifacts),
                        ),
                        (
                            "written manifest artifact count",
                            len(bundle.written_manifest_artifacts),
                        ),
                        *report_fields,
                    ),
                ),
            ),
        ),
    )


def build_processing_detail_records(
    bundle: GateExecutionBundle,
) -> tuple[GateProcessingDetailRecord, ...]:
    records: list[GateProcessingDetailRecord] = []
    for split_result in bundle.split_results:
        passed_ids = {entry.sample_id for entry in split_result.passed_entries}
        dropped_by_id = {entry.sample_id: entry for entry in split_result.dropped_entries}
        for match in split_result.source_matches:
            sample_id = (
                match.candidate_identity.keypoint.sample_key.value
                if match.candidate_identity
                else sample_id_from_translation(match.translation)
            )
            records.append(
                GateProcessingDetailRecord(
                    split=split_result.split,
                    sample_id=sample_id,
                    match_status=match.status.value,
                    passed=sample_id in passed_ids,
                    drop_stage=(
                        dropped_by_id[sample_id].drop_stage.value
                        if sample_id in dropped_by_id
                        else None
                    ),
                )
            )
    return tuple(records)


def build_processing_summary_records(
    bundle: GateExecutionBundle,
) -> tuple[GateProcessingSummaryRecord, ...]:
    return tuple(
        _processing_summary_record(bundle, split_result)
        for split_result in bundle.split_results
    )


def _processing_summary_record(
    bundle: GateExecutionBundle,
    split_result: GateSplitProcessingResult,
) -> GateProcessingSummaryRecord:
    dropped_entries = split_result.dropped_entries
    debug_ref_count = sum(1 for entry in dropped_entries if entry.debug_ref is not None)
    return GateProcessingSummaryRecord(
        split=split_result.split,
        processed_count=split_result.processed_count,
        prepared_sample_count=len(split_result.prepared_samples),
        passed_count=split_result.passed_count,
        dropped_count=split_result.dropped_count,
        materialize_dropped_debug_payloads=(
            bundle.workflow_result.config.materialize_dropped_debug_payloads
        ),
        pose_or_source_dropped_without_prepared_payload_count=sum(
            1
            for entry in dropped_entries
            if entry.drop_stage in {GateDropStage.SOURCE, GateDropStage.POSE}
        ),
        gate_dropped_prepared_sample_count=sum(
            1 for entry in dropped_entries if entry.drop_stage == GateDropStage.GATES
        ),
        dropped_debug_payload_written_count=len(split_result.dropped_debug_payloads),
        dropped_manifest_entries_with_debug_ref_count=debug_ref_count,
        dropped_manifest_entries_without_debug_ref_count=len(dropped_entries)
        - debug_ref_count,
    )


def build_gate_summary_records(
    bundle: GateExecutionBundle,
) -> tuple[GateSummaryRecord, ...]:
    return tuple(
        GateSummaryRecord(
            split=split_result.split,
            evaluated_count=len(split_result.gate_bundles),
            passed_count=sum(
                1 for gate in split_result.gate_bundles if gate.final_status.value == "passed"
            ),
            dropped_count=sum(
                1 for gate in split_result.gate_bundles if gate.final_status.value == "dropped"
            ),
        )
        for split_result in bundle.split_results
    )


def build_gate_detail_records(
    bundle: GateExecutionBundle,
) -> tuple[GateDetailRecord, ...]:
    return tuple(
        GateDetailRecord(
            split=split_result.split,
            sample_id=gate.sample_id,
            final_status=gate.final_status.value,
            terminal_gate=None if gate.terminal_gate is None else gate.terminal_gate.value,
            failed_gates=tuple(failed.value for failed in gate.failed_gates),
            decisions=tuple(
                GateDecisionReviewRecord(
                    gate=decision.gate.value,
                    status=decision.status.value,
                    issue_codes=tuple(code.value for code in decision.issue_codes),
                )
                for decision in gate.decisions
            ),
        )
        for split_result in bundle.split_results
        for gate in split_result.gate_bundles
    )


def build_source_issue_summary_records(
    bundle: GateExecutionBundle,
) -> tuple[GateSourceIssueSummaryRecord, ...]:
    return tuple(
        GateSourceIssueSummaryRecord(
            split=split_result.split,
            source_issue_count=sum(
                len(match.source_issues) for match in split_result.source_matches
            ),
            unmatched_count=sum(1 for match in split_result.source_matches if not match.matched),
        )
        for split_result in bundle.split_results
    )


def build_source_issue_detail_records(
    bundle: GateExecutionBundle,
) -> tuple[GateSourceIssueDetailRecord, ...]:
    records: list[GateSourceIssueDetailRecord] = []
    for split_result in bundle.split_results:
        for match in split_result.source_matches:
            for issue in match.source_issues:
                records.append(
                    GateSourceIssueDetailRecord(
                        split=split_result.split,
                        sample_id=sample_id_from_translation(match.translation),
                        issue_code=issue.code.value,
                        detail=issue.detail,
                    )
                )
            if not match.matched and match.unmatched_reason is not None:
                records.append(
                    GateSourceIssueDetailRecord(
                        split=split_result.split,
                        sample_id=sample_id_from_translation(match.translation),
                        issue_code=match.unmatched_reason.code.value,
                        detail=match.unmatched_reason.detail,
                    )
                )
    return tuple(records)


def build_report_table_records(
    bundle: GateExecutionBundle,
) -> GateReportTables:
    return gate_report_tables(build_gate_report(bundle))


def _count_labels(values: Iterable[object]) -> tuple[str, ...]:
    counts = Counter(str(value) for value in values)
    return tuple(f"{key}={count}" for key, count in sorted(counts.items()))


def _operation_failure_items(
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


def _runtime_check_failure_items(
    rows: tuple[GateRuntimeAssetRow, ...],
    *,
    limit: int = 5,
) -> tuple[WorkflowReviewItem, ...]:
    return tuple(
        review_item(
            f"failed: {row.label}",
            (
                ("scope", row.scope),
                ("path", row.path),
                ("exists", row.exists),
                ("valid", row.valid),
                ("message", row.message),
            ),
        )
        for row in rows[:limit]
    )


def _identity_failure_items(
    rows: tuple[GateIdentityInvariantRow, ...],
    *,
    limit: int = 5,
) -> tuple[WorkflowReviewItem, ...]:
    return tuple(_identity_invariant_item(row) for row in rows[:limit])


def _viability_anomaly_items(
    rows: tuple[GateViabilityDropRow, ...],
    *,
    limit: int = 5,
) -> tuple[WorkflowReviewItem, ...]:
    return tuple(_viability_drop_item(row) for row in rows[:limit])


def _runtime_asset_rows(
    verification: GateRuntimeVerification,
    scope: str,
) -> tuple[GateRuntimeAssetRow, ...]:
    return tuple(
        GateRuntimeAssetRow(
            label=check.label,
            path=check.path,
            exists=check.exists,
            valid=check.valid,
            message=check.message,
            scope=check.scope,
        )
        for check in verification.checks
        if check.scope == scope
    )


def _split_count_rows(bundle: GateExecutionBundle) -> tuple[GateSplitCountRow, ...]:
    return tuple(
        GateSplitCountRow(
            split=split_result.split,
            processed_count=split_result.processed_count,
            passed_count=split_result.passed_count,
            dropped_count=split_result.dropped_count,
        )
        for split_result in bundle.split_results
    )


def _identity_invariant_rows(
    bundle: GateExecutionBundle,
) -> tuple[GateIdentityInvariantRow, ...]:
    rows: list[GateIdentityInvariantRow] = []
    for split_result in bundle.split_results:
        for match in split_result.source_matches:
            if match.candidate_identity is None:
                continue
            rows.append(
                GateIdentityInvariantRow(
                    split=split_result.split,
                    sample_id=match.candidate_identity.keypoint.sample_key.value,
                    sentence_id=match.candidate_identity.translation.sentence_key.value,
                    status="passed",
                    issue_count=0,
                )
            )
    return tuple(rows)


def _viability_drop_rows(bundle: GateExecutionBundle) -> tuple[GateViabilityDropRow, ...]:
    rows: list[GateViabilityDropRow] = []
    dropped_by_key = {
        (split_result.split, entry.sample_id): entry
        for split_result in bundle.split_results
        for entry in split_result.dropped_entries
    }
    for split_result in bundle.split_results:
        for report in split_result.viability_reports:
            dropped = dropped_by_key.get((split_result.split, report.sample_id))
            rows.append(
                GateViabilityDropRow(
                    split=split_result.split,
                    sample_id=report.sample_id,
                    sentence_id=report.sentence_id,
                    viability_status=report.status.value,
                    issue_codes=tuple(issue.code.value for issue in report.issues),
                    dropped=dropped is not None,
                    drop_stage=None if dropped is None else dropped.drop_stage.value,
                )
            )
    return tuple(rows)


def _manifest_rows(result: GateWorkflowResult) -> tuple[GateManifestRow, ...]:
    return tuple(
        GateManifestRow(
            label=f"{manifest.partition} manifest [{manifest.split}]",
            path=manifest.path,
            partition=manifest.partition,
            split=manifest.split,
        )
        for manifest in result.output_summary.planned_manifest_outputs
    )


def _report_artifact_rows(result: GateWorkflowResult) -> tuple[GateReportArtifactRow, ...]:
    artifacts = result.output_summary.planned_report_outputs
    return (
        GateReportArtifactRow("summary markdown", artifacts.summary_markdown_path),
        GateReportArtifactRow("processing summary", artifacts.processing_summary_jsonl_path),
        GateReportArtifactRow("processing detail", artifacts.processing_detail_jsonl_path),
        GateReportArtifactRow("gate summary", artifacts.gate_summary_jsonl_path),
        GateReportArtifactRow("gate detail", artifacts.gate_detail_jsonl_path),
        GateReportArtifactRow("source issue summary", artifacts.source_issue_summary_jsonl_path),
        GateReportArtifactRow("source issue detail", artifacts.source_issue_detail_jsonl_path),
        GateReportArtifactRow("report index", artifacts.index_json_path),
    )


def _written_artifact_rows(bundle: GateExecutionBundle) -> tuple[GateWrittenArtifactRow, ...]:
    payload_rows = tuple(
        GateWrittenArtifactRow(
            label=payload.receipt.label,
            kind=payload.receipt.kind,
            path=payload.path,
            sha256=payload.sha256,
            execution_id=payload.execution_id,
            sample_id=payload.sample_id,
        )
        for payload in bundle.written_payload_artifacts
    )
    manifest_rows = tuple(
        GateWrittenArtifactRow(
            label=manifest.receipt.label,
            kind=manifest.receipt.kind,
            path=manifest.path,
            sha256=manifest.sha256,
            execution_id=manifest.execution_id,
        )
        for manifest in bundle.written_manifest_artifacts
    )
    return (*payload_rows, *manifest_rows)


def _operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    return review_item(
        operation.label,
        (
            ("kind", operation.__class__.__name__),
            ("live owner", operation.progress.live_owner if operation.progress else None),
        ),
    )


def _execution_result_item(result: OperationExecutionResult) -> WorkflowReviewItem:
    return review_item(
        result.label,
        (
            ("operation_kind", result.operation_kind),
            ("succeeded", result.succeeded),
            ("returncode", result.returncode),
            ("execution_mode", result.execution_mode),
        ),
    )


def _runtime_asset_item(row: GateRuntimeAssetRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("path", row.path),
            ("exists", row.exists),
            ("valid", row.valid),
            ("scope", row.scope),
            ("message", row.message),
        ),
    )


def _split_count_item(row: GateSplitCountRow) -> WorkflowReviewItem:
    return review_item(
        row.split,
        (
            ("processed", row.processed_count),
            ("passed", row.passed_count),
            ("dropped", row.dropped_count),
        ),
    )


def _identity_invariant_item(row: GateIdentityInvariantRow) -> WorkflowReviewItem:
    return review_item(
        f"{row.split}/{row.sample_id}",
        (
            ("sentence_id", row.sentence_id),
            ("status", row.status),
            ("issue_count", row.issue_count),
        ),
    )


def _viability_drop_item(row: GateViabilityDropRow) -> WorkflowReviewItem:
    return review_item(
        f"{row.split}/{row.sample_id}",
        (
            ("sentence_id", row.sentence_id),
            ("viability_status", row.viability_status),
            ("issue_codes", row.issue_codes),
            ("dropped", row.dropped),
            ("drop_stage", row.drop_stage),
        ),
    )


def _manifest_item(row: GateManifestRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (("path", row.path), ("partition", row.partition), ("split", row.split)),
    )


def _report_artifact_item(row: GateReportArtifactRow) -> WorkflowReviewItem:
    return review_item(row.label, (("path", row.path),))


def _written_artifact_item(row: GateWrittenArtifactRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("kind", row.kind),
            ("path", row.path),
            ("sha256", row.sha256),
            ("execution_id", row.execution_id),
            ("sample_id", row.sample_id),
        ),
    )


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "build_final_operator_summary_sections",
    "build_final_review_sections",
    "build_gate_detail_records",
    "build_gate_summary_records",
    "build_output_detail_sections",
    "build_output_summary_sections",
    "build_processing_detail_records",
    "build_processing_detail_sections",
    "build_processing_summary_records",
    "build_processing_summary_sections",
    "build_report_sections",
    "build_report_table_records",
    "build_runtime_plan_detail_sections",
    "build_runtime_plan_sections",
    "build_runtime_plan_summary_sections",
    "build_runtime_restore_detail_sections",
    "build_runtime_restore_sections",
    "build_runtime_restore_summary_sections",
    "build_runtime_verification_detail_sections",
    "build_runtime_verification_sections",
    "build_runtime_verification_summary_sections",
    "build_gate_report",
    "build_source_issue_detail_records",
    "build_source_issue_summary_records",
    "build_written_artifact_detail_sections",
    "build_written_artifact_summary_sections",
]
