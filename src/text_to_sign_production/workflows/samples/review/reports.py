from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.workflows.foundation.review import (
    render_review_sections_markdown,
    write_json,
    write_jsonl,
    write_markdown,
)
from text_to_sign_production.workflows.samples.constants import (
    SAMPLES_STAGE_REPORT_GATE_DETAIL_WRITE,
    SAMPLES_STAGE_REPORT_GATE_SUMMARY_WRITE,
    SAMPLES_STAGE_REPORT_PROCESSING_DETAIL_WRITE,
    SAMPLES_STAGE_REPORT_PROCESSING_SUMMARY_WRITE,
    SAMPLES_STAGE_REPORT_SOURCE_ISSUE_DETAIL_WRITE,
    SAMPLES_STAGE_REPORT_SOURCE_ISSUE_SUMMARY_WRITE,
    SAMPLES_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.samples.contracts import SamplesReportArtifacts
from text_to_sign_production.workflows.samples.processing import SamplesExecutionBundle
from text_to_sign_production.workflows.samples.review.sections import (
    build_final_review_sections,
    build_gate_detail_records,
    build_gate_summary_records,
    build_output_summary_sections,
    build_processing_detail_records,
    build_processing_summary_records,
    build_processing_summary_sections,
    build_runtime_verification_sections,
    build_source_issue_detail_records,
    build_source_issue_summary_records,
)


def write_samples_reports(
    *,
    bundle: SamplesExecutionBundle,
    progress_session: ProgressSession | None = None,
) -> SamplesReportArtifacts:
    progress_session = _visible_progress_session(progress_session)
    output_summary = bundle.workflow_result.output_summary
    artifacts = output_summary.report_artifacts

    summary_sections = (
        *build_runtime_verification_sections(bundle.workflow_result.runtime_verification),
        *build_processing_summary_sections(bundle),
        *build_output_summary_sections(bundle.workflow_result),
        *build_final_review_sections(bundle),
    )
    write_markdown(
        artifacts.summary_markdown_path,
        render_review_sections_markdown(summary_sections),
    )
    _write_jsonl_report(
        artifacts.processing_summary_jsonl_path,
        build_processing_summary_records(bundle),
        stage_id=SAMPLES_STAGE_REPORT_PROCESSING_SUMMARY_WRITE,
        label="processing summary report write",
        artifact_role="processing_summary",
        progress_session=progress_session,
    )
    _write_jsonl_report(
        artifacts.processing_detail_jsonl_path,
        build_processing_detail_records(bundle),
        stage_id=SAMPLES_STAGE_REPORT_PROCESSING_DETAIL_WRITE,
        label="processing detail report write",
        artifact_role="processing_detail",
        progress_session=progress_session,
    )
    _write_jsonl_report(
        artifacts.gate_summary_jsonl_path,
        build_gate_summary_records(bundle),
        stage_id=SAMPLES_STAGE_REPORT_GATE_SUMMARY_WRITE,
        label="gate summary report write",
        artifact_role="gate_summary",
        progress_session=progress_session,
    )
    _write_jsonl_report(
        artifacts.gate_detail_jsonl_path,
        build_gate_detail_records(bundle),
        stage_id=SAMPLES_STAGE_REPORT_GATE_DETAIL_WRITE,
        label="gate detail report write",
        artifact_role="gate_detail",
        progress_session=progress_session,
    )
    _write_jsonl_report(
        artifacts.source_issue_summary_jsonl_path,
        build_source_issue_summary_records(bundle),
        stage_id=SAMPLES_STAGE_REPORT_SOURCE_ISSUE_SUMMARY_WRITE,
        label="source issue summary report write",
        artifact_role="source_issue_summary",
        progress_session=progress_session,
    )
    _write_jsonl_report(
        artifacts.source_issue_detail_jsonl_path,
        build_source_issue_detail_records(bundle),
        stage_id=SAMPLES_STAGE_REPORT_SOURCE_ISSUE_DETAIL_WRITE,
        label="source issue detail report write",
        artifact_role="source_issue_detail",
        progress_session=progress_session,
    )
    write_json(
        artifacts.index_json_path,
        {
            "workflow": "samples",
            "reports": {
                "summary_markdown_path": artifacts.summary_markdown_path,
                "processing_summary_jsonl_path": artifacts.processing_summary_jsonl_path,
                "processing_detail_jsonl_path": artifacts.processing_detail_jsonl_path,
                "gate_summary_jsonl_path": artifacts.gate_summary_jsonl_path,
                "gate_detail_jsonl_path": artifacts.gate_detail_jsonl_path,
                "source_issue_summary_jsonl_path": artifacts.source_issue_summary_jsonl_path,
                "source_issue_detail_jsonl_path": artifacts.source_issue_detail_jsonl_path,
                "index_json_path": artifacts.index_json_path,
            },
            "outputs": {
                "manifest_outputs": tuple(
                    {
                        "partition": manifest.partition,
                        "split": manifest.split,
                        "path": manifest.path,
                    }
                    for manifest in output_summary.manifest_outputs
                ),
                "passed_samples_root": output_summary.passed_samples_root,
                "dropped_samples_root": output_summary.dropped_samples_root,
            },
        },
    )
    return artifacts


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=SAMPLES_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def _write_jsonl_report(
    path: Path,
    records: tuple[Mapping[str, object], ...],
    *,
    stage_id: str,
    label: str,
    artifact_role: str,
    progress_session: ProgressSession | None,
) -> None:
    if progress_session is not None and records:
        with progress_session.task(
            _jsonl_report_progress_spec(
                stage_id=stage_id,
                label=label,
                artifact_role=artifact_role,
            ),
            total=len(records),
        ) as progress_task:
            write_jsonl(path, records, progress=progress_task)
    else:
        write_jsonl(path, records)


def _jsonl_report_progress_spec(
    *,
    stage_id: str,
    label: str,
    artifact_role: str,
) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=SAMPLES_WORKFLOW_NAME,
        stage_id=stage_id,
        label=label,
        unit="record",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="jsonl_report_write",
        total_semantics="records written to one JSONL report artifact",
        bar_eligible=True,
        artifact_role=artifact_role,
    )
