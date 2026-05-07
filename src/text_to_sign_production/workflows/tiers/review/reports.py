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
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_DECISION_DETAIL_WRITE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import TiersReportArtifacts
from text_to_sign_production.workflows.tiers.processing import TiersExecutionBundle
from text_to_sign_production.workflows.tiers.review.sections import (
    build_calibration_detail_payload,
    build_calibration_sections,
    build_calibration_surfaces_payload,
    build_decision_detail_records,
    build_final_review_sections,
    build_output_summary_sections,
    build_processing_summary_sections,
    build_runtime_verification_sections,
)


def write_tiers_reports(
    *,
    bundle: TiersExecutionBundle,
    progress_session: ProgressSession | None = None,
) -> TiersReportArtifacts:
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
    write_markdown(
        artifacts.calibration_markdown_path,
        render_review_sections_markdown(build_calibration_sections(bundle)),
    )
    _write_decision_detail_jsonl(
        artifacts.decision_detail_jsonl_path,
        build_decision_detail_records(bundle),
        progress_session=progress_session,
    )
    write_json(
        artifacts.calibration_surfaces_json_path,
        build_calibration_surfaces_payload(bundle),
    )
    write_json(
        artifacts.calibration_detail_json_path,
        build_calibration_detail_payload(bundle),
    )
    write_json(
        artifacts.index_json_path,
        {
            "workflow": "tiers",
            "reports": {
                "summary_markdown_path": artifacts.summary_markdown_path,
                "calibration_markdown_path": artifacts.calibration_markdown_path,
                "decision_detail_jsonl_path": artifacts.decision_detail_jsonl_path,
                "calibration_surfaces_json_path": artifacts.calibration_surfaces_json_path,
                "calibration_detail_json_path": artifacts.calibration_detail_json_path,
                "index_json_path": artifacts.index_json_path,
            },
            "outputs": {
                "tiered_manifest_outputs": tuple(
                    {
                        "tier": output.tier,
                        "membership": output.membership,
                        "split": output.split,
                        "path": output.path,
                    }
                    for output in output_summary.tiered_manifest_outputs
                ),
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
        workflow_id=TIERS_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def _write_decision_detail_jsonl(
    path: Path,
    records: tuple[Mapping[str, object], ...],
    *,
    progress_session: ProgressSession | None,
) -> None:
    if progress_session is not None and records:
        with progress_session.task(
            _decision_detail_progress_spec(),
            total=len(records),
        ) as progress_task:
            write_jsonl(path, records, progress=progress_task)
    else:
        write_jsonl(path, records)


def _decision_detail_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_DECISION_DETAIL_WRITE,
        label="decision detail report write",
        unit="record",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="jsonl_report_write",
        total_semantics="records written to decision detail JSONL report",
        bar_eligible=True,
        artifact_role="decision_detail",
    )
