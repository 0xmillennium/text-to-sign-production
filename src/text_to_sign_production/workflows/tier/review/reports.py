from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.data.tier.reports.calibration import (
    TierCalibrationProgressSpecs,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import (
    JsonValue,
    render_review_sections_markdown,
    write_json,
    write_markdown,
)
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_CALIBRATION_ACTIVE_SPAN_DERIVATION,
    TIER_STAGE_CALIBRATION_BINDING_DISTRIBUTIONS,
    TIER_STAGE_CALIBRATION_FAMILY_PASS_SURFACES,
    TIER_STAGE_CALIBRATION_FAMILY_WATERFALLS,
    TIER_STAGE_DECISION_DETAIL_JSON_PROJECT,
    TIER_STAGE_DECISION_DETAIL_PAYLOAD_PROJECT,
    TIER_STAGE_REPORT_FILE_WRITE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import TierWrittenReportArtifacts
from text_to_sign_production.workflows.tier.contracts.review import (
    ActiveSpanDerivationReview,
    CalibrationDetailReviewPayload,
    CalibrationSurfacesReviewPayload,
    DistributionSummaryReview,
    FamilyPassSurfaceReview,
    FamilyWaterfallsReview,
    FamilyWaterfallStepReview,
    LeakageReviewSummary,
    MembershipCountReview,
    TierDecisionDetailReviewPayload,
    TierManifestOutputReview,
    TierReportIndexPayload,
)
from text_to_sign_production.workflows.tier.processing import TierExecutionBundle
from text_to_sign_production.workflows.tier.review.sections import (
    build_calibration_detail_review_payload,
    build_calibration_sections,
    build_calibration_surfaces_review_payload,
    build_decision_detail_review_payload,
    build_final_review_sections,
    build_output_summary_sections,
    build_processing_summary_sections,
    build_runtime_verification_sections,
    _tier_calibration_surfaces,
)


def write_tier_reports(
    *,
    bundle: TierExecutionBundle,
    progress_session: ProgressSession | None = None,
) -> TierWrittenReportArtifacts:
    progress_session = _visible_progress_session(progress_session)
    output_summary = bundle.workflow_result.output_summary
    artifacts = output_summary.planned_report_outputs

    summary_sections = (
        *build_runtime_verification_sections(bundle.workflow_result.runtime_verification),
        *build_processing_summary_sections(bundle),
        *build_output_summary_sections(bundle.workflow_result),
        *build_final_review_sections(bundle),
    )
    summary_markdown = render_review_sections_markdown(summary_sections)
    calibration_surfaces = _tier_calibration_surfaces(
        bundle,
        progress_session=progress_session,
        progress_specs=_calibration_progress_specs(),
    )
    calibration_markdown = render_review_sections_markdown(
        build_calibration_sections(
            bundle,
            calibration_surfaces=calibration_surfaces,
        )
    )
    decision_detail_payload = build_decision_detail_review_payload(
        bundle,
        progress_session=progress_session,
        progress_spec=_decision_detail_payload_projection_progress_spec(),
    )
    decision_detail_document = _decision_detail_json_document(
        decision_detail_payload,
        progress_session=progress_session,
    )
    calibration_surfaces_document = _calibration_surfaces_json_payload(
        build_calibration_surfaces_review_payload(
            bundle,
            calibration_surfaces=calibration_surfaces,
        )
    )
    calibration_detail_document = _calibration_detail_json_payload(
        build_calibration_detail_review_payload(
            bundle,
            calibration_surfaces=calibration_surfaces,
        )
    )
    index_document = _report_index_json_payload(
        TierReportIndexPayload(
            summary_markdown_path=artifacts.summary_markdown_path,
            calibration_markdown_path=artifacts.calibration_markdown_path,
            decision_detail_json_path=artifacts.decision_detail_json_path,
            calibration_surfaces_json_path=artifacts.calibration_surfaces_json_path,
            calibration_detail_json_path=artifacts.calibration_detail_json_path,
            index_json_path=artifacts.index_json_path,
            config_provenance=tuple(
                (
                    output_summary_config.label,
                    output_summary_config.original_path,
                    output_summary_config.execution_path,
                    output_summary_config.sha256,
                )
                for output_summary_config in (
                    bundle.workflow_result.execution_inputs.filters_config_provenance,
                    bundle.workflow_result.execution_inputs.tier_config_provenance,
                )
            ),
            planned_tiered_manifest_outputs=tuple(
                TierManifestOutputReview(
                    tier=output.tier,
                    membership=output.membership,
                    split=output.split,
                    path=output.path,
                )
                for output in output_summary.planned_tiered_manifest_outputs
            ),
        )
    )
    _write_report_files(
        (
            ("markdown", artifacts.summary_markdown_path, summary_markdown),
            ("markdown", artifacts.calibration_markdown_path, calibration_markdown),
            ("json", artifacts.decision_detail_json_path, decision_detail_document),
            ("json", artifacts.calibration_surfaces_json_path, calibration_surfaces_document),
            ("json", artifacts.calibration_detail_json_path, calibration_detail_document),
            ("json", artifacts.index_json_path, index_document),
        ),
        progress_session=progress_session,
    )
    return TierWrittenReportArtifacts(
        summary_markdown=written_file_receipt(
            "tier summary markdown",
            artifacts.summary_markdown_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="tier_report",
        ),
        calibration_markdown=written_file_receipt(
            "tier calibration markdown",
            artifacts.calibration_markdown_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="tier_report",
        ),
        decision_detail_json=written_file_receipt(
            "tier decision detail",
            artifacts.decision_detail_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="tier_report",
        ),
        calibration_surfaces_json=written_file_receipt(
            "tier calibration surfaces",
            artifacts.calibration_surfaces_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="tier_report",
        ),
        calibration_detail_json=written_file_receipt(
            "tier calibration detail",
            artifacts.calibration_detail_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="tier_report",
        ),
        index_json=written_file_receipt(
            "tier report index",
            artifacts.index_json_path,
            execution_id=bundle.workflow_result.execution_id,
            kind="tier_report",
        ),
    )


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=TIER_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )


def _decision_detail_json_document(
    payload: TierDecisionDetailReviewPayload,
    *,
    progress_session: ProgressSession | None,
) -> JsonValue:
    records = _decision_detail_records(payload, progress_session=progress_session)
    return {
        "schema_version": "tier.decision.detail.v1",
        "report_kind": "tier_decision_detail",
        "record_count": len(records),
        "records": records,
    }


def _decision_detail_payload_projection_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_DECISION_DETAIL_PAYLOAD_PROJECT,
        label="decision detail payload projection",
        unit="decision",
        owner_module="text_to_sign_production.workflows.tier.review.sections",
        split_behavior="global",
        operation_kind="projection",
        total_semantics="tier reports indexed and decision bundles projected",
        bar_eligible=True,
        allowed_counters=("indexed", "projected", "skipped", "failed"),
        artifact_role="decision_detail",
    )


def _calibration_progress_specs() -> TierCalibrationProgressSpecs:
    return TierCalibrationProgressSpecs(
        family_pass_surfaces=ProgressStageSpec(
            workflow_id=TIER_WORKFLOW_NAME,
            stage_id=TIER_STAGE_CALIBRATION_FAMILY_PASS_SURFACES,
            label="calibration family pass surfaces",
            unit="surface",
            owner_module="text_to_sign_production.data.tier.reports.calibration",
            split_behavior="global",
            operation_kind="projection",
            total_semantics="family pass calibration aggregate",
            bar_eligible=True,
            allowed_counters=("projected", "failed"),
            artifact_role="calibration",
        ),
        binding_metric_distributions=ProgressStageSpec(
            workflow_id=TIER_WORKFLOW_NAME,
            stage_id=TIER_STAGE_CALIBRATION_BINDING_DISTRIBUTIONS,
            label="calibration binding metric distributions",
            unit="distribution set",
            owner_module="text_to_sign_production.data.tier.reports.calibration",
            split_behavior="global",
            operation_kind="projection",
            total_semantics="binding metric calibration distributions",
            bar_eligible=True,
            allowed_counters=("projected", "failed"),
            artifact_role="calibration",
        ),
        family_waterfalls=ProgressStageSpec(
            workflow_id=TIER_WORKFLOW_NAME,
            stage_id=TIER_STAGE_CALIBRATION_FAMILY_WATERFALLS,
            label="calibration family waterfalls",
            unit="waterfall set",
            owner_module="text_to_sign_production.data.tier.reports.calibration",
            split_behavior="global",
            operation_kind="projection",
            total_semantics="family waterfall calibration aggregate",
            bar_eligible=True,
            allowed_counters=("projected", "failed"),
            artifact_role="calibration",
        ),
        active_span_derivation=ProgressStageSpec(
            workflow_id=TIER_WORKFLOW_NAME,
            stage_id=TIER_STAGE_CALIBRATION_ACTIVE_SPAN_DERIVATION,
            label="calibration active span derivation",
            unit="summary",
            owner_module="text_to_sign_production.data.tier.reports.calibration",
            split_behavior="global",
            operation_kind="projection",
            total_semantics="active span derivation calibration aggregate",
            bar_eligible=True,
            allowed_counters=("projected", "failed"),
            artifact_role="calibration",
        ),
    )


def _decision_detail_json_projection_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_DECISION_DETAIL_JSON_PROJECT,
        label="decision detail JSON projection",
        unit="record",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="projection",
        total_semantics="decision detail JSON records",
        bar_eligible=True,
        allowed_counters=("projected", "skipped", "failed"),
        artifact_role="decision_detail",
    )


def _decision_detail_records(
    payload: TierDecisionDetailReviewPayload,
    *,
    progress_session: ProgressSession | None,
) -> tuple[JsonValue, ...]:
    if progress_session is None:
        return tuple(_decision_detail_record_json(record) for record in payload.records)
    records: list[JsonValue] = []
    projected = 0
    with progress_session.task(
        _decision_detail_json_projection_progress_spec(),
        total=len(payload.records),
    ) as progress_task:
        for record in payload.records:
            records.append(_decision_detail_record_json(record))
            projected += 1
            progress_task.advance(counters={"projected": projected})
    return tuple(records)


def _decision_detail_record_json(record) -> JsonValue:
    return {
        "split": record.split,
        "sample_id": record.sample_id,
        "selected_tier": record.selected_tier,
        "tier_status": record.tier_status,
        "leakage_observed_max_severity": record.leakage_observed_max_severity,
        "leakage_admissible_tiers": record.leakage_admissible_tiers,
        "leakage_rejected_tiers": record.leakage_rejected_tiers,
        "report_summary": dict(record.report_summary.items()),
        "metric_rows": tuple(
            {
                "family": row.family,
                "metric_name": row.metric_name,
                "value": row.value,
            }
            for row in record.metric_rows
        ),
        "tier_rows": tuple(
            {
                "family": row.family,
                "status": row.status,
                "best_supported_tier": row.best_supported_tier,
                "supported_tiers": row.supported_tiers,
                "issue_count": row.issue_count,
            }
            for row in record.tier_rows
        ),
    }


def _write_report_files(
    files: tuple[tuple[str, Path, str | JsonValue], ...],
    *,
    progress_session: ProgressSession | None,
) -> None:
    if progress_session is None:
        for file_kind, path, payload in files:
            _write_report_file(file_kind, path, payload)
        return
    written = 0
    with progress_session.task(_report_file_write_progress_spec(), total=len(files)) as task:
        for file_kind, path, payload in files:
            _write_report_file(file_kind, path, payload)
            written += 1
            task.advance(counters={"written": written})


def _write_report_file(file_kind: str, path: Path, payload: str | JsonValue) -> None:
    if file_kind == "markdown":
        if not isinstance(payload, str):
            raise TypeError("markdown report payload must be text")
        write_markdown(path, payload)
        return
    if file_kind == "json":
        write_json(path, payload)
        return
    raise ValueError(f"Unsupported tier report file kind: {file_kind}")


def _report_file_write_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_REPORT_FILE_WRITE,
        label="tier report file write",
        unit="file",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="write",
        total_semantics="report files",
        bar_eligible=True,
        allowed_counters=("written", "failed"),
    )


def _calibration_surfaces_json_payload(
    payload: CalibrationSurfacesReviewPayload,
) -> JsonValue:
    return {
        "workflow": payload.workflow,
        "processed_count": payload.processed_count,
        "membership_counts_by_tier_split": tuple(
            _membership_count_json(row) for row in payload.membership_counts_by_tier_split
        ),
        "family_pass_surfaces": tuple(
            _family_pass_surface_json(row) for row in payload.family_pass_surfaces
        ),
        "binding_metric_distributions": tuple(
            _distribution_summary_json(row) for row in payload.binding_metric_distributions
        ),
        "family_waterfalls": _waterfalls_json(payload.family_waterfalls),
        "active_span_derivation": _active_span_derivation_json(payload.active_span_derivation),
        "leakage": _leakage_json(payload.leakage),
        "tier_report_count": payload.tier_report_count,
    }


def _calibration_detail_json_payload(
    payload: CalibrationDetailReviewPayload,
) -> JsonValue:
    return {
        "workflow": payload.workflow,
        "processed_count": payload.processed_count,
        "filter_config": {
            "family_count": payload.filter_config.family_count,
            "families": tuple(
                {
                    "family": family.family,
                    "threshold_level_count": family.threshold_level_count,
                    "levels": family.levels,
                }
                for family in payload.filter_config.families
            ),
        },
        "tier_policies": {
            "tier_order": payload.tier_policies.tier_order,
            "policy_count": payload.tier_policies.policy_count,
            "policies": tuple(
                {
                    "tier": policy.tier,
                    "max_allowed_leakage_severity": policy.max_allowed_leakage_severity,
                    "family_filter_levels": tuple(
                        {
                            "family": level.family,
                            "level": level.level,
                        }
                        for level in policy.family_filter_levels
                    ),
                }
                for policy in payload.tier_policies.policies
            ),
        },
        "membership_counts_by_tier_split": tuple(
            _membership_count_json(row) for row in payload.membership_counts_by_tier_split
        ),
        "family_pass_surfaces": tuple(
            _family_pass_surface_json(row) for row in payload.family_pass_surfaces
        ),
        "binding_metric_distributions": tuple(
            _distribution_summary_json(row) for row in payload.binding_metric_distributions
        ),
        "family_waterfalls": _waterfalls_json(payload.family_waterfalls),
        "active_span_derivation": _active_span_derivation_json(payload.active_span_derivation),
        "leakage": _leakage_json(payload.leakage),
        "tier_reports": tuple(dict(report.items()) for report in payload.tier_reports),
        "outputs": {
            "planned_tiered_manifest_outputs": tuple(
                _manifest_output_json(output) for output in payload.planned_tiered_manifest_outputs
            ),
        },
    }


def _report_index_json_payload(payload: TierReportIndexPayload) -> JsonValue:
    return {
        "workflow": "tier",
        "config_provenance": tuple(
            {
                "label": label,
                "original_path": original_path,
                "execution_path": execution_path,
                "sha256": sha256,
            }
            for label, original_path, execution_path, sha256 in payload.config_provenance
        ),
        "reports": {
            "summary_markdown_path": payload.summary_markdown_path,
            "calibration_markdown_path": payload.calibration_markdown_path,
            "decision_detail_json_path": payload.decision_detail_json_path,
            "calibration_surfaces_json_path": payload.calibration_surfaces_json_path,
            "calibration_detail_json_path": payload.calibration_detail_json_path,
            "index_json_path": payload.index_json_path,
        },
        "outputs": {
            "planned_tiered_manifest_outputs": tuple(
                _manifest_output_json(output) for output in payload.planned_tiered_manifest_outputs
            ),
        },
    }


def _membership_count_json(row: MembershipCountReview) -> JsonValue:
    return {
        "tier": row.tier,
        "split": row.split,
        "included_count": row.included_count,
        "excluded_count": row.excluded_count,
    }


def _family_pass_surface_json(row: FamilyPassSurfaceReview) -> JsonValue:
    return {
        "family": row.family,
        "loose_pass_count": row.loose_pass_count,
        "loose_pass_ratio": row.loose_pass_ratio,
        "clean_pass_count": row.clean_pass_count,
        "clean_pass_ratio": row.clean_pass_ratio,
        "tight_pass_count": row.tight_pass_count,
        "tight_pass_ratio": row.tight_pass_ratio,
    }


def _distribution_summary_json(row: DistributionSummaryReview) -> JsonValue:
    return {
        "family": row.family,
        "metric_name": row.metric_name,
        "min": row.minimum,
        "p05": row.p05,
        "p10": row.p10,
        "p25": row.p25,
        "p50": row.p50,
        "p75": row.p75,
        "p90": row.p90,
        "p95": row.p95,
        "max": row.maximum,
    }


def _waterfall_step_json(row: FamilyWaterfallStepReview) -> JsonValue:
    return {
        "tier": row.tier,
        "family": row.family,
        "remaining_sample_count_before_family": row.remaining_before_count,
        "remaining_sample_count_after_family": row.remaining_after_count,
        "dropped_sample_count": row.dropped_count,
    }


def _waterfalls_json(row: FamilyWaterfallsReview) -> JsonValue:
    return {
        "loose": tuple(_waterfall_step_json(step) for step in row.loose),
        "clean": tuple(_waterfall_step_json(step) for step in row.clean),
        "tight": tuple(_waterfall_step_json(step) for step in row.tight),
    }


def _active_span_derivation_json(row: ActiveSpanDerivationReview) -> JsonValue:
    return {
        "count_distributions": tuple(
            _distribution_summary_json(distribution) for distribution in row.count_distributions
        ),
        "fallback_used_count": row.fallback_used_count,
        "fallback_used_ratio": row.fallback_used_ratio,
    }


def _leakage_json(summary: LeakageReviewSummary) -> JsonValue:
    return {
        "sample_summary_count": summary.sample_summary_count,
        "pair_fact_count": summary.pair_fact_count,
        "affected_sample_count": summary.affected_sample_count,
        "max_severity_counts": tuple(
            {"name": row.name, "count": row.count} for row in summary.max_severity_counts
        ),
    }


def _manifest_output_json(output: TierManifestOutputReview) -> JsonValue:
    return {
        "tier": output.tier,
        "membership": output.membership,
        "split": output.split,
        "path": output.path,
    }
