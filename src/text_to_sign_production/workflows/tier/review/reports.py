from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.progress import (
    ProgressSession,
    ProgressStageSpec,
    TqdmProgressSink,
)
from text_to_sign_production.workflows.foundation.provenance import written_file_receipt
from text_to_sign_production.workflows.foundation.review import (
    JsonValue,
    render_review_sections_markdown,
    write_json,
    write_jsonl,
    write_markdown,
)
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_DECISION_DETAIL_WRITE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import TierWrittenReportArtifacts
from text_to_sign_production.workflows.tier.contracts.review import (
    ActiveSpanDerivationReview,
    CalibrationDetailReviewPayload,
    CalibrationSurfacesReviewPayload,
    DistributionSummaryReview,
    FamilyPassSurfaceReview,
    FamilyWaterfallStepReview,
    FamilyWaterfallsReview,
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
        build_decision_detail_review_payload(bundle),
        progress_session=progress_session,
    )
    write_json(
        artifacts.calibration_surfaces_json_path,
        _calibration_surfaces_json_payload(build_calibration_surfaces_review_payload(bundle)),
    )
    write_json(
        artifacts.calibration_detail_json_path,
        _calibration_detail_json_payload(build_calibration_detail_review_payload(bundle)),
    )
    write_json(
        artifacts.index_json_path,
        _report_index_json_payload(
            TierReportIndexPayload(
                summary_markdown_path=artifacts.summary_markdown_path,
                calibration_markdown_path=artifacts.calibration_markdown_path,
                decision_detail_jsonl_path=artifacts.decision_detail_jsonl_path,
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
        ),
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
        decision_detail_jsonl=written_file_receipt(
            "tier decision detail",
            artifacts.decision_detail_jsonl_path,
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


def _write_decision_detail_jsonl(
    path: Path,
    payload: TierDecisionDetailReviewPayload,
    *,
    progress_session: ProgressSession | None,
) -> None:
    records = _decision_detail_jsonl_records(payload)
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
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_DECISION_DETAIL_WRITE,
        label="decision detail report write",
        unit="record",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="jsonl_report_write",
        total_semantics="records written to decision detail JSONL report",
        bar_eligible=True,
        artifact_role="decision_detail",
    )


def _decision_detail_jsonl_records(
    payload: TierDecisionDetailReviewPayload,
) -> tuple[JsonValue, ...]:
    return tuple(
        {
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
        for record in payload.records
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
            _distribution_summary_json(row)
            for row in payload.binding_metric_distributions
        ),
        "family_waterfalls": _waterfalls_json(payload.family_waterfalls),
        "active_span_derivation": _active_span_derivation_json(
            payload.active_span_derivation
        ),
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
            _distribution_summary_json(row)
            for row in payload.binding_metric_distributions
        ),
        "family_waterfalls": _waterfalls_json(payload.family_waterfalls),
        "active_span_derivation": _active_span_derivation_json(
            payload.active_span_derivation
        ),
        "leakage": _leakage_json(payload.leakage),
        "tier_reports": tuple(dict(report.items()) for report in payload.tier_reports),
        "outputs": {
            "planned_tiered_manifest_outputs": tuple(
                _manifest_output_json(output)
                for output in payload.planned_tiered_manifest_outputs
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
            "decision_detail_jsonl_path": payload.decision_detail_jsonl_path,
            "calibration_surfaces_json_path": payload.calibration_surfaces_json_path,
            "calibration_detail_json_path": payload.calibration_detail_json_path,
            "index_json_path": payload.index_json_path,
        },
        "outputs": {
            "planned_tiered_manifest_outputs": tuple(
                _manifest_output_json(output)
                for output in payload.planned_tiered_manifest_outputs
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
            _distribution_summary_json(distribution)
            for distribution in row.count_distributions
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
