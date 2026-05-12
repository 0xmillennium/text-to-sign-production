from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from text_to_sign_production.core.ids import TierMembership, TierName
from text_to_sign_production.data.dataset.analysis import (
    summarize_checkpoint_handoff,
    summarize_passed_manifest,
)
from text_to_sign_production.data.tier.families.types import BindingQualityFamily
from text_to_sign_production.data.tier.policies.analysis import tier_membership_for_decision
from text_to_sign_production.data.tier.reports.calibration import (
    TierActiveSpanDerivationSummary,
    TierCalibrationSurfaces,
    TierFamilyPassSurface,
    TierFamilyWaterfallStep,
    TierFamilyWaterfalls,
    TierMetricDistributionSummary,
    build_tier_calibration_surfaces,
)
from text_to_sign_production.data.tier.reports.types import MetricReportRow, TierReportRow
from text_to_sign_production.workflows.foundation.execution import (
    ArchiveCreateOperation,
    ArchiveExtractOperation,
    ArchiveVerifyOperation,
    FileCopyOperation,
    OperationExecutionResult,
    WorkflowOperation,
    operation_kind,
)
from text_to_sign_production.workflows.foundation.review import (
    RenderableValue,
    WorkflowReviewItem,
    WorkflowReviewSection,
    review_item,
    review_section,
)
from text_to_sign_production.workflows.tier.contracts import (
    TieredManifestRow,
    TierMembershipCountRow,
    TierPlannedReportOutputRow,
    TierReportArtifactRow,
    TierRuntimeAssetRow,
    TierRuntimePlan,
    TierRuntimeRestoreResult,
    TierRuntimeVerification,
    TierWorkflowResult,
    TierWrittenReportArtifactRow,
    TierWrittenReportArtifacts,
)
from text_to_sign_production.workflows.tier.contracts.results import TieredManifestOutputPlan
from text_to_sign_production.workflows.tier.contracts.review import (
    CalibrationDetailReviewPayload,
    CalibrationSurfacesReviewPayload,
    ActiveSpanDerivationReview,
    DistributionSummaryReview,
    FamilyFilterLevelReview,
    FamilyPassSurfaceReview,
    FamilyWaterfallStepReview,
    FamilyWaterfallsReview,
    LeakageReviewSummary,
    MembershipCountReview,
    MetricRowReview,
    NamedCountReview,
    TierDecisionDetailReviewPayload,
    TierDecisionReviewRecord,
    TierFilterFamilyReview,
    TierFilterReviewSummary,
    TierManifestOutputReview,
    TierPolicyReview,
    TierPolicyReviewSummary,
    TierReportSummaryReview,
    TierRowReview,
)
from text_to_sign_production.workflows.tier.processing import (
    TierExecutionBundle,
    TierReportResult,
)

def build_runtime_plan_sections(
    plan: TierRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_plan_summary_sections(plan)


def build_runtime_plan_summary_sections(
    plan: TierRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime plan summary",
            (
                review_item(
                    "tier",
                    (
                        ("restore operation count", len(plan.restore_operations)),
                        (
                            "filters config execution path",
                            plan.execution_inputs.filters_config_provenance.execution_path,
                        ),
                        (
                            "tier config execution path",
                            plan.execution_inputs.tier_config_provenance.execution_path,
                        ),
                    ),
                ),
            ),
        ),
    )


def build_runtime_plan_detail_sections(
    plan: TierRuntimePlan,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Config provenance",
            (
                review_item(
                    plan.execution_inputs.filters_config_provenance.label,
                    (
                        (
                            "original_path",
                            plan.execution_inputs.filters_config_provenance.original_path,
                        ),
                        (
                            "execution_path",
                            plan.execution_inputs.filters_config_provenance.execution_path,
                        ),
                        ("sha256", plan.execution_inputs.filters_config_provenance.sha256),
                    ),
                ),
                review_item(
                    plan.execution_inputs.tier_config_provenance.label,
                    (
                        (
                            "original_path",
                            plan.execution_inputs.tier_config_provenance.original_path,
                        ),
                        (
                            "execution_path",
                            plan.execution_inputs.tier_config_provenance.execution_path,
                        ),
                        ("sha256", plan.execution_inputs.tier_config_provenance.sha256),
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
    result: TierRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_restore_summary_sections(result)


def build_runtime_restore_summary_sections(
    result: TierRuntimeRestoreResult,
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
    result: TierRuntimeRestoreResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Runtime restore execution details",
            tuple(_execution_result_item(execution) for execution in result.execution.results),
        ),
    )


def build_runtime_verification_sections(
    verification: TierRuntimeVerification,
) -> tuple[WorkflowReviewSection, ...]:
    return build_runtime_verification_summary_sections(verification)


def build_runtime_verification_summary_sections(
    verification: TierRuntimeVerification,
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
    verification: TierRuntimeVerification,
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
    bundle: TierExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    summary = bundle.workflow_result.output_summary
    checkpoint_handoff = summarize_checkpoint_handoff(
        bundle.catalog_bundle.payloads,
        bundle.catalog_bundle.manifests,
        (),
    )
    passed_manifest = summarize_passed_manifest(bundle.catalog_bundle.manifests)
    membership_rows = _membership_count_rows(bundle)
    return (
        review_section(
            "Processing overview",
            (
                review_item(
                    "tier processing",
                    (
                        ("loaded passed samples", summary.loaded_passed_sample_count),
                        ("quality facts", summary.quality_fact_count),
                        ("quality contexts", summary.quality_context_count),
                        ("quality metric bundles", summary.quality_metric_bundle_count),
                        ("leakage sample summaries", summary.leakage_sample_summary_count),
                        ("tier decisions", summary.tier_decision_count),
                        ("tier reports", summary.tier_report_count),
                        (
                            "planned tiered manifest outputs",
                            len(summary.planned_tiered_manifest_outputs),
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Checkpoint input integrity",
            (
                review_item(
                    "passed checkpoint",
                    (
                        ("payload_count", checkpoint_handoff.payload_count),
                        ("passed_manifest_count", checkpoint_handoff.passed_count),
                        ("coherent_passed_count", checkpoint_handoff.coherent_passed_count),
                        ("coherence_issue_count", checkpoint_handoff.coherence_issue_count),
                        (
                            "passed_manifest_validation_issue_count",
                            passed_manifest.validation_issue_count,
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Membership count summary",
            (
                review_item(
                    "membership",
                    (
                        ("tier/split row count", len(membership_rows)),
                        (
                            "included by tier",
                            _tier_included_count_labels(membership_rows),
                        ),
                    ),
                ),
            ),
        ),
    )


def build_processing_detail_sections(
    bundle: TierExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Membership counts by tier and split",
            tuple(_membership_count_item(row) for row in _membership_count_rows(bundle)),
        ),
    )


def build_calibration_sections(
    bundle: TierExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    return build_calibration_summary_sections(bundle)


def build_calibration_summary_sections(
    bundle: TierExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    calibration = _tier_calibration_surfaces(bundle)
    pass_surfaces = tuple(
        _family_pass_surface_review(row) for row in calibration.family_pass_surfaces
    )
    metric_distributions = tuple(
        _distribution_summary_review(row) for row in calibration.binding_metric_distributions
    )
    waterfalls = _waterfalls_review(calibration.family_waterfalls)
    active_span = _active_span_derivation_review(calibration.active_span_derivation)
    return (
        review_section(
            "Filter config summary",
            (
                review_item(
                    "filters",
                    _filter_review_summary(bundle).items(),
                ),
            ),
        ),
        review_section(
            "Tier policy summary",
            (
                review_item(
                    "policies",
                    _policy_review_summary(bundle).items(),
                ),
            ),
        ),
        review_section(
            "Leakage summary",
            (
                review_item(
                    "leakage",
                    _leakage_review_summary(bundle).items(),
                ),
            ),
        ),
        review_section(
            "Family pass calibration",
            (
                review_item(
                    "pass surfaces",
                    (
                        ("family count", len(pass_surfaces)),
                        ("families", tuple(surface.label for surface in pass_surfaces)),
                    ),
                ),
            ),
        ),
        review_section(
            "Binding metric distributions",
            (
                review_item(
                    "binding metrics",
                    (
                        ("metric count", len(metric_distributions)),
                        (
                            "distributions",
                            tuple(distribution.label for distribution in metric_distributions),
                        ),
                    ),
                ),
            ),
        ),
        review_section(
            "Tier family waterfalls",
            (
                review_item(
                    "waterfalls",
                    (
                        ("loose step count", len(waterfalls.loose)),
                        ("loose steps", tuple(step.label for step in waterfalls.loose)),
                        ("clean step count", len(waterfalls.clean)),
                        ("clean steps", tuple(step.label for step in waterfalls.clean)),
                        ("tight step count", len(waterfalls.tight)),
                        ("tight steps", tuple(step.label for step in waterfalls.tight)),
                    ),
                ),
            ),
        ),
        review_section(
            "Active span derivation calibration",
            (
                review_item(
                    "active span",
                    active_span.items(),
                ),
            ),
        ),
        review_section(
            "Tier report summary",
            (
                review_item(
                    "reports",
                    (
                        ("tier report count", len(bundle.tier_reports)),
                        (
                            "tier status breakdown",
                            _count_labels(
                                report.report.summary.tier_status.value
                                for report in bundle.tier_reports
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def build_calibration_detail_sections(
    bundle: TierExecutionBundle,
) -> tuple[WorkflowReviewSection, ...]:
    calibration = _tier_calibration_surfaces(bundle)
    waterfalls = _waterfalls_review(calibration.family_waterfalls)
    active_span = _active_span_derivation_review(calibration.active_span_derivation)
    return (
        review_section(
            "Family pass calibration details",
            tuple(
                _family_pass_surface_item(_family_pass_surface_review(row))
                for row in calibration.family_pass_surfaces
            ),
        ),
        review_section(
            "Binding metric distribution details",
            tuple(
                _distribution_summary_item(_distribution_summary_review(row))
                for row in calibration.binding_metric_distributions
            ),
        ),
        review_section(
            "Tier family waterfall details",
            (
                *tuple(_waterfall_step_item(row) for row in waterfalls.loose),
                *tuple(_waterfall_step_item(row) for row in waterfalls.clean),
                *tuple(_waterfall_step_item(row) for row in waterfalls.tight),
            ),
        ),
        review_section(
            "Active span derivation details",
            (
                review_item(
                    "fallback",
                    (
                        (
                            "fallback_used_count",
                            active_span.fallback_used_count,
                        ),
                        (
                            "fallback_used_ratio",
                            active_span.fallback_used_ratio,
                        ),
                    ),
                ),
                *tuple(
                    _distribution_summary_item(row)
                    for row in active_span.count_distributions
                ),
            ),
        ),
        review_section(
            "Tier report summary details",
            tuple(_tier_report_summary_item(report) for report in bundle.tier_reports),
        ),
    )


def build_output_summary_sections(
    result: TierWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    manifest_rows = _tiered_manifest_rows(result)
    report_rows = _planned_report_output_rows(result)
    return (
        review_section(
            "Planned output summary",
            (
                review_item(
                    "outputs",
                    (
                        ("planned tiered manifest output count", len(manifest_rows)),
                        ("planned report output count", len(report_rows)),
                        (
                            "summary markdown path",
                            result.output_summary.planned_report_outputs.summary_markdown_path,
                        ),
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
    result: TierWorkflowResult,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Planned tiered manifest outputs",
            tuple(_tiered_manifest_item(row) for row in _tiered_manifest_rows(result)),
        ),
        review_section(
            "Planned report outputs",
            tuple(_planned_report_output_item(row) for row in _planned_report_output_rows(result)),
        ),
    )


def build_final_review_sections(
    bundle: TierExecutionBundle,
    report_artifacts: TierWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    return build_final_operator_summary_sections(bundle, report_artifacts=report_artifacts)


def build_written_report_artifact_summary_sections(
    artifacts: TierWrittenReportArtifacts,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Written report artifact summary",
            (
                review_item(
                    "reports",
                    (
                        ("written report artifact count", 6),
                        ("report index path", artifacts.index_json_path),
                        ("report execution id", artifacts.execution_id),
                    ),
                ),
            ),
        ),
    )


def build_written_report_artifact_detail_sections(
    artifacts: TierWrittenReportArtifacts,
) -> tuple[WorkflowReviewSection, ...]:
    return (
        review_section(
            "Written report artifact details",
            tuple(
                _written_report_artifact_item(row)
                for row in _written_report_artifact_rows(artifacts)
            ),
        ),
    )


def build_final_operator_summary_sections(
    bundle: TierExecutionBundle,
    report_artifacts: TierWrittenReportArtifacts | None = None,
) -> tuple[WorkflowReviewSection, ...]:
    planned_outputs = bundle.workflow_result.output_summary.planned_report_outputs
    report_fields: tuple[tuple[str, RenderableValue], ...] = ()
    if report_artifacts is not None:
        report_fields = (("report index path", report_artifacts.index_json_path),)
    return (
        review_section(
            "Final operator summary",
            (
                review_item(
                    "tier",
                    (
                        ("workflow", "tier"),
                        ("processed_count", bundle.processed_count),
                        (
                            "written tiered manifest artifact count",
                            len(bundle.written_tiered_manifest_artifacts),
                        ),
                        ("tier report count", len(bundle.tier_reports)),
                        ("planned summary markdown path", planned_outputs.summary_markdown_path),
                        (
                            "planned calibration markdown path",
                            planned_outputs.calibration_markdown_path,
                        ),
                        *report_fields,
                    ),
                ),
            ),
        ),
    )


def build_decision_detail_review_payload(
    bundle: TierExecutionBundle,
) -> TierDecisionDetailReviewPayload:
    """Build the typed decision-detail projection consumed by JSONL writing."""
    records: list[TierDecisionReviewRecord] = []
    report_lookup = {
        (report.manifest.split, report.manifest.sample_id): report for report in bundle.tier_reports
    }
    for decision_bundle in bundle.decision_bundles:
        key = (decision_bundle.manifest.split, decision_bundle.manifest.sample_id)
        report_bundle = report_lookup[key]
        report = report_bundle.report
        leakage_decision = decision_bundle.decision.leakage_decision
        records.append(
            TierDecisionReviewRecord(
                split=decision_bundle.manifest.split.value,
                sample_id=decision_bundle.manifest.sample_id,
                selected_tier=(
                    None
                    if decision_bundle.decision.selected_tier is None
                    else decision_bundle.decision.selected_tier.value
                ),
                tier_status=decision_bundle.decision.status.value,
                leakage_observed_max_severity=(
                    None if leakage_decision is None else leakage_decision.observed_max_severity
                ),
                leakage_admissible_tiers=(
                    ()
                    if leakage_decision is None
                    else tuple(tier.value for tier in leakage_decision.admissible_tiers)
                ),
                leakage_rejected_tiers=(
                    ()
                    if leakage_decision is None
                    else tuple(tier.value for tier in leakage_decision.rejected_tiers)
                ),
                report_summary=_tier_report_summary_review(report_bundle),
                metric_rows=_metric_row_records(report.tables.metric_rows),
                tier_rows=_tier_row_records(report.tables.tier_rows),
            )
        )
    return TierDecisionDetailReviewPayload(records=tuple(records))


def build_calibration_surfaces_review_payload(
    bundle: TierExecutionBundle,
) -> CalibrationSurfacesReviewPayload:
    calibration = _tier_calibration_surfaces(bundle)
    return CalibrationSurfacesReviewPayload(
        workflow="tier",
        processed_count=bundle.processed_count,
        membership_counts_by_tier_split=_membership_count_review_rows(bundle),
        family_pass_surfaces=tuple(
            _family_pass_surface_review(row) for row in calibration.family_pass_surfaces
        ),
        binding_metric_distributions=tuple(
            _distribution_summary_review(row)
            for row in calibration.binding_metric_distributions
        ),
        family_waterfalls=_waterfalls_review(calibration.family_waterfalls),
        active_span_derivation=_active_span_derivation_review(
            calibration.active_span_derivation
        ),
        leakage=_leakage_review_summary(bundle),
        tier_report_count=len(bundle.tier_reports),
    )


def build_calibration_detail_review_payload(
    bundle: TierExecutionBundle,
) -> CalibrationDetailReviewPayload:
    calibration = _tier_calibration_surfaces(bundle)
    return CalibrationDetailReviewPayload(
        workflow="tier",
        processed_count=bundle.processed_count,
        filter_config=_filter_review_summary(bundle),
        tier_policies=_policy_review_summary(bundle),
        membership_counts_by_tier_split=_membership_count_review_rows(bundle),
        family_pass_surfaces=tuple(
            _family_pass_surface_review(row) for row in calibration.family_pass_surfaces
        ),
        binding_metric_distributions=tuple(
            _distribution_summary_review(row)
            for row in calibration.binding_metric_distributions
        ),
        family_waterfalls=_waterfalls_review(calibration.family_waterfalls),
        active_span_derivation=_active_span_derivation_review(
            calibration.active_span_derivation
        ),
        leakage=_leakage_review_summary(bundle),
        tier_reports=tuple(_tier_report_summary_review(report) for report in bundle.tier_reports),
        planned_tiered_manifest_outputs=tuple(
            _manifest_output_review(output)
            for output in bundle.workflow_result.output_summary.planned_tiered_manifest_outputs
        ),
    )


def _count_labels(values: Iterable[object]) -> tuple[str, ...]:
    counts = Counter(str(value) for value in values)
    return tuple(f"{key}={count}" for key, count in sorted(counts.items()))


def _tier_included_count_labels(
    rows: tuple[TierMembershipCountRow, ...],
) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[row.tier] += row.included_count
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
    rows: tuple[TierRuntimeAssetRow, ...],
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


def _operation_item(operation: WorkflowOperation) -> WorkflowReviewItem:
    operation_fields: list[tuple[str, RenderableValue]] = [
        ("operation kind", operation_kind(operation)),
        ("label", operation.label),
    ]
    operation_fields.extend(_operation_path_fields(operation))
    if operation.progress is not None:
        operation_fields.append(("progress stage id", operation.progress.stage.stage_id))
    return review_item(str(operation.label), operation_fields)


def _operation_path_fields(operation: WorkflowOperation) -> tuple[tuple[str, RenderableValue], ...]:
    if isinstance(operation, FileCopyOperation):
        fields: list[tuple[str, RenderableValue]] = [
            ("source_path", operation.source_path),
            ("target_path", operation.target_path),
        ]
        if operation.expected_input_bytes is not None:
            fields.append(("expected_input_bytes", operation.expected_input_bytes))
        return tuple(fields)
    if isinstance(operation, ArchiveExtractOperation):
        fields = [
            ("archive_path", operation.archive_path),
            ("extraction_root", operation.extraction_root),
        ]
        if operation.expected_input_bytes is not None:
            fields.append(("expected_input_bytes", operation.expected_input_bytes))
        return tuple(fields)
    if isinstance(operation, ArchiveCreateOperation | ArchiveVerifyOperation):
        return (("archive_path", operation.archive_path),)
    return ()


def _execution_result_item(result: OperationExecutionResult) -> WorkflowReviewItem:
    result_fields: list[tuple[str, RenderableValue]] = [
        ("operation_kind", result.operation_kind),
        ("succeeded", result.succeeded),
        ("returncode", result.returncode),
        ("execution_mode", result.execution_mode),
    ]
    if result.observed_outputs:
        result_fields.append(("observed_outputs", result.observed_outputs))
    return review_item(str(result.label), result_fields)


def _runtime_asset_rows(
    verification: TierRuntimeVerification,
    scope: str,
) -> tuple[TierRuntimeAssetRow, ...]:
    return tuple(
        TierRuntimeAssetRow(
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


def _runtime_asset_item(row: TierRuntimeAssetRow) -> WorkflowReviewItem:
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


def _membership_count_rows(
    bundle: TierExecutionBundle,
) -> tuple[TierMembershipCountRow, ...]:
    rows: list[TierMembershipCountRow] = []
    for tier in TierName:
        for split in bundle.workflow_result.config.splits:
            included = 0
            excluded = 0
            for decision_bundle in bundle.decision_bundles:
                if decision_bundle.manifest.split.value != split:
                    continue
                if (
                    tier_membership_for_decision(decision_bundle.decision, tier)
                    is TierMembership.INCLUDED
                ):
                    included += 1
                else:
                    excluded += 1
            rows.append(
                TierMembershipCountRow(
                    tier=tier.value,
                    split=split,
                    included_count=included,
                    excluded_count=excluded,
                )
            )
    return tuple(rows)


def _membership_count_item(row: TierMembershipCountRow) -> WorkflowReviewItem:
    return review_item(
        f"{row.tier} [{row.split}]",
        (
            ("tier", row.tier),
            ("split", row.split),
            ("included_count", row.included_count),
            ("excluded_count", row.excluded_count),
        ),
    )


def _membership_count_review_rows(
    bundle: TierExecutionBundle,
) -> tuple[MembershipCountReview, ...]:
    return tuple(
        MembershipCountReview(
            tier=row.tier,
            split=row.split,
            included_count=row.included_count,
            excluded_count=row.excluded_count,
        )
        for row in _membership_count_rows(bundle)
    )


def _tier_calibration_surfaces(bundle: TierExecutionBundle) -> TierCalibrationSurfaces:
    return build_tier_calibration_surfaces(
        quality_metrics=tuple(quality_bundle.metrics for quality_bundle in bundle.quality_bundles),
        quality_contexts=tuple(quality_bundle.context for quality_bundle in bundle.quality_bundles),
        tier_decisions=tuple(
            decision_bundle.decision for decision_bundle in bundle.decision_bundles
        ),
    )


def _family_pass_surface_review(row: TierFamilyPassSurface) -> FamilyPassSurfaceReview:
    return FamilyPassSurfaceReview(
        family=row.family.value,
        loose_pass_count=row.loose_pass_count,
        loose_pass_ratio=row.loose_pass_ratio,
        clean_pass_count=row.clean_pass_count,
        clean_pass_ratio=row.clean_pass_ratio,
        tight_pass_count=row.tight_pass_count,
        tight_pass_ratio=row.tight_pass_ratio,
    )


def _distribution_summary_review(
    row: TierMetricDistributionSummary,
) -> DistributionSummaryReview:
    return DistributionSummaryReview(
        family=row.family,
        metric_name=row.metric_name,
        minimum=row.minimum,
        p05=row.p05,
        p10=row.p10,
        p25=row.p25,
        p50=row.p50,
        p75=row.p75,
        p90=row.p90,
        p95=row.p95,
        maximum=row.maximum,
    )


def _waterfall_step_review(row: TierFamilyWaterfallStep) -> FamilyWaterfallStepReview:
    return FamilyWaterfallStepReview(
        tier=row.tier.value,
        family=row.family.value,
        remaining_before_count=row.remaining_before_count,
        remaining_after_count=row.remaining_after_count,
        dropped_count=row.dropped_count,
    )


def _waterfalls_review(row: TierFamilyWaterfalls) -> FamilyWaterfallsReview:
    return FamilyWaterfallsReview(
        loose=tuple(_waterfall_step_review(step) for step in row.loose),
        clean=tuple(_waterfall_step_review(step) for step in row.clean),
        tight=tuple(_waterfall_step_review(step) for step in row.tight),
    )


def _active_span_derivation_review(
    row: TierActiveSpanDerivationSummary,
) -> ActiveSpanDerivationReview:
    return ActiveSpanDerivationReview(
        count_distributions=tuple(
            _distribution_summary_review(distribution)
            for distribution in row.count_distributions
        ),
        fallback_used_count=row.fallback_used_count,
        fallback_used_ratio=row.fallback_used_ratio,
    )


def _family_pass_surface_item(row: FamilyPassSurfaceReview) -> WorkflowReviewItem:
    return review_item(
        row.family,
        (
            ("loose_pass_count", row.loose_pass_count),
            ("loose_pass_ratio", row.loose_pass_ratio),
            ("clean_pass_count", row.clean_pass_count),
            ("clean_pass_ratio", row.clean_pass_ratio),
            ("tight_pass_count", row.tight_pass_count),
            ("tight_pass_ratio", row.tight_pass_ratio),
        ),
    )


def _distribution_summary_item(row: DistributionSummaryReview) -> WorkflowReviewItem:
    return review_item(
        f"{row.family}.{row.metric_name}",
        (
            ("min", row.minimum),
            ("p05", row.p05),
            ("p10", row.p10),
            ("p25", row.p25),
            ("p50", row.p50),
            ("p75", row.p75),
            ("p90", row.p90),
            ("p95", row.p95),
            ("max", row.maximum),
        ),
    )


def _waterfall_step_item(row: FamilyWaterfallStepReview) -> WorkflowReviewItem:
    return review_item(
        f"{row.tier}.{row.family}",
        (
            ("tier", row.tier),
            ("family", row.family),
            ("remaining_sample_count_before_family", row.remaining_before_count),
            ("remaining_sample_count_after_family", row.remaining_after_count),
            ("dropped_sample_count", row.dropped_count),
        ),
    )


def _tiered_manifest_rows(
    result: TierWorkflowResult,
) -> tuple[TieredManifestRow, ...]:
    return tuple(
        TieredManifestRow(
            tier=output.tier,
            membership=output.membership,
            split=output.split,
            path=output.path,
        )
        for output in result.output_summary.planned_tiered_manifest_outputs
    )


def _tiered_manifest_item(row: TieredManifestRow) -> WorkflowReviewItem:
    return review_item(
        f"{row.tier}/{row.membership} [{row.split}]",
        (
            ("tier", row.tier),
            ("membership", row.membership),
            ("split", row.split),
            ("path", row.path),
        ),
    )


def _planned_report_output_rows(
    result: TierWorkflowResult,
) -> tuple[TierPlannedReportOutputRow, ...]:
    artifacts = result.output_summary.planned_report_outputs
    return (
        TierPlannedReportOutputRow("summary markdown", artifacts.summary_markdown_path),
        TierPlannedReportOutputRow("calibration markdown", artifacts.calibration_markdown_path),
        TierPlannedReportOutputRow("decision detail JSONL", artifacts.decision_detail_jsonl_path),
        TierPlannedReportOutputRow(
            "calibration surfaces JSON",
            artifacts.calibration_surfaces_json_path,
        ),
        TierPlannedReportOutputRow(
            "calibration detail JSON",
            artifacts.calibration_detail_json_path,
        ),
        TierPlannedReportOutputRow("report index JSON", artifacts.index_json_path),
    )


def _planned_report_output_item(row: TierReportArtifactRow) -> WorkflowReviewItem:
    return review_item(row.label, (("path", row.path),))


def _written_report_artifact_rows(
    artifacts: TierWrittenReportArtifacts,
) -> tuple[TierWrittenReportArtifactRow, ...]:
    return (
        TierWrittenReportArtifactRow(
            "summary markdown",
            artifacts.summary_markdown_path,
            artifacts.summary_markdown.sha256,
            artifacts.summary_markdown.execution_id,
        ),
        TierWrittenReportArtifactRow(
            "calibration markdown",
            artifacts.calibration_markdown_path,
            artifacts.calibration_markdown.sha256,
            artifacts.calibration_markdown.execution_id,
        ),
        TierWrittenReportArtifactRow(
            "decision detail JSONL",
            artifacts.decision_detail_jsonl_path,
            artifacts.decision_detail_jsonl.sha256,
            artifacts.decision_detail_jsonl.execution_id,
        ),
        TierWrittenReportArtifactRow(
            "calibration surfaces JSON",
            artifacts.calibration_surfaces_json_path,
            artifacts.calibration_surfaces_json.sha256,
            artifacts.calibration_surfaces_json.execution_id,
        ),
        TierWrittenReportArtifactRow(
            "calibration detail JSON",
            artifacts.calibration_detail_json_path,
            artifacts.calibration_detail_json.sha256,
            artifacts.calibration_detail_json.execution_id,
        ),
        TierWrittenReportArtifactRow(
            "report index JSON",
            artifacts.index_json_path,
            artifacts.index_json.sha256,
            artifacts.index_json.execution_id,
        ),
    )


def _written_report_artifact_item(row: TierWrittenReportArtifactRow) -> WorkflowReviewItem:
    return review_item(
        row.label,
        (
            ("path", row.path),
            ("sha256", row.sha256),
            ("execution_id", row.execution_id),
        ),
    )


def _tier_report_summary_item(report_bundle: TierReportResult) -> WorkflowReviewItem:
    summary = _tier_report_summary_review(report_bundle)
    return review_item(
        summary.label,
        summary.items(),
    )


def _tier_report_summary_review(report_bundle: TierReportResult) -> TierReportSummaryReview:
    summary = report_bundle.report.summary
    return TierReportSummaryReview(
        split=summary.split.value,
        sample_id=summary.sample_id,
        frame_count=summary.frame_count,
        duration_seconds=summary.duration_seconds,
        checkpoint_admission_status=summary.checkpoint_admission_status,
        gate_detail_status=summary.gate_detail_status,
        tier_status=summary.tier_status.value,
        selected_tier=None if summary.selected_tier is None else summary.selected_tier.value,
        max_leakage_severity=summary.max_leakage_severity.value,
    )


def _leakage_review_summary(bundle: TierExecutionBundle) -> LeakageReviewSummary:
    severity_counts: dict[str, int] = {}
    affected_sample_count = 0
    for summary in bundle.leakage_bundle.sample_summaries:
        severity = summary.max_severity.value
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        if summary.has_leakage:
            affected_sample_count += 1
    return LeakageReviewSummary(
        sample_summary_count=len(bundle.leakage_bundle.sample_summaries),
        pair_fact_count=len(bundle.leakage_bundle.pair_facts),
        affected_sample_count=affected_sample_count,
        max_severity_counts=tuple(
            NamedCountReview(name=name, count=count)
            for name, count in sorted(severity_counts.items())
        ),
    )


def _filter_review_summary(bundle: TierExecutionBundle) -> TierFilterReviewSummary:
    filters = bundle.filter_config
    family_sections = (
        ("oob", filters.oob),
        ("upper_body_support", filters.upper_body_support),
        ("manual_visibility", filters.manual_visibility),
        ("non_manual_visibility", filters.non_manual_visibility),
        ("confidence", filters.confidence),
        ("kinematic_naturalness", filters.kinematic_naturalness),
        ("tracking_quality", filters.tracking_quality),
        ("manual_detail", filters.manual_detail),
        ("non_manual_quality", filters.non_manual_quality),
        ("geometry", filters.geometry),
    )
    return TierFilterReviewSummary(
        family_count=len(family_sections),
        families=tuple(
            TierFilterFamilyReview(
                family=family,
                threshold_level_count=len(section.thresholds_by_level),
                levels=tuple(tier.value for tier in section.thresholds_by_level),
            )
            for family, section in family_sections
        ),
    )


def _policy_review_summary(bundle: TierExecutionBundle) -> TierPolicyReviewSummary:
    policies = bundle.tier_policies
    return TierPolicyReviewSummary(
        tier_order=tuple(tier.value for tier in policies.tier_order),
        policy_count=len(policies.policies_by_tier),
        policies=tuple(
            TierPolicyReview(
                tier=tier.value,
                max_allowed_leakage_severity=policy.max_allowed_leakage_severity.value,
                family_filter_levels=tuple(
                    FamilyFilterLevelReview(
                        family=family.value,
                        level=policy.family_filter_levels[family].value,
                    )
                    for family in BindingQualityFamily
                ),
            )
            for tier, policy in sorted(
                policies.policies_by_tier.items(),
                key=lambda item: item[0].value,
            )
        ),
    )


def _metric_row_records(rows: tuple[MetricReportRow, ...]) -> tuple[MetricRowReview, ...]:
    return tuple(
        MetricRowReview(
            family=row.family,
            metric_name=row.metric_name,
            value=row.value,
        )
        for row in rows
    )


def _tier_row_records(rows: tuple[TierReportRow, ...]) -> tuple[TierRowReview, ...]:
    return tuple(
        TierRowReview(
            family=row.family.value,
            status=row.status.value,
            best_supported_tier=(
                None if row.best_supported_tier is None else row.best_supported_tier.value
            ),
            supported_tiers=tuple(tier.value for tier in row.supported_tiers),
            issue_count=row.issue_count,
        )
        for row in rows
    )


def _manifest_output_review(output: TieredManifestOutputPlan) -> TierManifestOutputReview:
    return TierManifestOutputReview(
        tier=output.tier,
        membership=output.membership,
        split=output.split,
        path=output.path,
    )
