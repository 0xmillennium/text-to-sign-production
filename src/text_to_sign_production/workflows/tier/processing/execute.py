from __future__ import annotations

from uuid import uuid4

from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.core.progress import ProgressSession, TqdmProgressSink
from text_to_sign_production.data.tier.leakages import LeakageBundle, sample_leakage_summary
from text_to_sign_production.data.tier.reports import build_tier_report
from text_to_sign_production.workflows.tier.constants import TIER_WORKFLOW_NAME
from text_to_sign_production.workflows.tier.contracts import (
    TieredManifestOutputPlan,
    TierPlannedReportOutputs,
    TierRuntimeVerification,
    TierWorkflowConfig,
    TierWorkflowExecutionInputs,
    TierWorkflowInvariantError,
    TierWorkflowOutputSummary,
    TierWorkflowResult,
)
from text_to_sign_production.workflows.tier.layout import TierLayout
from text_to_sign_production.workflows.tier.processing.catalog import (
    load_tier_catalog_bundle,
)
from text_to_sign_production.workflows.tier.processing.decisions import (
    build_tier_decision_bundles,
)
from text_to_sign_production.workflows.tier.processing.leakages import (
    build_tier_leakage_bundle,
)
from text_to_sign_production.workflows.tier.processing.manifests import (
    write_tiered_manifests,
)
from text_to_sign_production.workflows.tier.processing.metrics import (
    build_tier_quality_bundles,
)
from text_to_sign_production.workflows.tier.processing.models import (
    TierDecisionResult,
    TierExecutionBundle,
    TierQualityBundle,
    TierReportResult,
)


def execute_tier_processing(
    *,
    config: TierWorkflowConfig,
    layout: TierLayout,
    execution_inputs: TierWorkflowExecutionInputs,
    runtime_verification: TierRuntimeVerification,
    progress_session: ProgressSession | None = None,
) -> TierExecutionBundle:
    if not runtime_verification.succeeded:
        raise TierWorkflowInvariantError("Runtime verification must succeed before processing")
    if layout.config != config:
        raise TierWorkflowInvariantError("layout.config must match the provided config")

    progress_session = _visible_progress_session(progress_session)
    execution_id = uuid4().hex
    catalog_bundle = load_tier_catalog_bundle(
        layout=layout,
        execution_inputs=execution_inputs,
        splits=config.splits,
        progress_session=progress_session,
    )
    quality_bundles = build_tier_quality_bundles(
        catalog_bundle,
        progress_session=progress_session,
    )
    leakage_bundle = build_tier_leakage_bundle(catalog_bundle)
    filter_config, tier_policies, decision_bundles = build_tier_decision_bundles(
        layout=layout,
        quality_bundles=quality_bundles,
        leakage_bundle=leakage_bundle,
        progress_session=progress_session,
    )
    tier_reports = _build_tier_reports(
        quality_bundles=quality_bundles,
        decision_bundles=decision_bundles,
        leakage_bundle=leakage_bundle,
    )
    written_tiered_manifest_artifacts = write_tiered_manifests(
        layout=layout,
        decision_bundles=decision_bundles,
        execution_id=execution_id,
        progress_session=progress_session,
    )
    workflow_result = TierWorkflowResult(
        execution_id=execution_id,
        config=config,
        execution_inputs=execution_inputs,
        runtime_verification=runtime_verification,
        output_summary=TierWorkflowOutputSummary(
            loaded_passed_sample_count=catalog_bundle.processed_count,
            quality_fact_count=len(quality_bundles),
            quality_context_count=len(quality_bundles),
            quality_metric_bundle_count=len(quality_bundles),
            leakage_sample_summary_count=len(leakage_bundle.sample_summaries),
            tier_decision_count=len(decision_bundles),
            tier_report_count=len(tier_reports),
            planned_tiered_manifest_outputs=_planned_tiered_manifest_outputs(layout),
            planned_report_outputs=TierPlannedReportOutputs(
                summary_markdown_path=layout.reports.summary_markdown_path,
                calibration_markdown_path=layout.reports.calibration_markdown_path,
                decision_detail_json_path=layout.reports.decision_detail_json_path,
                calibration_surfaces_json_path=layout.reports.calibration_surfaces_json_path,
                calibration_detail_json_path=layout.reports.calibration_detail_json_path,
                index_json_path=layout.reports.index_json_path,
            ),
        ),
    )
    return TierExecutionBundle(
        workflow_result=workflow_result,
        catalog_bundle=catalog_bundle,
        filter_config=filter_config,
        tier_policies=tier_policies,
        quality_bundles=quality_bundles,
        leakage_bundle=leakage_bundle,
        decision_bundles=decision_bundles,
        written_tiered_manifest_artifacts=written_tiered_manifest_artifacts,
        tier_reports=tier_reports,
    )


def _planned_tiered_manifest_outputs(layout: TierLayout) -> tuple[TieredManifestOutputPlan, ...]:
    outputs: list[TieredManifestOutputPlan] = []
    for tier in TierName:
        for membership in TierMembership:
            for split in tuple(SampleSplit(split) for split in layout.config.splits):
                outputs.append(
                    TieredManifestOutputPlan(
                        tier=tier.value,
                        membership=membership.value,
                        split=split.value,
                        path=layout.stores.runtime.manifests.tiered_manifest(
                            tier,
                            membership,
                            split,
                        ).path,
                    )
                )
    return tuple(outputs)


def _build_tier_reports(
    *,
    quality_bundles: tuple[TierQualityBundle, ...],
    decision_bundles: tuple[TierDecisionResult, ...],
    leakage_bundle: LeakageBundle,
) -> tuple[TierReportResult, ...]:
    if len(quality_bundles) != len(decision_bundles):
        raise TierWorkflowInvariantError("quality bundle and tier decision counts are misaligned")
    reports: list[TierReportResult] = []
    for quality_bundle, decision_bundle in zip(
        quality_bundles,
        decision_bundles,
        strict=True,
    ):
        if (
            quality_bundle.manifest.sample_id != decision_bundle.manifest.sample_id
            or quality_bundle.manifest.split is not decision_bundle.manifest.split
        ):
            raise TierWorkflowInvariantError("quality bundle and tier decision order is misaligned")
        reports.append(
            TierReportResult(
                sample=quality_bundle.sample,
                manifest=quality_bundle.manifest,
                report=build_tier_report(
                    quality_bundle.sample,
                    quality_bundle.facts,
                    quality_bundle.metrics,
                    sample_leakage_summary(
                        leakage_bundle,
                        split=quality_bundle.manifest.split,
                        sample_id=quality_bundle.manifest.sample_id,
                    ),
                    decision_bundle.decision,
                    decision_bundle.checkpoint_admission,
                ),
            )
        )
    return tuple(reports)


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=TIER_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )
