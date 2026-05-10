from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from text_to_sign_production.core.progress import ProgressSession, TqdmProgressSink
from text_to_sign_production.workflows.tiers.constants import TIERS_WORKFLOW_NAME
from text_to_sign_production.workflows.tiers.contracts import (
    TiersReportArtifacts,
    TiersRuntimeVerification,
    TiersWorkflowConfig,
    TiersWorkflowExecutionInputs,
    TiersWorkflowInvariantError,
    TiersWorkflowOutputSummary,
    TiersWorkflowResult,
)
from text_to_sign_production.workflows.tiers.layout import TiersLayout
from text_to_sign_production.workflows.tiers.processing.catalog import (
    load_tiers_catalog_bundle,
)
from text_to_sign_production.workflows.tiers.processing.decisions import (
    build_tiers_decision_bundles,
)
from text_to_sign_production.workflows.tiers.processing.leakages import (
    build_tiers_leakage_bundle,
)
from text_to_sign_production.workflows.tiers.processing.manifests import (
    write_tiered_manifests,
)
from text_to_sign_production.workflows.tiers.processing.metrics import (
    build_tiers_quality_bundles,
)
from text_to_sign_production.workflows.tiers.processing.models import (
    TiersDecisionBundle,
    TiersExecutionBundle,
    TiersQualityBundle,
    TiersReportBundle,
)


def execute_tiers_processing(
    *,
    config: TiersWorkflowConfig,
    layout: TiersLayout,
    execution_inputs: TiersWorkflowExecutionInputs,
    runtime_verification: TiersRuntimeVerification,
    progress_session: ProgressSession | None = None,
) -> TiersExecutionBundle:
    if not runtime_verification.succeeded:
        raise TiersWorkflowInvariantError("Runtime verification must succeed before processing")
    if layout.config != config:
        raise TiersWorkflowInvariantError("layout.config must match the provided config")

    progress_session = _visible_progress_session(progress_session)
    catalog_bundle = load_tiers_catalog_bundle(
        layout=layout,
        execution_inputs=execution_inputs,
        splits=config.splits,
        progress_session=progress_session,
    )
    quality_bundles = build_tiers_quality_bundles(
        catalog_bundle,
        progress_session=progress_session,
    )
    leakage_bundle = build_tiers_leakage_bundle(catalog_bundle)
    filter_config, tier_policies, decision_bundles = build_tiers_decision_bundles(
        layout=layout,
        quality_bundles=quality_bundles,
        leakage_bundle=leakage_bundle,
        progress_session=progress_session,
    )
    tiered_manifest_outputs = write_tiered_manifests(
        layout=layout,
        decision_bundles=decision_bundles,
        progress_session=progress_session,
    )
    quality_reports = _build_quality_reports(
        quality_bundles=quality_bundles,
        decision_bundles=decision_bundles,
        leakage_bundle=leakage_bundle,
    )
    workflow_result = TiersWorkflowResult(
        config=config,
        execution_inputs=execution_inputs,
        runtime_verification=runtime_verification,
        output_summary=TiersWorkflowOutputSummary(
            loaded_passed_sample_count=catalog_bundle.processed_count,
            quality_fact_count=len(quality_bundles),
            quality_context_count=len(quality_bundles),
            quality_metric_bundle_count=len(quality_bundles),
            leakage_sample_summary_count=len(leakage_bundle.sample_summaries),
            tier_decision_count=len(decision_bundles),
            quality_report_count=len(quality_reports),
            tiered_manifest_outputs=tiered_manifest_outputs,
            report_artifacts=TiersReportArtifacts(
                summary_markdown_path=layout.reports.summary_markdown_path,
                calibration_markdown_path=layout.reports.calibration_markdown_path,
                decision_detail_jsonl_path=layout.reports.decision_detail_jsonl_path,
                calibration_surfaces_json_path=layout.reports.calibration_surfaces_json_path,
                calibration_detail_json_path=layout.reports.calibration_detail_json_path,
                index_json_path=layout.reports.index_json_path,
            ),
        ),
    )
    return TiersExecutionBundle(
        workflow_result=workflow_result,
        catalog_bundle=catalog_bundle,
        filter_config=filter_config,
        tier_policies=tier_policies,
        quality_bundles=quality_bundles,
        leakage_bundle=leakage_bundle,
        decision_bundles=decision_bundles,
        tiered_manifest_outputs=tiered_manifest_outputs,
        quality_reports=quality_reports,
    )


def _build_quality_reports(
    *,
    quality_bundles: tuple[TiersQualityBundle, ...],
    decision_bundles: tuple[TiersDecisionBundle, ...],
    leakage_bundle: object,
) -> tuple[TiersReportBundle, ...]:
    if len(quality_bundles) != len(decision_bundles):
        raise TiersWorkflowInvariantError("quality bundle and tier decision counts are misaligned")
    build_quality_report = _load_quality_report_builder()
    reports: list[TiersReportBundle] = []
    for quality_bundle, decision_bundle in zip(
        quality_bundles,
        decision_bundles,
        strict=True,
    ):
        if (
            quality_bundle.manifest.sample_id != decision_bundle.manifest.sample_id
            or quality_bundle.manifest.split is not decision_bundle.manifest.split
        ):
            raise TiersWorkflowInvariantError(
                "quality bundle and tier decision order is misaligned"
            )
        reports.append(
            TiersReportBundle(
                sample=quality_bundle.sample,
                manifest=quality_bundle.manifest,
                report=build_quality_report(
                    quality_bundle.sample,
                    quality_bundle.facts,
                    quality_bundle.context,
                    quality_bundle.metrics,
                    leakage_bundle,
                    decision_bundle.decision,
                    None,
                ),
            )
        )
    return tuple(reports)


def _load_quality_report_builder() -> Callable[..., Any]:
    try:
        reports = import_module("text_to_sign_production.data.quality.reports")
    except ModuleNotFoundError as exc:
        raise TiersWorkflowInvariantError(
            "Unable to import data.quality.reports owner package."
        ) from exc
    return cast(Callable[..., Any], reports.build_quality_report)


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=TIERS_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )
