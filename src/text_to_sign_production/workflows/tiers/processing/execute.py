from __future__ import annotations

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
    build_tiers_decision_bundle,
)
from text_to_sign_production.workflows.tiers.processing.leakages import (
    build_tiers_leakage_bundle,
)
from text_to_sign_production.workflows.tiers.processing.manifests import (
    write_tiered_manifests,
)
from text_to_sign_production.workflows.tiers.processing.metrics import (
    build_tiers_metric_bundles,
)
from text_to_sign_production.workflows.tiers.processing.models import TiersExecutionBundle


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
    metric_bundles = build_tiers_metric_bundles(
        catalog_bundle,
        progress_session=progress_session,
    )
    leakage_bundle = build_tiers_leakage_bundle(
        catalog_bundle.manifests,
        metric_bundles,
        progress_session=progress_session,
    )
    filter_config, tier_policies, tier_bundle = build_tiers_decision_bundle(
        layout=layout,
        manifests=catalog_bundle.manifests,
        metric_bundles=metric_bundles,
        leakage_bundle=leakage_bundle,
        progress_session=progress_session,
    )
    tiered_manifest_outputs = write_tiered_manifests(
        layout=layout,
        manifests=catalog_bundle.manifests,
        tier_bundle=tier_bundle,
        progress_session=progress_session,
    )
    workflow_result = TiersWorkflowResult(
        config=config,
        execution_inputs=execution_inputs,
        runtime_verification=runtime_verification,
        output_summary=TiersWorkflowOutputSummary(
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
        metric_bundles=metric_bundles,
        leakage_bundle=leakage_bundle,
        tier_bundle=tier_bundle,
    )


def _visible_progress_session(
    progress_session: ProgressSession | None,
) -> ProgressSession:
    if progress_session is not None:
        return progress_session
    return ProgressSession(
        workflow_id=TIERS_WORKFLOW_NAME,
        sink=TqdmProgressSink(),
    )
