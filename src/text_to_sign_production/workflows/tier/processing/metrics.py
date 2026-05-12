from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.tier.quality import (
    TierQualityComputationError,
    compute_tier_quality,
)
from text_to_sign_production.workflows.tier.constants import (
    TIER_STAGE_METRIC_COMPUTE,
    TIER_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tier.contracts import TierWorkflowInvariantError
from text_to_sign_production.workflows.tier.processing.models import (
    TierCatalogBundle,
    TierQualityBundle,
    TierSampleBundle,
)


def build_tier_quality_bundles(
    catalog_bundle: TierCatalogBundle,
    *,
    progress_session: ProgressSession | None = None,
) -> tuple[TierQualityBundle, ...]:
    quality_bundles: list[TierQualityBundle] = []
    if progress_session is not None and catalog_bundle.processed_count > 0:
        with progress_session.task(
            _metric_compute_progress_spec(),
            total=catalog_bundle.processed_count,
        ) as progress_task:
            for sample_bundle in catalog_bundle.samples:
                quality_bundles.append(_build_quality_bundle(sample_bundle))
                progress_task.advance()
    else:
        quality_bundles = [
            _build_quality_bundle(sample_bundle) for sample_bundle in catalog_bundle.samples
        ]
    return tuple(quality_bundles)


def _build_quality_bundle(sample_bundle: TierSampleBundle) -> TierQualityBundle:
    try:
        quality = compute_tier_quality(sample_bundle.sample)
    except TierQualityComputationError as exc:
        raise TierWorkflowInvariantError(
            "Quality computation failed for "
            f"{sample_bundle.manifest.split.value}/{sample_bundle.manifest.sample_id}: "
            f"{exc}"
        ) from exc
    return TierQualityBundle(
        sample=sample_bundle.sample,
        manifest=sample_bundle.manifest,
        source_manifest_path=sample_bundle.source_manifest_path,
        source_manifest_sha256=sample_bundle.source_manifest_sha256,
        facts=quality.facts,
        context=quality.context,
        metrics=quality.metrics,
    )


def _metric_compute_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=TIER_STAGE_METRIC_COMPUTE,
        label="quality continuation",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="quality_continuation",
        total_semantics="PreparedSample payloads with facts, context, and metric bundles",
        bar_eligible=True,
    )
