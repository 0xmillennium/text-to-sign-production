from __future__ import annotations

from text_to_sign_production.core.ids import SampleSplit
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
        for split, sample_bundles in _samples_by_split(catalog_bundle.samples):
            with progress_session.task(
                _metric_compute_progress_spec(split),
                total=len(sample_bundles),
            ) as progress_task:
                for sample_bundle in sample_bundles:
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


def _samples_by_split(
    sample_bundles: tuple[TierSampleBundle, ...],
) -> tuple[tuple[SampleSplit, tuple[TierSampleBundle, ...]], ...]:
    splits = tuple(dict.fromkeys(sample.manifest.split for sample in sample_bundles))
    return tuple(
        (
            split,
            tuple(sample for sample in sample_bundles if sample.manifest.split is split),
        )
        for split in splits
    )


def _metric_compute_progress_spec(split: SampleSplit) -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIER_WORKFLOW_NAME,
        stage_id=f"{TIER_STAGE_METRIC_COMPUTE}.{split.value}",
        label=f"quality metrics [{split.value}]",
        unit="sample",
        owner_module=__name__,
        split_behavior="per_split",
        operation_kind="quality_continuation",
        total_semantics="PreparedSample payloads with facts, context, and metric bundles",
        bar_eligible=True,
    )
