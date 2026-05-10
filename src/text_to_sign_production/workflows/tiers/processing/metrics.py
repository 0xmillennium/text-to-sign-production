from __future__ import annotations

from text_to_sign_production.core.progress import ProgressSession, ProgressStageSpec
from text_to_sign_production.data.tier.context import (
    build_quality_context,
    validate_quality_context,
)
from text_to_sign_production.data.tier.facts import (
    build_quality_facts,
    validate_quality_facts_invariants,
)
from text_to_sign_production.data.tier.families import (
    build_quality_metric_bundle,
    validate_quality_metric_bundle,
)
from text_to_sign_production.workflows.tiers.constants import (
    TIERS_STAGE_METRIC_COMPUTE,
    TIERS_WORKFLOW_NAME,
)
from text_to_sign_production.workflows.tiers.contracts import TiersWorkflowInvariantError
from text_to_sign_production.workflows.tiers.processing.models import (
    TiersCatalogBundle,
    TiersQualityBundle,
    TiersSampleBundle,
)


def build_tiers_quality_bundles(
    catalog_bundle: TiersCatalogBundle,
    *,
    progress_session: ProgressSession | None = None,
) -> tuple[TiersQualityBundle, ...]:
    quality_bundles: list[TiersQualityBundle] = []
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


def _build_quality_bundle(sample_bundle: TiersSampleBundle) -> TiersQualityBundle:
    facts = build_quality_facts(sample_bundle.sample)
    fact_issues = validate_quality_facts_invariants(facts)
    if fact_issues:
        raise TiersWorkflowInvariantError(
            "Quality facts validation failed for "
            f"{sample_bundle.manifest.split.value}/{sample_bundle.manifest.sample_id}: "
            f"{fact_issues}"
        )

    context = build_quality_context(sample_bundle.sample, facts)
    context_issues = validate_quality_context(context)
    if context_issues:
        raise TiersWorkflowInvariantError(
            "Quality context validation failed for "
            f"{sample_bundle.manifest.split.value}/{sample_bundle.manifest.sample_id}: "
            f"{context_issues}"
        )

    metrics = build_quality_metric_bundle(sample_bundle.sample, facts, context)
    metric_issues = validate_quality_metric_bundle(metrics)
    if metric_issues:
        raise TiersWorkflowInvariantError(
            "Quality metric bundle validation failed for "
            f"{sample_bundle.manifest.split.value}/{sample_bundle.manifest.sample_id}: "
            f"{metric_issues}"
        )
    return TiersQualityBundle(
        sample=sample_bundle.sample,
        manifest=sample_bundle.manifest,
        facts=facts,
        context=context,
        metrics=metrics,
    )


def _metric_compute_progress_spec() -> ProgressStageSpec:
    return ProgressStageSpec(
        workflow_id=TIERS_WORKFLOW_NAME,
        stage_id=TIERS_STAGE_METRIC_COMPUTE,
        label="quality continuation",
        unit="sample",
        owner_module=__name__,
        split_behavior="global",
        operation_kind="quality_continuation",
        total_semantics="PreparedSample payloads with facts, context, and metric bundles",
        bar_eligible=True,
    )
