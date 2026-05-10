"""Quality semantic bundle bridge from checkpoint truth.

This module rebuilds downstream semantic bundles by calling the public
builders of the owning quality bounded contexts. It does NOT copy
facts/context/families logic into the bridge.
"""

from __future__ import annotations

from text_to_sign_production.data.tier._shared.types import (
    QualityPoseTruth,
    QualitySourceTruth,
)
from text_to_sign_production.data.tier.context.build import (
    build_quality_context_from_neutral_truth,
)
from text_to_sign_production.data.tier.context.types import (
    QualityContext,
    QualityContextNeutralBuildInput,
)
from text_to_sign_production.data.tier.facts.build import (
    build_quality_facts_from_neutral_truth,
)
from text_to_sign_production.data.tier.facts.types import (
    QualityFacts,
    QualityFactsBuildInput,
)
from text_to_sign_production.data.tier.families.compute import (
    build_quality_metric_bundle_from_neutral_truth,
)
from text_to_sign_production.data.tier.families.types import (
    QualityMetricBundle,
    QualityMetricNeutralBuildInput,
)


def build_quality_facts_from_checkpoint(
    source_truth: QualitySourceTruth,
    pose_truth: QualityPoseTruth,
) -> QualityFacts:
    """Build quality facts from checkpoint-rehydrated neutral truth.

    Delegates to the facts bounded context's neutral truth builder.
    """
    return build_quality_facts_from_neutral_truth(
        QualityFactsBuildInput(
            source_truth=source_truth,
            pose_truth=pose_truth,
        )
    )


def build_quality_context_from_checkpoint(
    source_truth: QualitySourceTruth,
    pose_truth: QualityPoseTruth,
    facts: QualityFacts,
) -> QualityContext:
    """Build quality context from checkpoint-rehydrated neutral truth.

    Delegates to the context bounded context's neutral truth builder.
    """
    return build_quality_context_from_neutral_truth(
        QualityContextNeutralBuildInput(
            source_truth=source_truth,
            pose_truth=pose_truth,
            quality_facts=facts,
        )
    )


def build_quality_metrics_from_checkpoint(
    source_truth: QualitySourceTruth,
    pose_truth: QualityPoseTruth,
    facts: QualityFacts,
    context: QualityContext,
) -> QualityMetricBundle:
    """Build quality metrics from checkpoint-rehydrated neutral truth.

    Delegates to the families bounded context's neutral truth builder.
    """
    return build_quality_metric_bundle_from_neutral_truth(
        QualityMetricNeutralBuildInput(
            source_truth=source_truth,
            pose_truth=pose_truth,
            quality_facts=facts,
            quality_context=context,
        )
    )


__all__ = [
    "build_quality_context_from_checkpoint",
    "build_quality_facts_from_checkpoint",
    "build_quality_metrics_from_checkpoint",
]
