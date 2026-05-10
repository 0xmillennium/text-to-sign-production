"""Thin orchestration for PreparedSample-derived quality-family metrics."""

from __future__ import annotations

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts
from text_to_sign_production.data.tier.families.confidence import compute_confidence_metrics
from text_to_sign_production.data.tier.families.geometry import compute_geometry_metrics
from text_to_sign_production.data.tier.families.kinematic_naturalness import (
    compute_kinematic_naturalness_metrics,
)
from text_to_sign_production.data.tier.families.length import (
    compute_length_diagnostic_metrics,
)
from text_to_sign_production.data.tier.families.manual_detail import (
    compute_manual_detail_metrics,
)
from text_to_sign_production.data.tier.families.manual_visibility import (
    compute_manual_visibility_metrics,
)
from text_to_sign_production.data.tier.families.non_manual_quality import (
    compute_non_manual_quality_metrics,
)
from text_to_sign_production.data.tier.families.non_manual_visibility import (
    compute_non_manual_visibility_metrics,
)
from text_to_sign_production.data.tier.families.oob import compute_oob_metrics
from text_to_sign_production.data.tier.families.text import compute_text_diagnostic_metrics
from text_to_sign_production.data.tier.families.tracking_quality import (
    compute_tracking_quality_metrics,
)
from text_to_sign_production.data.tier.families.types import (
    QualityMetricBuildInput,
    QualityMetricBundle,
)
from text_to_sign_production.data.tier.families.upper_body_support import (
    compute_upper_body_support_metrics,
)


def build_quality_metric_bundle(
    sample: PreparedSample,
    facts: QualityFacts,
    context: QualityContext,
) -> QualityMetricBundle:
    """Build composed family metrics from PreparedSample, facts, and context."""
    input = QualityMetricBuildInput(
        sample=sample,
        quality_facts=facts,
        quality_context=context,
    )
    return QualityMetricBundle(
        oob=compute_oob_metrics(input),
        upper_body_support=compute_upper_body_support_metrics(input),
        manual_visibility=compute_manual_visibility_metrics(input),
        non_manual_visibility=compute_non_manual_visibility_metrics(input),
        confidence=compute_confidence_metrics(input),
        kinematic_naturalness=compute_kinematic_naturalness_metrics(input),
        tracking_quality=compute_tracking_quality_metrics(input),
        manual_detail=compute_manual_detail_metrics(input),
        non_manual_quality=compute_non_manual_quality_metrics(input),
        geometry=compute_geometry_metrics(input),
        text=compute_text_diagnostic_metrics(input),
        length=compute_length_diagnostic_metrics(input),
    )


__all__ = ["build_quality_metric_bundle"]
