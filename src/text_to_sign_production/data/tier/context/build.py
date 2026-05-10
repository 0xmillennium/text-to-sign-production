"""Thin orchestration for PreparedSample-derived quality context."""

from __future__ import annotations

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context.face import build_face_region_context
from text_to_sign_production.data.tier.context.geometry import build_geometry_reference_context
from text_to_sign_production.data.tier.context.hands import build_representative_hand_context
from text_to_sign_production.data.tier.context.span import (
    build_active_span_context,
    build_representative_articulator_context,
)
from text_to_sign_production.data.tier.context.types import QualityContext
from text_to_sign_production.data.tier.facts import QualityFacts


def build_quality_context(sample: PreparedSample, facts: QualityFacts) -> QualityContext:
    """Build composed shared derived context from PreparedSample and facts."""
    active_span = build_active_span_context(sample, facts)
    representative_articulator = build_representative_articulator_context(sample, active_span)
    representative_hand = build_representative_hand_context(sample, active_span)
    face = build_face_region_context(sample, active_span, representative_hand)
    geometry = build_geometry_reference_context(sample, representative_hand)
    return QualityContext(
        active_span=active_span,
        representative_articulator=representative_articulator,
        representative_hand=representative_hand,
        face=face,
        geometry=geometry,
    )


__all__ = ["build_quality_context"]
