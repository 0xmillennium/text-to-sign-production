"""Policy-free per-sample metric computation layer."""

from __future__ import annotations

from text_to_sign_production.data.metrics.active_signing_span import (
    compute_active_signing_span_metrics,
)
from text_to_sign_production.data.metrics.compute import build_metric_bundle
from text_to_sign_production.data.metrics.confidence import compute_confidence_metrics
from text_to_sign_production.data.metrics.coverage import compute_coverage_metrics
from text_to_sign_production.data.metrics.face import compute_face_metrics
from text_to_sign_production.data.metrics.hand import compute_hand_metrics
from text_to_sign_production.data.metrics.length import compute_length_metrics
from text_to_sign_production.data.metrics.oob import compute_oob_metrics
from text_to_sign_production.data.metrics.temporal_coherence import (
    compute_temporal_coherence_metrics,
)
from text_to_sign_production.data.metrics.text import compute_text_metrics
from text_to_sign_production.data.metrics.types import (
    ActiveSigningSpanMetrics,
    ConfidenceMetrics,
    CoverageMetrics,
    FaceMetrics,
    HandMetrics,
    LengthMetrics,
    MetricBundle,
    MetricFamily,
    MetricValidationIssue,
    OobMetrics,
    TemporalCoherenceMetrics,
    TextMetrics,
    UpperBodySupportMetrics,
    ValidMetrics,
)
from text_to_sign_production.data.metrics.upper_body_support import (
    UPPER_BODY_SUPPORT_LANDMARK_INDICES,
    UPPER_BODY_SUPPORT_LANDMARKS,
    compute_upper_body_support_metrics,
)
from text_to_sign_production.data.metrics.valid import compute_valid_metrics
from text_to_sign_production.data.metrics.validate import validate_metric_bundle

__all__ = [
    "ActiveSigningSpanMetrics",
    "ConfidenceMetrics",
    "CoverageMetrics",
    "FaceMetrics",
    "HandMetrics",
    "LengthMetrics",
    "MetricBundle",
    "MetricFamily",
    "MetricValidationIssue",
    "OobMetrics",
    "TemporalCoherenceMetrics",
    "TextMetrics",
    "UPPER_BODY_SUPPORT_LANDMARK_INDICES",
    "UPPER_BODY_SUPPORT_LANDMARKS",
    "UpperBodySupportMetrics",
    "ValidMetrics",
    "build_metric_bundle",
    "compute_active_signing_span_metrics",
    "compute_confidence_metrics",
    "compute_coverage_metrics",
    "compute_face_metrics",
    "compute_hand_metrics",
    "compute_length_metrics",
    "compute_oob_metrics",
    "compute_temporal_coherence_metrics",
    "compute_text_metrics",
    "compute_upper_body_support_metrics",
    "compute_valid_metrics",
    "validate_metric_bundle",
]
