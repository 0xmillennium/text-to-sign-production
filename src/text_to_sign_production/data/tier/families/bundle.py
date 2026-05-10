"""Bundle organization helpers for quality-family metrics."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    DiagnosticQualityFamily,
    QualityMetricBundle,
)


def binding_family_names() -> tuple[BindingQualityFamily, ...]:
    """Return binding families in stable bundle order."""
    return tuple(BindingQualityFamily)


def diagnostic_family_names() -> tuple[DiagnosticQualityFamily, ...]:
    """Return diagnostic families in stable bundle order."""
    return tuple(DiagnosticQualityFamily)


def bundle_metric_groups(bundle: QualityMetricBundle) -> dict[str, object]:
    """Return first-class binding and diagnostic metric groups."""
    return {
        "binding": {
            BindingQualityFamily.OOB: bundle.oob,
            BindingQualityFamily.UPPER_BODY_SUPPORT: bundle.upper_body_support,
            BindingQualityFamily.MANUAL_VISIBILITY: bundle.manual_visibility,
            BindingQualityFamily.NON_MANUAL_VISIBILITY: bundle.non_manual_visibility,
            BindingQualityFamily.CONFIDENCE: bundle.confidence,
            BindingQualityFamily.KINEMATIC_NATURALNESS: bundle.kinematic_naturalness,
            BindingQualityFamily.TRACKING_QUALITY: bundle.tracking_quality,
            BindingQualityFamily.MANUAL_DETAIL: bundle.manual_detail,
            BindingQualityFamily.NON_MANUAL_QUALITY: bundle.non_manual_quality,
            BindingQualityFamily.GEOMETRY: bundle.geometry,
        },
        "diagnostic": {
            DiagnosticQualityFamily.TEXT: bundle.text,
            DiagnosticQualityFamily.LENGTH: bundle.length,
        },
    }


__all__ = ["binding_family_names", "bundle_metric_groups", "diagnostic_family_names"]
