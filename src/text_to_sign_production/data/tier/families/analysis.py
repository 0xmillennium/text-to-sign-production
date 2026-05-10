"""Read-only analysis helpers for quality-family metrics."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

from text_to_sign_production.data.tier.families.bundle import (
    binding_family_names,
    diagnostic_family_names,
)
from text_to_sign_production.data.tier.families.types import (
    BindingQualityFamily,
    DiagnosticQualityFamily,
    QualityMetricBundle,
)


def binding_metric_summary(
    bundle: QualityMetricBundle,
) -> dict[BindingQualityFamily, dict[str, float]]:
    """Return binding-family metric values keyed by family and metric name."""
    return {
        family: _numeric_dataclass_values(getattr(bundle, family.value))
        for family in binding_family_names()
    }


def diagnostic_metric_summary(
    bundle: QualityMetricBundle,
) -> dict[DiagnosticQualityFamily, dict[str, float]]:
    """Return diagnostic-family metric values keyed by family and metric name."""
    return {
        family: _numeric_dataclass_values(getattr(bundle, family.value))
        for family in diagnostic_family_names()
    }


def summarize_quality_metric_bundle(bundle: QualityMetricBundle) -> dict[str, object]:
    """Return a compact, read-only metric bundle inventory."""
    binding = binding_metric_summary(bundle)
    diagnostic = diagnostic_metric_summary(bundle)
    return {
        "binding_family_count": len(binding),
        "diagnostic_family_count": len(diagnostic),
        "binding_metric_count": sum(len(metrics) for metrics in binding.values()),
        "diagnostic_metric_count": sum(len(metrics) for metrics in diagnostic.values()),
        "binding_families": tuple(family.value for family in binding),
        "diagnostic_families": tuple(family.value for family in diagnostic),
    }


def _numeric_dataclass_values(value: object) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in fields(cast(Any, value)):
        metric_value = getattr(value, field.name)
        if metric_value is None:
            continue
        if isinstance(metric_value, int | float):
            metrics[field.name] = float(metric_value)
    return metrics


__all__ = [
    "binding_metric_summary",
    "diagnostic_metric_summary",
    "summarize_quality_metric_bundle",
]
