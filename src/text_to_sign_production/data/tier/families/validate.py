"""Validation for quality-family metric objects."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, fields, is_dataclass

from text_to_sign_production.data.tier.families.types import QualityMetricBundle


class FamilyValidationIssueCode(enum.StrEnum):
    """Controlled validation issue codes for family metrics."""

    NON_FINITE_METRIC = "non_finite_metric"
    RATIO_OUT_OF_RANGE = "ratio_out_of_range"
    NEGATIVE_COUNT = "negative_count"


@dataclass(frozen=True, slots=True)
class FamilyValidationIssue:
    """Structured family-metric validation issue."""

    code: FamilyValidationIssueCode
    message: str
    field_path: str


def validate_quality_metric_bundle(
    bundle: QualityMetricBundle,
) -> tuple[FamilyValidationIssue, ...]:
    """Validate metric-domain field structure without applying policy decisions."""
    issues: list[FamilyValidationIssue] = []
    for family_field in fields(bundle):
        family_value = getattr(bundle, family_field.name)
        if not is_dataclass(family_value):
            continue
        for metric_field in fields(family_value):
            value = getattr(family_value, metric_field.name)
            path = f"{family_field.name}.{metric_field.name}"
            issues.extend(_validate_metric_value(path, value))
    return tuple(issues)


def _validate_metric_value(path: str, value: object) -> tuple[FamilyValidationIssue, ...]:
    issues: list[FamilyValidationIssue] = []
    if value is None:
        return ()
    if isinstance(value, int):
        if value < 0:
            issues.append(
                FamilyValidationIssue(
                    code=FamilyValidationIssueCode.NEGATIVE_COUNT,
                    message="Count metric cannot be negative.",
                    field_path=path,
                )
            )
        return tuple(issues)
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append(
                FamilyValidationIssue(
                    code=FamilyValidationIssueCode.NON_FINITE_METRIC,
                    message="Metric must be finite.",
                    field_path=path,
                )
            )
            return tuple(issues)
        if path.endswith("_ratio") and not 0.0 <= value <= 1.0:
            issues.append(
                FamilyValidationIssue(
                    code=FamilyValidationIssueCode.RATIO_OUT_OF_RANGE,
                    message="Ratio metric must be within [0, 1].",
                    field_path=path,
                )
            )
    return tuple(issues)


__all__ = [
    "FamilyValidationIssue",
    "FamilyValidationIssueCode",
    "validate_quality_metric_bundle",
]
