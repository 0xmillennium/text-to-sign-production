"""Stable public surface for building tier reports."""

from text_to_sign_production.data.tier.reports.build import build_tier_report
from text_to_sign_production.data.tier.reports.calibration import (
    TierActiveSpanDerivationSummary,
    TierCalibrationProgressSpecs,
    TierCalibrationSurfaces,
    TierFamilyPassSurface,
    TierFamilyWaterfallStep,
    TierFamilyWaterfalls,
    TierMetricDistributionSummary,
    build_tier_calibration_surfaces,
)
from text_to_sign_production.data.tier.reports.types import (
    TierReportBundle,
)

__all__ = [
    "TierActiveSpanDerivationSummary",
    "TierCalibrationProgressSpecs",
    "TierCalibrationSurfaces",
    "TierFamilyPassSurface",
    "TierFamilyWaterfallStep",
    "TierFamilyWaterfalls",
    "TierMetricDistributionSummary",
    "TierReportBundle",
    "build_tier_calibration_surfaces",
    "build_tier_report",
]
