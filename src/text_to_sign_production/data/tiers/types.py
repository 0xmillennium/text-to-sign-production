"""Typed models for deterministic tier decisions."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.leakages.types import LeakageSampleRef, LeakageSeverity

SplitScope: TypeAlias = SampleSplit | None


class FilterLevel(enum.StrEnum):
    """Strict threshold levels loaded from filters.yaml."""

    LOOSE = "loose"
    CLEAN = "clean"
    TIGHT = "tight"


class TierName(enum.StrEnum):
    """Fixed tier identities owned by tier policy."""

    LOOSE = "loose"
    CLEAN = "clean"
    TIGHT = "tight"


class BindingTierFamily(enum.StrEnum):
    """Metric families that are binding for tier inclusion decisions."""

    OOB = "oob"
    COVERAGE = "coverage"
    HAND = "hand"
    FACE = "face"
    CONFIDENCE = "confidence"
    TEXT = "text"
    LENGTH = "length"


class DiagnosticMetric(enum.StrEnum):
    """Computed metrics kept visible for diagnostics without tier veto power."""

    VALID_FRAME_RATIO = "valid_frame_ratio"
    INVALID_FRAME_RATIO = "invalid_frame_ratio"
    ZEROED_CANONICAL_JOINT_FRAME_RATIO = "zeroed_canonical_joint_frame_ratio"
    FRAMES_PER_TOKEN = "frames_per_token"
    FRAMES_PER_CHARACTER = "frames_per_character"


class MetricPolicyRole(enum.StrEnum):
    """Whether a calibration metric participates in tier veto policy."""

    BINDING = "binding"
    DIAGNOSTIC = "diagnostic"


class NearThresholdSide(enum.StrEnum):
    """Pass/fail side for nearest-threshold sample records."""

    PASSING = "passing"
    FAILING = "failing"


@dataclass(frozen=True, slots=True)
class MetricDistributionRecord:
    """Split-aware distribution summary for a calibration metric."""

    split: SplitScope
    role: MetricPolicyRole
    family: str
    metric_key: str
    sample_count: int
    missing_count: int
    unique_value_count: int
    minimum: float | int | None
    p5: float | int | None
    p25: float | int | None
    p50: float | int | None
    p75: float | int | None
    p95: float | int | None
    p99: float | int | None
    maximum: float | int | None


@dataclass(frozen=True, slots=True)
class TierMetricPassFailRecord:
    """Pass/fail counts for one binding metric under one tier and split."""

    tier_name: TierName
    split: SampleSplit
    role: MetricPolicyRole
    family: BindingTierFamily
    metric_key: str
    pass_count: int
    fail_count: int


@dataclass(frozen=True, slots=True)
class BlockerFrequencyRecord:
    """Aggregated blocker frequency for tier decisions."""

    tier_name: TierName
    split: SampleSplit
    role: MetricPolicyRole
    family: BindingTierFamily
    metric_key: str
    blocker_count: int


@dataclass(frozen=True, slots=True)
class CoFailureRecord:
    """Family-level co-failure count for excluded tier decisions."""

    tier_name: TierName
    split: SampleSplit
    left_family: BindingTierFamily
    right_family: BindingTierFamily
    decision_count: int


@dataclass(frozen=True, slots=True)
class SaturatedMetricRecord:
    """Heuristic saturated-metric finding for calibration review."""

    split: SplitScope
    tier_name: TierName | None
    role: MetricPolicyRole
    family: str
    metric_key: str
    sample_count: int
    pass_rate: float | None
    fail_rate: float | None
    unique_value_count: int | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NearThresholdSampleRecord:
    """Sample nearest to a binding threshold under one tier."""

    tier_name: TierName
    split: SampleSplit
    role: MetricPolicyRole
    family: BindingTierFamily
    metric_key: str
    sample_id: str
    side: NearThresholdSide
    actual_value: float | int | None
    expected_value: float | int
    comparison: str
    threshold_distance: float


@dataclass(frozen=True, slots=True)
class TierDeltaRecord:
    """Samples included in a looser tier but excluded by the next stricter tier."""

    split: SampleSplit
    from_tier: TierName
    to_tier: TierName
    sample_id: str


@dataclass(frozen=True, slots=True)
class TierDeltaSummaryRecord:
    """Split-level count of samples lost between adjacent tiers."""

    split: SampleSplit
    from_tier: TierName
    to_tier: TierName
    sample_count: int


@dataclass(frozen=True, slots=True)
class CoverageFamilySummaryRecord:
    """Dedicated coverage-family summary by split and channel dimension."""

    split: SplitScope
    metric_key: str
    sample_count: int
    minimum: float | int | None
    p50: float | int | None
    p95: float | int | None
    maximum: float | int | None


@dataclass(frozen=True, slots=True)
class ConfidenceChannelSummaryRecord:
    """Dedicated channel-aware confidence summary by split."""

    split: SplitScope
    metric_key: str
    sample_count: int
    minimum: float | int | None
    p50: float | int | None
    p95: float | int | None
    maximum: float | int | None


@dataclass(frozen=True, slots=True)
class TierCalibrationSurfaces:
    """All typed calibration outputs for one tier run."""

    metric_distributions: tuple[MetricDistributionRecord, ...]
    pass_fail_counts: tuple[TierMetricPassFailRecord, ...]
    primary_blockers: tuple[BlockerFrequencyRecord, ...]
    all_blockers: tuple[BlockerFrequencyRecord, ...]
    cofailures: tuple[CoFailureRecord, ...]
    saturated_metrics: tuple[SaturatedMetricRecord, ...]
    near_threshold_samples: tuple[NearThresholdSampleRecord, ...]
    tier_delta_samples: tuple[TierDeltaRecord, ...]
    tier_delta_summaries: tuple[TierDeltaSummaryRecord, ...]
    coverage_summaries: tuple[CoverageFamilySummaryRecord, ...]
    confidence_channel_summaries: tuple[ConfidenceChannelSummaryRecord, ...]


@dataclass(frozen=True, slots=True)
class OobThresholds:
    """Thresholds for out-of-bounds metrics."""

    max_out_of_bounds_ratio: float


@dataclass(frozen=True, slots=True)
class CoverageThresholds:
    """Thresholds for coverage metrics."""

    min_body_landmark_coverage_ratio: float
    min_any_hand_landmark_coverage_ratio: float
    min_face_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class HandThresholds:
    """Thresholds for temporal hand availability metrics."""

    min_any_hand_available_frame_ratio: float
    max_any_hand_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class FaceThresholds:
    """Thresholds for temporal face availability metrics."""

    min_face_available_frame_ratio: float


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Thresholds for channel-aware confidence quality metrics."""

    min_body_mean_confidence: float
    min_left_hand_mean_confidence: float
    min_right_hand_mean_confidence: float
    min_face_mean_confidence: float


@dataclass(frozen=True, slots=True)
class TextThresholds:
    """Thresholds for text metrics."""

    min_character_count: int
    min_token_count: int


@dataclass(frozen=True, slots=True)
class LengthThresholds:
    """Thresholds for length metrics."""

    min_num_frames: int
    min_duration_seconds: float


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Strict typed filter thresholds for every family and level."""

    oob: Mapping[FilterLevel, OobThresholds]
    coverage: Mapping[FilterLevel, CoverageThresholds]
    hand: Mapping[FilterLevel, HandThresholds]
    face: Mapping[FilterLevel, FaceThresholds]
    confidence: Mapping[FilterLevel, ConfidenceThresholds]
    text: Mapping[FilterLevel, TextThresholds]
    length: Mapping[FilterLevel, LengthThresholds]

    def __post_init__(self) -> None:
        for field_name in (
            "oob",
            "coverage",
            "hand",
            "face",
            "confidence",
            "text",
            "length",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Composition policy for a named tier."""

    tier_name: TierName
    family_levels: Mapping[BindingTierFamily, FilterLevel]
    max_allowed_leakage_severity: LeakageSeverity

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_levels",
            MappingProxyType(dict(self.family_levels)),
        )


@dataclass(frozen=True, slots=True)
class TierMetricFailure:
    """A deterministic family-level metric failure."""

    family: BindingTierFamily
    metric_key: str
    reason_code: str
    actual_value: float | int | None
    expected_value: float | int | None
    comparison: str
    applied_level: FilterLevel


@dataclass(frozen=True, slots=True)
class TierLeakageFailure:
    """A deterministic leakage-severity failure."""

    reason_code: str
    actual_max_severity: LeakageSeverity
    allowed_max_severity: LeakageSeverity
    matched_samples: tuple[LeakageSampleRef, ...]


@dataclass(frozen=True, slots=True)
class TierDecision:
    """Decision for one split/sample/tier identity."""

    sample_id: str
    split: SampleSplit
    tier_name: TierName
    included: bool
    metric_failures: tuple[TierMetricFailure, ...]
    leakage_failure: TierLeakageFailure | None
    max_leakage_severity: LeakageSeverity
    applied_family_levels: Mapping[BindingTierFamily, FilterLevel]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_failures",
            tuple(self.metric_failures),
        )
        object.__setattr__(
            self,
            "applied_family_levels",
            MappingProxyType(dict(self.applied_family_levels)),
        )


@dataclass(frozen=True, slots=True)
class TierBundle:
    """All deterministic tier decisions."""

    decisions: tuple[TierDecision, ...]


@dataclass(frozen=True, slots=True)
class IncludedTierSurfaceEntry:
    """File-free tier selection surface entry."""

    sample_id: str
    split: SampleSplit
    tier_name: TierName


@dataclass(frozen=True, slots=True)
class ExcludedTierSurfaceEntry:
    """File-free tier selection surface entry."""

    sample_id: str
    split: SampleSplit
    tier_name: TierName


@dataclass(frozen=True, slots=True)
class TierValidationIssue:
    """A specific issue found during tier bundle validation."""

    code: str
    message: str
