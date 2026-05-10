"""Public type source of truth for deterministic tier policy and decisions."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from text_to_sign_production.core.ids import SampleSplit, TierMembership, TierName
from text_to_sign_production.legacy_data._shared.types import ValidationIssue
from text_to_sign_production.legacy_data.leakages.types import LeakageSampleRef, LeakageSeverity

# ---------------------------------------------------------------------------
# Vocabulary enums
# ---------------------------------------------------------------------------

SplitScope: TypeAlias = SampleSplit | None


class FilterLevel(enum.StrEnum):
    """Strict threshold levels loaded from filters.yaml."""

    LOOSE = "loose"
    CLEAN = "clean"
    TIGHT = "tight"


class BindingTierFamily(enum.StrEnum):
    """Metric families that are binding for tier membership decisions."""

    OOB = "oob"
    UPPER_BODY_SUPPORT = "upper_body_support"
    MANUAL_VISIBILITY = "manual_visibility"
    NON_MANUAL_VISIBILITY = "non_manual_visibility"
    CONFIDENCE = "confidence"
    KINEMATIC_NATURALNESS = "kinematic_naturalness"
    TRACKING_QUALITY = "tracking_quality"
    MANUAL_DETAIL = "manual_detail"
    NON_MANUAL_QUALITY = "non_manual_quality"
    GEOMETRY = "geometry"


class DiagnosticTierCategory(enum.StrEnum):
    """Non-binding diagnostic metric categories in the tier role registry."""

    ACTIVE_SIGNING_SPAN = "active_signing_span"
    COVERAGE = "coverage"
    VALID = "valid"
    TEXT = "text"
    LENGTH = "length"


class DiagnosticMetric(enum.StrEnum):
    """Computed metrics kept visible for diagnostics without tier veto power."""

    ACTIVE_SIGNING_SPAN_START_FRAME_INDEX = "active_signing_span_start_frame_index"
    ACTIVE_SIGNING_SPAN_END_FRAME_INDEX_EXCLUSIVE = "active_signing_span_end_frame_index_exclusive"
    ACTIVE_SIGNING_SPAN_FRAME_COUNT = "active_signing_span_frame_count"
    ACTIVE_SIGNING_SPAN_FRAME_RATIO = "active_signing_span_frame_ratio"
    TRIMMED_PREFIX_FRAME_COUNT = "trimmed_prefix_frame_count"
    TRIMMED_PREFIX_FRAME_RATIO = "trimmed_prefix_frame_ratio"
    TRIMMED_SUFFIX_FRAME_COUNT = "trimmed_suffix_frame_count"
    TRIMMED_SUFFIX_FRAME_RATIO = "trimmed_suffix_frame_ratio"
    SUSTAINED_ANY_HAND_EVIDENCE_FRAME_RATIO = "sustained_any_hand_evidence_frame_ratio"
    SUSTAINED_UPPER_BODY_EVIDENCE_FRAME_RATIO = "sustained_upper_body_evidence_frame_ratio"
    SUSTAINED_MOTION_EVIDENCE_FRAME_RATIO = "sustained_motion_evidence_frame_ratio"
    FULL_CLIP_UPPER_BODY_SUPPORT_LANDMARK_COVERAGE_RATIO = (
        "full_clip_upper_body_support_landmark_coverage_ratio"
    )
    FULL_BODY_LANDMARK_COVERAGE_RATIO = "full_body_landmark_coverage_ratio"
    LEFT_HAND_LANDMARK_COVERAGE_RATIO = "left_hand_landmark_coverage_ratio"
    RIGHT_HAND_LANDMARK_COVERAGE_RATIO = "right_hand_landmark_coverage_ratio"
    ANY_HAND_LANDMARK_COVERAGE_RATIO = "any_hand_landmark_coverage_ratio"
    FACE_LANDMARK_COVERAGE_RATIO = "face_landmark_coverage_ratio"
    WHOLE_CLIP_LEFT_HAND_AVAILABLE_FRAME_RATIO = "whole_clip_left_hand_available_frame_ratio"
    WHOLE_CLIP_RIGHT_HAND_AVAILABLE_FRAME_RATIO = "whole_clip_right_hand_available_frame_ratio"
    WHOLE_CLIP_ANY_HAND_AVAILABLE_FRAME_RATIO = "whole_clip_any_hand_available_frame_ratio"
    FACE_AVAILABLE_FRAME_RATIO = "face_available_frame_ratio"
    FACE_UNAVAILABLE_FRAME_RATIO = "face_unavailable_frame_ratio"
    ACTIVE_SPAN_LEFT_HAND_AVAILABLE_MEAN_CONFIDENCE = (
        "active_span_left_hand_available_mean_confidence"
    )
    ACTIVE_SPAN_RIGHT_HAND_AVAILABLE_MEAN_CONFIDENCE = (
        "active_span_right_hand_available_mean_confidence"
    )
    FULL_CLIP_BODY_AVAILABLE_MEAN_CONFIDENCE = "full_clip_body_available_mean_confidence"
    FACE_AVAILABLE_MEAN_CONFIDENCE = "face_available_mean_confidence"
    OVERALL_AVAILABLE_MEAN_CONFIDENCE = "overall_available_mean_confidence"
    BODY_NONZERO_CONFIDENCE_RATIO = "body_nonzero_confidence_ratio"
    LEFT_HAND_NONZERO_CONFIDENCE_RATIO = "left_hand_nonzero_confidence_ratio"
    RIGHT_HAND_NONZERO_CONFIDENCE_RATIO = "right_hand_nonzero_confidence_ratio"
    FACE_NONZERO_CONFIDENCE_RATIO = "face_nonzero_confidence_ratio"
    OVERALL_NONZERO_CONFIDENCE_RATIO = "overall_nonzero_confidence_ratio"
    VALID_FRAME_COUNT = "valid_frame_count"
    VALID_FRAME_RATIO = "valid_frame_ratio"
    INVALID_FRAME_COUNT = "invalid_frame_count"
    INVALID_FRAME_RATIO = "invalid_frame_ratio"
    ZEROED_CANONICAL_JOINT_FRAME_COUNT = "zeroed_canonical_joint_frame_count"
    ZEROED_CANONICAL_JOINT_FRAME_RATIO = "zeroed_canonical_joint_frame_ratio"
    CHARACTER_COUNT = "character_count"
    TOKEN_COUNT = "token_count"
    NUM_FRAMES = "num_frames"
    DURATION_SECONDS = "duration_seconds"
    FRAMES_PER_TOKEN = "frames_per_token"
    FRAMES_PER_CHARACTER = "frames_per_character"


class MetricPolicyRole(enum.StrEnum):
    """Whether a metric participates in tier veto policy."""

    BINDING = "binding"
    DIAGNOSTIC = "diagnostic"


_BINDING_TIER_FAMILY_DISPLAY_LABELS: Mapping[BindingTierFamily, str] = MappingProxyType(
    {
        BindingTierFamily.OOB: "out-of-bounds coordinates",
        BindingTierFamily.UPPER_BODY_SUPPORT: "upper-body support",
        BindingTierFamily.MANUAL_VISIBILITY: "manual visibility",
        BindingTierFamily.NON_MANUAL_VISIBILITY: "non-manual visibility",
        BindingTierFamily.CONFIDENCE: "available-landmark confidence",
        BindingTierFamily.KINEMATIC_NATURALNESS: "kinematic naturalness",
        BindingTierFamily.TRACKING_QUALITY: "tracking quality",
        BindingTierFamily.MANUAL_DETAIL: "manual detail",
        BindingTierFamily.NON_MANUAL_QUALITY: "non-manual quality",
        BindingTierFamily.GEOMETRY: "geometry",
    }
)


def binding_tier_family_display_label(family: BindingTierFamily) -> str:
    """Return the user-facing label for a binding tier family."""
    if not isinstance(family, BindingTierFamily):
        raise TypeError(f"Expected BindingTierFamily, got {type(family)!r}.")
    return _BINDING_TIER_FAMILY_DISPLAY_LABELS[family]


# ---------------------------------------------------------------------------
# Threshold / config types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OobThresholds:
    """Thresholds for out-of-bounds metrics."""

    max_out_of_bounds_ratio: float


@dataclass(frozen=True, slots=True)
class UpperBodySupportThresholds:
    """Thresholds for active-span upper-body landmark-density support metrics."""

    min_active_span_upper_body_support_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class ManualVisibilityThresholds:
    """Thresholds for active-signing-span manual visibility metrics."""

    min_active_span_any_hand_available_frame_ratio: float
    max_active_span_any_hand_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualVisibilityThresholds:
    """Thresholds for active-signing-span non-manual visibility metrics."""

    min_active_span_face_available_frame_ratio: float
    max_active_span_face_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Thresholds for confidence over available/nonzero landmarks."""

    min_active_span_body_available_mean_confidence: float
    min_active_span_any_hand_available_mean_confidence: float


@dataclass(frozen=True, slots=True)
class KinematicNaturalnessThresholds:
    """Thresholds for active-span motion and missing-transition pathology metrics."""

    max_active_span_abrupt_motion_frame_ratio: float
    max_active_span_discontinuity_frame_ratio: float
    max_active_span_frozen_run_ratio: float


@dataclass(frozen=True, slots=True)
class TrackingQualityThresholds:
    """Thresholds for persisted tracking-quality metrics."""

    max_tracked_target_missing_frame_ratio: float
    max_person_tracking_continuity_break_ratio: float
    max_person_tracking_reanchor_ratio: float


@dataclass(frozen=True, slots=True)
class ManualDetailThresholds:
    """Thresholds for active-span representative-hand detail metrics."""

    min_active_span_representative_hand_landmark_coverage_ratio: float
    min_active_span_representative_hand_fingertip_coverage_ratio: float
    min_active_span_representative_hand_distal_chain_coverage_ratio: float
    max_active_span_representative_hand_detail_dropout_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualQualityThresholds:
    """Thresholds for active-span non-manual detail metrics."""

    min_active_span_face_landmark_coverage_ratio: float
    min_active_span_upper_face_landmark_coverage_ratio: float
    min_active_span_lower_face_landmark_coverage_ratio: float
    max_active_span_face_detail_dropout_run_ratio: float
    min_active_span_manual_face_overlap_frame_ratio: float


@dataclass(frozen=True, slots=True)
class GeometryThresholds:
    """Thresholds for sample-internal geometric consistency metrics."""

    max_active_span_upper_body_bone_length_outlier_frame_ratio: float
    max_active_span_representative_hand_bone_length_outlier_frame_ratio: float
    max_active_span_cross_channel_scale_outlier_frame_ratio: float


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Strict typed filter thresholds for every family and level."""

    oob: Mapping[FilterLevel, OobThresholds]
    upper_body_support: Mapping[FilterLevel, UpperBodySupportThresholds]
    manual_visibility: Mapping[FilterLevel, ManualVisibilityThresholds]
    non_manual_visibility: Mapping[FilterLevel, NonManualVisibilityThresholds]
    confidence: Mapping[FilterLevel, ConfidenceThresholds]
    kinematic_naturalness: Mapping[FilterLevel, KinematicNaturalnessThresholds]
    tracking_quality: Mapping[FilterLevel, TrackingQualityThresholds]
    manual_detail: Mapping[FilterLevel, ManualDetailThresholds]
    non_manual_quality: Mapping[FilterLevel, NonManualQualityThresholds]
    geometry: Mapping[FilterLevel, GeometryThresholds]

    def __post_init__(self) -> None:
        for field_name in (
            "oob",
            "upper_body_support",
            "manual_visibility",
            "non_manual_visibility",
            "confidence",
            "kinematic_naturalness",
            "tracking_quality",
            "manual_detail",
            "non_manual_quality",
            "geometry",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )


# ---------------------------------------------------------------------------
# Policy types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Composition policy for a named tier."""

    tier_name: TierName
    family_levels: Mapping[BindingTierFamily, FilterLevel]
    max_allowed_leakage_severity: LeakageSeverity

    def __post_init__(self) -> None:
        if not isinstance(self.tier_name, TierName):
            raise TypeError(f"tier_name must be TierName, got {type(self.tier_name)!r}.")
        if not isinstance(self.max_allowed_leakage_severity, LeakageSeverity):
            raise TypeError(
                "max_allowed_leakage_severity must be LeakageSeverity, got "
                f"{type(self.max_allowed_leakage_severity)!r}."
            )
        strict_family_levels: dict[BindingTierFamily, FilterLevel] = {}
        for family, level in self.family_levels.items():
            if not isinstance(family, BindingTierFamily):
                raise TypeError(
                    f"family_levels keys must be BindingTierFamily, got {type(family)!r}."
                )
            if not isinstance(level, FilterLevel):
                raise TypeError(f"family_levels values must be FilterLevel, got {type(level)!r}.")
            strict_family_levels[family] = level
        object.__setattr__(
            self,
            "family_levels",
            MappingProxyType(strict_family_levels),
        )


# ---------------------------------------------------------------------------
# Decision / failure / bundle types
# ---------------------------------------------------------------------------


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

    def __post_init__(self) -> None:
        if not isinstance(self.family, BindingTierFamily):
            raise TypeError(f"family must be BindingTierFamily, got {type(self.family)!r}.")
        if not isinstance(self.applied_level, FilterLevel):
            raise TypeError(f"applied_level must be FilterLevel, got {type(self.applied_level)!r}.")


@dataclass(frozen=True, slots=True)
class TierLeakageFailure:
    """A deterministic leakage-severity failure."""

    reason_code: str
    actual_max_severity: LeakageSeverity
    allowed_max_severity: LeakageSeverity
    matched_samples: tuple[LeakageSampleRef, ...]


@dataclass(frozen=True, slots=True)
class TierMembershipRecord:
    """Thin membership projection for materializing tier manifests and archives."""

    sample_id: str
    split: SampleSplit
    tier_name: TierName
    membership: TierMembership

    def __post_init__(self) -> None:
        if not isinstance(self.split, SampleSplit):
            raise TypeError(f"split must be SampleSplit, got {type(self.split)!r}.")
        if not isinstance(self.tier_name, TierName):
            raise TypeError(f"tier_name must be TierName, got {type(self.tier_name)!r}.")
        if not isinstance(self.membership, TierMembership):
            raise TypeError(f"membership must be TierMembership, got {type(self.membership)!r}.")


@dataclass(frozen=True, slots=True)
class TierDecisionDetail:
    """Explanation detail for one split/sample/tier membership decision."""

    sample_id: str
    split: SampleSplit
    tier_name: TierName
    membership: TierMembership
    metric_failures: tuple[TierMetricFailure, ...]
    leakage_failure: TierLeakageFailure | None
    max_leakage_severity: LeakageSeverity
    applied_family_levels: Mapping[BindingTierFamily, FilterLevel]

    def __post_init__(self) -> None:
        if not isinstance(self.split, SampleSplit):
            raise TypeError(f"split must be SampleSplit, got {type(self.split)!r}.")
        if not isinstance(self.tier_name, TierName):
            raise TypeError(f"tier_name must be TierName, got {type(self.tier_name)!r}.")
        if not isinstance(self.membership, TierMembership):
            raise TypeError(f"membership must be TierMembership, got {type(self.membership)!r}.")
        if not isinstance(self.max_leakage_severity, LeakageSeverity):
            raise TypeError(
                f"max_leakage_severity must be LeakageSeverity, got "
                f"{type(self.max_leakage_severity)!r}."
            )
        object.__setattr__(self, "metric_failures", tuple(self.metric_failures))
        strict_family_levels: dict[BindingTierFamily, FilterLevel] = {}
        for family, level in self.applied_family_levels.items():
            if not isinstance(family, BindingTierFamily):
                raise TypeError(
                    f"applied_family_levels keys must be BindingTierFamily, got {type(family)!r}."
                )
            if not isinstance(level, FilterLevel):
                raise TypeError(
                    f"applied_family_levels values must be FilterLevel, got {type(level)!r}."
                )
            strict_family_levels[family] = level
        object.__setattr__(
            self,
            "applied_family_levels",
            MappingProxyType(strict_family_levels),
        )


@dataclass(frozen=True, slots=True)
class TierBundle:
    """All deterministic tier memberships and explanation details."""

    memberships: tuple[TierMembershipRecord, ...]
    decision_details: tuple[TierDecisionDetail, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "memberships", tuple(self.memberships))
        object.__setattr__(self, "decision_details", tuple(self.decision_details))


# ---------------------------------------------------------------------------
# Analysis output record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TierInclusionCountRecord:
    """Split-aware inclusion/exclusion counts under one tier."""

    tier_name: TierName
    split: SampleSplit
    included_count: int
    excluded_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class TierMetricPassFailRecord:
    """Pass/fail counts for one binding metric under one tier and split."""

    tier_name: TierName
    split: SampleSplit
    family: BindingTierFamily
    metric_key: str
    pass_count: int
    fail_count: int


@dataclass(frozen=True, slots=True)
class BlockerFrequencyRecord:
    """Aggregated metric blocker frequency for tier decisions."""

    tier_name: TierName
    split: SampleSplit
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


class NearThresholdSide(enum.StrEnum):
    """Pass/fail side for nearest-threshold sample records."""

    PASSING = "passing"
    FAILING = "failing"


@dataclass(frozen=True, slots=True)
class NearThresholdSampleRecord:
    """Sample nearest to a binding threshold under one tier."""

    tier_name: TierName
    split: SampleSplit
    family: BindingTierFamily
    metric_key: str
    sample_id: str
    side: NearThresholdSide
    actual_value: float | int | None
    expected_value: float | int
    comparison: str
    threshold_distance: float


@dataclass(frozen=True, slots=True)
class NearThresholdSummaryRecord:
    """Count of near-threshold samples by tier, split, metric, and side."""

    tier_name: TierName
    split: SampleSplit
    family: BindingTierFamily
    metric_key: str
    side: NearThresholdSide
    sample_count: int


@dataclass(frozen=True, slots=True)
class TierDeltaRecord:
    """Sample included in a looser tier but excluded by the next stricter tier."""

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


# ---------------------------------------------------------------------------
# Validation aliases / small helpers
# ---------------------------------------------------------------------------


TierValidationIssue = ValidationIssue


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "BindingTierFamily",
    "BlockerFrequencyRecord",
    "CoFailureRecord",
    "ConfidenceThresholds",
    "DiagnosticMetric",
    "DiagnosticTierCategory",
    "FilterConfig",
    "FilterLevel",
    "GeometryThresholds",
    "KinematicNaturalnessThresholds",
    "ManualDetailThresholds",
    "ManualVisibilityThresholds",
    "MetricPolicyRole",
    "NearThresholdSampleRecord",
    "NearThresholdSide",
    "NearThresholdSummaryRecord",
    "NonManualVisibilityThresholds",
    "NonManualQualityThresholds",
    "OobThresholds",
    "SplitScope",
    "TrackingQualityThresholds",
    "TierBundle",
    "TierDecisionDetail",
    "TierDeltaRecord",
    "TierDeltaSummaryRecord",
    "TierInclusionCountRecord",
    "TierLeakageFailure",
    "TierMembership",
    "TierMembershipRecord",
    "TierMetricFailure",
    "TierMetricPassFailRecord",
    "TierName",
    "TierPolicy",
    "TierValidationIssue",
    "UpperBodySupportThresholds",
    "binding_tier_family_display_label",
]
