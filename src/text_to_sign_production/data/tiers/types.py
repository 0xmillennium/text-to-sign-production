"""Typed models for deterministic tier decisions."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.data.leakages.types import LeakageSampleRef, LeakageSeverity


class FilterLevel(enum.StrEnum):
    """Strict threshold levels loaded from filters.yaml."""

    LOOSE = "loose"
    CLEAN = "clean"
    TIGHT = "tight"


class MetricFamily(enum.StrEnum):
    """Metric families owned by the tier decision layer."""

    OOB = "oob"
    HAND = "hand"
    FACE = "face"
    VALID = "valid"
    CONFIDENCE = "confidence"
    TEXT = "text"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class OobThresholds:
    """Thresholds for out-of-bounds metrics."""

    max_out_of_bounds_ratio: float


@dataclass(frozen=True, slots=True)
class HandThresholds:
    """Thresholds for hand-presence metrics."""

    min_any_hand_nonzero_frame_ratio: float


@dataclass(frozen=True, slots=True)
class FaceThresholds:
    """Thresholds for face-presence metrics."""

    min_face_nonzero_frame_ratio: float
    max_face_missing_frame_ratio: float


@dataclass(frozen=True, slots=True)
class ValidThresholds:
    """Thresholds for valid-frame metrics."""

    min_valid_frame_ratio: float
    max_zeroed_canonical_joint_frame_ratio: float


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Thresholds for confidence metrics."""

    min_overall_mean_confidence: float
    min_overall_nonzero_confidence_ratio: float


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

    oob: dict[FilterLevel, OobThresholds]
    hand: dict[FilterLevel, HandThresholds]
    face: dict[FilterLevel, FaceThresholds]
    valid: dict[FilterLevel, ValidThresholds]
    confidence: dict[FilterLevel, ConfidenceThresholds]
    text: dict[FilterLevel, TextThresholds]
    length: dict[FilterLevel, LengthThresholds]


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """Composition policy for a named tier."""

    tier_name: str
    family_levels: dict[str, FilterLevel]
    max_allowed_leakage_severity: LeakageSeverity


@dataclass(frozen=True, slots=True)
class TierMetricFailure:
    """A deterministic family-level metric failure."""

    family: str
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
    split: str
    tier_name: str
    included: bool
    metric_failures: tuple[TierMetricFailure, ...]
    leakage_failure: TierLeakageFailure | None
    max_leakage_severity: LeakageSeverity
    applied_family_levels: dict[str, FilterLevel]


@dataclass(frozen=True, slots=True)
class TierBundle:
    """All deterministic tier decisions."""

    decisions: tuple[TierDecision, ...]


@dataclass(frozen=True, slots=True)
class IncludedTierSurfaceEntry:
    """File-free included surface entry."""

    sample_id: str
    split: str
    tier_name: str


@dataclass(frozen=True, slots=True)
class ExcludedTierSurfaceEntry:
    """File-free excluded surface entry."""

    sample_id: str
    split: str
    tier_name: str
    metric_failures: tuple[TierMetricFailure, ...]
    leakage_failure: TierLeakageFailure | None


@dataclass(frozen=True, slots=True)
class TierValidationIssue:
    """A specific issue found during tier bundle validation."""

    code: str
    message: str
