"""Typed tier-filter parsing from ``configs/data/filters.yaml``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Generic, TypeVar

import yaml

from text_to_sign_production.core.ids import TierName

DEFAULT_TIER_FILTERS_CONFIG_PATH = Path("configs/data/filters.yaml")

ThresholdT = TypeVar("ThresholdT")


@dataclass(frozen=True, slots=True)
class OobTierThresholds:
    """OOB thresholds for one filter level."""

    max_out_of_bounds_ratio: float


@dataclass(frozen=True, slots=True)
class UpperBodySupportTierThresholds:
    """Upper-body support thresholds for one filter level."""

    min_active_span_upper_body_support_landmark_coverage_ratio: float


@dataclass(frozen=True, slots=True)
class ManualVisibilityTierThresholds:
    """Manual visibility thresholds for one filter level."""

    min_active_span_any_hand_available_frame_ratio: float
    max_active_span_any_hand_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualVisibilityTierThresholds:
    """Non-manual visibility thresholds for one filter level."""

    min_active_span_face_available_frame_ratio: float
    max_active_span_face_unavailable_run_ratio: float


@dataclass(frozen=True, slots=True)
class ConfidenceTierThresholds:
    """Confidence thresholds for one filter level."""

    min_active_span_body_observed_mean_confidence: float
    min_active_span_any_hand_observed_mean_confidence: float


@dataclass(frozen=True, slots=True)
class KinematicNaturalnessTierThresholds:
    """Kinematic naturalness thresholds for one filter level."""

    max_comparable_transition_abrupt_ratio: float
    max_comparable_transition_discontinuity_ratio: float
    max_active_span_frozen_run_ratio: float


@dataclass(frozen=True, slots=True)
class TrackingQualityTierThresholds:
    """Tracking quality thresholds for one filter level."""

    max_tracked_target_missing_frame_ratio: float
    max_person_tracking_continuity_break_ratio: float
    max_person_tracking_reanchor_ratio: float


@dataclass(frozen=True, slots=True)
class ManualDetailTierThresholds:
    """Manual detail thresholds for one filter level."""

    min_active_span_representative_hand_landmark_coverage_ratio: float
    min_active_span_representative_hand_fingertip_coverage_ratio: float
    min_active_span_representative_hand_distal_chain_coverage_ratio: float
    max_active_span_representative_hand_detail_dropout_run_ratio: float


@dataclass(frozen=True, slots=True)
class NonManualQualityTierThresholds:
    """Non-manual quality thresholds for one filter level."""

    min_active_span_face_landmark_coverage_ratio: float
    min_active_span_upper_face_landmark_coverage_ratio: float
    min_active_span_lower_face_landmark_coverage_ratio: float
    max_active_span_face_detail_dropout_run_ratio: float
    min_active_span_face_available_given_manual_frame_ratio: float


@dataclass(frozen=True, slots=True)
class GeometryTierThresholds:
    """Geometry consistency thresholds for one filter level."""

    max_active_span_upper_body_bone_length_outlier_frame_ratio: float
    max_active_span_cross_channel_scale_outlier_frame_ratio: float


@dataclass(frozen=True, slots=True)
class _TierFilterSection(Generic[ThresholdT]):
    thresholds_by_level: Mapping[TierName, ThresholdT]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "thresholds_by_level",
            MappingProxyType(dict(self.thresholds_by_level)),
        )

    def thresholds_for(self, level: TierName) -> ThresholdT:
        """Return thresholds for a parsed filter level."""
        return self.thresholds_by_level[level]


@dataclass(frozen=True, slots=True)
class OobTierFilters(_TierFilterSection[OobTierThresholds]):
    """Typed OOB tier filters."""


@dataclass(frozen=True, slots=True)
class UpperBodySupportTierFilters(_TierFilterSection[UpperBodySupportTierThresholds]):
    """Typed upper-body-support tier filters."""


@dataclass(frozen=True, slots=True)
class ManualVisibilityTierFilters(_TierFilterSection[ManualVisibilityTierThresholds]):
    """Typed manual-visibility tier filters."""


@dataclass(frozen=True, slots=True)
class NonManualVisibilityTierFilters(_TierFilterSection[NonManualVisibilityTierThresholds]):
    """Typed non-manual-visibility tier filters."""


@dataclass(frozen=True, slots=True)
class ConfidenceTierFilters(_TierFilterSection[ConfidenceTierThresholds]):
    """Typed confidence tier filters."""


@dataclass(frozen=True, slots=True)
class KinematicNaturalnessTierFilters(_TierFilterSection[KinematicNaturalnessTierThresholds]):
    """Typed kinematic-naturalness tier filters."""


@dataclass(frozen=True, slots=True)
class TrackingQualityTierFilters(_TierFilterSection[TrackingQualityTierThresholds]):
    """Typed tracking-quality tier filters."""


@dataclass(frozen=True, slots=True)
class ManualDetailTierFilters(_TierFilterSection[ManualDetailTierThresholds]):
    """Typed manual-detail tier filters."""


@dataclass(frozen=True, slots=True)
class NonManualQualityTierFilters(_TierFilterSection[NonManualQualityTierThresholds]):
    """Typed non-manual-quality tier filters."""


@dataclass(frozen=True, slots=True)
class GeometryTierFilters(_TierFilterSection[GeometryTierThresholds]):
    """Typed geometry tier filters."""


@dataclass(frozen=True, slots=True)
class TierFiltersConfig:
    """Typed tier filter authority for all binding families."""

    oob: OobTierFilters
    upper_body_support: UpperBodySupportTierFilters
    manual_visibility: ManualVisibilityTierFilters
    non_manual_visibility: NonManualVisibilityTierFilters
    confidence: ConfidenceTierFilters
    kinematic_naturalness: KinematicNaturalnessTierFilters
    tracking_quality: TrackingQualityTierFilters
    manual_detail: ManualDetailTierFilters
    non_manual_quality: NonManualQualityTierFilters
    geometry: GeometryTierFilters


def load_tier_filters_config(path: Path = DEFAULT_TIER_FILTERS_CONFIG_PATH) -> TierFiltersConfig:
    """Load and parse tier filters from YAML."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid tier filters YAML: {exc}") from exc

    config = parse_tier_filters_config_mapping(loaded)
    from text_to_sign_production.data.tier.policies.validate import (
        validate_tier_filters_config,
    )

    issues = validate_tier_filters_config(config)
    if issues:
        raise ValueError(f"Invalid tier filters config: {issues}")
    return config


def parse_tier_filters_config_mapping(value: object) -> TierFiltersConfig:
    """Parse a loaded YAML mapping into typed tier filters."""
    root = _require_mapping(value, "tier filters root")
    _require_exact_keys(root, ("families",), "tier filters root")
    families = _require_mapping(root["families"], "families")
    _require_exact_keys(
        families,
        (
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
        ),
        "families",
    )
    return TierFiltersConfig(
        oob=OobTierFilters(
            _parse_threshold_levels(
                families["oob"],
                "oob",
                ("max_out_of_bounds_ratio",),
                lambda item: OobTierThresholds(
                    max_out_of_bounds_ratio=_require_ratio(
                        item["max_out_of_bounds_ratio"],
                        "oob.max_out_of_bounds_ratio",
                    )
                ),
            )
        ),
        upper_body_support=UpperBodySupportTierFilters(
            _parse_threshold_levels(
                families["upper_body_support"],
                "upper_body_support",
                ("min_active_span_upper_body_support_landmark_coverage_ratio",),
                lambda item: UpperBodySupportTierThresholds(
                    min_active_span_upper_body_support_landmark_coverage_ratio=_require_ratio(
                        item["min_active_span_upper_body_support_landmark_coverage_ratio"],
                        "upper_body_support.min_active_span_upper_body_support_landmark_coverage_ratio",
                    )
                ),
            )
        ),
        manual_visibility=ManualVisibilityTierFilters(
            _parse_threshold_levels(
                families["manual_visibility"],
                "manual_visibility",
                (
                    "min_active_span_any_hand_available_frame_ratio",
                    "max_active_span_any_hand_unavailable_run_ratio",
                ),
                lambda item: ManualVisibilityTierThresholds(
                    min_active_span_any_hand_available_frame_ratio=_require_ratio(
                        item["min_active_span_any_hand_available_frame_ratio"],
                        "manual_visibility.min_active_span_any_hand_available_frame_ratio",
                    ),
                    max_active_span_any_hand_unavailable_run_ratio=_require_ratio(
                        item["max_active_span_any_hand_unavailable_run_ratio"],
                        "manual_visibility.max_active_span_any_hand_unavailable_run_ratio",
                    ),
                ),
            )
        ),
        non_manual_visibility=NonManualVisibilityTierFilters(
            _parse_threshold_levels(
                families["non_manual_visibility"],
                "non_manual_visibility",
                (
                    "min_active_span_face_available_frame_ratio",
                    "max_active_span_face_unavailable_run_ratio",
                ),
                lambda item: NonManualVisibilityTierThresholds(
                    min_active_span_face_available_frame_ratio=_require_ratio(
                        item["min_active_span_face_available_frame_ratio"],
                        "non_manual_visibility.min_active_span_face_available_frame_ratio",
                    ),
                    max_active_span_face_unavailable_run_ratio=_require_ratio(
                        item["max_active_span_face_unavailable_run_ratio"],
                        "non_manual_visibility.max_active_span_face_unavailable_run_ratio",
                    ),
                ),
            )
        ),
        confidence=ConfidenceTierFilters(
            _parse_threshold_levels(
                families["confidence"],
                "confidence",
                (
                    "min_active_span_body_observed_mean_confidence",
                    "min_active_span_any_hand_observed_mean_confidence",
                ),
                lambda item: ConfidenceTierThresholds(
                    min_active_span_body_observed_mean_confidence=_require_ratio(
                        item["min_active_span_body_observed_mean_confidence"],
                        "confidence.min_active_span_body_observed_mean_confidence",
                    ),
                    min_active_span_any_hand_observed_mean_confidence=_require_ratio(
                        item["min_active_span_any_hand_observed_mean_confidence"],
                        "confidence.min_active_span_any_hand_observed_mean_confidence",
                    ),
                ),
            )
        ),
        kinematic_naturalness=KinematicNaturalnessTierFilters(
            _parse_threshold_levels(
                families["kinematic_naturalness"],
                "kinematic_naturalness",
                (
                    "max_comparable_transition_abrupt_ratio",
                    "max_comparable_transition_discontinuity_ratio",
                    "max_active_span_frozen_run_ratio",
                ),
                lambda item: KinematicNaturalnessTierThresholds(
                    max_comparable_transition_abrupt_ratio=_require_ratio(
                        item["max_comparable_transition_abrupt_ratio"],
                        "kinematic_naturalness.max_comparable_transition_abrupt_ratio",
                    ),
                    max_comparable_transition_discontinuity_ratio=_require_ratio(
                        item["max_comparable_transition_discontinuity_ratio"],
                        "kinematic_naturalness.max_comparable_transition_discontinuity_ratio",
                    ),
                    max_active_span_frozen_run_ratio=_require_ratio(
                        item["max_active_span_frozen_run_ratio"],
                        "kinematic_naturalness.max_active_span_frozen_run_ratio",
                    ),
                ),
            )
        ),
        tracking_quality=TrackingQualityTierFilters(
            _parse_threshold_levels(
                families["tracking_quality"],
                "tracking_quality",
                (
                    "max_tracked_target_missing_frame_ratio",
                    "max_person_tracking_continuity_break_ratio",
                    "max_person_tracking_reanchor_ratio",
                ),
                lambda item: TrackingQualityTierThresholds(
                    max_tracked_target_missing_frame_ratio=_require_ratio(
                        item["max_tracked_target_missing_frame_ratio"],
                        "tracking_quality.max_tracked_target_missing_frame_ratio",
                    ),
                    max_person_tracking_continuity_break_ratio=_require_ratio(
                        item["max_person_tracking_continuity_break_ratio"],
                        "tracking_quality.max_person_tracking_continuity_break_ratio",
                    ),
                    max_person_tracking_reanchor_ratio=_require_ratio(
                        item["max_person_tracking_reanchor_ratio"],
                        "tracking_quality.max_person_tracking_reanchor_ratio",
                    ),
                ),
            )
        ),
        manual_detail=ManualDetailTierFilters(
            _parse_threshold_levels(
                families["manual_detail"],
                "manual_detail",
                (
                    "min_active_span_representative_hand_landmark_coverage_ratio",
                    "min_active_span_representative_hand_fingertip_coverage_ratio",
                    "min_active_span_representative_hand_distal_chain_coverage_ratio",
                    "max_active_span_representative_hand_detail_dropout_run_ratio",
                ),
                lambda item: ManualDetailTierThresholds(
                    min_active_span_representative_hand_landmark_coverage_ratio=_require_ratio(
                        item["min_active_span_representative_hand_landmark_coverage_ratio"],
                        "manual_detail.min_active_span_representative_hand_landmark_coverage_ratio",
                    ),
                    min_active_span_representative_hand_fingertip_coverage_ratio=_require_ratio(
                        item["min_active_span_representative_hand_fingertip_coverage_ratio"],
                        "manual_detail.min_active_span_representative_hand_fingertip_coverage_ratio",
                    ),
                    min_active_span_representative_hand_distal_chain_coverage_ratio=_require_ratio(
                        item["min_active_span_representative_hand_distal_chain_coverage_ratio"],
                        "manual_detail.min_active_span_representative_hand_distal_chain_coverage_ratio",
                    ),
                    max_active_span_representative_hand_detail_dropout_run_ratio=_require_ratio(
                        item["max_active_span_representative_hand_detail_dropout_run_ratio"],
                        "manual_detail.max_active_span_representative_hand_detail_dropout_run_ratio",
                    ),
                ),
            )
        ),
        non_manual_quality=NonManualQualityTierFilters(
            _parse_threshold_levels(
                families["non_manual_quality"],
                "non_manual_quality",
                (
                    "min_active_span_face_landmark_coverage_ratio",
                    "min_active_span_upper_face_landmark_coverage_ratio",
                    "min_active_span_lower_face_landmark_coverage_ratio",
                    "max_active_span_face_detail_dropout_run_ratio",
                    "min_active_span_face_available_given_manual_frame_ratio",
                ),
                lambda item: NonManualQualityTierThresholds(
                    min_active_span_face_landmark_coverage_ratio=_require_ratio(
                        item["min_active_span_face_landmark_coverage_ratio"],
                        "non_manual_quality.min_active_span_face_landmark_coverage_ratio",
                    ),
                    min_active_span_upper_face_landmark_coverage_ratio=_require_ratio(
                        item["min_active_span_upper_face_landmark_coverage_ratio"],
                        "non_manual_quality.min_active_span_upper_face_landmark_coverage_ratio",
                    ),
                    min_active_span_lower_face_landmark_coverage_ratio=_require_ratio(
                        item["min_active_span_lower_face_landmark_coverage_ratio"],
                        "non_manual_quality.min_active_span_lower_face_landmark_coverage_ratio",
                    ),
                    max_active_span_face_detail_dropout_run_ratio=_require_ratio(
                        item["max_active_span_face_detail_dropout_run_ratio"],
                        "non_manual_quality.max_active_span_face_detail_dropout_run_ratio",
                    ),
                    min_active_span_face_available_given_manual_frame_ratio=_require_ratio(
                        item["min_active_span_face_available_given_manual_frame_ratio"],
                        "non_manual_quality.min_active_span_face_available_given_manual_frame_ratio",
                    ),
                ),
            )
        ),
        geometry=GeometryTierFilters(
            _parse_threshold_levels(
                families["geometry"],
                "geometry",
                (
                    "max_active_span_upper_body_bone_length_outlier_frame_ratio",
                    "max_active_span_cross_channel_scale_outlier_frame_ratio",
                ),
                lambda item: GeometryTierThresholds(
                    max_active_span_upper_body_bone_length_outlier_frame_ratio=_require_ratio(
                        item["max_active_span_upper_body_bone_length_outlier_frame_ratio"],
                        "geometry.max_active_span_upper_body_bone_length_outlier_frame_ratio",
                    ),
                    max_active_span_cross_channel_scale_outlier_frame_ratio=_require_ratio(
                        item["max_active_span_cross_channel_scale_outlier_frame_ratio"],
                        "geometry.max_active_span_cross_channel_scale_outlier_frame_ratio",
                    ),
                ),
            )
        ),
    )


def _parse_threshold_levels(
    value: object,
    label: str,
    expected_threshold_keys: tuple[str, ...],
    factory: Callable[[Mapping[str, object]], ThresholdT],
) -> Mapping[TierName, ThresholdT]:
    levels = _require_mapping(value, label)
    _require_exact_keys(levels, tuple(tier.value for tier in TierName), label)
    parsed: dict[TierName, ThresholdT] = {}
    for tier in TierName:
        item = _require_mapping(levels[tier.value], f"{label}.{tier.value}")
        _require_exact_keys(item, expected_threshold_keys, f"{label}.{tier.value}")
        parsed[tier] = factory(item)
    return parsed


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected_keys: tuple[str, ...],
    label: str,
) -> None:
    expected = set(expected_keys)
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} must contain exactly {sorted(expected)} "
            f"(missing={sorted(expected - actual)}, unknown={sorted(actual - expected)})."
        )


def _require_ratio(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be a numeric ratio.")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be within [0, 1].")
    return parsed


__all__ = [
    "ConfidenceTierFilters",
    "ConfidenceTierThresholds",
    "DEFAULT_TIER_FILTERS_CONFIG_PATH",
    "GeometryTierFilters",
    "GeometryTierThresholds",
    "KinematicNaturalnessTierFilters",
    "KinematicNaturalnessTierThresholds",
    "ManualDetailTierFilters",
    "ManualDetailTierThresholds",
    "ManualVisibilityTierFilters",
    "ManualVisibilityTierThresholds",
    "NonManualQualityTierFilters",
    "NonManualQualityTierThresholds",
    "NonManualVisibilityTierFilters",
    "NonManualVisibilityTierThresholds",
    "OobTierFilters",
    "OobTierThresholds",
    "TierFiltersConfig",
    "TrackingQualityTierFilters",
    "TrackingQualityTierThresholds",
    "UpperBodySupportTierFilters",
    "UpperBodySupportTierThresholds",
    "load_tier_filters_config",
    "parse_tier_filters_config_mapping",
]
