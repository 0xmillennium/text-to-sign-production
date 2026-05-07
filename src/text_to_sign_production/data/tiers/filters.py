"""Top-level loading and dispatch for semantically explicit tier filter thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml

from text_to_sign_production.data.tiers._shared.parsing import require_exact_keys, require_mapping
from text_to_sign_production.data.tiers.coherence import (
    parse_temporal_coherence_thresholds,
)
from text_to_sign_production.data.tiers.confidence import parse_confidence_thresholds
from text_to_sign_production.data.tiers.face import parse_face_thresholds
from text_to_sign_production.data.tiers.geometry import parse_geometry_thresholds
from text_to_sign_production.data.tiers.hand import parse_hand_thresholds
from text_to_sign_production.data.tiers.manual_detail import parse_manual_detail_thresholds
from text_to_sign_production.data.tiers.non_manual_quality import (
    parse_non_manual_quality_thresholds,
)
from text_to_sign_production.data.tiers.oob import parse_oob_thresholds
from text_to_sign_production.data.tiers.roles import BINDING_TIER_FAMILIES
from text_to_sign_production.data.tiers.support import (
    parse_upper_body_support_thresholds,
)
from text_to_sign_production.data.tiers.tracking import parse_tracking_quality_thresholds
from text_to_sign_production.data.tiers.types import FilterConfig, FilterLevel

ThresholdT = TypeVar("ThresholdT")

_FAMILY_KEYS = tuple(family.value for family in BINDING_TIER_FAMILIES)
_LEVEL_ORDER = (FilterLevel.LOOSE, FilterLevel.CLEAN, FilterLevel.TIGHT)


def load_filter_config(path: str | Path) -> FilterConfig:
    """Load filters.yaml into a strict typed threshold config."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid filters YAML: {exc}") from exc
    return parse_filter_config(loaded)


def parse_filter_config(payload: object) -> FilterConfig:
    """Parse a loaded YAML object into a strict typed threshold config."""
    root = require_mapping(payload, "filters root")
    require_exact_keys(root, ("families",), "filters root")

    families = require_mapping(root["families"], "families")
    require_exact_keys(families, _FAMILY_KEYS, "families")

    config = FilterConfig(
        oob=parse_oob_thresholds(families["oob"]),
        upper_body_support=parse_upper_body_support_thresholds(families["upper_body_support"]),
        manual_visibility=parse_hand_thresholds(families["manual_visibility"]),
        non_manual_visibility=parse_face_thresholds(families["non_manual_visibility"]),
        confidence=parse_confidence_thresholds(families["confidence"]),
        kinematic_naturalness=parse_temporal_coherence_thresholds(
            families["kinematic_naturalness"]
        ),
        tracking_quality=parse_tracking_quality_thresholds(families["tracking_quality"]),
        manual_detail=parse_manual_detail_thresholds(families["manual_detail"]),
        non_manual_quality=parse_non_manual_quality_thresholds(families["non_manual_quality"]),
        geometry=parse_geometry_thresholds(families["geometry"]),
    )
    validate_filter_config(config)
    return config


def validate_filter_config(config: FilterConfig) -> None:
    """Validate cross-level strictness for binding tier veto thresholds."""
    _require_nonincreasing(
        config.oob,
        "max_out_of_bounds_ratio",
        "oob.max_out_of_bounds_ratio",
    )
    _require_nondecreasing(
        config.upper_body_support,
        "min_active_span_upper_body_support_landmark_coverage_ratio",
        "upper_body_support.min_active_span_upper_body_support_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.manual_visibility,
        "min_active_span_any_hand_available_frame_ratio",
        "manual_visibility.min_active_span_any_hand_available_frame_ratio",
    )
    _require_nonincreasing(
        config.manual_visibility,
        "max_active_span_any_hand_unavailable_run_ratio",
        "manual_visibility.max_active_span_any_hand_unavailable_run_ratio",
    )
    _require_nondecreasing(
        config.non_manual_visibility,
        "min_active_span_face_available_frame_ratio",
        "non_manual_visibility.min_active_span_face_available_frame_ratio",
    )
    _require_nonincreasing(
        config.non_manual_visibility,
        "max_active_span_face_unavailable_run_ratio",
        "non_manual_visibility.max_active_span_face_unavailable_run_ratio",
    )
    _require_nondecreasing(
        config.confidence,
        "min_active_span_body_available_mean_confidence",
        "confidence.min_active_span_body_available_mean_confidence",
    )
    _require_nondecreasing(
        config.confidence,
        "min_active_span_any_hand_available_mean_confidence",
        "confidence.min_active_span_any_hand_available_mean_confidence",
    )
    _require_nonincreasing(
        config.kinematic_naturalness,
        "max_active_span_abrupt_motion_frame_ratio",
        "kinematic_naturalness.max_active_span_abrupt_motion_frame_ratio",
    )
    _require_nonincreasing(
        config.kinematic_naturalness,
        "max_active_span_discontinuity_frame_ratio",
        "kinematic_naturalness.max_active_span_discontinuity_frame_ratio",
    )
    _require_nonincreasing(
        config.kinematic_naturalness,
        "max_active_span_frozen_run_ratio",
        "kinematic_naturalness.max_active_span_frozen_run_ratio",
    )
    _require_nonincreasing(
        config.tracking_quality,
        "max_tracked_target_missing_frame_ratio",
        "tracking_quality.max_tracked_target_missing_frame_ratio",
    )
    _require_nonincreasing(
        config.tracking_quality,
        "max_person_tracking_continuity_break_ratio",
        "tracking_quality.max_person_tracking_continuity_break_ratio",
    )
    _require_nonincreasing(
        config.tracking_quality,
        "max_person_tracking_reanchor_ratio",
        "tracking_quality.max_person_tracking_reanchor_ratio",
    )
    _require_nondecreasing(
        config.manual_detail,
        "min_active_span_representative_hand_landmark_coverage_ratio",
        "manual_detail.min_active_span_representative_hand_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.manual_detail,
        "min_active_span_representative_hand_fingertip_coverage_ratio",
        "manual_detail.min_active_span_representative_hand_fingertip_coverage_ratio",
    )
    _require_nondecreasing(
        config.manual_detail,
        "min_active_span_representative_hand_distal_chain_coverage_ratio",
        "manual_detail.min_active_span_representative_hand_distal_chain_coverage_ratio",
    )
    _require_nonincreasing(
        config.manual_detail,
        "max_active_span_representative_hand_detail_dropout_run_ratio",
        "manual_detail.max_active_span_representative_hand_detail_dropout_run_ratio",
    )
    _require_nondecreasing(
        config.non_manual_quality,
        "min_active_span_face_landmark_coverage_ratio",
        "non_manual_quality.min_active_span_face_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.non_manual_quality,
        "min_active_span_upper_face_landmark_coverage_ratio",
        "non_manual_quality.min_active_span_upper_face_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.non_manual_quality,
        "min_active_span_lower_face_landmark_coverage_ratio",
        "non_manual_quality.min_active_span_lower_face_landmark_coverage_ratio",
    )
    _require_nonincreasing(
        config.non_manual_quality,
        "max_active_span_face_detail_dropout_run_ratio",
        "non_manual_quality.max_active_span_face_detail_dropout_run_ratio",
    )
    _require_nondecreasing(
        config.non_manual_quality,
        "min_active_span_manual_face_overlap_frame_ratio",
        "non_manual_quality.min_active_span_manual_face_overlap_frame_ratio",
    )
    _require_nonincreasing(
        config.geometry,
        "max_active_span_upper_body_bone_length_outlier_frame_ratio",
        "geometry.max_active_span_upper_body_bone_length_outlier_frame_ratio",
    )
    _require_nonincreasing(
        config.geometry,
        "max_active_span_representative_hand_bone_length_outlier_frame_ratio",
        "geometry.max_active_span_representative_hand_bone_length_outlier_frame_ratio",
    )
    _require_nonincreasing(
        config.geometry,
        "max_active_span_cross_channel_scale_outlier_frame_ratio",
        "geometry.max_active_span_cross_channel_scale_outlier_frame_ratio",
    )


def _require_nondecreasing(
    thresholds_by_level: dict[FilterLevel, ThresholdT] | object,
    attr_name: str,
    label: str,
) -> None:
    values = _ordered_values(thresholds_by_level, attr_name)
    if values != sorted(values):
        raise ValueError(
            f"{label} must be monotonic non-decreasing across loose/clean/tight, got {values}"
        )


def _require_nonincreasing(
    thresholds_by_level: dict[FilterLevel, ThresholdT] | object,
    attr_name: str,
    label: str,
) -> None:
    values = _ordered_values(thresholds_by_level, attr_name)
    if values != sorted(values, reverse=True):
        raise ValueError(
            f"{label} must be monotonic non-increasing across loose/clean/tight, got {values}"
        )


def _ordered_values(thresholds_by_level: object, attr_name: str) -> list[float | int]:
    return [
        getattr(thresholds_by_level[level], attr_name)  # type: ignore[index]
        for level in _LEVEL_ORDER
    ]
