"""Top-level loading and dispatch for configs/data/filters.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml

from text_to_sign_production.data.tiers._shared.parsing import require_exact_keys, require_mapping
from text_to_sign_production.data.tiers.confidence import parse_confidence_thresholds
from text_to_sign_production.data.tiers.coverage import parse_coverage_thresholds
from text_to_sign_production.data.tiers.face import parse_face_thresholds
from text_to_sign_production.data.tiers.hand import parse_hand_thresholds
from text_to_sign_production.data.tiers.length import parse_length_thresholds
from text_to_sign_production.data.tiers.oob import parse_oob_thresholds
from text_to_sign_production.data.tiers.text import parse_text_thresholds
from text_to_sign_production.data.tiers.types import BindingTierFamily, FilterConfig, FilterLevel

ThresholdT = TypeVar("ThresholdT")

_FAMILY_KEYS = tuple(family.value for family in BindingTierFamily)
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
        coverage=parse_coverage_thresholds(families["coverage"]),
        hand=parse_hand_thresholds(families["hand"]),
        face=parse_face_thresholds(families["face"]),
        confidence=parse_confidence_thresholds(families["confidence"]),
        text=parse_text_thresholds(families["text"]),
        length=parse_length_thresholds(families["length"]),
    )
    validate_filter_config(config)
    return config


def validate_filter_config(config: FilterConfig) -> None:
    """Validate cross-level strictness for binding tier family thresholds."""
    _require_nonincreasing(
        config.oob,
        "max_out_of_bounds_ratio",
        "oob.max_out_of_bounds_ratio",
    )
    _require_nondecreasing(
        config.coverage,
        "min_body_landmark_coverage_ratio",
        "coverage.min_body_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.coverage,
        "min_any_hand_landmark_coverage_ratio",
        "coverage.min_any_hand_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.coverage,
        "min_face_landmark_coverage_ratio",
        "coverage.min_face_landmark_coverage_ratio",
    )
    _require_nondecreasing(
        config.hand,
        "min_any_hand_available_frame_ratio",
        "hand.min_any_hand_available_frame_ratio",
    )
    _require_nonincreasing(
        config.hand,
        "max_any_hand_unavailable_run_ratio",
        "hand.max_any_hand_unavailable_run_ratio",
    )
    _require_nondecreasing(
        config.face,
        "min_face_available_frame_ratio",
        "face.min_face_available_frame_ratio",
    )
    _require_nondecreasing(
        config.confidence,
        "min_body_mean_confidence",
        "confidence.min_body_mean_confidence",
    )
    _require_nondecreasing(
        config.confidence,
        "min_left_hand_mean_confidence",
        "confidence.min_left_hand_mean_confidence",
    )
    _require_nondecreasing(
        config.confidence,
        "min_right_hand_mean_confidence",
        "confidence.min_right_hand_mean_confidence",
    )
    _require_nondecreasing(
        config.confidence,
        "min_face_mean_confidence",
        "confidence.min_face_mean_confidence",
    )
    _require_nondecreasing(config.text, "min_character_count", "text.min_character_count")
    _require_nondecreasing(config.text, "min_token_count", "text.min_token_count")
    _require_nondecreasing(config.length, "min_num_frames", "length.min_num_frames")
    _require_nondecreasing(
        config.length,
        "min_duration_seconds",
        "length.min_duration_seconds",
    )


def _require_nondecreasing(
    thresholds_by_level: dict[FilterLevel, ThresholdT] | object,
    attr_name: str,
    label: str,
) -> None:
    values = _ordered_values(thresholds_by_level, attr_name)
    if values != sorted(values):
        raise ValueError(
            f"{label} must be monotonic non-decreasing across loose/clean/tight, "
            f"got {values}"
        )


def _require_nonincreasing(
    thresholds_by_level: dict[FilterLevel, ThresholdT] | object,
    attr_name: str,
    label: str,
) -> None:
    values = _ordered_values(thresholds_by_level, attr_name)
    if values != sorted(values, reverse=True):
        raise ValueError(
            f"{label} must be monotonic non-increasing across loose/clean/tight, "
            f"got {values}"
        )


def _ordered_values(thresholds_by_level: object, attr_name: str) -> list[float | int]:
    return [
        getattr(thresholds_by_level[level], attr_name)  # type: ignore[index]
        for level in _LEVEL_ORDER
    ]
