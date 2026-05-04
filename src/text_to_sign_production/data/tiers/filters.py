"""Strict loading and validation for configs/data/filters.yaml."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias, TypeVar, cast

import yaml

from text_to_sign_production.data.tiers.types import (
    ConfidenceThresholds,
    FaceThresholds,
    FilterConfig,
    FilterLevel,
    HandThresholds,
    LengthThresholds,
    OobThresholds,
    TextThresholds,
    ValidThresholds,
)

ThresholdT = TypeVar("ThresholdT")
FamilyLevels: TypeAlias = dict[FilterLevel, ThresholdT]

_LEVEL_KEYS = tuple(level.value for level in FilterLevel)
_FAMILY_KEYS = ("oob", "hand", "face", "valid", "confidence", "text", "length")


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
    root = _require_mapping(payload, "filters root")
    _require_exact_keys(root, ("families",), "filters root")

    families = _require_mapping(root["families"], "families")
    _require_exact_keys(families, _FAMILY_KEYS, "families")

    return FilterConfig(
        oob=_parse_family_levels(
            families["oob"],
            "oob",
            ("max_out_of_bounds_ratio",),
            lambda values: OobThresholds(
                max_out_of_bounds_ratio=_require_ratio(
                    values["max_out_of_bounds_ratio"],
                    "oob.max_out_of_bounds_ratio",
                )
            ),
        ),
        hand=_parse_family_levels(
            families["hand"],
            "hand",
            ("min_any_hand_nonzero_frame_ratio",),
            lambda values: HandThresholds(
                min_any_hand_nonzero_frame_ratio=_require_ratio(
                    values["min_any_hand_nonzero_frame_ratio"],
                    "hand.min_any_hand_nonzero_frame_ratio",
                )
            ),
        ),
        face=_parse_family_levels(
            families["face"],
            "face",
            ("min_face_nonzero_frame_ratio", "max_face_missing_frame_ratio"),
            lambda values: FaceThresholds(
                min_face_nonzero_frame_ratio=_require_ratio(
                    values["min_face_nonzero_frame_ratio"],
                    "face.min_face_nonzero_frame_ratio",
                ),
                max_face_missing_frame_ratio=_require_ratio(
                    values["max_face_missing_frame_ratio"],
                    "face.max_face_missing_frame_ratio",
                ),
            ),
        ),
        valid=_parse_family_levels(
            families["valid"],
            "valid",
            ("min_valid_frame_ratio", "max_zeroed_canonical_joint_frame_ratio"),
            lambda values: ValidThresholds(
                min_valid_frame_ratio=_require_ratio(
                    values["min_valid_frame_ratio"],
                    "valid.min_valid_frame_ratio",
                ),
                max_zeroed_canonical_joint_frame_ratio=_require_ratio(
                    values["max_zeroed_canonical_joint_frame_ratio"],
                    "valid.max_zeroed_canonical_joint_frame_ratio",
                ),
            ),
        ),
        confidence=_parse_family_levels(
            families["confidence"],
            "confidence",
            ("min_overall_mean_confidence", "min_overall_nonzero_confidence_ratio"),
            lambda values: ConfidenceThresholds(
                min_overall_mean_confidence=_require_ratio(
                    values["min_overall_mean_confidence"],
                    "confidence.min_overall_mean_confidence",
                ),
                min_overall_nonzero_confidence_ratio=_require_ratio(
                    values["min_overall_nonzero_confidence_ratio"],
                    "confidence.min_overall_nonzero_confidence_ratio",
                ),
            ),
        ),
        text=_parse_family_levels(
            families["text"],
            "text",
            ("min_character_count", "min_token_count"),
            lambda values: TextThresholds(
                min_character_count=_require_nonnegative_int(
                    values["min_character_count"],
                    "text.min_character_count",
                ),
                min_token_count=_require_nonnegative_int(
                    values["min_token_count"],
                    "text.min_token_count",
                ),
            ),
        ),
        length=_parse_family_levels(
            families["length"],
            "length",
            ("min_num_frames", "min_duration_seconds"),
            lambda values: LengthThresholds(
                min_num_frames=_require_positive_int(
                    values["min_num_frames"],
                    "length.min_num_frames",
                ),
                min_duration_seconds=_require_positive_float(
                    values["min_duration_seconds"],
                    "length.min_duration_seconds",
                ),
            ),
        ),
    )


def _parse_family_levels(
    payload: object,
    family_name: str,
    threshold_keys: tuple[str, ...],
    factory: Callable[[Mapping[str, object]], ThresholdT],
) -> FamilyLevels[ThresholdT]:
    levels = _require_mapping(payload, family_name)
    _require_exact_keys(levels, _LEVEL_KEYS, family_name)

    parsed: FamilyLevels[ThresholdT] = {}
    for level in FilterLevel:
        level_payload = _require_mapping(levels[level.value], f"{family_name}.{level.value}")
        _require_exact_keys(level_payload, threshold_keys, f"{family_name}.{level.value}")
        parsed[level] = factory(level_payload)
    return parsed


def _require_mapping(payload: object, name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    for key in payload:
        if not isinstance(key, str):
            raise ValueError(f"{name} contains non-string key {key!r}")
    return cast(Mapping[str, object], payload)


def _require_exact_keys(
    payload: Mapping[str, object],
    expected_keys: tuple[str, ...],
    name: str,
) -> None:
    actual = set(payload)
    expected = set(expected_keys)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"{name} keys must be exactly {list(expected_keys)} ({', '.join(details)})"
        )


def _require_ratio(value: object, name: str) -> float:
    number = _require_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}")
    return number


def _require_positive_float(value: object, name: str) -> float:
    number = _require_number(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value
