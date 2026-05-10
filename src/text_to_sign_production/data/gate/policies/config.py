"""Typed samples admission-gate configuration parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from text_to_sign_production.data.gate.policies.validate import (
    validate_gates_config,
)

DEFAULT_GATES_CONFIG_PATH = Path("configs/data/gates.yaml")


@dataclass(frozen=True, slots=True)
class SourceGateThresholds:
    """Source admission-gate thresholds."""

    min_character_count: int
    min_token_count: int
    fail_on_source_issues: bool = True


@dataclass(frozen=True, slots=True)
class FramesGateThresholds:
    """Frame and duration admission-gate thresholds."""

    min_frame_count: int
    min_valid_frame_count: int
    min_duration_seconds: float
    max_duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BodyGateThresholds:
    """Body-channel admission-gate thresholds."""

    min_body_nonzero_frames: int | None = None
    min_body_available_frame_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class HandGateThresholds:
    """Manual-channel admission-gate thresholds."""

    min_any_hand_nonzero_frames: int | None = None
    min_any_hand_available_frame_ratio: float | None = None
    max_tracked_target_missing_frame_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class FaceGateThresholds:
    """Face-channel admission-gate thresholds."""

    min_face_nonzero_frames: int | None = None
    min_face_available_frame_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class GatesConfig:
    """Typed configuration for all samples admission gates."""

    source: SourceGateThresholds
    frames: FramesGateThresholds
    body: BodyGateThresholds
    hand: HandGateThresholds
    face: FaceGateThresholds


def load_gates_config(path: Path = DEFAULT_GATES_CONFIG_PATH) -> GatesConfig:
    """Load and parse gate configuration from YAML."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid gates YAML: {exc}") from exc
    config = parse_gates_config_mapping(loaded)
    issues = validate_gates_config(config)
    if issues:
        raise ValueError(f"Invalid gates config: {issues}")
    return config


def parse_gates_config_mapping(value: object) -> GatesConfig:
    """Parse raw gate configuration mapping into typed threshold sections."""
    data = _require_mapping(value, "Gates config")
    if "integrity" in data:
        return _parse_legacy_config(data)
    return _parse_sectioned_config(data)


def _parse_legacy_config(data: Mapping[str, object]) -> GatesConfig:
    integrity = _require_mapping(data["integrity"], "integrity")
    tracking = _require_mapping(data["tracking_integrity"], "tracking_integrity")
    channel_presence = _require_mapping(data["channel_presence"], "channel_presence")
    text_sanity = _require_mapping(data["text_sanity"], "text_sanity")
    channels = _require_mapping(channel_presence["channels"], "channel_presence.channels")
    body = _require_mapping(channels["body"], "channel_presence.channels.body")
    face = _require_mapping(channels["face"], "channel_presence.channels.face")
    return GatesConfig(
        source=SourceGateThresholds(
            min_character_count=_require_nonnegative_int(
                text_sanity["min_character_count"],
                "text_sanity.min_character_count",
            ),
            min_token_count=_require_nonnegative_int(
                text_sanity["min_token_count"],
                "text_sanity.min_token_count",
            ),
        ),
        frames=FramesGateThresholds(
            min_frame_count=_require_positive_int(
                integrity["min_num_frames"],
                "integrity.min_num_frames",
            ),
            min_valid_frame_count=_require_positive_int(
                integrity["min_valid_frames"],
                "integrity.min_valid_frames",
            ),
            min_duration_seconds=_require_positive_float(
                integrity["min_duration_seconds"],
                "integrity.min_duration_seconds",
            ),
        ),
        body=BodyGateThresholds(
            min_body_nonzero_frames=_require_nonnegative_int(
                body["min_nonzero_frames"],
                "channel_presence.channels.body.min_nonzero_frames",
            ),
        ),
        hand=HandGateThresholds(
            min_any_hand_nonzero_frames=_require_nonnegative_int(
                channel_presence["min_any_hand_nonzero_frames"],
                "channel_presence.min_any_hand_nonzero_frames",
            ),
            max_tracked_target_missing_frame_ratio=_require_ratio(
                tracking["max_tracked_target_missing_frame_ratio"],
                "tracking_integrity.max_tracked_target_missing_frame_ratio",
            ),
        ),
        face=FaceGateThresholds(
            min_face_nonzero_frames=_require_nonnegative_int(
                face["min_nonzero_frames"],
                "channel_presence.channels.face.min_nonzero_frames",
            ),
        ),
    )


def _parse_sectioned_config(data: Mapping[str, object]) -> GatesConfig:
    source = _require_mapping(data["source"], "source")
    frames = _require_mapping(data["frames"], "frames")
    body = _require_mapping(data["body"], "body")
    hand = _require_mapping(data["hand"], "hand")
    face = _require_mapping(data["face"], "face")
    return GatesConfig(
        source=SourceGateThresholds(
            min_character_count=_require_nonnegative_int(
                source["min_character_count"],
                "source.min_character_count",
            ),
            min_token_count=_require_nonnegative_int(
                source["min_token_count"],
                "source.min_token_count",
            ),
            fail_on_source_issues=_optional_bool(source.get("fail_on_source_issues"), True),
        ),
        frames=FramesGateThresholds(
            min_frame_count=_require_positive_int(
                frames["min_frame_count"],
                "frames.min_frame_count",
            ),
            min_valid_frame_count=_require_positive_int(
                frames["min_valid_frame_count"],
                "frames.min_valid_frame_count",
            ),
            min_duration_seconds=_require_positive_float(
                frames["min_duration_seconds"],
                "frames.min_duration_seconds",
            ),
            max_duration_seconds=_optional_positive_float(
                frames.get("max_duration_seconds"),
                "frames.max_duration_seconds",
            ),
        ),
        body=BodyGateThresholds(
            min_body_nonzero_frames=_optional_nonnegative_int(
                body.get("min_body_nonzero_frames"),
                "body.min_body_nonzero_frames",
            ),
            min_body_available_frame_ratio=_optional_ratio(
                body.get("min_body_available_frame_ratio"),
                "body.min_body_available_frame_ratio",
            ),
        ),
        hand=HandGateThresholds(
            min_any_hand_nonzero_frames=_optional_nonnegative_int(
                hand.get("min_any_hand_nonzero_frames"),
                "hand.min_any_hand_nonzero_frames",
            ),
            min_any_hand_available_frame_ratio=_optional_ratio(
                hand.get("min_any_hand_available_frame_ratio"),
                "hand.min_any_hand_available_frame_ratio",
            ),
            max_tracked_target_missing_frame_ratio=_optional_ratio(
                hand.get("max_tracked_target_missing_frame_ratio"),
                "hand.max_tracked_target_missing_frame_ratio",
            ),
        ),
        face=FaceGateThresholds(
            min_face_nonzero_frames=_optional_nonnegative_int(
                face.get("min_face_nonzero_frames"),
                "face.min_face_nonzero_frames",
            ),
            min_face_available_frame_ratio=_optional_ratio(
                face.get("min_face_available_frame_ratio"),
                "face.min_face_available_frame_ratio",
            ),
        ),
    )


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return value


def _require_positive_int(value: object, name: str) -> int:
    parsed = _int_value(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _require_nonnegative_int(value: object, name: str) -> int:
    parsed = _int_value(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, name)


def _require_ratio(value: object, name: str) -> float:
    parsed = _float_value(value)
    if parsed is None or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be within [0, 1].")
    return parsed


def _optional_ratio(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _require_ratio(value, name)


def _require_positive_float(value: object, name: str) -> float:
    parsed = _float_value(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _optional_positive_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _require_positive_float(value, name)


def _optional_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean config value, got {value!r}.")
    return value


def _int_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


__all__ = [
    "BodyGateThresholds",
    "DEFAULT_GATES_CONFIG_PATH",
    "FaceGateThresholds",
    "FramesGateThresholds",
    "GatesConfig",
    "HandGateThresholds",
    "SourceGateThresholds",
    "load_gates_config",
    "parse_gates_config_mapping",
]
