"""Configuration models for structural gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS

_TOP_LEVEL_KEYS = ("min_valid_frames", "max_out_of_bounds_ratio", "channels")
_CHANNEL_CONFIG_KEYS = ("min_nonzero_frames",)


@dataclass(frozen=True, slots=True)
class ChannelGateConfig:
    """Structural configuration for a specific canonical channel."""

    min_nonzero_frames: int


@dataclass(frozen=True, slots=True)
class GatesConfig:
    """Configuration for all structural gates."""

    min_valid_frames: int
    max_out_of_bounds_ratio: float
    channel_configs: Mapping[str, ChannelGateConfig]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_configs",
            MappingProxyType(dict(self.channel_configs)),
        )


def load_gates_config(path: Path) -> GatesConfig:
    """Load and validate the structural gates configuration."""
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    data = _require_mapping(loaded, "Gates config")
    _require_exact_keys(data, _TOP_LEVEL_KEYS, "Gates config")

    min_valid = _require_positive_int(data["min_valid_frames"], "min_valid_frames")
    max_oob = _require_ratio(data["max_out_of_bounds_ratio"], "max_out_of_bounds_ratio")

    raw_channels = _require_mapping(data["channels"], "channels")
    _require_exact_keys(raw_channels, CANONICAL_POSE_CHANNELS, "channels")

    parsed_channels: dict[str, ChannelGateConfig] = {}
    for channel in CANONICAL_POSE_CHANNELS:
        ch_data = _require_mapping(raw_channels[channel], f"channels.{channel}")
        _require_exact_keys(ch_data, _CHANNEL_CONFIG_KEYS, f"channels.{channel}")

        min_nonzero = _require_nonnegative_int(
            ch_data["min_nonzero_frames"],
            f"channels.{channel}.min_nonzero_frames",
        )
        if min_nonzero > min_valid:
            raise ValueError(
                f"channels.{channel}.min_nonzero_frames must be <= min_valid_frames, "
                f"got {min_nonzero}>{min_valid}."
            )
        parsed_channels[channel] = ChannelGateConfig(min_nonzero_frames=min_nonzero)

    config = GatesConfig(
        min_valid_frames=min_valid,
        max_out_of_bounds_ratio=max_oob,
        channel_configs=parsed_channels,
    )

    from text_to_sign_production.data.gates.validate import validate_gates_config

    issues = validate_gates_config(config)
    if issues:
        raise ValueError(f"Invalid gates config: {issues}")

    return config


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping.")
    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{name} contains non-string key {key!r}.")
    return cast(Mapping[str, object], value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected_keys: tuple[str, ...],
    name: str,
) -> None:
    actual = set(value)
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
            f"{name} keys must be exactly {list(expected_keys)} ({', '.join(details)})."
        )


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return value


def _require_ratio(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}.")
    return number
