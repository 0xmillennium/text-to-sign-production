"""Configuration models for structural gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ChannelGateConfig:
    """Structural configuration for a specific canonical channel."""

    min_nonzero_frames: int


@dataclass(frozen=True, slots=True)
class GatesConfig:
    """Configuration for all structural gates."""

    min_valid_frames: int
    max_out_of_bounds_ratio: float
    channel_configs: dict[str, ChannelGateConfig]


def load_gates_config(path: Path) -> GatesConfig:
    """Load and validate the structural gates configuration."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Gates config must be a YAML object.")

    if "min_valid_frames" not in data:
        raise ValueError("Missing 'min_valid_frames' in config.")
    min_valid = data["min_valid_frames"]
    if not isinstance(min_valid, int):
        raise ValueError(f"'min_valid_frames' must be an integer, got {type(min_valid)}")

    if "max_out_of_bounds_ratio" not in data:
        raise ValueError("Missing 'max_out_of_bounds_ratio' in config.")
    max_oob = data["max_out_of_bounds_ratio"]
    if not isinstance(max_oob, (int, float)):
        raise ValueError(f"'max_out_of_bounds_ratio' must be numeric, got {type(max_oob)}")

    if "channels" not in data:
        raise ValueError("Missing 'channels' in config.")
    raw_channels = data["channels"]
    if not isinstance(raw_channels, dict):
        raise ValueError("'channels' must be a mapping.")

    from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS

    channel_configs = {}
    for channel in CANONICAL_POSE_CHANNELS:
        if channel not in raw_channels:
            raise ValueError(f"Missing configuration for canonical channel '{channel}'.")
        ch_data = raw_channels[channel]
        if not isinstance(ch_data, dict):
            raise ValueError(f"Channel config for '{channel}' must be a mapping.")
        if "min_nonzero_frames" not in ch_data:
            raise ValueError(f"Missing 'min_nonzero_frames' for channel '{channel}'.")

        min_nonzero = ch_data["min_nonzero_frames"]
        if not isinstance(min_nonzero, int):
            raise ValueError(f"'min_nonzero_frames' for '{channel}' must be an integer.")

        channel_configs[channel] = ChannelGateConfig(min_nonzero_frames=min_nonzero)

    unknown_channels = set(raw_channels.keys()) - set(CANONICAL_POSE_CHANNELS)
    if unknown_channels:
        raise ValueError(f"Unknown channels in config: {unknown_channels}")

    config = GatesConfig(
        min_valid_frames=min_valid,
        max_out_of_bounds_ratio=float(max_oob),
        channel_configs=channel_configs,
    )

    from text_to_sign_production.data.gates.validate import validate_gates_config

    issues = validate_gates_config(config)
    if issues:
        raise ValueError(f"Invalid gates config: {issues}")

    return config
