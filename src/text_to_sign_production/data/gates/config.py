"""Loading orchestration for structural gates configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from text_to_sign_production.data._shared.validate import (
    float_value,
    int_value,
    is_positive_finite_number,
    is_unit_interval,
    key_delta,
    non_string_keys,
    require_string_key_mapping,
)
from text_to_sign_production.data.gates.types import (
    ChannelGateConfig,
    ChannelPresenceGateConfig,
    GatesConfig,
    IntegrityGateConfig,
    TextSanityGateConfig,
    TrackingIntegrityGateConfig,
)
from text_to_sign_production.data.gates.validate import validate_gates_config

_TOP_LEVEL_KEYS = (
    "integrity",
    "tracking_integrity",
    "channel_presence",
    "text_sanity",
)
_INTEGRITY_KEYS = (
    "min_valid_frames",
    "max_out_of_bounds_ratio",
    "min_num_frames",
    "min_duration_seconds",
)
_TRACKING_INTEGRITY_KEYS = (
    "max_tracked_target_missing_frame_ratio",
    "max_zeroed_canonical_joint_frame_ratio",
    "max_person_tracking_continuity_break_ratio",
    "max_person_tracking_reanchor_ratio",
)
_CHANNEL_PRESENCE_KEYS = (
    "min_any_hand_nonzero_frames",
    "channels",
)
_TEXT_SANITY_KEYS = (
    "min_character_count",
    "min_token_count",
)
_CHANNEL_CONFIG_KEYS = ("min_nonzero_frames",)
_CHANNEL_GATE_CONFIG_CHANNELS = ("body", "face")


def load_gates_config(path: Path) -> GatesConfig:
    """Load, parse, and validate the structural gates configuration."""
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
    """Parse a raw gates configuration mapping into typed config objects."""
    data = _require_mapping(value, "Gates config")
    _require_exact_keys(data, _TOP_LEVEL_KEYS, "Gates config")

    integrity_data = _require_mapping(data["integrity"], "integrity")
    _require_exact_keys(integrity_data, _INTEGRITY_KEYS, "integrity")
    min_valid = _require_positive_int(
        integrity_data["min_valid_frames"],
        "integrity.min_valid_frames",
    )
    max_oob = _require_ratio(
        integrity_data["max_out_of_bounds_ratio"],
        "integrity.max_out_of_bounds_ratio",
    )
    min_num_frames = _require_positive_int(
        integrity_data["min_num_frames"],
        "integrity.min_num_frames",
    )
    min_duration_seconds = _require_positive_float(
        integrity_data["min_duration_seconds"],
        "integrity.min_duration_seconds",
    )

    tracking_data = _require_mapping(data["tracking_integrity"], "tracking_integrity")
    _require_exact_keys(tracking_data, _TRACKING_INTEGRITY_KEYS, "tracking_integrity")

    channel_presence_data = _require_mapping(data["channel_presence"], "channel_presence")
    _require_exact_keys(channel_presence_data, _CHANNEL_PRESENCE_KEYS, "channel_presence")
    min_any_hand = _require_nonnegative_int(
        channel_presence_data["min_any_hand_nonzero_frames"],
        "channel_presence.min_any_hand_nonzero_frames",
    )
    if min_any_hand > min_valid:
        raise ValueError(
            "channel_presence.min_any_hand_nonzero_frames must be <= "
            "integrity.min_valid_frames, "
            f"got {min_any_hand}>{min_valid}."
        )

    raw_channels = _require_mapping(channel_presence_data["channels"], "channel_presence.channels")
    _require_exact_keys(raw_channels, _CHANNEL_GATE_CONFIG_CHANNELS, "channel_presence.channels")

    parsed_channels: dict[str, ChannelGateConfig] = {}
    for channel in _CHANNEL_GATE_CONFIG_CHANNELS:
        ch_data = _require_mapping(
            raw_channels[channel],
            f"channel_presence.channels.{channel}",
        )
        _require_exact_keys(
            ch_data,
            _CHANNEL_CONFIG_KEYS,
            f"channel_presence.channels.{channel}",
        )

        min_nonzero = _require_nonnegative_int(
            ch_data["min_nonzero_frames"],
            f"channel_presence.channels.{channel}.min_nonzero_frames",
        )
        if min_nonzero > min_valid:
            raise ValueError(
                f"channel_presence.channels.{channel}.min_nonzero_frames must be <= "
                "integrity.min_valid_frames, "
                f"got {min_nonzero}>{min_valid}."
            )
        parsed_channels[channel] = ChannelGateConfig(min_nonzero_frames=min_nonzero)

    text_sanity_data = _require_mapping(data["text_sanity"], "text_sanity")
    _require_exact_keys(text_sanity_data, _TEXT_SANITY_KEYS, "text_sanity")

    return GatesConfig(
        integrity=IntegrityGateConfig(
            min_valid_frames=min_valid,
            max_out_of_bounds_ratio=max_oob,
            min_num_frames=min_num_frames,
            min_duration_seconds=min_duration_seconds,
        ),
        tracking_integrity=TrackingIntegrityGateConfig(
            max_tracked_target_missing_frame_ratio=_require_ratio(
                tracking_data["max_tracked_target_missing_frame_ratio"],
                "tracking_integrity.max_tracked_target_missing_frame_ratio",
            ),
            max_zeroed_canonical_joint_frame_ratio=_require_ratio(
                tracking_data["max_zeroed_canonical_joint_frame_ratio"],
                "tracking_integrity.max_zeroed_canonical_joint_frame_ratio",
            ),
            max_person_tracking_continuity_break_ratio=_require_ratio(
                tracking_data["max_person_tracking_continuity_break_ratio"],
                "tracking_integrity.max_person_tracking_continuity_break_ratio",
            ),
            max_person_tracking_reanchor_ratio=_require_ratio(
                tracking_data["max_person_tracking_reanchor_ratio"],
                "tracking_integrity.max_person_tracking_reanchor_ratio",
            ),
        ),
        channel_presence=ChannelPresenceGateConfig(
            min_any_hand_nonzero_frames=min_any_hand,
            channels=parsed_channels,
        ),
        text_sanity=TextSanityGateConfig(
            min_character_count=_require_nonnegative_int(
                text_sanity_data["min_character_count"],
                "text_sanity.min_character_count",
            ),
            min_token_count=_require_nonnegative_int(
                text_sanity_data["min_token_count"],
                "text_sanity.min_token_count",
            ),
        ),
    )


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    mapping = require_string_key_mapping(value)
    if mapping is None:
        if isinstance(value, Mapping):
            bad_keys = non_string_keys(value)
            if bad_keys:
                raise ValueError(f"{name} contains non-string key {bad_keys[0]!r}.")
        raise ValueError(f"{name} must be a YAML mapping.")
    return mapping


def _require_exact_keys(
    value: Mapping[str, object],
    expected_keys: tuple[str, ...],
    name: str,
) -> None:
    missing, unknown = key_delta(value, expected_keys)
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
    parsed = int_value(value)
    if parsed is None:
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return parsed


def _require_nonnegative_int(value: object, name: str) -> int:
    parsed = int_value(value)
    if parsed is None:
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return parsed


def _require_ratio(value: object, name: str) -> float:
    number = float_value(value)
    if number is None:
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    if not is_unit_interval(number):
        raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}.")
    return number


def _require_positive_float(value: object, name: str) -> float:
    number = float_value(value)
    if number is None:
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    if not is_positive_finite_number(number):
        raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    return number
