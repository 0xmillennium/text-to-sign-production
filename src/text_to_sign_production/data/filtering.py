"""Minimal structural filtering for normalized candidate samples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    CORE_CHANNELS,
    LEGACY_REQUIRED_CORE_CHANNELS,
)
from .jsonl import count_jsonl_records, iter_jsonl, write_jsonl
from .schemas import NormalizedManifestEntry
from .utils import (
    ensure_directory,
    iter_with_progress,
    repo_relative_path,
    resolve_data_root,
    utc_timestamp,
)

FILTER_CONFIG_SCHEMA_VERSION = 2
SUPPORTED_FILTER_CONFIG_SCHEMA_VERSIONS = (1, FILTER_CONFIG_SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """The structural filtering policy for Dataset Build export."""

    schema_version: int
    require_nonempty_text: bool
    require_positive_duration: bool
    require_keypoints_dir: bool
    require_frames: bool
    drop_on_sample_parse_error: bool
    require_at_least_one_valid_frame: bool
    minimum_nonzero_frames_per_core_channel: int
    required_all_core_channels: tuple[str, ...]
    required_any_core_channel_groups: tuple[tuple[str, ...], ...]


def _require_boolean_config_value(payload: dict[str, Any], field_name: str) -> bool:
    value = _require_config_field(payload, field_name)
    if not isinstance(value, bool):
        raise ValueError(
            f"Filter config field {field_name!r} must be a boolean, got {type(value).__name__}."
        )
    return value


def _require_nonnegative_int_config_value(payload: dict[str, Any], field_name: str) -> int:
    value = _require_config_field(payload, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "Filter config field "
            f"{field_name!r} must be a non-negative integer, got {type(value).__name__}."
        )
    if value < 0:
        raise ValueError(
            f"Filter config field {field_name!r} must be a non-negative integer, got {value}."
        )
    return int(value)


def _validate_core_channel_list(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"Filter config field {field_name!r} must be a list of core channel names."
        )
    if not value:
        raise ValueError(f"Filter config field {field_name!r} must not be empty.")

    seen_channels: set[str] = set()
    for channel in value:
        if not isinstance(channel, str):
            raise ValueError(
                f"Filter config field {field_name!r} must contain only core channel names."
            )
        if channel not in CORE_CHANNELS:
            expected = ", ".join(CORE_CHANNELS)
            raise ValueError(
                f"Filter config field {field_name!r} contains unsupported core channel "
                f"{channel!r}; expected one of: {expected}."
            )
        if channel in seen_channels:
            raise ValueError(
                f"Filter config field {field_name!r} contains duplicate core channel {channel!r}."
            )
        seen_channels.add(channel)

    return tuple(channel for channel in CORE_CHANNELS if channel in seen_channels)


def _require_core_channel_list_config_value(
    payload: dict[str, Any],
    field_name: str,
) -> tuple[str, ...]:
    return _validate_core_channel_list(_require_config_field(payload, field_name), field_name)


def _require_core_channel_groups_config_value(
    payload: dict[str, Any],
    field_name: str,
) -> tuple[tuple[str, ...], ...]:
    value = _require_config_field(payload, field_name)
    if not isinstance(value, list):
        raise ValueError(
            f"Filter config field {field_name!r} must be a list of core channel groups."
        )
    if not value:
        raise ValueError(f"Filter config field {field_name!r} must not be empty.")

    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(value):
        canonical_group = _validate_core_channel_list(
            group,
            f"{field_name}[{index}]",
        )
        if canonical_group in groups:
            raise ValueError(
                f"Filter config field {field_name!r} contains duplicate core channel group "
                f"{'|'.join(canonical_group)!r}."
            )
        groups.append(canonical_group)
    return tuple(groups)


def _require_config_field(payload: dict[str, Any], field_name: str) -> Any:
    if field_name not in payload:
        raise ValueError(f"Filter config is missing required field {field_name!r}.")
    return payload[field_name]


def _require_schema_version(payload: dict[str, Any], path: Path) -> int:
    schema_version = _require_config_field(payload, "schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(
            "Filter config field 'schema_version' must be an integer, "
            f"got {type(schema_version).__name__} in {path}."
        )
    if schema_version not in SUPPORTED_FILTER_CONFIG_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported filter config schema_version {schema_version!r} in {path}; "
            f"expected one of {SUPPORTED_FILTER_CONFIG_SCHEMA_VERSIONS}."
        )
    return int(schema_version)


def load_filter_config(path: Path) -> FilterConfig:
    """Load a supported filtering policy from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in filter config {path}.")
    schema_version = _require_schema_version(payload, path)
    required_all_core_channels = (
        LEGACY_REQUIRED_CORE_CHANNELS
        if schema_version == 1
        else _require_core_channel_list_config_value(payload, "required_all_core_channels")
    )
    required_any_core_channel_groups = (
        ()
        if schema_version == 1
        else _require_core_channel_groups_config_value(
            payload,
            "required_any_core_channel_groups",
        )
    )
    return FilterConfig(
        schema_version=schema_version,
        require_nonempty_text=_require_boolean_config_value(payload, "require_nonempty_text"),
        require_positive_duration=_require_boolean_config_value(
            payload, "require_positive_duration"
        ),
        require_keypoints_dir=_require_boolean_config_value(payload, "require_keypoints_dir"),
        require_frames=_require_boolean_config_value(payload, "require_frames"),
        drop_on_sample_parse_error=_require_boolean_config_value(
            payload, "drop_on_sample_parse_error"
        ),
        require_at_least_one_valid_frame=_require_boolean_config_value(
            payload, "require_at_least_one_valid_frame"
        ),
        minimum_nonzero_frames_per_core_channel=_require_nonnegative_int_config_value(
            payload,
            "minimum_nonzero_frames_per_core_channel",
        ),
        required_all_core_channels=required_all_core_channels,
        required_any_core_channel_groups=required_any_core_channel_groups,
    )


def _core_channel_is_usable(
    entry: NormalizedManifestEntry,
    config: FilterConfig,
    channel: str,
) -> bool:
    return (
        entry.core_channel_nonzero_frames.get(channel, 0)
        >= config.minimum_nonzero_frames_per_core_channel
    )


def _core_channel_group_drop_reason(group: tuple[str, ...]) -> str:
    return f"unusable_core_channel_group:{'|'.join(group)}"


def determine_drop_reasons(entry: NormalizedManifestEntry, config: FilterConfig) -> list[str]:
    """Return deterministic drop reasons for one normalized candidate."""

    drop_reasons: list[str] = []
    if config.require_nonempty_text and not entry.text.strip():
        drop_reasons.append("missing_text")
    if config.require_positive_duration and entry.end_time <= entry.start_time:
        drop_reasons.append("invalid_time_range")
    if config.require_keypoints_dir and entry.source_keypoints_dir is None:
        drop_reasons.append("missing_keypoints_dir")
    if config.require_frames and entry.num_frames <= 0:
        drop_reasons.append("zero_frames")
    if config.drop_on_sample_parse_error and entry.sample_parse_error is not None:
        drop_reasons.append("sample_parse_error")
    if config.require_at_least_one_valid_frame and entry.frame_valid_count <= 0:
        drop_reasons.append("all_frames_invalid")
    for channel in config.required_all_core_channels:
        if not _core_channel_is_usable(entry, config, channel):
            drop_reasons.append(f"unusable_core_channel:{channel}")
    for group in config.required_any_core_channel_groups:
        if not any(_core_channel_is_usable(entry, config, channel) for channel in group):
            drop_reasons.append(_core_channel_group_drop_reason(group))
    if (
        entry.sample_path is None
        and entry.source_keypoints_dir is not None
        and "sample_parse_error" not in drop_reasons
    ):
        drop_reasons.append("missing_sample_file")
    return drop_reasons


def filter_split(
    split: str,
    config: FilterConfig,
    *,
    normalized_manifest_path: Path | str,
    filtered_manifest_path: Path | str,
) -> tuple[list[NormalizedManifestEntry], dict[str, Any]]:
    """Filter one split-specific normalized manifest."""

    input_path = Path(normalized_manifest_path)
    output_path = Path(filtered_manifest_path)
    ensure_directory(output_path.parent)

    kept_entries: list[NormalizedManifestEntry] = []
    drop_reason_counter: Counter[str] = Counter()
    dropped_examples: list[dict[str, Any]] = []
    dropped_records: list[dict[str, Any]] = []
    total_entries = 0
    input_total = count_jsonl_records(input_path)

    for record in iter_with_progress(
        iter_jsonl(input_path),
        total=input_total,
        desc=f"Filter {split}",
        unit="records",
    ):
        total_entries += 1
        entry = NormalizedManifestEntry.from_record(record)
        drop_reasons = determine_drop_reasons(entry, config)
        if drop_reasons:
            drop_reason_counter.update(drop_reasons)
            dropped_records.append(
                {
                    **entry.to_record(),
                    "drop_reasons": list(drop_reasons),
                    "quality_tier": "dropped",
                }
            )
            if len(dropped_examples) < 20:
                dropped_examples.append(
                    {"sample_id": entry.sample_id, "drop_reasons": drop_reasons}
                )
            continue
        kept_entries.append(entry)

    write_jsonl(output_path, kept_entries)
    return kept_entries, {
        "split": split,
        "input_samples": total_entries,
        "kept_samples": len(kept_entries),
        "dropped_samples": total_entries - len(kept_entries),
        "drop_reason_counts": {
            key: drop_reason_counter[key] for key in sorted(drop_reason_counter)
        },
        "dropped_examples": dropped_examples,
        "dropped_records": dropped_records,
    }


def filter_all_splits(
    config_path: Path,
    *,
    splits: tuple[str, ...],
    normalized_manifests_root: Path | str,
    filtered_manifests_root: Path | str,
    data_root: Path | str,
) -> dict[str, Any]:
    """Apply structural filtering to every official split and return filter facts."""

    config = load_filter_config(config_path)
    resolved_data_root = resolve_data_root(data_root)
    resolved_normalized_manifests_root = Path(normalized_manifests_root)
    resolved_filtered_manifests_root = Path(filtered_manifests_root)
    resolved_config_path = Path(config_path).expanduser().resolve()
    try:
        report_config_path = repo_relative_path(resolved_config_path, data_root=resolved_data_root)
    except ValueError:
        report_config_path = config_path.name

    generated_at = utc_timestamp()
    report: dict[str, Any] = {
        "generated_at": generated_at,
        "config_path": report_config_path,
        "splits": {},
    }
    for split in splits:
        _, split_report = filter_split(
            split,
            config,
            normalized_manifest_path=(
                resolved_normalized_manifests_root / f"normalized_{split}.jsonl"
            ),
            filtered_manifest_path=resolved_filtered_manifests_root / f"filtered_{split}.jsonl",
        )
        report["splits"][split] = {
            "generated_at": generated_at,
            "config_path": report_config_path,
            **split_report,
        }

    return report
