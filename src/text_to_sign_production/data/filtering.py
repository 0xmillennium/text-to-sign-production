"""Minimal structural filtering for normalized candidate samples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import (
    FILTERED_MANIFESTS_ROOT,
    INTERIM_REPORTS_ROOT,
    NORMALIZED_MANIFESTS_ROOT,
    REQUIRED_CORE_CHANNELS,
    SPLITS,
)
from .jsonl import iter_jsonl, write_json, write_jsonl
from .schemas import NormalizedManifestEntry
from .utils import ensure_directory, repo_relative_path, resolve_repo_path, utc_timestamp

FILTER_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """The structural filtering policy for v1 dataset export."""

    require_nonempty_text: bool
    require_positive_duration: bool
    require_keypoints_dir: bool
    require_frames: bool
    drop_on_sample_parse_error: bool
    require_at_least_one_valid_frame: bool
    minimum_nonzero_frames_per_core_channel: int


def load_filter_config(path: Path) -> FilterConfig:
    """Load the v1 filtering policy from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in filter config {path}.")
    schema_version = payload.get("schema_version")
    if schema_version != FILTER_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported filter config schema_version {schema_version!r} in {path}; "
            f"expected {FILTER_CONFIG_SCHEMA_VERSION}."
        )
    return FilterConfig(
        require_nonempty_text=bool(payload["require_nonempty_text"]),
        require_positive_duration=bool(payload["require_positive_duration"]),
        require_keypoints_dir=bool(payload["require_keypoints_dir"]),
        require_frames=bool(payload["require_frames"]),
        drop_on_sample_parse_error=bool(payload["drop_on_sample_parse_error"]),
        require_at_least_one_valid_frame=bool(payload["require_at_least_one_valid_frame"]),
        minimum_nonzero_frames_per_core_channel=int(
            payload["minimum_nonzero_frames_per_core_channel"]
        ),
    )


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
    for channel in REQUIRED_CORE_CHANNELS:
        if (
            entry.core_channel_nonzero_frames.get(channel, 0)
            < config.minimum_nonzero_frames_per_core_channel
        ):
            drop_reasons.append(f"unusable_core_channel:{channel}")
    if (
        entry.sample_path is None
        and entry.source_keypoints_dir is not None
        and "sample_parse_error" not in drop_reasons
    ):
        drop_reasons.append("missing_sample_file")
    return drop_reasons


def filter_split(
    split: str, config: FilterConfig
) -> tuple[list[NormalizedManifestEntry], dict[str, Any]]:
    """Filter one split-specific normalized manifest."""

    input_path = NORMALIZED_MANIFESTS_ROOT / f"normalized_{split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Normalized manifest not found: {input_path}")

    ensure_directory(FILTERED_MANIFESTS_ROOT)

    kept_entries: list[NormalizedManifestEntry] = []
    drop_reason_counter: Counter[str] = Counter()
    dropped_examples: list[dict[str, Any]] = []
    total_entries = 0

    for record in iter_jsonl(input_path):
        total_entries += 1
        entry = NormalizedManifestEntry.from_record(record)
        drop_reasons = determine_drop_reasons(entry, config)
        if drop_reasons:
            drop_reason_counter.update(drop_reasons)
            if len(dropped_examples) < 20:
                dropped_examples.append(
                    {"sample_id": entry.sample_id, "drop_reasons": drop_reasons}
                )
            continue
        kept_entries.append(entry)

    write_jsonl(FILTERED_MANIFESTS_ROOT / f"filtered_{split}.jsonl", kept_entries)
    return kept_entries, {
        "split": split,
        "input_samples": total_entries,
        "kept_samples": len(kept_entries),
        "dropped_samples": total_entries - len(kept_entries),
        "drop_reason_counts": {
            key: drop_reason_counter[key] for key in sorted(drop_reason_counter)
        },
        "dropped_examples": dropped_examples,
    }


def filter_all_splits(config_path: Path, *, splits: tuple[str, ...] = SPLITS) -> dict[str, Any]:
    """Apply structural filtering to every official split."""

    config = load_filter_config(config_path)
    ensure_directory(INTERIM_REPORTS_ROOT)
    resolved_config_path = resolve_repo_path(config_path)
    try:
        report_config_path = repo_relative_path(resolved_config_path)
    except ValueError:
        report_config_path = config_path.name

    report: dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "config_path": report_config_path,
        "splits": {},
    }
    for split in splits:
        _, split_report = filter_split(split, config)
        report["splits"][split] = split_report

    write_json(INTERIM_REPORTS_ROOT / "filter-report.json", report)
    return report
