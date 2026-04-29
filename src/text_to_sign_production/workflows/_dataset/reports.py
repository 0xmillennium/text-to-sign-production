"""Dataset workflow-owned report artifact writing and cleanup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import ensure_dir
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.data.constants import SPLITS
from text_to_sign_production.data.jsonl import write_jsonl
from text_to_sign_production.data.quality import QUALITY_TIERS
from text_to_sign_production.workflows._dataset.types import DatasetWorkflowError


def write_interim_assumption_reports(
    assumption_report: Mapping[str, Any],
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> None:
    """Write split-scoped interim assumption JSON reports."""

    duplicate_sample_ids = _duplicate_sample_ids(assumption_report)
    sample_id_to_splits = _sample_id_to_splits_from_raw_manifests(layout=layout, splits=splits)
    duplicate_ids_by_split = {
        split: [
            sample_id
            for sample_id in duplicate_sample_ids
            if split in set(sample_id_to_splits[sample_id])
        ][:20]
        for split in splits
    }
    for split in splits:
        split_payload = dict(_report_split_mapping(assumption_report, "assumption_report", split))
        split_payload["compared_splits"] = list(splits)
        split_payload["split_integrity"] = {
            "sample_id_overlap_detected_with_compared_splits": bool(duplicate_ids_by_split[split]),
            "duplicate_sample_ids_with_compared_splits": duplicate_ids_by_split[split],
        }
        _write_json(_interim_assumption_report_path(layout, split), split_payload)


def write_interim_filter_reports(
    filter_report: Mapping[str, Any],
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> None:
    """Write split-scoped interim filter JSON reports."""

    for split in splits:
        _write_json(
            _interim_filter_report_path(layout, split),
            _report_split_mapping(filter_report, "filter_report", split),
        )


def write_processed_reports(
    *,
    assumption_report: Mapping[str, Any],
    export_report: Mapping[str, Any],
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> None:
    """Write processed JSON and Markdown report artifacts."""

    quality_report = _report_mapping(export_report, "quality_report")
    split_report = _report_mapping(export_report, "split_report")
    quality_outputs = _optional_report_mapping(export_report, "quality_outputs")
    for split in splits:
        _write_json(
            _processed_data_quality_report_json_path(layout, split),
            _report_split_mapping(quality_report, "quality_report", split),
        )
        _write_json(
            _processed_split_report_json_path(layout, split),
            _report_split_mapping(split_report, "split_report", split),
        )
        assumption_split = _report_split_mapping(assumption_report, "assumption_report", split)
        quality_split = _report_split_mapping(quality_report, "quality_report", split)
        split_payload = _report_split_mapping(split_report, "split_report", split)
        _write_text(
            _processed_assumption_report_markdown_path(layout, split),
            _assumption_markdown(split=split, split_data=assumption_split),
        )
        _write_text(
            _processed_data_quality_report_markdown_path(layout, split),
            _data_quality_markdown(split=split, split_data=quality_split),
        )
        _write_text(
            _processed_split_report_markdown_path(layout, split),
            _split_markdown(split=split, split_data=split_payload),
        )
        if quality_outputs is not None:
            expanded_reports = _report_mapping(quality_outputs, "quality_reports")
            expanded_quality_split = _mapping_value(expanded_reports, split)
            _write_json(_processed_quality_report_json_path(layout, split), expanded_quality_split)
            _write_text(
                _processed_quality_report_markdown_path(layout, split),
                _expanded_quality_markdown(split=split, split_data=expanded_quality_split),
            )
    if quality_outputs is not None:
        leakage_report = _mapping_value(quality_outputs, "leakage_report")
        _write_json(_processed_leakage_report_json_path(layout), leakage_report)
        _write_text(
            _processed_leakage_report_markdown_path(layout),
            _leakage_markdown(leakage_report),
        )


def write_quality_tier_manifests(
    *,
    export_report: Mapping[str, Any],
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, dict[str, Path]]:
    """Write quality-tier manifests that reference existing processed samples."""

    quality_outputs = _optional_report_mapping(export_report, "quality_outputs")
    if quality_outputs is None:
        return {split: {} for split in splits}
    tiers_by_split = _report_mapping(quality_outputs, "tiers_by_split")
    written: dict[str, dict[str, Path]] = {}
    for split in splits:
        split_tiers = _mapping_value(tiers_by_split, split)
        split_paths: dict[str, Path] = {}
        for tier in QUALITY_TIERS:
            path = _processed_tier_manifest_path(layout, tier=tier, split=split)
            records = _sequence_value(split_tiers, tier)
            write_jsonl(path, list(records))
            split_paths[tier] = path
        written[split] = split_paths
    return written


def remove_stale_split_files(
    root: Path,
    *,
    filename_template: str,
    requested_splits: tuple[str, ...],
    all_splits: tuple[str, ...] = SPLITS,
) -> None:
    """Remove known split-specific files for unrequested official splits."""

    for split in all_splits:
        if split in requested_splits:
            continue
        path = root / filename_template.format(split=split)
        if not path.exists():
            continue
        if not path.is_file():
            raise DatasetWorkflowError(f"Expected stale split output path to be a file: {path}")
        path.unlink()


def remove_stale_processed_report_files(
    *,
    layout: ProjectLayout,
    requested_splits: tuple[str, ...],
) -> None:
    remove_stale_split_files(
        _processed_assumption_reports_root(layout),
        filename_template="{split}.md",
        requested_splits=requested_splits,
    )
    for root in (
        _processed_data_quality_reports_root(layout),
        _processed_quality_reports_root(layout),
        _processed_split_reports_root(layout),
    ):
        remove_stale_split_files(
            root,
            filename_template="{split}.json",
            requested_splits=requested_splits,
        )
        remove_stale_split_files(
            root,
            filename_template="{split}.md",
            requested_splits=requested_splits,
        )
    remove_stale_split_files(
        _processed_tiers_root(layout) / "broad",
        filename_template="{split}.jsonl",
        requested_splits=requested_splits,
    )
    remove_stale_split_files(
        _processed_tiers_root(layout) / "quality",
        filename_template="{split}.jsonl",
        requested_splits=requested_splits,
    )
    remove_stale_split_files(
        _processed_tiers_root(layout) / "audit_low_quality",
        filename_template="{split}.jsonl",
        requested_splits=requested_splits,
    )
    remove_stale_split_files(
        _processed_tiers_root(layout) / "dropped",
        filename_template="{split}.jsonl",
        requested_splits=requested_splits,
    )


def _report_mapping(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise DatasetWorkflowError(f"export_report is missing {key!r}.")
    return value


def _optional_report_mapping(report: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = report.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DatasetWorkflowError(f"export_report field {key!r} must be a mapping or null.")
    return value


def _mapping_value(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        raise DatasetWorkflowError(f"Report mapping is missing mapping field {key!r}.")
    return value


def _sequence_value(report: Mapping[str, Any], key: str) -> list[Any]:
    value = report.get(key)
    if not isinstance(value, list):
        raise DatasetWorkflowError(f"Report mapping is missing list field {key!r}.")
    return value


def _report_split_mapping(
    report: Mapping[str, Any],
    report_name: str,
    split: str,
) -> Mapping[str, Any]:
    splits_payload = report.get("splits")
    if not isinstance(splits_payload, Mapping):
        raise DatasetWorkflowError(f"{report_name} is missing a splits mapping.")
    split_payload = splits_payload.get(split)
    if not isinstance(split_payload, Mapping):
        raise DatasetWorkflowError(f"{report_name} split {split} must be a mapping.")
    return split_payload


def _duplicate_sample_ids(assumption_report: Mapping[str, Any]) -> list[str]:
    split_integrity = assumption_report.get("split_integrity")
    if not isinstance(split_integrity, Mapping):
        return []
    duplicate_sample_ids = split_integrity.get("duplicate_sample_ids")
    if not isinstance(duplicate_sample_ids, list):
        return []
    return [str(sample_id) for sample_id in duplicate_sample_ids]


def _sample_id_to_splits_from_raw_manifests(
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, list[str]]:
    artifact_layout = DatasetArtifactLayout(layout)
    sample_id_to_splits: dict[str, list[str]] = {}
    for split in splits:
        raw_manifest_path = artifact_layout.raw_manifest_path(split)
        with raw_manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise DatasetWorkflowError(
                        "Raw manifest record must be a JSON object while writing reports: "
                        f"{raw_manifest_path}:{line_number}"
                    )
                sample_id = payload.get("sample_id")
                if isinstance(sample_id, str):
                    sample_id_to_splits.setdefault(sample_id, []).append(split)
    return sample_id_to_splits


def _dataset_report_paths(
    *,
    layout: ProjectLayout,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "interim_assumption_report_paths": {
            split: _interim_assumption_report_path(layout, split) for split in splits
        },
        "interim_filter_report_paths": {
            split: _interim_filter_report_path(layout, split) for split in splits
        },
        "interim_report_paths": {
            **{
                f"assumption_{split}": _interim_assumption_report_path(layout, split)
                for split in splits
            },
            **{f"filter_{split}": _interim_filter_report_path(layout, split) for split in splits},
        },
        "processed_report_paths": {
            **{
                f"assumption_{split}_markdown": (
                    _processed_assumption_report_markdown_path(layout, split)
                )
                for split in splits
            },
            **{
                f"data_quality_{split}_json": (
                    _processed_data_quality_report_json_path(layout, split)
                )
                for split in splits
            },
            **{
                f"data_quality_{split}_markdown": (
                    _processed_data_quality_report_markdown_path(layout, split)
                )
                for split in splits
            },
            **{
                f"quality_{split}_json": _processed_quality_report_json_path(layout, split)
                for split in splits
            },
            **{
                f"quality_{split}_markdown": (
                    _processed_quality_report_markdown_path(layout, split)
                )
                for split in splits
            },
            **{
                f"split_{split}_json": _processed_split_report_json_path(layout, split)
                for split in splits
            },
            **{
                f"split_{split}_markdown": _processed_split_report_markdown_path(layout, split)
                for split in splits
            },
            "leakage_global_json": _processed_leakage_report_json_path(layout),
            "leakage_global_markdown": _processed_leakage_report_markdown_path(layout),
        },
        "quality_report_paths": {
            **{
                f"{split}_json": _processed_quality_report_json_path(layout, split)
                for split in splits
            },
            **{
                f"{split}_markdown": _processed_quality_report_markdown_path(layout, split)
                for split in splits
            },
        },
        "leakage_report_paths": {
            "global_json": _processed_leakage_report_json_path(layout),
            "global_markdown": _processed_leakage_report_markdown_path(layout),
        },
        "tier_manifest_paths": {
            tier: {
                split: _processed_tier_manifest_path(layout, tier=tier, split=split)
                for split in splits
            }
            for tier in QUALITY_TIERS
        },
    }


def _interim_reports_root(layout: ProjectLayout) -> Path:
    return layout.interim_root / "reports"


def _interim_assumption_reports_root(layout: ProjectLayout) -> Path:
    return _interim_reports_root(layout) / "assumption"


def _interim_filter_reports_root(layout: ProjectLayout) -> Path:
    return _interim_reports_root(layout) / "filter"


def _processed_v1_reports_root(layout: ProjectLayout) -> Path:
    return layout.processed_root / "v1" / "reports"


def _processed_assumption_reports_root(layout: ProjectLayout) -> Path:
    return _processed_v1_reports_root(layout) / "assumption"


def _processed_data_quality_reports_root(layout: ProjectLayout) -> Path:
    return _processed_v1_reports_root(layout) / "data_quality"


def _processed_quality_reports_root(layout: ProjectLayout) -> Path:
    return _processed_v1_reports_root(layout) / "quality"


def _processed_leakage_reports_root(layout: ProjectLayout) -> Path:
    return _processed_v1_reports_root(layout) / "leakage"


def _processed_split_reports_root(layout: ProjectLayout) -> Path:
    return _processed_v1_reports_root(layout) / "split"


def _processed_tiers_root(layout: ProjectLayout) -> Path:
    return layout.processed_root / "v1" / "tiers"


def _interim_assumption_report_path(layout: ProjectLayout, split: str) -> Path:
    return _interim_assumption_reports_root(layout) / f"{_require_split(split)}.json"


def _interim_filter_report_path(layout: ProjectLayout, split: str) -> Path:
    return _interim_filter_reports_root(layout) / f"{_require_split(split)}.json"


def _processed_assumption_report_markdown_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_assumption_reports_root(layout) / f"{_require_split(split)}.md"


def _processed_data_quality_report_json_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_data_quality_reports_root(layout) / f"{_require_split(split)}.json"


def _processed_data_quality_report_markdown_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_data_quality_reports_root(layout) / f"{_require_split(split)}.md"


def _processed_quality_report_json_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_quality_reports_root(layout) / f"{_require_split(split)}.json"


def _processed_quality_report_markdown_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_quality_reports_root(layout) / f"{_require_split(split)}.md"


def _processed_leakage_report_json_path(layout: ProjectLayout) -> Path:
    return _processed_leakage_reports_root(layout) / "global.json"


def _processed_leakage_report_markdown_path(layout: ProjectLayout) -> Path:
    return _processed_leakage_reports_root(layout) / "global.md"


def _processed_split_report_json_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_split_reports_root(layout) / f"{_require_split(split)}.json"


def _processed_split_report_markdown_path(layout: ProjectLayout, split: str) -> Path:
    return _processed_split_reports_root(layout) / f"{_require_split(split)}.md"


def _processed_tier_manifest_path(layout: ProjectLayout, *, tier: str, split: str) -> Path:
    if tier not in QUALITY_TIERS:
        raise DatasetWorkflowError(f"Unsupported quality tier {tier!r}.")
    return _processed_tiers_root(layout) / tier / f"{_require_split(split)}.jsonl"


def _require_split(split: str) -> str:
    if split not in SPLITS:
        expected = ", ".join(SPLITS)
        raise DatasetWorkflowError(f"Unsupported split {split!r}; expected one of: {expected}.")
    return split


def _write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent, label=f"{path.name} parent directory")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent, label=f"{path.name} parent directory")
    path.write_text(text, encoding="utf-8")


def _assumption_markdown(*, split: str, split_data: Mapping[str, Any]) -> str:
    video_metadata = dict(split_data["video_metadata"])
    openpose_schema = dict(split_data["openpose_schema"])
    unmatched_examples = [str(value) for value in split_data.get("unmatched_examples", [])][:10]
    channel_lengths = dict(openpose_schema.get("channel_lengths", {}))
    lines = [
        f"# Assumption Report - {split}",
        "",
        f"Generated at: `{split_data['generated_at']}`",
        "",
        f"- translation rows: `{split_data['translation_row_count']}`",
        f"- matched samples: `{split_data['matched_sample_count']}`",
        f"- unmatched samples: `{split_data['unmatched_sample_count']}`",
        f"- readable video metadata: `{video_metadata['readable_count']}`",
        f"- unreadable video metadata: `{video_metadata['unreadable_count']}`",
        f"- first-frame people counter: `{split_data['first_frame_people_counter']}`",
        "",
        "## Unmatched examples",
        "",
        *_markdown_bullets(unmatched_examples),
        "",
        "## Video metadata",
        "",
        f"- dimensions: `{video_metadata.get('dimension_counts', {})}`",
        f"- fps: `{video_metadata.get('fps_counts', {})}`",
        "",
        "## OpenPose 2D channel lengths",
        "",
    ]
    if channel_lengths:
        for channel_name in sorted(channel_lengths):
            lines.append(f"- {channel_name}: `{channel_lengths[channel_name]}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Schema deviations",
            "",
            f"- counts: `{openpose_schema.get('deviation_counts', {})}`",
            "",
        ]
    )
    return "\n".join(lines)


def _data_quality_markdown(*, split: str, split_data: Mapping[str, Any]) -> str:
    lines = [
        f"# Data Quality Report - {split}",
        "",
        f"Generated at: `{split_data['generated_at']}`",
        "",
        f"- processed samples: `{split_data['processed_sample_count']}`",
        f"- dropped samples: `{split_data['dropped_sample_count']}`",
        f"- drop reasons: `{split_data['drop_reason_counts']}`",
        f"- multi-person samples: `{split_data['multi_person_samples']}`",
        f"- multi-person frames: `{split_data['multi_person_frames']}`",
        f"- max people per frame: `{split_data['max_people_per_frame']}`",
        f"- parse/schema issue samples: `{split_data['parse_or_schema_issue_samples']}`",
        f"- out-of-bounds coordinates: `{split_data['out_of_bounds_coordinate_count']}`",
        (
            f"- frames with any zeroed parser-required point: "
            f"`{split_data['frames_with_any_zeroed_parser_required_point']}`"
        ),
        (
            f"- zeroed parser-required point frame ratio: "
            f"`{split_data['frames_with_any_zeroed_parser_required_point_ratio']}`"
        ),
        (
            "This audit/debug signal counts frames where at least one parser-required raw "
            "OpenPose point row is zero-filled. It does not mean the whole frame is unusable "
            "or that all required joints are missing."
        ),
        f"- text length summary: `{split_data['text_length']}`",
        f"- frame count summary: `{split_data['frame_count']}`",
        f"- FPS summary: `{split_data['fps']}`",
        f"- face-missing ratio summary: `{split_data['face_missing_ratio']}`",
        "",
    ]
    return "\n".join(lines)


def _split_markdown(*, split: str, split_data: Mapping[str, Any]) -> str:
    lines = [
        f"# Split Integrity Report - {split}",
        "",
        f"Generated at: `{split_data['generated_at']}`",
        "",
        f"- compared splits: `{split_data['compared_splits']}`",
        (f"- cross-split overlap available: `{split_data['cross_split_overlap_available']}`"),
        f"- raw samples: `{split_data['raw_samples']}`",
        f"- processed samples: `{split_data['processed_samples']}`",
        f"- raw videos: `{split_data['raw_video_count']}`",
        f"- processed videos: `{split_data['processed_video_count']}`",
        (
            f"- sample-id overlap with compared splits: "
            f"`{split_data['sample_id_overlap_with_compared_splits']}`"
        ),
        (
            f"- video overlap with compared splits: "
            f"`{split_data['video_overlap_with_compared_splits']}`"
        ),
        "",
    ]
    if not bool(split_data["cross_split_overlap_available"]):
        lines.extend(
            [
                "This report was generated for a single-split run; no other split was "
                "available for cross-split comparison.",
                "",
            ]
        )
    return "\n".join(lines)


def _expanded_quality_markdown(*, split: str, split_data: Mapping[str, Any]) -> str:
    counts = dict(split_data["sample_counts"])
    text_normalization = dict(split_data["text_normalization"])
    lines = [
        f"# Expanded Quality Report - {split}",
        "",
        f"Generated at: `{split_data['generated_at']}`",
        "",
        f"- broad samples: `{counts.get('broad', 0)}`",
        f"- quality samples: `{counts.get('quality', 0)}`",
        f"- audit low-quality samples: `{counts.get('audit_low_quality', 0)}`",
        f"- dropped samples: `{counts.get('dropped', 0)}`",
        f"- sequence length percentiles: `{split_data['sequence_length_percentiles']}`",
        f"- text length percentiles: `{split_data['text_length_percentiles']}`",
        (
            f"- normalized text length percentiles: "
            f"`{split_data['normalized_text_length_percentiles']}`"
        ),
        f"- token length: `{split_data['token_length']}`",
        (
            f"- duplicated normalized text count: "
            f"`{text_normalization['duplicated_normalized_text_count']}`"
        ),
        "",
        "## Channel Coverage",
        "",
    ]
    channels = dict(split_data["channels"])
    for channel in sorted(channels):
        channel_payload = dict(channels[channel])
        lines.append(f"- {channel}: `{channel_payload['coverage_histogram']}`")
    lines.extend(
        [
            "",
            "## Low-Quality Kept Examples",
            "",
            *_markdown_bullets(
                [
                    f"{record['sample_id']}: {record['quality_reasons']}"
                    for record in split_data.get("kept_low_quality_examples", [])
                ][:10]
            ),
            "",
            "## Dropped Examples",
            "",
            *_markdown_bullets(
                [
                    f"{record['sample_id']}: {record['drop_reasons']}"
                    for record in split_data.get("dropped_examples", [])
                ][:10]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _leakage_markdown(split_data: Mapping[str, Any]) -> str:
    lines = [
        "# Leakage Report - Global",
        "",
        f"Generated at: `{split_data['generated_at']}`",
        f"- blocking_for_complete: `{split_data['blocking_for_complete']}`",
        "",
        "## Cross-Split Overlaps",
        "",
    ]
    overlaps = dict(split_data["overlaps"])
    for field_name in sorted(overlaps):
        lines.append(f"- {field_name}: `{overlaps[field_name]}`")
    lines.extend(
        [
            "",
            "## Near-Duplicate Normalized Transcript Groups",
            "",
            *_markdown_bullets(
                [
                    f"{group['normalized_text']}: {group['records']}"
                    for group in split_data.get("near_duplicate_normalized_transcript_groups", [])
                ][:10]
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_bullets(values: list[str]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {value}" for value in values]


__all__: list[str] = []
