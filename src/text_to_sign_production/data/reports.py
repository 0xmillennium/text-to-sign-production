"""Report construction and rendering for the Sprint 2 data pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from .constants import PROCESSED_REPORTS_ROOT, SPLITS
from .schemas import ProcessedManifestEntry, RawManifestEntry
from .utils import summarize_numbers

T = TypeVar("T")


def _records_for_split(
    records_by_split: Mapping[str, list[T]],
    *,
    mapping_name: str,
    split: str,
) -> list[T]:
    try:
        return records_by_split[split]
    except KeyError as exc:
        raise ValueError(f"{mapping_name} is missing requested split: {split}") from exc


def _numeric_summary(values: Iterable[int | float]) -> dict[str, float | int | None]:
    summary = summarize_numbers(values)
    min_value = summary["min"]
    max_value = summary["max"]
    mean_value = summary["mean"]
    median_value = summary["median"]
    if (
        min_value is not None
        and max_value is not None
        and mean_value is not None
        and median_value is not None
    ):
        summary["min"] = round(float(min_value), 3)
        summary["max"] = round(float(max_value), 3)
        summary["mean"] = round(float(mean_value), 3)
        summary["median"] = round(float(median_value), 3)
    return summary


def build_quality_report(
    *,
    final_records_by_split: Mapping[str, list[ProcessedManifestEntry]],
    filter_report: Mapping[str, Any],
    generated_at: str,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Construct the machine-readable processed data-quality report."""

    quality_report: dict[str, Any] = {"generated_at": generated_at, "splits": {}}
    for split in splits:
        final_records = _records_for_split(
            final_records_by_split,
            mapping_name="final_records_by_split",
            split=split,
        )
        split_filter_report = dict(filter_report["splits"][split])
        frame_counts = [record.num_frames for record in final_records]
        text_lengths = [len(record.text) for record in final_records]
        fps_values = [record.fps for record in final_records if record.fps is not None]
        face_missing_ratios = [
            (record.face_missing_frame_count / record.num_frames)
            for record in final_records
            if record.num_frames > 0
        ]
        quality_report["splits"][split] = {
            "processed_sample_count": len(final_records),
            "dropped_sample_count": split_filter_report["dropped_samples"],
            "drop_reason_counts": split_filter_report["drop_reason_counts"],
            "text_length": _numeric_summary(text_lengths),
            "frame_count": _numeric_summary(frame_counts),
            "fps": _numeric_summary(fps_values),
            "face_missing_ratio": _numeric_summary(face_missing_ratios),
            "multi_person_samples": sum(
                1 for record in final_records if record.multi_person_frame_count > 0
            ),
            "multi_person_frames": sum(record.multi_person_frame_count for record in final_records),
            "max_people_per_frame": max(
                (record.max_people_per_frame for record in final_records), default=0
            ),
            "parse_or_schema_issue_samples": sum(
                1 for record in final_records if record.frame_issue_counts
            ),
            "out_of_bounds_coordinate_count": sum(
                record.out_of_bounds_coordinate_count for record in final_records
            ),
            "frames_with_any_zeroed_required_joint": sum(
                record.frames_with_any_zeroed_required_joint for record in final_records
            ),
            "readable_video_metadata_samples": sum(
                1
                for record in final_records
                if record.video_metadata_error is None and record.fps is not None
            ),
        }
    return quality_report


def build_split_report(
    *,
    raw_records_by_split: Mapping[str, list[RawManifestEntry]],
    final_records_by_split: Mapping[str, list[ProcessedManifestEntry]],
    generated_at: str,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Construct the machine-readable split integrity report."""

    split_report: dict[str, Any] = {"generated_at": generated_at, "splits": {}}
    requested_splits = tuple(splits)
    for split in requested_splits:
        raw_entries = _records_for_split(
            raw_records_by_split,
            mapping_name="raw_records_by_split",
            split=split,
        )
        final_records = _records_for_split(
            final_records_by_split,
            mapping_name="final_records_by_split",
            split=split,
        )
        video_ids = {record.source_video_id for record in final_records}
        split_report["splits"][split] = {
            "raw_samples": len(raw_entries),
            "processed_samples": len(final_records),
            "raw_video_count": len({entry.video_id for entry in raw_entries}),
            "processed_video_count": len(video_ids),
            "sample_id_overlap_with_other_splits": {},
        }

    split_names = {
        split: {
            record.sample_id
            for record in _records_for_split(
                final_records_by_split,
                mapping_name="final_records_by_split",
                split=split,
            )
        }
        for split in requested_splits
    }
    for split in requested_splits:
        overlaps = {
            other_split: sorted(split_names[split].intersection(split_names[other_split]))[:10]
            for other_split in requested_splits
            if other_split != split
        }
        split_report["splits"][split]["sample_id_overlap_with_other_splits"] = overlaps
    split_report["official_split_names"] = list(requested_splits)
    video_id_overlap: dict[str, int] = {}
    for left_index, left_split in enumerate(requested_splits):
        left_video_ids = {
            record.source_video_id
            for record in _records_for_split(
                final_records_by_split,
                mapping_name="final_records_by_split",
                split=left_split,
            )
        }
        for right_split in requested_splits[left_index + 1 :]:
            right_video_ids = {
                record.source_video_id
                for record in _records_for_split(
                    final_records_by_split,
                    mapping_name="final_records_by_split",
                    split=right_split,
                )
            }
            video_id_overlap[f"{left_split}_{right_split}"] = len(
                left_video_ids.intersection(right_video_ids)
            )
    split_report["video_id_overlap"] = video_id_overlap
    return split_report


def write_markdown_reports(
    assumption_report: Mapping[str, Any],
    quality_report: Mapping[str, Any],
    split_report: Mapping[str, Any],
    *,
    splits: tuple[str, ...] = SPLITS,
) -> None:
    """Write human-readable Markdown reports under the processed reports root."""

    assumption_lines = [
        "# Assumption Report",
        "",
        f"Generated at: `{assumption_report['generated_at']}`",
        "",
    ]
    for split in splits:
        split_data = assumption_report["splits"][split]
        assumption_lines.extend(
            [
                f"## {split}",
                "",
                f"- translation rows: `{split_data['translation_row_count']}`",
                f"- matched samples: `{split_data['matched_sample_count']}`",
                f"- unmatched samples: `{split_data['unmatched_sample_count']}`",
                f"- readable video metadata: `{split_data['video_metadata']['readable_count']}`",
                (
                    f"- unreadable video metadata: "
                    f"`{split_data['video_metadata']['unreadable_count']}`"
                ),
                f"- first-frame people counter: `{split_data['first_frame_people_counter']}`",
                f"- schema deviation counts: `{split_data['openpose_schema']['deviation_counts']}`",
                "",
            ]
        )
    (PROCESSED_REPORTS_ROOT / "assumption-report.md").write_text(
        "\n".join(assumption_lines), encoding="utf-8"
    )

    quality_lines = [
        "# Data Quality Report",
        "",
        f"Generated at: `{quality_report['generated_at']}`",
        "",
    ]
    for split in splits:
        split_data = quality_report["splits"][split]
        quality_lines.extend(
            [
                f"## {split}",
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
                    f"- frames with any zeroed required joint: "
                    f"`{split_data['frames_with_any_zeroed_required_joint']}`"
                ),
                f"- text length summary: `{split_data['text_length']}`",
                f"- frame count summary: `{split_data['frame_count']}`",
                f"- FPS summary: `{split_data['fps']}`",
                f"- face-missing ratio summary: `{split_data['face_missing_ratio']}`",
                "",
            ]
        )
    (PROCESSED_REPORTS_ROOT / "data-quality-report.md").write_text(
        "\n".join(quality_lines), encoding="utf-8"
    )

    split_lines = [
        "# Split Integrity Report",
        "",
        f"Generated at: `{split_report['generated_at']}`",
        "",
        f"Official split names: `{split_report['official_split_names']}`",
        f"Video overlap summary: `{split_report['video_id_overlap']}`",
        "",
    ]
    for split in splits:
        split_data = split_report["splits"][split]
        split_lines.extend(
            [
                f"## {split}",
                "",
                f"- raw samples: `{split_data['raw_samples']}`",
                f"- processed samples: `{split_data['processed_samples']}`",
                f"- raw videos: `{split_data['raw_video_count']}`",
                f"- processed videos: `{split_data['processed_video_count']}`",
                (
                    f"- sample-id overlap with other splits: "
                    f"`{split_data['sample_id_overlap_with_other_splits']}`"
                ),
                "",
            ]
        )
    (PROCESSED_REPORTS_ROOT / "split-report.md").write_text(
        "\n".join(split_lines), encoding="utf-8"
    )
