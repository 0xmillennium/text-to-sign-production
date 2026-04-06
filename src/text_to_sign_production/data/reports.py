"""Report generation for the Sprint 2 data pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .constants import (
    FILTERED_MANIFESTS_ROOT,
    PROCESSED_MANIFESTS_ROOT,
    PROCESSED_REPORTS_ROOT,
    RAW_MANIFESTS_ROOT,
    SPLITS,
)
from .jsonl import read_jsonl, write_json, write_jsonl
from .schemas import NormalizedManifestEntry, ProcessedManifestEntry, RawManifestEntry
from .utils import ensure_directory, summarize_numbers, utc_timestamp


def _load_raw_records(split: str) -> list[RawManifestEntry]:
    return [
        RawManifestEntry.from_record(record)
        for record in read_jsonl(RAW_MANIFESTS_ROOT / f"raw_{split}.jsonl")
    ]


def _load_filtered_records(split: str) -> list[NormalizedManifestEntry]:
    return [
        NormalizedManifestEntry.from_record(record)
        for record in read_jsonl(FILTERED_MANIFESTS_ROOT / f"filtered_{split}.jsonl")
    ]


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


def _build_processed_manifest_entry(entry: NormalizedManifestEntry) -> ProcessedManifestEntry:
    if entry.sample_path is None:
        raise ValueError(f"Filtered entry {entry.sample_id} is missing sample_path.")
    if entry.source_keypoints_dir is None:
        raise ValueError(f"Filtered entry {entry.sample_id} is missing source_keypoints_dir.")

    return ProcessedManifestEntry(
        sample_id=entry.sample_id,
        processed_schema_version=entry.processed_schema_version,
        text=entry.text,
        split=entry.split,
        fps=entry.fps,
        num_frames=entry.num_frames,
        sample_path=entry.sample_path,
        source_video_id=entry.source_video_id,
        source_sentence_id=entry.source_sentence_id,
        source_sentence_name=entry.source_sentence_name,
        selected_person_index=entry.selected_person_index,
        multi_person_frame_count=entry.multi_person_frame_count,
        max_people_per_frame=entry.max_people_per_frame,
        source_metadata_path=entry.source_metadata_path,
        source_keypoints_dir=entry.source_keypoints_dir,
        source_video_path=entry.source_video_path,
        video_width=entry.video_width,
        video_height=entry.video_height,
        video_metadata_error=entry.video_metadata_error,
        frame_valid_count=entry.frame_valid_count,
        frame_invalid_count=entry.frame_invalid_count,
        face_missing_frame_count=entry.face_missing_frame_count,
        out_of_bounds_coordinate_count=entry.out_of_bounds_coordinate_count,
        frames_with_any_zeroed_required_joint=entry.frames_with_any_zeroed_required_joint,
        frame_issue_counts=entry.frame_issue_counts,
        core_channel_nonzero_frames=entry.core_channel_nonzero_frames,
    )


def export_final_manifests(
    assumption_report: dict[str, Any],
    filter_report: dict[str, Any],
) -> dict[str, Any]:
    """Build final processed manifests plus Markdown and JSON reports."""

    ensure_directory(PROCESSED_MANIFESTS_ROOT)
    ensure_directory(PROCESSED_REPORTS_ROOT)

    final_records_by_split: dict[str, list[ProcessedManifestEntry]] = {}
    raw_records_by_split: dict[str, list[RawManifestEntry]] = {}
    split_report: dict[str, Any] = {"generated_at": utc_timestamp(), "splits": {}}
    quality_report: dict[str, Any] = {"generated_at": utc_timestamp(), "splits": {}}

    for split in SPLITS:
        filtered_entries = _load_filtered_records(split)
        raw_entries = _load_raw_records(split)
        raw_records_by_split[split] = raw_entries
        final_records = [
            _build_processed_manifest_entry(entry)
            for entry in filtered_entries
            if entry.sample_path is not None
        ]
        write_jsonl(PROCESSED_MANIFESTS_ROOT / f"{split}.jsonl", final_records)
        final_records_by_split[split] = final_records

        video_ids = {record.source_video_id for record in final_records}
        frame_counts = [record.num_frames for record in final_records]
        text_lengths = [len(record.text) for record in final_records]
        fps_values = [record.fps for record in final_records if record.fps is not None]
        face_missing_ratios = [
            (record.face_missing_frame_count / record.num_frames)
            for record in final_records
            if record.num_frames > 0
        ]

        split_report["splits"][split] = {
            "raw_samples": len(raw_entries),
            "processed_samples": len(final_records),
            "raw_video_count": len({entry.video_id for entry in raw_entries}),
            "processed_video_count": len(video_ids),
            "sample_id_overlap_with_other_splits": [],
        }
        quality_report["splits"][split] = {
            "processed_sample_count": len(final_records),
            "dropped_sample_count": filter_report["splits"][split]["dropped_samples"],
            "drop_reason_counts": filter_report["splits"][split]["drop_reason_counts"],
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

    split_names = {
        split: {record.sample_id for record in records}
        for split, records in final_records_by_split.items()
    }
    for split in SPLITS:
        overlaps = {
            other_split: sorted(split_names[split].intersection(split_names[other_split]))[:10]
            for other_split in SPLITS
            if other_split != split
        }
        split_report["splits"][split]["sample_id_overlap_with_other_splits"] = overlaps
    split_report["official_split_names"] = list(SPLITS)
    split_report["video_id_overlap"] = {
        "train_val": len(
            {record.source_video_id for record in final_records_by_split["train"]}.intersection(
                {record.source_video_id for record in final_records_by_split["val"]}
            )
        ),
        "train_test": len(
            {record.source_video_id for record in final_records_by_split["train"]}.intersection(
                {record.source_video_id for record in final_records_by_split["test"]}
            )
        ),
        "val_test": len(
            {record.source_video_id for record in final_records_by_split["val"]}.intersection(
                {record.source_video_id for record in final_records_by_split["test"]}
            )
        ),
    }

    write_json(PROCESSED_REPORTS_ROOT / "data-quality-report.json", quality_report)
    write_json(PROCESSED_REPORTS_ROOT / "split-report.json", split_report)
    write_markdown_reports(assumption_report, filter_report, quality_report, split_report)
    return {
        "quality_report": quality_report,
        "split_report": split_report,
    }


def _render_summary_list(summary: dict[str, float | int | None]) -> list[str]:
    return [
        f"count: {summary['count']}",
        f"min: {summary['min']}",
        f"median: {summary['median']}",
        f"mean: {summary['mean']}",
        f"max: {summary['max']}",
    ]


def write_markdown_reports(
    assumption_report: dict[str, Any],
    filter_report: dict[str, Any],
    quality_report: dict[str, Any],
    split_report: dict[str, Any],
) -> None:
    """Write human-readable Markdown reports under the processed reports root."""

    assumption_lines = [
        "# Assumption Report",
        "",
        f"Generated at: `{assumption_report['generated_at']}`",
        "",
    ]
    for split in SPLITS:
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
    for split in SPLITS:
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
    for split in SPLITS:
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
