"""Report construction and rendering for the Dataset Build data pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from .constants import SPLITS
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


def _report_split_mapping(
    report: Mapping[str, Any],
    *,
    report_name: str,
    split: str,
) -> Mapping[str, Any]:
    splits_payload = report.get("splits")
    if not isinstance(splits_payload, Mapping):
        raise ValueError(f"{report_name} is missing a splits mapping.")
    try:
        split_payload = splits_payload[split]
    except KeyError as exc:
        raise ValueError(f"{report_name} is missing requested split: {split}") from exc
    if not isinstance(split_payload, Mapping):
        raise ValueError(f"{report_name} split {split} must be a mapping.")
    return split_payload


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
        split_filter_report = dict(
            _report_split_mapping(filter_report, report_name="filter_report", split=split)
        )
        frame_counts = [record.num_frames for record in final_records]
        total_frame_count = sum(frame_counts)
        text_lengths = [len(record.text) for record in final_records]
        fps_values = [record.fps for record in final_records if record.fps is not None]
        face_missing_ratios = [
            (record.face_missing_frame_count / record.num_frames)
            for record in final_records
            if record.num_frames > 0
        ]
        frames_with_zeroed_parser_required_point = sum(
            record.frames_with_any_zeroed_required_joint for record in final_records
        )
        quality_report["splits"][split] = {
            "generated_at": generated_at,
            "split": split,
            "processed_sample_count": len(final_records),
            "dropped_sample_count": split_filter_report["dropped_samples"],
            "drop_reason_counts": split_filter_report["drop_reason_counts"],
            "total_frame_count": total_frame_count,
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
            "frames_with_any_zeroed_parser_required_point": (
                frames_with_zeroed_parser_required_point
            ),
            "frames_with_any_zeroed_parser_required_point_ratio": (
                round(frames_with_zeroed_parser_required_point / total_frame_count, 6)
                if total_frame_count > 0
                else None
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
    sample_ids_by_split: dict[str, set[str]] = {}
    video_ids_by_split: dict[str, set[str]] = {}
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
        sample_ids = {record.sample_id for record in final_records}
        video_ids = {record.source_video_id for record in final_records}
        sample_ids_by_split[split] = sample_ids
        video_ids_by_split[split] = video_ids
        split_report["splits"][split] = {
            "generated_at": generated_at,
            "split": split,
            "compared_splits": list(requested_splits),
            "cross_split_overlap_available": len(requested_splits) > 1,
            "raw_samples": len(raw_entries),
            "processed_samples": len(final_records),
            "raw_video_count": len({entry.video_id for entry in raw_entries}),
            "processed_video_count": len(video_ids),
            "sample_id_overlap_with_other_splits": {},
            "sample_id_overlap_with_compared_splits": {},
            "video_overlap_with_compared_splits": {},
        }

    for split in requested_splits:
        overlaps = {
            other_split: sorted(
                sample_ids_by_split[split].intersection(sample_ids_by_split[other_split])
            )[:10]
            for other_split in requested_splits
            if other_split != split
        }
        video_overlaps = {
            other_split: len(
                video_ids_by_split[split].intersection(video_ids_by_split[other_split])
            )
            for other_split in requested_splits
            if other_split != split
        }
        split_report["splits"][split]["sample_id_overlap_with_other_splits"] = overlaps
        split_report["splits"][split]["sample_id_overlap_with_compared_splits"] = overlaps
        split_report["splits"][split]["video_overlap_with_compared_splits"] = video_overlaps
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
