"""Processed manifest export orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from text_to_sign_production.core.progress import ProgressReporter

from .jsonl import read_jsonl, write_jsonl
from .quality import build_quality_outputs, load_quality_tier_config
from .reports import build_quality_report, build_split_report
from .schemas import NormalizedManifestEntry, ProcessedManifestEntry, RawManifestEntry
from .utils import (
    ensure_directory,
    iter_with_progress,
    resolve_data_root,
    utc_timestamp,
)
from .validate import validate_processed_sample_path


def _load_raw_records(
    split: str,
    *,
    raw_manifests_root: Path,
) -> list[RawManifestEntry]:
    return [
        RawManifestEntry.from_record(record)
        for record in read_jsonl(raw_manifests_root / f"raw_{split}.jsonl")
    ]


def _load_filtered_records(
    split: str,
    *,
    filtered_manifests_root: Path,
) -> list[NormalizedManifestEntry]:
    return [
        NormalizedManifestEntry.from_record(record)
        for record in read_jsonl(filtered_manifests_root / f"filtered_{split}.jsonl")
    ]


def _build_processed_manifest_entry(
    entry: NormalizedManifestEntry,
    *,
    processed_samples_root: Path,
    data_root: Path | str,
) -> ProcessedManifestEntry:
    if entry.sample_path is None:
        raise ValueError(f"Filtered entry {entry.sample_id} is missing sample_path.")
    if entry.source_keypoints_dir is None:
        raise ValueError(f"Filtered entry {entry.sample_id} is missing source_keypoints_dir.")
    validate_processed_sample_path(
        entry.sample_path,
        split=entry.split,
        sample_id=entry.sample_id,
        processed_samples_root=processed_samples_root,
        data_root=data_root,
    )

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


def _validate_report_splits(
    report_name: str, report: Mapping[str, Any], requested_splits: tuple[str, ...]
) -> None:
    splits_payload = report.get("splits")
    if not isinstance(splits_payload, Mapping):
        raise ValueError(f"{report_name} is missing a splits mapping.")
    available_splits = set(splits_payload)
    missing_splits = [split for split in requested_splits if split not in available_splits]
    if missing_splits:
        missing_display = ", ".join(missing_splits)
        raise ValueError(f"{report_name} is missing requested splits: {missing_display}")


def export_final_manifests(
    assumption_report: Mapping[str, Any],
    filter_report: Mapping[str, Any],
    *,
    splits: tuple[str, ...],
    raw_manifests_root: Path | str,
    filtered_manifests_root: Path | str,
    processed_manifests_root: Path | str,
    processed_samples_root: Path | str,
    data_root: Path | str,
    quality_config_path: Path | str | None = None,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Build final processed manifests and return processed report facts."""

    resolve_data_root(data_root)
    resolved_raw_manifests_root = Path(raw_manifests_root)
    resolved_filtered_manifests_root = Path(filtered_manifests_root)
    resolved_processed_manifests_root = Path(processed_manifests_root)
    resolved_processed_samples_root = Path(processed_samples_root)
    ensure_directory(resolved_processed_manifests_root)

    requested_splits = tuple(splits)
    _validate_report_splits("assumption_report", assumption_report, requested_splits)
    _validate_report_splits("filter_report", filter_report, requested_splits)

    raw_records_by_split: dict[str, list[RawManifestEntry]] = {}
    final_records_by_split: dict[str, list[ProcessedManifestEntry]] = {}
    generated_at = utc_timestamp()
    for split in requested_splits:
        raw_records_by_split[split] = _load_raw_records(
            split,
            raw_manifests_root=resolved_raw_manifests_root,
        )
        filtered_entries = _load_filtered_records(
            split,
            filtered_manifests_root=resolved_filtered_manifests_root,
        )
        final_records_by_split[split] = [
            _build_processed_manifest_entry(
                entry,
                processed_samples_root=resolved_processed_samples_root,
                data_root=data_root,
            )
            for entry in iter_with_progress(
                filtered_entries,
                total=len(filtered_entries),
                desc=f"Export processed manifest {split}",
                unit="records",
            )
        ]

    for split, final_records in final_records_by_split.items():
        write_jsonl(resolved_processed_manifests_root / f"{split}.jsonl", final_records)

    quality_report = build_quality_report(
        final_records_by_split=final_records_by_split,
        filter_report=filter_report,
        generated_at=generated_at,
        splits=requested_splits,
    )
    split_report = build_split_report(
        raw_records_by_split=raw_records_by_split,
        final_records_by_split=final_records_by_split,
        generated_at=generated_at,
        splits=requested_splits,
    )
    quality_outputs: dict[str, Any] | None = None
    if quality_config_path is not None:
        quality_config = load_quality_tier_config(quality_config_path)
        quality_outputs = build_quality_outputs(
            records_by_split=final_records_by_split,
            dropped_records_by_split=_dropped_records_by_split(filter_report, requested_splits),
            config=quality_config,
            data_root=data_root,
            generated_at=generated_at,
            reporter=reporter,
            splits=requested_splits,
        )

    return {
        "quality_report": quality_report,
        "split_report": split_report,
        "quality_outputs": quality_outputs,
    }


def _dropped_records_by_split(
    filter_report: Mapping[str, Any],
    requested_splits: tuple[str, ...],
) -> dict[str, list[Mapping[str, Any]]]:
    splits_payload = filter_report.get("splits")
    if not isinstance(splits_payload, Mapping):
        return {split: [] for split in requested_splits}
    dropped: dict[str, list[Mapping[str, Any]]] = {}
    for split in requested_splits:
        split_payload = splits_payload.get(split)
        if not isinstance(split_payload, Mapping):
            dropped[split] = []
            continue
        raw_records = split_payload.get("dropped_records", [])
        if not isinstance(raw_records, list):
            dropped[split] = []
            continue
        dropped[split] = [record for record in raw_records if isinstance(record, Mapping)]
    return dropped
