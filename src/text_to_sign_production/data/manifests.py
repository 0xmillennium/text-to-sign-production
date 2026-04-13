"""Processed manifest export and processed sample path policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .constants import (
    FILTERED_MANIFESTS_ROOT,
    PROCESSED_MANIFESTS_ROOT,
    PROCESSED_REPORTS_ROOT,
    RAW_MANIFESTS_ROOT,
    SPLITS,
)
from .jsonl import read_jsonl, write_jsonl
from .reports import build_quality_report, build_split_report, write_markdown_reports
from .schemas import NormalizedManifestEntry, ProcessedManifestEntry, RawManifestEntry
from .utils import (
    ensure_directory,
    resolve_repo_path,
    utc_timestamp,
    write_json,
)


def _processed_samples_root() -> Path:
    return resolve_repo_path("data/processed/v1/samples").resolve()


def resolve_processed_sample_path(path: Path | str) -> Path:
    """Resolve and validate a processed sample path stored in a manifest."""

    raw_value = str(path)
    normalized_value = raw_value.strip()
    if not normalized_value:
        raise ValueError("Processed sample_path must be a non-empty repo-relative .npz path.")

    candidate = Path(normalized_value)
    if candidate.is_absolute():
        raise ValueError(f"Processed sample_path must be repo-relative: {normalized_value}")
    if candidate.suffix != ".npz":
        raise ValueError(f"Processed sample_path must end with .npz: {normalized_value}")

    processed_samples_root = _processed_samples_root()
    resolved = resolve_repo_path(candidate)
    if not resolved.is_relative_to(processed_samples_root):
        raise ValueError(
            f"Processed sample_path must stay under data/processed/v1/samples: {normalized_value}"
        )

    relative_path = resolved.relative_to(processed_samples_root)
    if len(relative_path.parts) != 2:
        raise ValueError(
            "Processed sample_path must follow "
            "data/processed/v1/samples/<split>/<sample_id>.npz: "
            f"{normalized_value}"
        )

    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"Processed sample_path must resolve to a file: {normalized_value}")
    return resolved


def validate_processed_sample_path(
    path: Path | str,
    *,
    split: str,
    sample_id: str,
) -> Path:
    """Resolve and validate a processed sample path against its manifest identity."""

    resolved = resolve_processed_sample_path(path)
    relative_path = resolved.relative_to(_processed_samples_root())
    relative_split, filename = relative_path.parts

    if relative_split != split:
        raise ValueError(
            f"Processed sample_path split does not match manifest split {split}: {path}"
        )

    expected_filename = f"{sample_id}.npz"
    if filename != expected_filename:
        raise ValueError(
            f"Processed sample_path filename does not match manifest sample_id {sample_id}: {path}"
        )

    return resolved


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


def _build_processed_manifest_entry(entry: NormalizedManifestEntry) -> ProcessedManifestEntry:
    if entry.sample_path is None:
        raise ValueError(f"Filtered entry {entry.sample_id} is missing sample_path.")
    if entry.source_keypoints_dir is None:
        raise ValueError(f"Filtered entry {entry.sample_id} is missing source_keypoints_dir.")
    validate_processed_sample_path(
        entry.sample_path,
        split=entry.split,
        sample_id=entry.sample_id,
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
    available_splits = set(dict(report.get("splits", {})))
    missing_splits = [split for split in requested_splits if split not in available_splits]
    if missing_splits:
        missing_display = ", ".join(missing_splits)
        raise ValueError(f"{report_name} is missing requested splits: {missing_display}")


def _remove_stale_manifest_files(requested_splits: tuple[str, ...]) -> None:
    for split in SPLITS:
        if split in requested_splits:
            continue
        manifest_path = PROCESSED_MANIFESTS_ROOT / f"{split}.jsonl"
        if not manifest_path.exists():
            continue
        if not manifest_path.is_file():
            raise ValueError(f"Expected processed manifest path to be a file: {manifest_path}")
        manifest_path.unlink()


def export_final_manifests(
    assumption_report: Mapping[str, Any],
    filter_report: Mapping[str, Any],
    *,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """Build final processed manifests plus JSON and Markdown reports."""

    ensure_directory(PROCESSED_MANIFESTS_ROOT)
    ensure_directory(PROCESSED_REPORTS_ROOT)

    requested_splits = tuple(splits)
    _validate_report_splits("assumption_report", assumption_report, requested_splits)
    _validate_report_splits("filter_report", filter_report, requested_splits)

    raw_records_by_split: dict[str, list[RawManifestEntry]] = {}
    final_records_by_split: dict[str, list[ProcessedManifestEntry]] = {}
    generated_at = utc_timestamp()
    for split in requested_splits:
        raw_records_by_split[split] = _load_raw_records(split)
        filtered_entries = _load_filtered_records(split)
        final_records_by_split[split] = [
            _build_processed_manifest_entry(entry) for entry in filtered_entries
        ]

    _remove_stale_manifest_files(requested_splits)
    for split, final_records in final_records_by_split.items():
        write_jsonl(PROCESSED_MANIFESTS_ROOT / f"{split}.jsonl", final_records)

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

    write_json(PROCESSED_REPORTS_ROOT / "data-quality-report.json", quality_report)
    write_json(PROCESSED_REPORTS_ROOT / "split-report.json", split_report)
    write_markdown_reports(
        assumption_report=assumption_report,
        quality_report=quality_report,
        split_report=split_report,
        splits=requested_splits,
    )
    return {
        "quality_report": quality_report,
        "split_report": split_report,
    }
