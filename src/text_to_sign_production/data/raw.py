"""Raw How2Sign ingestion helpers for Dataset Build."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import paths as core_paths
from ..core.paths import ProjectLayout, data_root_relative_path, repo_relative_path
from ..core.progress import iter_with_progress
from .constants import (
    EXPECTED_TRANSLATION_COLUMNS,
    SPLITS,
)
from .jsonl import write_jsonl
from .mp4 import read_video_metadata
from .openpose import inspect_first_frame
from .schemas import RawManifestEntry
from .utils import (
    ensure_directory,
    remove_stale_split_files,
    utc_timestamp,
    write_json,
)


def _counter_to_sorted_mapping(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


@dataclass(frozen=True, slots=True)
class SplitPaths:
    """Paths for one official How2Sign split."""

    split: str
    translation_path: Path
    keypoints_json_root: Path
    video_root: Path


def _manifest_data_path(path: Path, *, data_root: Path | str | None = None) -> str:
    if data_root is None:
        return repo_relative_path(path)
    return (Path("data") / data_root_relative_path(path, data_root=data_root)).as_posix()


def _layout_from_data_root(data_root: Path | str | None = None) -> ProjectLayout:
    if data_root is None:
        return ProjectLayout(core_paths.DEFAULT_REPO_ROOT)
    resolved_data_root = Path(data_root).expanduser().resolve()
    if resolved_data_root.name != "data":
        raise ValueError(
            f"data_root must point at the project data directory: {resolved_data_root}"
        )
    return ProjectLayout(resolved_data_root.parent)


def get_split_paths(
    split: str,
    *,
    layout: ProjectLayout | None = None,
    data_root: Path | str | None = None,
) -> SplitPaths:
    """Resolve split paths from a caller-provided project layout."""

    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    resolved_layout = _layout_from_data_root(data_root) if layout is None else layout
    split_root = resolved_layout.raw_bfh_keypoints_split_root(split)
    return SplitPaths(
        split=split,
        translation_path=resolved_layout.how2sign_translation_file(split),
        keypoints_json_root=split_root / "json",
        video_root=split_root / "video",
    )


def read_translation_rows(paths: SplitPaths) -> list[dict[str, str]]:
    """Read and validate one tab-delimited translation file stored with a `.csv` name."""

    with paths.translation_path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")
        if first_line == "":
            raise ValueError(f"Translation file {paths.translation_path} is empty.")
        if "\t" not in first_line:
            raise ValueError(f"Translation file {paths.translation_path} is not tab-delimited.")
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Translation file {paths.translation_path} is empty.")
        observed_columns = tuple(reader.fieldnames)
        if observed_columns != EXPECTED_TRANSLATION_COLUMNS:
            raise ValueError(
                f"Unexpected translation columns in {paths.translation_path}: {observed_columns}"
            )
        return [dict(row) for row in reader]


def build_raw_manifest_for_split(
    split: str,
    *,
    split_paths: SplitPaths | None = None,
    data_root: Path | str | None = None,
) -> tuple[list[RawManifestEntry], dict[str, Any]]:
    """Build a raw manifest and schema summary for one split."""

    paths = split_paths if split_paths is not None else get_split_paths(split, data_root=data_root)
    rows = read_translation_rows(paths)
    translation_path_value = _manifest_data_path(paths.translation_path, data_root=data_root)
    keypoints_json_root_value = _manifest_data_path(paths.keypoints_json_root, data_root=data_root)
    video_root_value = _manifest_data_path(paths.video_root, data_root=data_root)

    top_level_key_counter: Counter[str] = Counter()
    person_key_counter: Counter[str] = Counter()
    channel_length_counter: dict[str, Counter[str]] = {}
    first_frame_people_counter: Counter[str] = Counter()
    schema_deviation_counter: Counter[str] = Counter()
    video_dimension_counter: Counter[str] = Counter()
    video_fps_counter: Counter[str] = Counter()

    manifest_entries: list[RawManifestEntry] = []
    unmatched_examples: list[str] = []
    readable_video_metadata = 0
    unreadable_video_metadata = 0

    for row in iter_with_progress(
        rows,
        total=len(rows),
        desc=f"Raw manifest {split}",
        unit="rows",
    ):
        sentence_name = row["SENTENCE_NAME"]
        keypoints_dir = paths.keypoints_json_root / sentence_name
        video_path = paths.video_root / f"{sentence_name}.mp4"

        has_face: bool | None = None
        num_frames = 0
        video_width: int | None = None
        video_height: int | None = None
        video_fps: float | None = None
        video_metadata_error: str | None = None

        keypoints_dir_value = (
            _manifest_data_path(keypoints_dir, data_root=data_root)
            if keypoints_dir.is_dir()
            else None
        )
        if keypoints_dir_value is None:
            unmatched_examples.append(sentence_name)
        else:
            frame_paths = list(keypoints_dir.glob("*.json"))
            num_frames = len(frame_paths)
            if num_frames == 0:
                schema_deviation_counter["missing_frame_json_files"] += 1
            else:
                inspection = inspect_first_frame(min(frame_paths, key=lambda path: path.name))
                has_face = inspection.has_face
                first_frame_people_counter[str(inspection.people_count)] += 1
                top_level_key_counter.update(inspection.top_level_keys)
                person_key_counter.update(inspection.person_keys)
                for channel_key, channel_length in inspection.channel_lengths.items():
                    channel_counter = channel_length_counter.setdefault(channel_key, Counter())
                    channel_counter[str(channel_length)] += 1
                schema_deviation_counter.update(inspection.issue_codes)

            metadata = read_video_metadata(video_path)
            video_width = metadata.width
            video_height = metadata.height
            video_fps = metadata.fps
            video_metadata_error = metadata.error
            if metadata.error is None:
                readable_video_metadata += 1
            else:
                unreadable_video_metadata += 1
                schema_deviation_counter[f"video_metadata:{metadata.error}"] += 1
            if metadata.width is not None and metadata.height is not None:
                video_dimension_counter[f"{metadata.width}x{metadata.height}"] += 1
            if metadata.fps is not None:
                video_fps_counter[str(metadata.fps)] += 1

        manifest_entries.append(
            RawManifestEntry(
                sample_id=sentence_name,
                source_split=split,
                video_id=row["VIDEO_ID"],
                video_name=row["VIDEO_NAME"],
                sentence_id=row["SENTENCE_ID"],
                sentence_name=sentence_name,
                text=row["SENTENCE"],
                start_time=float(row["START_REALIGNED"]),
                end_time=float(row["END_REALIGNED"]),
                keypoints_dir=keypoints_dir_value,
                source_metadata_path=translation_path_value,
                has_face=has_face,
                num_frames=num_frames,
                source_video_path=_manifest_data_path(video_path, data_root=data_root),
                video_width=video_width,
                video_height=video_height,
                video_fps=video_fps,
                video_metadata_error=video_metadata_error,
            )
        )

    matched_count = sum(1 for entry in manifest_entries if entry.keypoints_dir is not None)
    if matched_count == 0:
        raise ValueError(f"No matched raw samples found for split {split}.")

    report: dict[str, Any] = {
        "split": split,
        "translation_path": translation_path_value,
        "keypoints_json_root": keypoints_json_root_value,
        "video_root": video_root_value,
        "translation_row_count": len(rows),
        "matched_sample_count": matched_count,
        "unmatched_sample_count": len(rows) - matched_count,
        "unmatched_examples": unmatched_examples[:20],
        "first_frame_people_counter": _counter_to_sorted_mapping(first_frame_people_counter),
        "openpose_schema": {
            "top_level_keys": _counter_to_sorted_mapping(top_level_key_counter),
            "person_keys": _counter_to_sorted_mapping(person_key_counter),
            "channel_lengths": {
                key: _counter_to_sorted_mapping(counter)
                for key, counter in sorted(channel_length_counter.items())
            },
            "deviation_counts": _counter_to_sorted_mapping(schema_deviation_counter),
        },
        "video_metadata": {
            "readable_count": readable_video_metadata,
            "unreadable_count": unreadable_video_metadata,
            "dimension_counts": _counter_to_sorted_mapping(video_dimension_counter),
            "fps_counts": _counter_to_sorted_mapping(video_fps_counter),
        },
    }
    return manifest_entries, report


def build_raw_manifests(
    *,
    splits: tuple[str, ...] = SPLITS,
    split_paths_by_split: Mapping[str, SplitPaths] | None = None,
    raw_manifests_root: Path | str | None = None,
    interim_reports_root: Path | str | None = None,
    raw_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build raw manifests and the machine-readable assumption report."""

    layout = _layout_from_data_root(data_root)
    resolved_raw_manifests_root = (
        layout.raw_manifests_root if raw_manifests_root is None else Path(raw_manifests_root)
    )
    resolved_interim_reports_root = (
        layout.interim_reports_root if interim_reports_root is None else Path(interim_reports_root)
    )
    resolved_raw_root = layout.how2sign_root if raw_root is None else Path(raw_root)
    ensure_directory(resolved_raw_manifests_root)
    ensure_directory(resolved_interim_reports_root)
    raw_root_value = _manifest_data_path(resolved_raw_root, data_root=data_root)

    report: dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "raw_root": raw_root_value,
        "translation_columns": list(EXPECTED_TRANSLATION_COLUMNS),
        "splits": {},
    }
    all_sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []

    for split in splits:
        manifest_entries, split_report = build_raw_manifest_for_split(
            split,
            split_paths=(None if split_paths_by_split is None else split_paths_by_split[split]),
            data_root=data_root,
        )
        write_jsonl(resolved_raw_manifests_root / f"raw_{split}.jsonl", manifest_entries)
        report["splits"][split] = split_report
        for entry in manifest_entries:
            if entry.sample_id in all_sample_ids:
                duplicate_sample_ids.append(entry.sample_id)
            all_sample_ids.add(entry.sample_id)

    remove_stale_split_files(
        resolved_raw_manifests_root,
        filename_template="raw_{split}.jsonl",
        requested_splits=splits,
        all_splits=SPLITS,
    )
    report["split_integrity"] = {
        "sample_id_overlap_detected": bool(duplicate_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids[:20],
    }
    write_json(resolved_interim_reports_root / "assumption-report.json", report)
    return report
