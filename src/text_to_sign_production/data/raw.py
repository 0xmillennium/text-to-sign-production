"""Raw How2Sign ingestion helpers for Sprint 2."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ops.progress import iter_with_progress
from .constants import (
    BFH_KEYPOINTS_ROOT,
    EXPECTED_TRANSLATION_COLUMNS,
    INTERIM_REPORTS_ROOT,
    RAW_MANIFESTS_ROOT,
    SPLIT_TO_KEYPOINT_DIR,
    SPLIT_TO_TRANSLATION_FILE,
    SPLITS,
    TRANSLATIONS_DIR,
)
from .jsonl import write_jsonl
from .mp4 import read_video_metadata
from .openpose import inspect_first_frame
from .schemas import RawManifestEntry
from .utils import (
    ensure_directory,
    remove_stale_split_files,
    repo_relative_path,
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


def get_split_paths(split: str) -> SplitPaths:
    """Resolve canonical split paths under the raw root."""

    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    split_dir = BFH_KEYPOINTS_ROOT / SPLIT_TO_KEYPOINT_DIR[split] / "openpose_output"
    return SplitPaths(
        split=split,
        translation_path=TRANSLATIONS_DIR / SPLIT_TO_TRANSLATION_FILE[split],
        keypoints_json_root=split_dir / "json",
        video_root=split_dir / "video",
    )


def _ensure_raw_layout(paths: SplitPaths) -> None:
    missing_paths = [
        str(path)
        for path in (paths.translation_path, paths.keypoints_json_root, paths.video_root)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing expected raw split roots: " + ", ".join(sorted(missing_paths))
        )


def read_translation_rows(paths: SplitPaths) -> list[dict[str, str]]:
    """Read and validate one tab-delimited translation file stored with a `.csv` name."""

    _ensure_raw_layout(paths)
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


def build_raw_manifest_for_split(split: str) -> tuple[list[RawManifestEntry], dict[str, Any]]:
    """Build a raw manifest and schema summary for one split."""

    paths = get_split_paths(split)
    rows = read_translation_rows(paths)
    translation_path_value = repo_relative_path(paths.translation_path)
    keypoints_json_root_value = repo_relative_path(paths.keypoints_json_root)
    video_root_value = repo_relative_path(paths.video_root)
    if (
        translation_path_value is None
        or keypoints_json_root_value is None
        or video_root_value is None
    ):
        raise ValueError(
            f"Expected canonical raw paths to stay under the repo root for split {split}."
        )

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

        keypoints_dir_value = repo_relative_path(keypoints_dir) if keypoints_dir.is_dir() else None
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
                source_video_path=repo_relative_path(video_path),
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


def build_raw_manifests(*, splits: tuple[str, ...] = SPLITS) -> dict[str, Any]:
    """Build raw manifests and the machine-readable assumption report."""

    ensure_directory(RAW_MANIFESTS_ROOT)
    ensure_directory(INTERIM_REPORTS_ROOT)
    raw_root_value = repo_relative_path(BFH_KEYPOINTS_ROOT.parent)
    if raw_root_value is None:
        raise ValueError("Expected the canonical raw root to stay under the repo root.")

    report: dict[str, Any] = {
        "generated_at": utc_timestamp(),
        "raw_root": raw_root_value,
        "translation_columns": list(EXPECTED_TRANSLATION_COLUMNS),
        "splits": {},
    }
    all_sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []

    for split in splits:
        manifest_entries, split_report = build_raw_manifest_for_split(split)
        write_jsonl(RAW_MANIFESTS_ROOT / f"raw_{split}.jsonl", manifest_entries)
        report["splits"][split] = split_report
        for entry in manifest_entries:
            if entry.sample_id in all_sample_ids:
                duplicate_sample_ids.append(entry.sample_id)
            all_sample_ids.add(entry.sample_id)

    remove_stale_split_files(
        RAW_MANIFESTS_ROOT,
        filename_template="raw_{split}.jsonl",
        requested_splits=splits,
        all_splits=SPLITS,
    )
    report["split_integrity"] = {
        "sample_id_overlap_detected": bool(duplicate_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids[:20],
    }
    write_json(INTERIM_REPORTS_ROOT / "assumption-report.json", report)
    return report
