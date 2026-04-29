"""Normalization and `.npz` export helpers for Dataset Build."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .constants import (
    CORE_CHANNELS,
    OPENPOSE_CHANNEL_SPECS,
    PROCESSED_SCHEMA_VERSION,
)
from .jsonl import count_jsonl_records, iter_jsonl, write_jsonl
from .openpose import parse_frame
from .schemas import NormalizedManifestEntry, RawManifestEntry
from .utils import (
    ensure_directory,
    iter_with_progress,
    manifest_data_path,
    resolve_data_root,
    resolve_manifest_path,
)


def _manifest_data_path(path: Path, *, data_root: Path | str) -> str:
    return manifest_data_path(path, data_root=data_root)


def _empty_channel_tensor(
    num_frames: int,
    channel: str,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    _, point_count, _ = OPENPOSE_CHANNEL_SPECS[channel]
    return (
        np.zeros((num_frames, point_count, 2), dtype=np.float32),
        np.zeros((num_frames, point_count), dtype=np.float32),
    )


def normalize_split(
    split: str,
    *,
    raw_manifest_path: Path | str,
    normalized_manifest_path: Path | str,
    processed_samples_split_root: Path | str,
    data_root: Path | str,
) -> list[NormalizedManifestEntry]:
    """Normalize all raw-manifest rows for one split."""

    resolve_data_root(data_root)
    resolved_raw_manifest_path = Path(raw_manifest_path)
    resolved_normalized_manifest_path = Path(normalized_manifest_path)
    resolved_processed_samples_split_root = Path(processed_samples_split_root)

    ensure_directory(resolved_normalized_manifest_path.parent)
    ensure_directory(resolved_processed_samples_split_root)

    total_records = count_jsonl_records(resolved_raw_manifest_path)
    normalized_entries: list[NormalizedManifestEntry] = []
    for record in iter_with_progress(
        iter_jsonl(resolved_raw_manifest_path),
        total=total_records,
        desc=f"Normalize {split}",
        unit="samples",
    ):
        raw_entry = RawManifestEntry.from_record(record)
        normalized_entries.append(
            normalize_sample(
                raw_entry,
                processed_samples_split_root=resolved_processed_samples_split_root,
                data_root=data_root,
            )
        )

    write_jsonl(resolved_normalized_manifest_path, normalized_entries)
    return normalized_entries


def normalize_all_splits(
    *,
    splits: tuple[str, ...],
    raw_manifests_root: Path | str,
    normalized_manifests_root: Path | str,
    processed_samples_root: Path | str,
    data_root: Path | str,
) -> None:
    """Normalize every official split."""

    resolve_data_root(data_root)
    resolved_raw_manifests_root = Path(raw_manifests_root)
    resolved_normalized_manifests_root = Path(normalized_manifests_root)
    resolved_processed_samples_root = Path(processed_samples_root)
    for split in splits:
        normalize_split(
            split,
            raw_manifest_path=resolved_raw_manifests_root / f"raw_{split}.jsonl",
            normalized_manifest_path=(
                resolved_normalized_manifests_root / f"normalized_{split}.jsonl"
            ),
            processed_samples_split_root=resolved_processed_samples_root / split,
            data_root=data_root,
        )


def normalize_sample(
    raw_entry: RawManifestEntry,
    *,
    processed_samples_split_root: Path | str,
    data_root: Path | str,
) -> NormalizedManifestEntry:
    """Normalize one raw sample and export a compressed `.npz` when possible."""

    if raw_entry.keypoints_dir is None:
        return NormalizedManifestEntry(
            sample_id=raw_entry.sample_id,
            processed_schema_version=PROCESSED_SCHEMA_VERSION,
            text=raw_entry.text,
            split=raw_entry.source_split,
            start_time=raw_entry.start_time,
            end_time=raw_entry.end_time,
            num_frames=raw_entry.num_frames,
            sample_path=None,
            source_video_id=raw_entry.video_id,
            source_sentence_id=raw_entry.sentence_id,
            source_sentence_name=raw_entry.sentence_name,
            source_metadata_path=raw_entry.source_metadata_path,
            source_keypoints_dir=None,
            source_video_path=raw_entry.source_video_path,
            fps=raw_entry.video_fps,
            video_width=raw_entry.video_width,
            video_height=raw_entry.video_height,
            video_metadata_error=raw_entry.video_metadata_error,
            selected_person_index=0,
            multi_person_frame_count=0,
            max_people_per_frame=0,
            frame_valid_count=0,
            frame_invalid_count=0,
            face_missing_frame_count=0,
            out_of_bounds_coordinate_count=0,
            frames_with_any_zeroed_required_joint=0,
            frame_issue_counts={},
            core_channel_nonzero_frames={channel: 0 for channel in CORE_CHANNELS},
            sample_parse_error=None,
        )

    keypoints_dir = resolve_manifest_path(raw_entry.keypoints_dir, data_root=data_root)
    frame_paths = sorted(keypoints_dir.glob("*.json"))
    if not frame_paths:
        return NormalizedManifestEntry(
            sample_id=raw_entry.sample_id,
            processed_schema_version=PROCESSED_SCHEMA_VERSION,
            text=raw_entry.text,
            split=raw_entry.source_split,
            start_time=raw_entry.start_time,
            end_time=raw_entry.end_time,
            num_frames=0,
            sample_path=None,
            source_video_id=raw_entry.video_id,
            source_sentence_id=raw_entry.sentence_id,
            source_sentence_name=raw_entry.sentence_name,
            source_metadata_path=raw_entry.source_metadata_path,
            source_keypoints_dir=raw_entry.keypoints_dir,
            source_video_path=raw_entry.source_video_path,
            fps=raw_entry.video_fps,
            video_width=raw_entry.video_width,
            video_height=raw_entry.video_height,
            video_metadata_error=raw_entry.video_metadata_error,
            selected_person_index=0,
            multi_person_frame_count=0,
            max_people_per_frame=0,
            frame_valid_count=0,
            frame_invalid_count=0,
            face_missing_frame_count=0,
            out_of_bounds_coordinate_count=0,
            frames_with_any_zeroed_required_joint=0,
            frame_issue_counts={"missing_frame_json_files": 1},
            core_channel_nonzero_frames={channel: 0 for channel in CORE_CHANNELS},
            sample_parse_error=None,
        )

    num_frames = len(frame_paths)
    coord_tensors = {}
    confidence_tensors = {}
    for channel in OPENPOSE_CHANNEL_SPECS:
        coord_tensors[channel], confidence_tensors[channel] = _empty_channel_tensor(
            num_frames, channel
        )

    people_per_frame = np.zeros((num_frames,), dtype=np.int16)
    frame_valid_mask = np.zeros((num_frames,), dtype=np.bool_)

    issue_counter: Counter[str] = Counter()
    face_missing_frame_count = 0
    out_of_bounds_coordinate_count = 0
    frames_with_any_zeroed_required_joint = 0
    multi_person_frame_count = 0
    max_people_per_frame = 0

    try:
        for index, frame_path in enumerate(frame_paths):
            parsed = parse_frame(frame_path)
            people_per_frame[index] = parsed.people_count
            frame_valid_mask[index] = parsed.frame_valid
            max_people_per_frame = max(max_people_per_frame, parsed.people_count)
            if parsed.people_count > 1:
                multi_person_frame_count += 1
            if parsed.face_missing:
                face_missing_frame_count += 1
            if parsed.has_any_zeroed_required_joint:
                frames_with_any_zeroed_required_joint += 1
            out_of_bounds_coordinate_count += parsed.out_of_bounds_coordinate_count
            issue_counter.update(parsed.issue_codes)

            for channel in OPENPOSE_CHANNEL_SPECS:
                coord_tensors[channel][index] = parsed.coords[channel]
                confidence_tensors[channel][index] = parsed.confidences[channel]
    except Exception as exc:  # pragma: no cover - defensive sample-level guard
        return NormalizedManifestEntry(
            sample_id=raw_entry.sample_id,
            processed_schema_version=PROCESSED_SCHEMA_VERSION,
            text=raw_entry.text,
            split=raw_entry.source_split,
            start_time=raw_entry.start_time,
            end_time=raw_entry.end_time,
            num_frames=num_frames,
            sample_path=None,
            source_video_id=raw_entry.video_id,
            source_sentence_id=raw_entry.sentence_id,
            source_sentence_name=raw_entry.sentence_name,
            source_metadata_path=raw_entry.source_metadata_path,
            source_keypoints_dir=raw_entry.keypoints_dir,
            source_video_path=raw_entry.source_video_path,
            fps=raw_entry.video_fps,
            video_width=raw_entry.video_width,
            video_height=raw_entry.video_height,
            video_metadata_error=raw_entry.video_metadata_error,
            selected_person_index=0,
            multi_person_frame_count=multi_person_frame_count,
            max_people_per_frame=max_people_per_frame,
            frame_valid_count=int(frame_valid_mask.sum()),
            frame_invalid_count=int(num_frames - int(frame_valid_mask.sum())),
            face_missing_frame_count=face_missing_frame_count,
            out_of_bounds_coordinate_count=out_of_bounds_coordinate_count,
            frames_with_any_zeroed_required_joint=frames_with_any_zeroed_required_joint,
            frame_issue_counts={str(key): int(value) for key, value in issue_counter.items()},
            core_channel_nonzero_frames={channel: 0 for channel in CORE_CHANNELS},
            sample_parse_error=f"{exc.__class__.__name__}:{exc}",
        )

    core_channel_nonzero_frames = {
        channel: int(np.count_nonzero(np.any(confidence_tensors[channel] > 0.0, axis=1)))
        for channel in CORE_CHANNELS
    }
    resolve_data_root(data_root)
    sample_output_path = Path(processed_samples_split_root) / f"{raw_entry.sample_id}.npz"
    ensure_directory(sample_output_path.parent)
    np.savez_compressed(
        sample_output_path,
        processed_schema_version=np.asarray(PROCESSED_SCHEMA_VERSION),
        body=coord_tensors["body"],
        body_confidence=confidence_tensors["body"],
        left_hand=coord_tensors["left_hand"],
        left_hand_confidence=confidence_tensors["left_hand"],
        right_hand=coord_tensors["right_hand"],
        right_hand_confidence=confidence_tensors["right_hand"],
        face=coord_tensors["face"],
        face_confidence=confidence_tensors["face"],
        people_per_frame=people_per_frame,
        selected_person_index=np.asarray(0, dtype=np.int16),
        frame_valid_mask=frame_valid_mask,
    )

    frame_valid_count = int(frame_valid_mask.sum())
    return NormalizedManifestEntry(
        sample_id=raw_entry.sample_id,
        processed_schema_version=PROCESSED_SCHEMA_VERSION,
        text=raw_entry.text,
        split=raw_entry.source_split,
        start_time=raw_entry.start_time,
        end_time=raw_entry.end_time,
        num_frames=num_frames,
        sample_path=_manifest_data_path(sample_output_path, data_root=data_root),
        source_video_id=raw_entry.video_id,
        source_sentence_id=raw_entry.sentence_id,
        source_sentence_name=raw_entry.sentence_name,
        source_metadata_path=raw_entry.source_metadata_path,
        source_keypoints_dir=raw_entry.keypoints_dir,
        source_video_path=raw_entry.source_video_path,
        fps=raw_entry.video_fps,
        video_width=raw_entry.video_width,
        video_height=raw_entry.video_height,
        video_metadata_error=raw_entry.video_metadata_error,
        selected_person_index=0,
        multi_person_frame_count=multi_person_frame_count,
        max_people_per_frame=max_people_per_frame,
        frame_valid_count=frame_valid_count,
        frame_invalid_count=num_frames - frame_valid_count,
        face_missing_frame_count=face_missing_frame_count,
        out_of_bounds_coordinate_count=out_of_bounds_coordinate_count,
        frames_with_any_zeroed_required_joint=frames_with_any_zeroed_required_joint,
        frame_issue_counts={str(key): int(value) for key, value in sorted(issue_counter.items())},
        core_channel_nonzero_frames=core_channel_nonzero_frames,
        sample_parse_error=None,
    )
