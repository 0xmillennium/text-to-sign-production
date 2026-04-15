"""Typed record models for the Dataset Build data pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass(slots=True)
class VideoMetadata:
    """Minimal MP4 header metadata needed by the processed manifest."""

    width: int | None
    height: int | None
    fps: float | None
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawManifestEntry:
    """A single row in a split-specific raw manifest."""

    sample_id: str
    source_split: str
    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    text: str
    start_time: float
    end_time: float
    keypoints_dir: str | None
    source_metadata_path: str
    has_face: bool | None
    num_frames: int
    source_video_path: str | None = None
    video_width: int | None = None
    video_height: int | None = None
    video_fps: float | None = None
    video_metadata_error: str | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> RawManifestEntry:
        return cls(
            sample_id=str(record["sample_id"]),
            source_split=str(record["source_split"]),
            video_id=str(record["video_id"]),
            video_name=str(record["video_name"]),
            sentence_id=str(record["sentence_id"]),
            sentence_name=str(record["sentence_name"]),
            text=str(record["text"]),
            start_time=float(record["start_time"]),
            end_time=float(record["end_time"]),
            keypoints_dir=record.get("keypoints_dir"),
            source_metadata_path=str(record["source_metadata_path"]),
            has_face=record.get("has_face"),
            num_frames=int(record["num_frames"]),
            source_video_path=record.get("source_video_path"),
            video_width=_int_or_none(record.get("video_width")),
            video_height=_int_or_none(record.get("video_height")),
            video_fps=_float_or_none(record.get("video_fps")),
            video_metadata_error=record.get("video_metadata_error"),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NormalizedManifestEntry:
    """A normalized candidate sample written before structural filtering."""

    sample_id: str
    processed_schema_version: str
    text: str
    split: str
    start_time: float
    end_time: float
    num_frames: int
    sample_path: str | None
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    source_metadata_path: str
    source_keypoints_dir: str | None
    source_video_path: str | None
    fps: float | None
    video_width: int | None
    video_height: int | None
    video_metadata_error: str | None
    selected_person_index: int
    multi_person_frame_count: int
    max_people_per_frame: int
    frame_valid_count: int
    frame_invalid_count: int
    face_missing_frame_count: int
    out_of_bounds_coordinate_count: int
    frames_with_any_zeroed_required_joint: int
    frame_issue_counts: dict[str, int] = field(default_factory=dict)
    core_channel_nonzero_frames: dict[str, int] = field(default_factory=dict)
    sample_parse_error: str | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> NormalizedManifestEntry:
        return cls(
            sample_id=str(record["sample_id"]),
            processed_schema_version=str(record["processed_schema_version"]),
            text=str(record["text"]),
            split=str(record["split"]),
            start_time=float(record["start_time"]),
            end_time=float(record["end_time"]),
            num_frames=int(record["num_frames"]),
            sample_path=record.get("sample_path"),
            source_video_id=str(record["source_video_id"]),
            source_sentence_id=str(record["source_sentence_id"]),
            source_sentence_name=str(record["source_sentence_name"]),
            source_metadata_path=str(record["source_metadata_path"]),
            source_keypoints_dir=record.get("source_keypoints_dir"),
            source_video_path=record.get("source_video_path"),
            fps=_float_or_none(record.get("fps")),
            video_width=_int_or_none(record.get("video_width")),
            video_height=_int_or_none(record.get("video_height")),
            video_metadata_error=record.get("video_metadata_error"),
            selected_person_index=int(record["selected_person_index"]),
            multi_person_frame_count=int(record["multi_person_frame_count"]),
            max_people_per_frame=int(record["max_people_per_frame"]),
            frame_valid_count=int(record["frame_valid_count"]),
            frame_invalid_count=int(record["frame_invalid_count"]),
            face_missing_frame_count=int(record["face_missing_frame_count"]),
            out_of_bounds_coordinate_count=int(record["out_of_bounds_coordinate_count"]),
            frames_with_any_zeroed_required_joint=int(
                record["frames_with_any_zeroed_required_joint"]
            ),
            frame_issue_counts={
                str(key): int(value)
                for key, value in dict(record.get("frame_issue_counts", {})).items()
            },
            core_channel_nonzero_frames={
                str(key): int(value)
                for key, value in dict(record.get("core_channel_nonzero_frames", {})).items()
            },
            sample_parse_error=record.get("sample_parse_error"),
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessedManifestEntry:
    """A final training-facing processed manifest entry."""

    sample_id: str
    processed_schema_version: str
    text: str
    split: str
    fps: float | None
    num_frames: int
    sample_path: str
    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str
    selected_person_index: int
    multi_person_frame_count: int
    max_people_per_frame: int
    source_metadata_path: str
    source_keypoints_dir: str
    source_video_path: str | None
    video_width: int | None
    video_height: int | None
    video_metadata_error: str | None
    frame_valid_count: int
    frame_invalid_count: int
    face_missing_frame_count: int
    out_of_bounds_coordinate_count: int
    frames_with_any_zeroed_required_joint: int
    frame_issue_counts: dict[str, int] = field(default_factory=dict)
    core_channel_nonzero_frames: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ProcessedManifestEntry:
        return cls(
            sample_id=str(record["sample_id"]),
            processed_schema_version=str(record["processed_schema_version"]),
            text=str(record["text"]),
            split=str(record["split"]),
            fps=_float_or_none(record.get("fps")),
            num_frames=int(record["num_frames"]),
            sample_path=str(record["sample_path"]),
            source_video_id=str(record["source_video_id"]),
            source_sentence_id=str(record["source_sentence_id"]),
            source_sentence_name=str(record["source_sentence_name"]),
            selected_person_index=int(record["selected_person_index"]),
            multi_person_frame_count=int(record["multi_person_frame_count"]),
            max_people_per_frame=int(record["max_people_per_frame"]),
            source_metadata_path=str(record["source_metadata_path"]),
            source_keypoints_dir=str(record["source_keypoints_dir"]),
            source_video_path=record.get("source_video_path"),
            video_width=_int_or_none(record.get("video_width")),
            video_height=_int_or_none(record.get("video_height")),
            video_metadata_error=record.get("video_metadata_error"),
            frame_valid_count=int(record["frame_valid_count"]),
            frame_invalid_count=int(record["frame_invalid_count"]),
            face_missing_frame_count=int(record["face_missing_frame_count"]),
            out_of_bounds_coordinate_count=int(record["out_of_bounds_coordinate_count"]),
            frames_with_any_zeroed_required_joint=int(
                record["frames_with_any_zeroed_required_joint"]
            ),
            frame_issue_counts={
                str(key): int(value)
                for key, value in dict(record.get("frame_issue_counts", {})).items()
            },
            core_channel_nonzero_frames={
                str(key): int(value)
                for key, value in dict(record.get("core_channel_nonzero_frames", {})).items()
            },
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationIssue:
    """A machine-readable manifest validation issue."""

    severity: str
    code: str
    message: str
    sample_id: str | None = None
    split: str | None = None
    path: str | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
