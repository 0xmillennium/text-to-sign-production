"""Construction and record conversions for manifest entries."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from text_to_sign_production.data.samples.schema import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.data.samples.types import (
    DroppedManifestEntry,
    FrameQualitySummary,
    JsonValue,
    ManifestEntry,
    PassedManifestEntry,
    SampleStatus,
    SelectedPersonMetadata,
)


def build_passed_entry(
    sample_id: str,
    text: str,
    split: str,
    num_frames: int,
    fps: float | None,
    sample_path: str,
    source_video_id: str,
    source_sentence_id: str,
    source_sentence_name: str,
    selected_person: SelectedPersonMetadata,
    frame_quality: FrameQualitySummary,
) -> PassedManifestEntry:
    """Build a semantically valid passed manifest entry."""
    return PassedManifestEntry(
        sample_id=sample_id,
        schema_version=PROCESSED_SCHEMA_VERSION,
        status=SampleStatus.PASSED,
        text=text,
        split=split,
        num_frames=num_frames,
        fps=fps,
        sample_path=sample_path,
        source_video_id=source_video_id,
        source_sentence_id=source_sentence_id,
        source_sentence_name=source_sentence_name,
        selected_person=selected_person,
        frame_quality=frame_quality,
    )


def build_dropped_entry(
    sample_id: str,
    split: str,
    drop_stage: str,
    drop_reasons: tuple[str, ...],
    debug_only: bool = False,
    sample_path: str | None = None,
    drop_details: dict[str, JsonValue] | None = None,
    text: str | None = None,
    num_frames: int | None = None,
    fps: float | None = None,
    selected_person: SelectedPersonMetadata | None = None,
    frame_quality: FrameQualitySummary | None = None,
) -> DroppedManifestEntry:
    """Build a semantically valid dropped manifest entry."""
    return DroppedManifestEntry(
        sample_id=sample_id,
        schema_version=PROCESSED_SCHEMA_VERSION,
        status=SampleStatus.DROPPED,
        split=split,
        drop_stage=drop_stage,
        drop_reasons=drop_reasons,
        debug_only=debug_only,
        sample_path=sample_path,
        drop_details=drop_details or {},
        text=text,
        num_frames=num_frames,
        fps=fps,
        selected_person=selected_person,
        frame_quality=frame_quality,
    )


def manifest_entry_from_record(record: Mapping[str, Any]) -> ManifestEntry:
    """Parse a manifest entry record dictionary into its typed model."""
    status = record.get("status")

    if status == SampleStatus.PASSED.value:
        selected_person_record = _require_mapping(record["selected_person"], "selected_person")
        frame_quality_record = _require_mapping(record["frame_quality"], "frame_quality")
        return PassedManifestEntry(
            sample_id=str(record["sample_id"]),
            schema_version=str(record["schema_version"]),
            status=SampleStatus.PASSED,
            text=str(record["text"]),
            split=str(record["split"]),
            num_frames=int(record["num_frames"]),
            fps=float(record["fps"]) if record.get("fps") is not None else None,
            sample_path=str(record["sample_path"]),
            source_video_id=str(record["source_video_id"]),
            source_sentence_id=str(record["source_sentence_id"]),
            source_sentence_name=str(record["source_sentence_name"]),
            selected_person=_selected_person_from_record(selected_person_record),
            frame_quality=_frame_quality_from_record(frame_quality_record),
        )
    if status == SampleStatus.DROPPED.value:
        dropped_selected_person_record = _optional_mapping(
            record.get("selected_person"), "selected_person"
        )
        dropped_frame_quality_record = _optional_mapping(
            record.get("frame_quality"), "frame_quality"
        )
        return DroppedManifestEntry(
            sample_id=str(record["sample_id"]),
            schema_version=str(record["schema_version"]),
            status=SampleStatus.DROPPED,
            split=str(record["split"]),
            drop_stage=str(record["drop_stage"]),
            drop_reasons=_drop_reasons_from_record(record["drop_reasons"]),
            debug_only=_bool_from_record(record["debug_only"]),
            sample_path=(
                str(record["sample_path"]) if record.get("sample_path") is not None else None
            ),
            drop_details=cast(dict[str, JsonValue], dict(record.get("drop_details", {}))),
            text=str(record["text"]) if "text" in record else None,
            num_frames=int(record["num_frames"]) if record.get("num_frames") is not None else None,
            fps=float(record["fps"]) if record.get("fps") is not None else None,
            selected_person=(
                _selected_person_from_record(dropped_selected_person_record)
                if dropped_selected_person_record is not None
                else None
            ),
            frame_quality=(
                _frame_quality_from_record(dropped_frame_quality_record)
                if dropped_frame_quality_record is not None
                else None
            ),
        )
    raise ValueError(f"Unknown or missing status '{status}' in record.")


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Manifest field {field_name!r} must be a mapping.")
    return value


def _optional_mapping(value: object, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, field_name)


def _selected_person_from_record(record: Mapping[str, Any]) -> SelectedPersonMetadata:
    return SelectedPersonMetadata(
        index=int(record["index"]),
        multi_person_frame_count=int(record["multi_person_frame_count"]),
        max_people_per_frame=int(record["max_people_per_frame"]),
    )


def _frame_quality_from_record(record: Mapping[str, Any]) -> FrameQualitySummary:
    return FrameQualitySummary(
        valid_frame_count=int(record["valid_frame_count"]),
        invalid_frame_count=int(record["invalid_frame_count"]),
        face_missing_frame_count=int(record["face_missing_frame_count"]),
        out_of_bounds_coordinate_count=int(record["out_of_bounds_coordinate_count"]),
        frames_with_any_zeroed_required_joint=int(record["frames_with_any_zeroed_required_joint"]),
        frame_issue_counts=_int_mapping(record.get("frame_issue_counts", {})),
        channel_nonzero_frames=_int_mapping(record.get("channel_nonzero_frames", {})),
    )


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping of string keys to integer counts.")
    return {str(key): int(item) for key, item in value.items()}


def _drop_reasons_from_record(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Manifest drop_reasons must be a sequence of strings.")
    return tuple(str(reason) for reason in value)


def _bool_from_record(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Expected a boolean value, got {type(value).__name__}.")
    return value
