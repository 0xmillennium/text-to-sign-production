"""Internal record parsing helpers for sample contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.samples.types import (
    FrameQualitySummary,
    SelectedPersonMetadata,
)


def require_mapping(value: object, field_name: str, *, surface: str) -> Mapping[str, Any]:
    """Require a mapping while naming the sample surface being parsed."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{surface} field {field_name!r} must be a mapping.")
    return value


def optional_mapping(
    value: object, field_name: str, *, surface: str
) -> Mapping[str, Any] | None:
    """Return an optional mapping while rejecting non-mapping values."""
    if value is None:
        return None
    return require_mapping(value, field_name, surface=surface)


def sample_split_from_record(value: object, field_name: str) -> SampleSplit:
    """Parse the canonical split identity from a record value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string split value.")
    try:
        return SampleSplit(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of {[split.value for split in SampleSplit]}, "
            f"got {value!r}."
        ) from exc


def selected_person_from_record(record: Mapping[str, Any]) -> SelectedPersonMetadata:
    """Parse selected-person metadata from a record mapping."""
    return SelectedPersonMetadata(
        index=int(record["index"]),
        multi_person_frame_count=int(record["multi_person_frame_count"]),
        max_people_per_frame=int(record["max_people_per_frame"]),
    )


def frame_quality_from_record(record: Mapping[str, Any]) -> FrameQualitySummary:
    """Parse frame-quality metadata from a record mapping."""
    return FrameQualitySummary(
        valid_frame_count=int(record["valid_frame_count"]),
        invalid_frame_count=int(record["invalid_frame_count"]),
        face_missing_frame_count=int(record["face_missing_frame_count"]),
        out_of_bounds_coordinate_count=int(record["out_of_bounds_coordinate_count"]),
        frames_with_any_zeroed_canonical_joint=int(
            record["frames_with_any_zeroed_canonical_joint"]
        ),
        frame_issue_counts=int_mapping(record.get("frame_issue_counts", {})),
        channel_nonzero_frames=int_mapping(record.get("channel_nonzero_frames", {})),
    )


def int_mapping(value: object) -> dict[str, int]:
    """Parse a string-keyed integer count mapping."""
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping of string keys to integer counts.")
    return {str(key): int(item) for key, item in value.items()}


def string_tuple_from_sequence(value: object, field_name: str) -> tuple[str, ...]:
    """Parse a non-string sequence of text values."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    return tuple(str(item) for item in value)


def bool_from_record(value: object, field_name: str) -> bool:
    """Parse a strict boolean field."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean value, got {type(value).__name__}.")
    return value
