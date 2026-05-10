"""Internal record parsing helpers for sample contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, cast

from text_to_sign_production.core.ids import SampleSplit, SampleStatus
from text_to_sign_production.legacy_data._shared.validate import format_keys, missing_keys
from text_to_sign_production.legacy_data.samples.types import (
    FrameQualitySummary,
    SelectedPersonMetadata,
)


def require_record_keys(
    record: Mapping[str, Any],
    required_keys: frozenset[str],
    *,
    surface: str,
) -> None:
    """Require top-level or nested record keys before typed construction."""
    missing = missing_keys(record, set(required_keys))
    if missing:
        raise KeyError(f"{surface} missing required keys: {format_keys(missing)}.")


def reject_record_keys(
    record: Mapping[str, Any],
    forbidden_keys: frozenset[str],
    *,
    surface: str,
) -> None:
    """Reject serialized fields that do not belong to the parsed record variant."""
    unexpected = forbidden_keys & set(record)
    if unexpected:
        raise KeyError(f"{surface} cannot contain keys: {format_keys(unexpected)}.")


def require_mapping(value: object, field_name: str, *, surface: str) -> Mapping[str, Any]:
    """Require a mapping while naming the sample surface being parsed."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{surface} field {field_name!r} must be a mapping.")
    return value


def optional_mapping(value: object, field_name: str, *, surface: str) -> Mapping[str, Any] | None:
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
            f"{field_name} must be one of {[split.value for split in SampleSplit]}, got {value!r}."
        ) from exc


def sample_status_from_record(value: object, field_name: str) -> SampleStatus:
    """Parse a serialized manifest status value."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string status value.")
    try:
        return SampleStatus(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of {[status.value for status in SampleStatus]}, "
            f"got {value!r}."
        ) from exc


def text_from_record(value: object, field_name: str) -> str:
    """Parse a required serialized text value without changing its content."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    return value


def optional_text_from_record(value: object, field_name: str) -> str | None:
    """Parse an optional serialized text value."""
    if value is None:
        return None
    return text_from_record(value, field_name)


def int_from_record(value: object, field_name: str) -> int:
    """Parse an integer scalar from a record value."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not a boolean.")
    if isinstance(value, Integral):
        return int(value)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be parseable as an integer.") from exc


def optional_int_from_record(value: object, field_name: str) -> int | None:
    """Parse an optional integer scalar from a record value."""
    if value is None:
        return None
    return int_from_record(value, field_name)


def optional_float_from_record(value: object, field_name: str) -> float | None:
    """Parse an optional float scalar from a record value."""
    if value is None:
        return None
    return float_from_record(value, field_name)


def float_from_record(value: object, field_name: str) -> float:
    """Parse a float scalar from a record value."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number, not a boolean.")
    if isinstance(value, Real):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be parseable as a number.") from exc


def selected_person_from_record(record: Mapping[str, Any]) -> SelectedPersonMetadata:
    """Parse selected-person metadata from a record mapping."""
    from text_to_sign_production.legacy_data.samples.schema import REQUIRED_SELECTED_PERSON_KEYS

    require_record_keys(record, REQUIRED_SELECTED_PERSON_KEYS, surface="Selected person")
    return SelectedPersonMetadata(
        index=int_from_record(record["index"], "selected_person.index"),
        multi_person_frame_count=int_from_record(
            record["multi_person_frame_count"],
            "selected_person.multi_person_frame_count",
        ),
        max_people_per_frame=int_from_record(
            record["max_people_per_frame"],
            "selected_person.max_people_per_frame",
        ),
    )


def frame_quality_from_record(record: Mapping[str, Any]) -> FrameQualitySummary:
    """Parse frame-quality metadata from a record mapping."""
    from text_to_sign_production.legacy_data.samples.schema import REQUIRED_FRAME_QUALITY_KEYS

    require_record_keys(record, REQUIRED_FRAME_QUALITY_KEYS, surface="Frame quality")
    return FrameQualitySummary(
        valid_frame_count=int_from_record(
            record["valid_frame_count"],
            "frame_quality.valid_frame_count",
        ),
        invalid_frame_count=int_from_record(
            record["invalid_frame_count"],
            "frame_quality.invalid_frame_count",
        ),
        face_missing_frame_count=int_from_record(
            record["face_missing_frame_count"],
            "frame_quality.face_missing_frame_count",
        ),
        out_of_bounds_coordinate_count=int_from_record(
            record["out_of_bounds_coordinate_count"],
            "frame_quality.out_of_bounds_coordinate_count",
        ),
        frames_with_any_zeroed_canonical_joint=int_from_record(
            record["frames_with_any_zeroed_canonical_joint"],
            "frame_quality.frames_with_any_zeroed_canonical_joint",
        ),
        tracked_target_missing_frame_count=int_from_record(
            record["tracked_target_missing_frame_count"],
            "frame_quality.tracked_target_missing_frame_count",
        ),
        tracked_target_missing_frame_ratio=float_from_record(
            record["tracked_target_missing_frame_ratio"],
            "frame_quality.tracked_target_missing_frame_ratio",
        ),
        person_tracking_continuity_break_count=int_from_record(
            record["person_tracking_continuity_break_count"],
            "frame_quality.person_tracking_continuity_break_count",
        ),
        person_tracking_continuity_break_ratio=float_from_record(
            record["person_tracking_continuity_break_ratio"],
            "frame_quality.person_tracking_continuity_break_ratio",
        ),
        person_tracking_reanchor_count=int_from_record(
            record["person_tracking_reanchor_count"],
            "frame_quality.person_tracking_reanchor_count",
        ),
        person_tracking_reanchor_ratio=float_from_record(
            record["person_tracking_reanchor_ratio"],
            "frame_quality.person_tracking_reanchor_ratio",
        ),
        frame_issue_counts=int_mapping(record.get("frame_issue_counts", {})),
        channel_nonzero_frames=int_mapping(record.get("channel_nonzero_frames", {})),
    )


def int_mapping(value: object) -> dict[str, int]:
    """Parse a string-keyed integer count mapping."""
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping of string keys to integer counts.")
    parsed: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("Expected a mapping with string keys.")
        parsed[key] = int_from_record(item, f"{key} count")
    return parsed


def string_tuple_from_sequence(value: object, field_name: str) -> tuple[str, ...]:
    """Parse a non-string sequence of text values."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings.")
    parsed = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} values must be strings.")
        parsed.append(item)
    return tuple(parsed)


def bool_from_record(value: object, field_name: str) -> bool:
    """Parse a strict boolean field."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean value, got {type(value).__name__}.")
    return value
