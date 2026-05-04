"""Validation of sample payloads and manifest entries against contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from text_to_sign_production.data._shared.identities import VALID_SAMPLE_SPLITS
from text_to_sign_production.data.samples._shared.validation import ValidationSeverity
from text_to_sign_production.data.samples.schema import (
    DROPPED_ONLY_MANIFEST_KEYS,
    PASSED_ONLY_MANIFEST_KEYS,
    PROCESSED_SCHEMA_VERSION,
    REQUIRED_DROPPED_MANIFEST_KEYS,
    REQUIRED_FRAME_QUALITY_KEYS,
    REQUIRED_PASSED_MANIFEST_KEYS,
    REQUIRED_PAYLOAD_KEYS,
    REQUIRED_POSE_CHANNEL_KEYS,
    REQUIRED_SELECTED_PERSON_KEYS,
)
from text_to_sign_production.data.samples.types import (
    DroppedManifestEntry,
    PassedManifestEntry,
    ProcessedSamplePayload,
    SampleStatus,
    SampleValidationIssue,
)


def validate_payload_record(record: Mapping[str, Any]) -> list[SampleValidationIssue]:
    """Validate a raw payload dictionary representation."""
    issues: list[SampleValidationIssue] = []
    _validate_required_keys(
        issues,
        record=record,
        required_keys=REQUIRED_PAYLOAD_KEYS,
        code="missing_payload_keys",
        label="Payload",
    )
    _validate_schema_version(issues, record=record, label="Payload")

    _validate_required_text_field(issues, record, "sample_id", label="Payload sample_id")
    _validate_required_text_field(issues, record, "text", label="Payload text")
    _validate_split(issues, record.get("split"), label="Payload split")

    num_frames = _validate_frame_count(
        issues,
        record.get("num_frames"),
        label="Payload num_frames",
        allow_zero=False,
    )
    _validate_fps(
        issues,
        record.get("fps"),
        label="Payload fps",
        required_key_present="fps" in record,
    )

    _validate_selected_person_record(
        issues,
        record.get("selected_person"),
        num_frames=num_frames,
        label="Payload selected_person",
    )
    _validate_frame_quality_record(
        issues,
        record.get("frame_quality"),
        num_frames=num_frames,
        label="Payload frame_quality",
    )
    _validate_pose_record(
        issues,
        record.get("pose"),
        num_frames=num_frames,
        label="Payload pose",
    )

    return issues


def validate_payload(payload: ProcessedSamplePayload) -> list[SampleValidationIssue]:
    """Validate a typed processed sample payload."""
    return validate_payload_record(payload.to_record())


def validate_manifest_record(record: Mapping[str, Any]) -> list[SampleValidationIssue]:
    """Validate a raw manifest entry dictionary before parsing."""
    issues: list[SampleValidationIssue] = []
    status = record.get("status")

    if status == SampleStatus.PASSED.value:
        _validate_passed_record(issues, record)
    elif status == SampleStatus.DROPPED.value:
        _validate_dropped_record(issues, record)
    else:
        _add_issue(
            issues,
            "invalid_status",
            f"Manifest entry status must be 'passed' or 'dropped', got {status!r}.",
        )
        _validate_schema_version(issues, record=record, label="Manifest entry")

    return issues


def validate_passed_entry(entry: PassedManifestEntry) -> list[SampleValidationIssue]:
    """Validate domain invariants of a parsed passed manifest entry."""
    issues = validate_manifest_record(entry.to_record())
    if entry.status != SampleStatus.PASSED:
        _add_issue(
            issues,
            "invalid_passed_entry_status",
            f"PassedManifestEntry status must be 'passed', got {entry.status!r}.",
        )
    return issues


def validate_dropped_entry(entry: DroppedManifestEntry) -> list[SampleValidationIssue]:
    """Validate domain invariants of a parsed dropped manifest entry."""
    issues = validate_manifest_record(entry.to_record())
    if entry.status != SampleStatus.DROPPED:
        _add_issue(
            issues,
            "invalid_dropped_entry_status",
            f"DroppedManifestEntry status must be 'dropped', got {entry.status!r}.",
        )
    return issues


def _validate_passed_record(issues: list[SampleValidationIssue], record: Mapping[str, Any]) -> None:
    _validate_required_keys(
        issues,
        record=record,
        required_keys=REQUIRED_PASSED_MANIFEST_KEYS,
        code="missing_passed_keys",
        label="Passed manifest entry",
    )
    _validate_schema_version(issues, record=record, label="Passed manifest entry")

    unexpected_dropped_keys = DROPPED_ONLY_MANIFEST_KEYS & set(record)
    if unexpected_dropped_keys:
        _add_issue(
            issues,
            "passed_entry_has_dropped_fields",
            f"Passed manifest entry cannot contain dropped-only fields: "
            f"{_format_keys(unexpected_dropped_keys)}.",
        )

    _validate_required_text_field(issues, record, "sample_id", label="Passed manifest sample_id")
    _validate_required_text_field(issues, record, "text", label="Passed manifest text")
    _validate_split(issues, record.get("split"), label="Passed manifest split")
    _validate_required_text_field(
        issues, record, "sample_path", label="Passed manifest sample_path"
    )
    _validate_required_text_field(
        issues, record, "source_video_id", label="Passed manifest source_video_id"
    )
    _validate_required_text_field(
        issues, record, "source_sentence_id", label="Passed manifest source_sentence_id"
    )
    _validate_required_text_field(
        issues,
        record,
        "source_sentence_name",
        label="Passed manifest source_sentence_name",
    )

    num_frames = _validate_frame_count(
        issues,
        record.get("num_frames"),
        label="Passed manifest num_frames",
        allow_zero=False,
    )
    _validate_fps(
        issues,
        record.get("fps"),
        label="Passed manifest fps",
        required_key_present="fps" in record,
    )
    _validate_selected_person_record(
        issues,
        record.get("selected_person"),
        num_frames=num_frames,
        label="Passed manifest selected_person",
    )
    _validate_frame_quality_record(
        issues,
        record.get("frame_quality"),
        num_frames=num_frames,
        label="Passed manifest frame_quality",
    )


def _validate_dropped_record(
    issues: list[SampleValidationIssue],
    record: Mapping[str, Any],
) -> None:
    _validate_required_keys(
        issues,
        record=record,
        required_keys=REQUIRED_DROPPED_MANIFEST_KEYS,
        code="missing_dropped_keys",
        label="Dropped manifest entry",
    )
    _validate_schema_version(issues, record=record, label="Dropped manifest entry")

    unexpected_passed_keys = PASSED_ONLY_MANIFEST_KEYS & set(record)
    if unexpected_passed_keys:
        _add_issue(
            issues,
            "dropped_entry_has_passed_fields",
            f"Dropped manifest entry cannot contain passed-only fields: "
            f"{_format_keys(unexpected_passed_keys)}.",
        )

    _validate_required_text_field(issues, record, "sample_id", label="Dropped manifest sample_id")
    _validate_split(issues, record.get("split"), label="Dropped manifest split")
    _validate_required_text_field(issues, record, "drop_stage", label="Dropped manifest drop_stage")
    _validate_drop_reasons(issues, record.get("drop_reasons"))
    _validate_drop_details(issues, record.get("drop_details"))

    debug_only = _validate_debug_only(issues, record.get("debug_only"))
    sample_path_present = "sample_path" in record and record.get("sample_path") is not None
    if sample_path_present:
        _validate_optional_text_value(
            issues,
            record.get("sample_path"),
            label="Dropped manifest sample_path",
        )
        if debug_only is False:
            _add_issue(
                issues,
                "materialized_dropped_sample_not_debug_only",
                "Dropped manifest entries with sample_path must set debug_only to true.",
            )
    elif debug_only is True:
        _add_issue(
            issues,
            "debug_only_dropped_entry_missing_sample_path",
            "Dropped manifest entries with debug_only=true must include sample_path.",
        )

    text_present = "text" in record and record.get("text") is not None
    if text_present:
        _validate_optional_text_value(issues, record.get("text"), label="Dropped manifest text")

    num_frames = None
    if "num_frames" in record and record.get("num_frames") is not None:
        num_frames = _validate_frame_count(
            issues,
            record.get("num_frames"),
            label="Dropped manifest num_frames",
            allow_zero=True,
        )
    if sample_path_present and num_frames is None:
        _add_issue(
            issues,
            "materialized_dropped_sample_missing_num_frames",
            "Materialized dropped entries must include num_frames.",
        )

    if "fps" in record:
        _validate_fps(
            issues,
            record.get("fps"),
            label="Dropped manifest fps",
            required_key_present=True,
        )

    selected_person_present = (
        "selected_person" in record and record.get("selected_person") is not None
    )
    if selected_person_present:
        _validate_selected_person_record(
            issues,
            record.get("selected_person"),
            num_frames=num_frames,
            label="Dropped manifest selected_person",
        )

    frame_quality_present = "frame_quality" in record and record.get("frame_quality") is not None
    if frame_quality_present:
        if num_frames is None:
            _add_issue(
                issues,
                "dropped_frame_quality_missing_num_frames",
                "Dropped entries with frame_quality must include num_frames.",
            )
        _validate_frame_quality_record(
            issues,
            record.get("frame_quality"),
            num_frames=num_frames,
            label="Dropped manifest frame_quality",
        )


def _validate_required_keys(
    issues: list[SampleValidationIssue],
    *,
    record: Mapping[str, Any],
    required_keys: frozenset[str],
    code: str,
    label: str,
) -> None:
    missing_keys = required_keys - set(record)
    if missing_keys:
        _add_issue(
            issues,
            code,
            f"{label} missing required keys: {_format_keys(missing_keys)}.",
        )


def _validate_schema_version(
    issues: list[SampleValidationIssue],
    *,
    record: Mapping[str, Any],
    label: str,
) -> None:
    if "schema_version" not in record:
        return
    schema_version = record.get("schema_version")
    if schema_version != PROCESSED_SCHEMA_VERSION:
        _add_issue(
            issues,
            "invalid_schema_version",
            f"{label} schema_version must be {PROCESSED_SCHEMA_VERSION!r}, got {schema_version!r}.",
        )


def _validate_required_text_field(
    issues: list[SampleValidationIssue],
    record: Mapping[str, Any],
    field_name: str,
    *,
    label: str,
) -> None:
    if field_name not in record:
        return
    _validate_optional_text_value(issues, record.get(field_name), label=label)


def _validate_optional_text_value(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        _add_issue(issues, "invalid_text_field", f"{label} must be a non-empty string.")


def _validate_split(issues: list[SampleValidationIssue], value: object, *, label: str) -> None:
    if not isinstance(value, str) or value not in VALID_SAMPLE_SPLITS:
        _add_issue(
            issues,
            "invalid_split",
            f"{label} must be one of {_format_keys(VALID_SAMPLE_SPLITS)}, got {value!r}.",
        )


def _validate_frame_count(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
    allow_zero: bool,
) -> int | None:
    frame_count = _int_value(value)
    if frame_count is None:
        _add_issue(issues, "invalid_num_frames", f"{label} must be an integer.")
        return None
    if frame_count < 0 or (frame_count == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        _add_issue(
            issues,
            "invalid_num_frames",
            f"{label} must be {comparator}, got {frame_count}.",
        )
    return frame_count


def _validate_fps(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
    required_key_present: bool,
) -> None:
    if value is None:
        return
    if not required_key_present:
        return
    fps = _float_value(value)
    if fps is None or fps <= 0.0 or not math.isfinite(fps):
        _add_issue(
            issues,
            "invalid_fps",
            f"{label} must be a positive finite number when present, got {value!r}.",
        )


def _validate_selected_person_record(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    num_frames: int | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _add_issue(issues, "invalid_selected_person", f"{label} must be a mapping.")
        return

    _validate_required_keys(
        issues,
        record=value,
        required_keys=REQUIRED_SELECTED_PERSON_KEYS,
        code="missing_selected_person_keys",
        label=label,
    )

    index = _validate_count_field(issues, value, "index", label=label)
    multi_person_frame_count = _validate_count_field(
        issues,
        value,
        "multi_person_frame_count",
        label=label,
    )
    max_people_per_frame = _validate_count_field(
        issues,
        value,
        "max_people_per_frame",
        label=label,
    )

    if max_people_per_frame is not None and max_people_per_frame < 1:
        _add_issue(
            issues,
            "invalid_selected_person",
            f"{label}.max_people_per_frame must be >= 1.",
        )
    if (
        index is not None
        and max_people_per_frame is not None
        and max_people_per_frame >= 1
        and index >= max_people_per_frame
    ):
        _add_issue(
            issues,
            "invalid_selected_person",
            f"{label}.index must be less than max_people_per_frame.",
        )
    if (
        multi_person_frame_count is not None
        and num_frames is not None
        and multi_person_frame_count > num_frames
    ):
        _add_issue(
            issues,
            "selected_person_frame_count_mismatch",
            f"{label}.multi_person_frame_count cannot exceed num_frames.",
        )


def _validate_frame_quality_record(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    num_frames: int | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _add_issue(issues, "invalid_frame_quality", f"{label} must be a mapping.")
        return

    _validate_required_keys(
        issues,
        record=value,
        required_keys=REQUIRED_FRAME_QUALITY_KEYS,
        code="missing_frame_quality_keys",
        label=label,
    )

    valid_frame_count = _validate_count_field(issues, value, "valid_frame_count", label=label)
    invalid_frame_count = _validate_count_field(
        issues,
        value,
        "invalid_frame_count",
        label=label,
    )
    face_missing_frame_count = _validate_count_field(
        issues,
        value,
        "face_missing_frame_count",
        label=label,
    )
    _validate_count_field(issues, value, "out_of_bounds_coordinate_count", label=label)
    zeroed_count = _validate_count_field(
        issues,
        value,
        "frames_with_any_zeroed_canonical_joint",
        label=label,
    )

    if (
        valid_frame_count is not None
        and invalid_frame_count is not None
        and num_frames is not None
        and valid_frame_count + invalid_frame_count != num_frames
    ):
        _add_issue(
            issues,
            "frame_quality_count_mismatch",
            f"{label} valid_frame_count + invalid_frame_count must equal num_frames.",
        )

    for field_name, count in (
        ("face_missing_frame_count", face_missing_frame_count),
        ("frames_with_any_zeroed_canonical_joint", zeroed_count),
    ):
        if count is not None and num_frames is not None and count > num_frames:
            _add_issue(
                issues,
                "frame_quality_count_mismatch",
                f"{label}.{field_name} cannot exceed num_frames.",
            )

    _validate_count_mapping(
        issues,
        value.get("frame_issue_counts"),
        required_keys=None,
        num_frames=None,
        label=f"{label}.frame_issue_counts",
    )
    _validate_count_mapping(
        issues,
        value.get("channel_nonzero_frames"),
        required_keys=_canonical_pose_channels(),
        num_frames=num_frames,
        label=f"{label}.channel_nonzero_frames",
    )


def _validate_pose_record(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    num_frames: int | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _add_issue(issues, "invalid_pose", f"{label} must be a mapping.")
        return

    observed_channels = set(value)
    canonical_channels = _canonical_pose_channels()
    missing_channels = set(canonical_channels) - observed_channels
    extra_channels = observed_channels - set(canonical_channels)
    if missing_channels:
        _add_issue(
            issues,
            "missing_pose_channels",
            f"{label} missing channels: {_format_keys(missing_channels)}.",
        )
    if extra_channels:
        _add_issue(
            issues,
            "unexpected_pose_channels",
            f"{label} has non-canonical channels: {_format_keys(extra_channels)}.",
        )

    for channel_name in canonical_channels:
        channel_record = value.get(channel_name)
        _validate_pose_channel_record(
            issues,
            channel_record,
            channel_name=channel_name,
            num_frames=num_frames,
            label=f"{label}.{channel_name}",
        )


def _validate_pose_channel_record(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    channel_name: str,
    num_frames: int | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _add_issue(issues, "invalid_pose_channel", f"{label} must be a mapping.")
        return

    _validate_required_keys(
        issues,
        record=value,
        required_keys=REQUIRED_POSE_CHANNEL_KEYS,
        code="missing_pose_channel_keys",
        label=label,
    )

    expected_joint_count = _pose_channel_joint_counts()[channel_name]

    coordinates = value.get("coordinates")
    confidence = value.get("confidence")
    if coordinates is None:
        _add_issue(issues, "missing_pose_coordinates", f"{label}.coordinates is required.")
    if confidence is None:
        _add_issue(issues, "missing_pose_confidence", f"{label}.confidence is required.")

    coordinates_shape = _shape_of(coordinates)
    if coordinates_shape is not None and num_frames is not None:
        expected_coordinates_shape = (
            num_frames,
            expected_joint_count,
            _pose_coordinate_dimensions(),
        )
        if coordinates_shape != expected_coordinates_shape:
            _add_issue(
                issues,
                "invalid_pose_coordinates_shape",
                f"{label}.coordinates shape must be {expected_coordinates_shape}, "
                f"got {coordinates_shape}.",
            )

    confidence_shape = _shape_of(confidence)
    if confidence_shape is not None and num_frames is not None:
        expected_confidence_shape = (num_frames, expected_joint_count)
        if confidence_shape != expected_confidence_shape:
            _add_issue(
                issues,
                "invalid_pose_confidence_shape",
                f"{label}.confidence shape must be {expected_confidence_shape}, "
                f"got {confidence_shape}.",
            )


def _validate_count_field(
    issues: list[SampleValidationIssue],
    record: Mapping[str, Any],
    field_name: str,
    *,
    label: str,
) -> int | None:
    if field_name not in record:
        return None
    value = _int_value(record.get(field_name))
    if value is None:
        _add_issue(
            issues,
            "invalid_count_field",
            f"{label}.{field_name} must be a non-negative integer.",
        )
        return None
    if value < 0:
        _add_issue(
            issues,
            "invalid_count_field",
            f"{label}.{field_name} must be non-negative, got {value}.",
        )
    return value


def _validate_count_mapping(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    required_keys: Sequence[str] | None,
    num_frames: int | None,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        _add_issue(issues, "invalid_count_mapping", f"{label} must be a mapping.")
        return

    observed_keys = {str(key) for key in value}
    if required_keys is not None:
        missing_keys = set(required_keys) - observed_keys
        if missing_keys:
            _add_issue(
                issues,
                "missing_count_mapping_keys",
                f"{label} missing keys: {_format_keys(missing_keys)}.",
            )

    for key, raw_count in value.items():
        if not isinstance(key, str) or not key.strip():
            _add_issue(issues, "invalid_count_mapping_key", f"{label} keys must be non-empty.")
        count = _int_value(raw_count)
        if count is None or count < 0:
            _add_issue(
                issues,
                "invalid_count_mapping_value",
                f"{label}.{key} must be a non-negative integer.",
            )
        elif num_frames is not None and count > num_frames:
            _add_issue(
                issues,
                "count_mapping_frame_mismatch",
                f"{label}.{key} cannot exceed num_frames.",
            )


def _validate_drop_reasons(issues: list[SampleValidationIssue], value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _add_issue(
            issues,
            "invalid_drop_reasons",
            "Dropped manifest drop_reasons must be a non-empty sequence of strings.",
        )
        return
    if not value:
        _add_issue(
            issues,
            "invalid_drop_reasons",
            "Dropped manifest drop_reasons must contain at least one reason.",
        )
        return

    normalized_reasons = []
    for reason in value:
        if not isinstance(reason, str) or not reason.strip():
            _add_issue(
                issues,
                "invalid_drop_reasons",
                "Dropped manifest drop_reasons values must be non-empty strings.",
            )
            continue
        normalized_reasons.append(reason)

    if len(set(normalized_reasons)) != len(normalized_reasons):
        _add_issue(
            issues,
            "duplicate_drop_reasons",
            "Dropped manifest drop_reasons must not contain duplicates.",
        )


def _validate_debug_only(issues: list[SampleValidationIssue], value: object) -> bool | None:
    if not isinstance(value, bool):
        _add_issue(
            issues,
            "invalid_debug_only",
            f"Dropped manifest debug_only must be a boolean, got {value!r}.",
        )
        return None
    return value


def _validate_drop_details(issues: list[SampleValidationIssue], value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        _add_issue(
            issues,
            "invalid_drop_details",
            "Dropped manifest drop_details must be a mapping.",
        )
        return
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            _add_issue(
                issues,
                "invalid_drop_details_key",
                "Dropped manifest drop_details keys must be non-empty strings.",
            )
        if not _is_json_value(item):
            _add_issue(
                issues,
                "invalid_drop_details_value",
                f"Dropped manifest drop_details[{key!r}] must be JSON-compatible.",
            )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _int_value(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _float_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def _shape_of(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def _add_issue(issues: list[SampleValidationIssue], code: str, message: str) -> None:
    issues.append(
        SampleValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=code,
            message=message,
        )
    )


def _format_keys(keys: Sequence[str] | set[str] | frozenset[str]) -> str:
    return ", ".join(sorted(keys))


def _canonical_pose_channels() -> tuple[str, ...]:
    from text_to_sign_production.data.pose.schema import CANONICAL_POSE_CHANNELS

    return CANONICAL_POSE_CHANNELS


def _pose_channel_joint_counts() -> dict[str, int]:
    from text_to_sign_production.data.pose.schema import POSE_CHANNEL_JOINT_COUNTS

    return POSE_CHANNEL_JOINT_COUNTS


def _pose_coordinate_dimensions() -> int:
    from text_to_sign_production.data.pose.schema import POSE_COORDINATE_DIMENSIONS

    return POSE_COORDINATE_DIMENSIONS
