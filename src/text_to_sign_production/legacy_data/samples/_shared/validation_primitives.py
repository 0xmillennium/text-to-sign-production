"""Leaf validation primitives for sample-domain validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from text_to_sign_production.legacy_data._shared.types import ValidationSeverity
from text_to_sign_production.legacy_data._shared.validate import (
    float_value,
    format_keys,
    int_value,
    is_json_value,
    is_non_empty_text,
    is_positive_finite_number,
    shape_tuple,
)
from text_to_sign_production.legacy_data.samples.schema import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.legacy_data.samples.types import JsonValue, SampleValidationIssue


def add_issue(issues: list[SampleValidationIssue], code: str, message: str) -> None:
    """Append one sample validation error."""
    issues.append(
        SampleValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=code,
            message=message,
        )
    )


def validate_schema_version_value(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
) -> None:
    """Validate a typed object's schema version value."""
    if value != PROCESSED_SCHEMA_VERSION:
        add_issue(
            issues,
            "invalid_schema_version",
            f"{label} schema_version must be {PROCESSED_SCHEMA_VERSION!r}, got {value!r}.",
        )


def validate_non_empty_text(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
    code: str = "invalid_text_field",
) -> None:
    """Validate a typed text field as non-empty text."""
    if not is_non_empty_text(value):
        add_issue(issues, code, f"{label} must be a non-empty string.")


def validate_frame_count_value(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
    allow_zero: bool,
) -> int | None:
    """Validate and return a typed frame count."""
    frame_count = int_value(value)
    if frame_count is None:
        add_issue(issues, "invalid_num_frames", f"{label} must be an integer.")
        return None
    if frame_count < 0 or (frame_count == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        add_issue(
            issues,
            "invalid_num_frames",
            f"{label} must be {comparator}, got {frame_count}.",
        )
    return frame_count


def validate_optional_fps_value(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
) -> None:
    """Validate optional FPS as a positive finite number when present."""
    if value is None:
        return
    fps = float_value(value)
    if fps is None or not is_positive_finite_number(fps):
        add_issue(
            issues,
            "invalid_fps",
            f"{label} must be a positive finite number when present, got {value!r}.",
        )


def validate_non_negative_count(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    label: str,
) -> int | None:
    """Validate and return a non-negative integer count."""
    count = int_value(value)
    if count is None:
        add_issue(issues, "invalid_count_field", f"{label} must be a non-negative integer.")
        return None
    if count < 0:
        add_issue(issues, "invalid_count_field", f"{label} must be non-negative, got {count}.")
    return count


def validate_count_mapping_values(
    issues: list[SampleValidationIssue],
    value: Mapping[str, int],
    *,
    required_keys: Sequence[str] | None,
    num_frames: int | None,
    label: str,
) -> None:
    """Validate a typed mapping of string keys to non-negative integer counts."""
    observed_keys = set(value)
    if required_keys is not None:
        missing = set(required_keys) - observed_keys
        if missing:
            add_issue(
                issues,
                "missing_count_mapping_keys",
                f"{label} missing keys: {format_keys(missing)}.",
            )

    for key, raw_count in value.items():
        if not is_non_empty_text(key):
            add_issue(issues, "invalid_count_mapping_key", f"{label} keys must be non-empty.")
        count = validate_non_negative_count(issues, raw_count, label=f"{label}.{key}")
        if count is not None and num_frames is not None and count > num_frames:
            add_issue(
                issues,
                "count_mapping_frame_mismatch",
                f"{label}.{key} cannot exceed num_frames.",
            )


def validate_array_shape(
    issues: list[SampleValidationIssue],
    value: object,
    *,
    missing_code: str,
    shape_code: str,
    expected_shape: tuple[int, ...] | None,
    label: str,
) -> None:
    """Validate an array-like value has the expected shape when shape is known."""
    if value is None:
        add_issue(issues, missing_code, f"{label} is required.")
        return
    actual_shape = shape_tuple(value)
    if actual_shape is None:
        add_issue(issues, shape_code, f"{label} must expose an array-like shape.")
        return
    if expected_shape is not None and actual_shape != expected_shape:
        add_issue(
            issues,
            shape_code,
            f"{label} shape must be {expected_shape}, got {actual_shape}.",
        )


def validate_json_mapping_values(
    issues: list[SampleValidationIssue],
    value: Mapping[str, JsonValue],
    *,
    label: str,
) -> None:
    """Validate JSON-compatible values inside a typed mapping."""
    for key, item in value.items():
        if not is_non_empty_text(key):
            add_issue(issues, "invalid_drop_details_key", f"{label} keys must be non-empty.")
        if not is_json_value(item):
            add_issue(
                issues,
                "invalid_drop_details_value",
                f"{label}[{key!r}] must be JSON-compatible.",
            )
