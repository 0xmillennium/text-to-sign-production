"""Shared primitive validation helpers for data-layer contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path


def int_value(value: object) -> int | None:
    """Return a strict non-bool integer value, or ``None`` when invalid."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def float_value(value: object) -> float | None:
    """Return a strict non-bool real value, or ``None`` when invalid."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def is_non_bool_int(value: object) -> bool:
    """Return whether a value is an integer but not a boolean."""
    return int_value(value) is not None


def is_finite_number(value: object) -> bool:
    """Return whether a value is a finite non-bool real number."""
    number = float_value(value)
    return number is not None and math.isfinite(number)


def is_unit_interval(value: object) -> bool:
    """Return whether a value is a finite number in [0, 1]."""
    number = float_value(value)
    return number is not None and math.isfinite(number) and 0.0 <= number <= 1.0


def is_positive_finite_number(value: object) -> bool:
    """Return whether a value is finite and greater than zero."""
    number = float_value(value)
    return number is not None and math.isfinite(number) and number > 0.0


def is_non_empty_text(value: object) -> bool:
    """Return whether a value is a string containing non-whitespace text."""
    return isinstance(value, str) and bool(value.strip())


def is_enum_member(value: object, enum_values: object) -> bool:
    """Return whether a value belongs to an enum-like container."""
    return value in enum_values  # type: ignore[operator]


def missing_keys(
    record: Mapping[str, object], required_keys: Sequence[str] | set[str]
) -> list[str]:
    """Return required keys absent from a mapping, sorted for stable messages."""
    return sorted(set(required_keys) - set(record))


def key_delta(
    record: Mapping[str, object],
    expected_keys: Sequence[str] | set[str],
) -> tuple[list[str], list[str]]:
    """Return missing and unknown keys against an exact key set."""
    actual = set(record)
    expected = set(expected_keys)
    return sorted(expected - actual), sorted(actual - expected)


def path_has_name(value: Path | str) -> bool:
    """Return whether a path-like value has a non-empty final name."""
    path = Path(value)
    return bool(str(path).strip()) and bool(path.name)


def is_sorted_unique_sequence(values: Sequence[object]) -> bool:
    """Return whether sequence values are sorted and unique by natural ordering."""
    try:
        return tuple(values) == tuple(sorted(set(values)))
    except TypeError:
        return False


def is_json_value(value: object) -> bool:
    """Return whether a value is JSON-compatible and finite where numeric."""
    if value is None or isinstance(value, str | bool | int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_value(item) for key, item in value.items())
    return False


def shape_tuple(value: object) -> tuple[int, ...] | None:
    """Return a tuple shape from array-like values, or ``None`` when absent/invalid."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError):
        return None


def counts_sum_to_total(left: int, right: int, total: int) -> bool:
    """Return whether two counts partition a total exactly."""
    return left + right == total


def count_within_total(count: int, total: int) -> bool:
    """Return whether a count is within [0, total]."""
    return 0 <= count <= total


def require_string_key_mapping(value: object) -> Mapping[str, object] | None:
    """Return a mapping when all keys are strings, otherwise ``None``."""
    if not isinstance(value, Mapping):
        return None
    if any(not isinstance(key, str) for key in value):
        return None
    return value


def non_string_keys(value: Mapping[object, object]) -> list[object]:
    """Return mapping keys that are not strings."""
    return [key for key in value if not isinstance(key, str)]


def format_keys(keys: Sequence[str] | set[str] | frozenset[str]) -> str:
    """Format keys in sorted deterministic order."""
    return ", ".join(sorted(keys))
