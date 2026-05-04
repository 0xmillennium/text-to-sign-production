"""Internal strict parsing helpers for tier configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeVar, cast

StrEnumT = TypeVar("StrEnumT")


def require_mapping(payload: object, name: str) -> Mapping[str, object]:
    """Require a string-keyed mapping."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    for key in payload:
        if not isinstance(key, str):
            raise ValueError(f"{name} contains non-string key {key!r}")
    return cast(Mapping[str, object], payload)


def require_exact_keys(
    payload: Mapping[str, object],
    expected_keys: tuple[str, ...],
    name: str,
) -> None:
    """Require exactly the named keys, rejecting missing and unknown fields."""
    actual = set(payload)
    expected = set(expected_keys)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            f"{name} keys must be exactly {list(expected_keys)} ({', '.join(details)})"
        )


def require_ratio(value: object, name: str) -> float:
    """Require a finite ratio in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}")
    return number


def require_nonnegative_int(value: object, name: str) -> int:
    """Require a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def require_positive_int(value: object, name: str) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def require_positive_float(value: object, name: str) -> float:
    """Require a positive finite number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def parse_enum_value(enum_type: type[StrEnumT], value: object, name: str) -> StrEnumT:
    """Parse a StrEnum by serialized value, not by member name."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as exc:
        values = [item.value for item in enum_type]  # type: ignore[attr-defined]
        raise ValueError(f"{name} must be one of {values}, got {value!r}") from exc
