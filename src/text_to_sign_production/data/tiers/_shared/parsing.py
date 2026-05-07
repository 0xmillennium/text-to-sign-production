"""Internal strict parsing helpers for tier configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast

from text_to_sign_production.data._shared.validate import (
    float_value,
    int_value,
    is_positive_finite_number,
    is_unit_interval,
    key_delta,
    non_string_keys,
    require_string_key_mapping,
)

StrEnumT = TypeVar("StrEnumT")


def require_mapping(payload: object, name: str) -> Mapping[str, object]:
    """Require a string-keyed mapping."""
    mapping = require_string_key_mapping(payload)
    if mapping is None:
        if isinstance(payload, Mapping):
            bad_keys = non_string_keys(payload)
            if bad_keys:
                raise ValueError(f"{name} contains non-string key {bad_keys[0]!r}")
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], mapping)


def require_exact_keys(
    payload: Mapping[str, object],
    expected_keys: tuple[str, ...],
    name: str,
) -> None:
    """Require exactly the named keys, rejecting missing and unknown fields."""
    missing, unknown = key_delta(payload, expected_keys)
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
    number = float_value(value)
    if number is None:
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not is_unit_interval(number):
        raise ValueError(f"{name} must be finite and within [0, 1], got {value!r}")
    return number


def require_nonnegative_int(value: object, name: str) -> int:
    """Require a non-negative integer."""
    parsed = int_value(value)
    if parsed is None:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return parsed


def require_positive_int(value: object, name: str) -> int:
    """Require a positive integer."""
    parsed = int_value(value)
    if parsed is None:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def require_positive_float(value: object, name: str) -> float:
    """Require a positive finite number."""
    number = float_value(value)
    if number is None:
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not is_positive_finite_number(number):
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
