"""Validation helpers for modeling research specifications."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class ModelingResearchSpecError(ValueError):
    """Raised when a modeling research specification is structurally invalid."""


@dataclass(frozen=True, slots=True)
class ValidationRequirement:
    """A required validation condition for a research-backed artifact."""

    identifier: str
    description: str
    required_for_exit: bool = True

    def __post_init__(self) -> None:
        require_non_empty(self.identifier, field_name="identifier")
        require_non_empty(self.description, field_name="description")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable validation requirement."""

        return {
            "identifier": self.identifier,
            "description": self.description,
            "required_for_exit": self.required_for_exit,
        }


def require_non_empty(value: str, *, field_name: str) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise ModelingResearchSpecError(f"{field_name} must be non-empty text.")
    return value


def require_unique_strings(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    """Validate that string values are non-empty and unique."""

    if not isinstance(values, tuple):
        raise ModelingResearchSpecError(f"{field_name} must be a tuple.")
    seen: set[str] = set()
    for value in values:
        require_non_empty(value, field_name=field_name)
        if value in seen:
            raise ModelingResearchSpecError(f"{field_name} must contain unique values.")
        seen.add(value)
    return values


def require_unique_enum_values(
    values: tuple[enum.StrEnum, ...],
    *,
    field_name: str,
) -> tuple[enum.StrEnum, ...]:
    """Validate that enum values are unique."""

    if not isinstance(values, tuple):
        raise ModelingResearchSpecError(f"{field_name} must be a tuple.")
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, enum.StrEnum):
            raise ModelingResearchSpecError(f"{field_name} must contain StrEnum values.")
        if value.value in seen:
            raise ModelingResearchSpecError(f"{field_name} must contain unique values.")
        seen.add(value.value)
    return values


def require_unique_enum_members(
    values: tuple[enum.StrEnum, ...],
    *,
    enum_type: type[enum.StrEnum],
    field_name: str,
) -> tuple[enum.StrEnum, ...]:
    """Validate that values are unique members of the expected enum type."""

    if not isinstance(values, tuple):
        raise ModelingResearchSpecError(f"{field_name} must be a tuple.")
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, enum_type):
            raise ModelingResearchSpecError(
                f"{field_name} must contain {enum_type.__name__} values."
            )
        if not value.value.strip():
            raise ModelingResearchSpecError(f"{field_name} must not contain blank enum values.")
        if value.value in seen:
            raise ModelingResearchSpecError(f"{field_name} must contain unique values.")
        seen.add(value.value)
    return values


def ensure_disjoint_enum_values(
    left: tuple[enum.StrEnum, ...],
    right: tuple[enum.StrEnum, ...],
    *,
    left_name: str,
    right_name: str,
) -> None:
    """Raise if two enum sequences share any string values."""

    require_unique_enum_values(left, field_name=left_name)
    require_unique_enum_values(right, field_name=right_name)
    overlap = {item.value for item in left}.intersection(item.value for item in right)
    if overlap:
        values = ", ".join(sorted(overlap))
        raise ModelingResearchSpecError(
            f"{left_name} and {right_name} must be disjoint; shared values: {values}."
        )


__all__ = [
    "ModelingResearchSpecError",
    "ValidationRequirement",
    "ensure_disjoint_enum_values",
    "require_non_empty",
    "require_unique_enum_members",
    "require_unique_enum_values",
    "require_unique_strings",
]
