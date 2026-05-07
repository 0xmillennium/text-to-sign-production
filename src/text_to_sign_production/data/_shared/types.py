"""Shared low-level data-layer type leaves."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ValidationSeverity(enum.StrEnum):
    """Typed severity values for validation issue surfaces."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Generic validation issue with a stable machine code and human message."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SevereValidationIssue:
    """Generic validation issue carrying a severity value."""

    severity: ValidationSeverity
    code: str
    message: str
