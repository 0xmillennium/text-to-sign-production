"""Internal validation primitives for sample contracts."""

from __future__ import annotations

import enum


class ValidationSeverity(enum.StrEnum):
    """Typed severity values for sample validation surfaces."""

    ERROR = "error"
    WARNING = "warning"
