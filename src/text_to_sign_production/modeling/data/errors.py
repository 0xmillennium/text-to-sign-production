"""Errors for canonical modeling data contracts."""

from __future__ import annotations


class ModelingDataError(ValueError):
    """Raised when modeling data contracts are violated."""


__all__ = [
    "ModelingDataError",
]
