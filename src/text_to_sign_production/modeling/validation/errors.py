"""Validation-domain errors for model validation artifacts and computation."""

from __future__ import annotations


class ModelValidationError(ValueError):
    """Raised when model validation inputs, records, or artifacts are invalid."""


__all__ = ["ModelValidationError"]
