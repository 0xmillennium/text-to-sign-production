"""Errors for articulator-aware foundation contracts."""

from __future__ import annotations


class ArticulatorAwareError(ValueError):
    """Raised when articulator-aware foundation inputs or artifacts are invalid."""


__all__ = ["ArticulatorAwareError"]
