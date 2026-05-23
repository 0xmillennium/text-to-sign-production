"""Errors raised by semantic-consistency objective foundation contracts."""

from __future__ import annotations


class SemanticConsistencyError(ValueError):
    """Raised when semantic-consistency foundation input or materialization is invalid."""


__all__ = ["SemanticConsistencyError"]
