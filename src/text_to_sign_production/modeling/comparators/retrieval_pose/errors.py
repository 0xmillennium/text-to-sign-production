"""Errors for the whole-pose retrieval comparator."""

from __future__ import annotations


class RetrievalPoseComparatorError(ValueError):
    """Raised when retrieval comparator configuration or execution is invalid."""


__all__ = ["RetrievalPoseComparatorError"]
