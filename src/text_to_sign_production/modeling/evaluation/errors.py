"""Evaluation protocol errors."""

from __future__ import annotations


class EvaluationProtocolError(RuntimeError):
    """Raised when the evaluation protocol cannot be interpreted or executed."""


__all__ = ["EvaluationProtocolError"]
