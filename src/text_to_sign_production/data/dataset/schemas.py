"""Dataset artifact schema version constants.

This module is intentionally dependency-free within the dataset package so
that both ``dropped_payloads`` (IO) and ``validate`` (semantic validation)
can import from it without creating a circular dependency.
"""

from __future__ import annotations

DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION = "dropped_sample.v1"

__all__ = ["DROPPED_SAMPLE_PAYLOAD_SCHEMA_VERSION"]
