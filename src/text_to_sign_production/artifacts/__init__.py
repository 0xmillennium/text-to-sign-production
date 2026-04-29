"""Generated artifact indexing and selection helpers."""

from __future__ import annotations

from text_to_sign_production.artifacts.index import SampleIndex, build_sample_index
from text_to_sign_production.artifacts.refs import (
    DuplicateSampleIdError,
    ManifestSampleRecord,
    ProcessedSampleDataError,
    SampleSelectionError,
    TimingMetadata,
)
from text_to_sign_production.artifacts.select import select_sample

__all__ = [
    "DuplicateSampleIdError",
    "ManifestSampleRecord",
    "ProcessedSampleDataError",
    "SampleIndex",
    "SampleSelectionError",
    "TimingMetadata",
    "build_sample_index",
    "select_sample",
]
