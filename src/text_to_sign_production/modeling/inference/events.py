"""Typed domain progress events for baseline inference/export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias


@dataclass(frozen=True, slots=True)
class QualitativeSampleExported:
    sample_index: int
    total_samples: int
    sample_id: str
    reference_artifact_path: Path
    prediction_artifact_path: Path


@dataclass(frozen=True, slots=True)
class SplitPredictionRecordWritten:
    record_index: int
    total_records: int
    split: str
    sample_id: str
    prediction_sample_path: Path


InferenceProgressEvent: TypeAlias = QualitativeSampleExported | SplitPredictionRecordWritten


class InferenceProgressSink(Protocol):
    def emit(self, event: InferenceProgressEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpInferenceProgressSink:
    def emit(self, event: InferenceProgressEvent) -> None:
        return


__all__ = [
    "InferenceProgressEvent",
    "InferenceProgressSink",
    "NoOpInferenceProgressSink",
    "QualitativeSampleExported",
    "SplitPredictionRecordWritten",
]
