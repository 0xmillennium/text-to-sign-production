from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.workflows.samples.contracts.config import (
    SamplesWorkflowInputError,
)


@dataclass(frozen=True, slots=True)
class SamplesRuntimeAssetRow:
    label: str
    path: Path
    exists: bool

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class SamplesSplitCountRow:
    split: str
    processed_count: int
    passed_count: int
    dropped_count: int

    def __post_init__(self) -> None:
        _validate_non_empty_text("split", self.split)
        _ensure_non_negative("processed_count", self.processed_count)
        _ensure_non_negative("passed_count", self.passed_count)
        _ensure_non_negative("dropped_count", self.dropped_count)
        object.__setattr__(self, "split", self.split.strip())


@dataclass(frozen=True, slots=True)
class SamplesReportArtifactRow:
    label: str
    path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class SamplesPublishTargetRow:
    label: str
    kind: str
    source_path: Path
    target_path: Path

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class SamplesManifestRow:
    label: str
    path: Path
    partition: str
    split: str

    def __post_init__(self) -> None:
        _validate_non_empty_text("label", self.label)
        _validate_non_empty_text("partition", self.partition)
        _validate_non_empty_text("split", self.split)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "partition", self.partition.strip())
        object.__setattr__(self, "split", self.split.strip())


def _validate_non_empty_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SamplesWorkflowInputError(f"{field_name} must be a non-empty string")


def _ensure_non_negative(field_name: str, value: int) -> None:
    if value < 0:
        raise SamplesWorkflowInputError(f"{field_name} must be >= 0")
