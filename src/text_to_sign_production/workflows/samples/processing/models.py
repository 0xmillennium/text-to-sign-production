from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.data.gates import ProcessingDecision
from text_to_sign_production.data.samples import (
    DroppedManifestEntry,
    DroppedMaterializationLifecycle,
    PassedManifestEntry,
    ProcessedSamplePayload,
)
from text_to_sign_production.data.sources import (
    SourceCandidate,
    SourceMatchResult,
    TranslationRow,
)
from text_to_sign_production.workflows.samples.contracts import SamplesWorkflowResult


@dataclass(frozen=True, slots=True)
class SamplesSourceBundle:
    translation: TranslationRow
    video_path: Path
    match: SourceMatchResult
    candidate: SourceCandidate | None


@dataclass(frozen=True, slots=True)
class SamplesPayloadMaterialization:
    payload: ProcessedSamplePayload | None
    payload_path: Path | None
    payload_relative_path: str | None
    dropped_materialization: DroppedMaterializationLifecycle | None
    not_attempted_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SamplesSplitProcessingResult:
    split: str
    source_matches: tuple[SourceMatchResult, ...]
    decisions: tuple[ProcessingDecision, ...]
    passed_entries: tuple[PassedManifestEntry, ...]
    dropped_entries: tuple[DroppedManifestEntry, ...]

    @property
    def processed_count(self) -> int:
        return len(self.source_matches)

    @property
    def passed_count(self) -> int:
        return len(self.passed_entries)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped_entries)


@dataclass(frozen=True, slots=True)
class SamplesExecutionBundle:
    workflow_result: SamplesWorkflowResult
    split_results: tuple[SamplesSplitProcessingResult, ...]
