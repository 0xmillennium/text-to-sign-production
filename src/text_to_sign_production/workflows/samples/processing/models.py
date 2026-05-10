from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.gate.sources import SourceMatchResult, TranslationSourceRecord
from text_to_sign_production.workflows.samples.contracts import SamplesWorkflowResult


@dataclass(frozen=True, slots=True)
class SamplesSourceBundle:
    translation: TranslationSourceRecord
    match: SourceMatchResult


@dataclass(frozen=True, slots=True)
class SamplesPayloadOutput:
    sample: PreparedSample
    path: Path
    payload_ref: str
    status: SampleStatus


@dataclass(frozen=True, slots=True)
class SamplesSplitProcessingResult:
    split: str
    source_matches: tuple[SourceMatchResult, ...]
    prepared_samples: tuple[PreparedSample, ...]
    gate_bundles: tuple[GateDecisionBundle, ...]
    passed_payloads: tuple[SamplesPayloadOutput, ...]
    dropped_debug_payloads: tuple[SamplesPayloadOutput, ...]
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


__all__ = [
    "SamplesExecutionBundle",
    "SamplesPayloadOutput",
    "SamplesSourceBundle",
    "SamplesSplitProcessingResult",
]
