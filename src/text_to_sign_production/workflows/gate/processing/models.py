from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    DroppedSample,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.dataset.confidence import (
    ConfidenceCanonicalizationSummary,
)
from text_to_sign_production.data.gate.sources import SourceMatchResult
from text_to_sign_production.data.gate.sources.types import CandidateViabilityReport
from text_to_sign_production.workflows.gate.contracts import GateWorkflowResult
from text_to_sign_production.workflows.gate.contracts.results import (
    GateWrittenManifestArtifact,
    GateWrittenPayloadArtifact,
)


@dataclass(frozen=True, slots=True)
class GatePayloadOutput:
    sample: PreparedSample
    path: Path
    payload_ref: str
    status: SampleStatus


@dataclass(frozen=True, slots=True)
class GateDroppedSamplePayloadOutput:
    sample: DroppedSample
    path: Path
    payload_ref: str


@dataclass(frozen=True, slots=True)
class GateSplitProcessingResult:
    split: str
    source_matches: tuple[SourceMatchResult, ...]
    viability_reports: tuple[CandidateViabilityReport, ...]
    prepared_samples: tuple[PreparedSample, ...]
    gate_bundles: tuple[GateDecisionBundle, ...]
    passed_payloads: tuple[GatePayloadOutput, ...]
    dropped_sample_payloads: tuple[GateDroppedSamplePayloadOutput, ...]
    passed_entries: tuple[PassedManifestEntry, ...]
    dropped_entries: tuple[DroppedManifestEntry, ...]
    confidence_canonicalization_summaries: tuple[ConfidenceCanonicalizationSummary, ...]

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
class GateExecutionBundle:
    workflow_result: GateWorkflowResult
    split_results: tuple[GateSplitProcessingResult, ...]
    written_payload_artifacts: tuple[GateWrittenPayloadArtifact, ...]
    written_manifest_artifacts: tuple[GateWrittenManifestArtifact, ...]


__all__ = [
    "GateExecutionBundle",
    "GateDroppedSamplePayloadOutput",
    "GatePayloadOutput",
    "GateSplitProcessingResult",
]
