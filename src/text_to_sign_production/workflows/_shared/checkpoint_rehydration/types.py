"""Type system for checkpoint rehydration bridge."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from text_to_sign_production.data.tier._shared.types import (
    QualityPoseTruth,
    QualitySourceTruth,
)
from text_to_sign_production.data.tier.context.types import QualityContext
from text_to_sign_production.data.tier.facts.types import QualityFacts
from text_to_sign_production.data.tier.families.types import QualityMetricBundle
from text_to_sign_production.data.gate.types import (
    PassedManifestEntry,
    ProcessedSamplePayload,
)


class CheckpointRehydrationIssueCode(enum.StrEnum):
    """Machine-readable issue codes for checkpoint rehydration validation."""

    MANIFEST_PAYLOAD_IDENTITY_MISMATCH = "manifest_payload_identity_mismatch"
    MANIFEST_PAYLOAD_SPLIT_MISMATCH = "manifest_payload_split_mismatch"
    MANIFEST_PAYLOAD_TEXT_MISMATCH = "manifest_payload_text_mismatch"
    MANIFEST_PAYLOAD_FRAME_COUNT_MISMATCH = "manifest_payload_frame_count_mismatch"
    NORMALIZED_TEXT_HANDOFF_MISSING = "normalized_text_handoff_missing"
    SOURCE_TRUTH_INCOHERENT = "source_truth_incoherent"
    POSE_TRUTH_INCOHERENT = "pose_truth_incoherent"
    POSE_TRUTH_FRAME_COUNT_MISMATCH = "pose_truth_frame_count_mismatch"
    BUNDLE_FACTS_MISSING = "bundle_facts_missing"
    BUNDLE_CONTEXT_MISSING = "bundle_context_missing"
    BUNDLE_METRICS_MISSING = "bundle_metrics_missing"
    BUNDLE_SOURCE_TRUTH_MISSING = "bundle_source_truth_missing"
    BUNDLE_POSE_TRUTH_MISSING = "bundle_pose_truth_missing"


@dataclass(frozen=True, slots=True)
class CheckpointRehydrationIssue:
    """Structured issue from checkpoint rehydration validation."""

    code: CheckpointRehydrationIssueCode
    message: str
    field_path: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointRehydrationInput:
    """Typed input boundary for checkpoint rehydration.

    Consumes the gate checkpoint authority (manifest + payload) as the
    sole input for rebuilding downstream quality bundles.
    """

    passed_manifest: PassedManifestEntry
    payload: ProcessedSamplePayload


@dataclass(frozen=True, slots=True)
class CheckpointRehydrationBundle:
    """Rehydrated downstream quality bundle from checkpoint truth.

    Contains the full set of authority-neutral downstream truth and
    semantic quality bundles needed for downstream quality continuation.
    Does NOT contain gate decisions, tier decisions, or reports.
    """

    source_truth: QualitySourceTruth
    pose_truth: QualityPoseTruth
    facts: QualityFacts
    context: QualityContext
    metrics: QualityMetricBundle


__all__ = [
    "CheckpointRehydrationBundle",
    "CheckpointRehydrationInput",
    "CheckpointRehydrationIssue",
    "CheckpointRehydrationIssueCode",
]
