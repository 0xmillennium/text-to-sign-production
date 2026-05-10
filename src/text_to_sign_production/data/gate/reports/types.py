"""Typed samples-stage report projection products."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceCoverageSection:
    """Source coverage projection for the samples stage."""

    sample_count: int
    split_counts: dict[str, int]
    samples_with_source_issues: int


@dataclass(frozen=True, slots=True)
class MatchingOutcomesSection:
    """Matching outcome projection for the samples stage."""

    matched_count: int
    dropped_count: int


@dataclass(frozen=True, slots=True)
class PoseHealthSection:
    """Pose health projection for the samples stage."""

    payload_count: int
    total_frame_count: int
    total_valid_frame_count: int


@dataclass(frozen=True, slots=True)
class GateOutcomesSection:
    """Gate outcome projection for the samples stage."""

    evaluated_count: int
    passed_count: int
    dropped_count: int
    failed_gate_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CheckpointIntegritySection:
    """Checkpoint integrity projection for the samples stage."""

    passed_manifest_count: int
    dropped_manifest_count: int
    coherent_passed_count: int
    canonical_normalized_text_missing_count: int


@dataclass(frozen=True, slots=True)
class SamplesReportBundle:
    """Root samples-stage report bundle."""

    schema_version: str
    source_coverage: SourceCoverageSection
    matching_outcomes: MatchingOutcomesSection
    pose_health: PoseHealthSection
    gate_outcomes: GateOutcomesSection
    checkpoint_integrity: CheckpointIntegritySection


__all__ = [
    "CheckpointIntegritySection",
    "GateOutcomesSection",
    "MatchingOutcomesSection",
    "PoseHealthSection",
    "SamplesReportBundle",
    "SourceCoverageSection",
]
