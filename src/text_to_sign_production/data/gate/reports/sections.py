"""Section builders for samples-stage report projections."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.gate.reports.types import (
    CheckpointIntegritySection,
    GateOutcomesSection,
    MatchingOutcomesSection,
    PoseHealthSection,
    SourceCoverageSection,
)
from text_to_sign_production.data.dataset.validate import validate_payload_manifest_coherence


def build_source_coverage_section(
    payloads: Sequence[PreparedSample],
) -> SourceCoverageSection:
    """Build source coverage projection from PreparedSample payloads."""
    split_counts = Counter(payload.source.split.value for payload in payloads)
    return SourceCoverageSection(
        sample_count=len(payloads),
        split_counts=dict(sorted(split_counts.items())),
        samples_with_source_issues=sum(
            1 for payload in payloads if payload.source.source_issue_codes
        ),
    )


def build_matching_outcomes_section(
    passed_entries: Sequence[PassedManifestEntry],
    dropped_entries: Sequence[DroppedManifestEntry],
) -> MatchingOutcomesSection:
    """Build matching outcome projection from final manifests."""
    return MatchingOutcomesSection(
        matched_count=len(passed_entries),
        dropped_count=len(dropped_entries),
    )


def build_pose_health_section(payloads: Sequence[PreparedSample]) -> PoseHealthSection:
    """Build pose health projection from PreparedSample payloads."""
    return PoseHealthSection(
        payload_count=len(payloads),
        total_frame_count=sum(payload.pose.frame_count for payload in payloads),
        total_valid_frame_count=sum(
            int(np.count_nonzero(payload.pose.valid_frame_mask)) for payload in payloads
        ),
    )


def build_gate_outcomes_section(
    bundles: Sequence[GateDecisionBundle],
) -> GateOutcomesSection:
    """Build gate outcome projection from samples admission decisions."""
    failed_counter = Counter(gate.value for bundle in bundles for gate in bundle.failed_gates)
    return GateOutcomesSection(
        evaluated_count=len(bundles),
        passed_count=sum(1 for bundle in bundles if bundle.final_status is SampleStatus.PASSED),
        dropped_count=sum(1 for bundle in bundles if bundle.final_status is SampleStatus.DROPPED),
        failed_gate_counts=dict(sorted(failed_counter.items())),
    )


def build_checkpoint_integrity_section(
    payloads: Sequence[PreparedSample],
    passed_entries: Sequence[PassedManifestEntry],
    dropped_entries: Sequence[DroppedManifestEntry],
) -> CheckpointIntegritySection:
    """Build checkpoint integrity projection from payloads and manifests."""
    payload_by_id = {payload.source.sample_id: payload for payload in payloads}
    coherent = 0
    for entry in passed_entries:
        payload = payload_by_id.get(entry.sample_id)
        if payload is not None and not validate_payload_manifest_coherence(payload, entry):
            coherent += 1
    return CheckpointIntegritySection(
        passed_manifest_count=len(passed_entries),
        dropped_manifest_count=len(dropped_entries),
        coherent_passed_count=coherent,
        canonical_normalized_text_missing_count=sum(
            1 for entry in passed_entries if not entry.canonical_normalized_text.strip()
        ),
    )


__all__ = [
    "build_checkpoint_integrity_section",
    "build_gate_outcomes_section",
    "build_matching_outcomes_section",
    "build_pose_health_section",
    "build_source_coverage_section",
]
