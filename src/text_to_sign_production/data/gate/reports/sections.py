"""Section builders for gate-stage report projections."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    GateDropStage,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.dataset.analysis import (
    summarize_checkpoint_handoff,
    summarize_dropped_manifest,
    summarize_passed_manifest,
    summarize_prepared_sample_payload,
)
from text_to_sign_production.data.gate.reports.types import (
    CheckpointIntegritySection,
    DroppedDebugPayloadSection,
    GateOutcomesSection,
    ManifestOutcomesSection,
    PoseHealthSection,
    SourceCoverageSection,
)


def build_source_coverage_section(
    payloads: Sequence[PreparedSample],
) -> SourceCoverageSection:
    """Build source coverage projection from PreparedSample payloads."""
    split_counts = Counter(payload.source.split.value for payload in payloads)
    sample_summaries = tuple(summarize_prepared_sample_payload(payload) for payload in payloads)
    return SourceCoverageSection(
        prepared_sample_count=len(payloads),
        split_counts=dict(sorted(split_counts.items())),
        prepared_samples_with_source_issues=sum(
            1 for payload in payloads if payload.source.source_issue_codes
        ),
        source_complete_count=sum(1 for summary in sample_summaries if summary.source_complete),
        validation_issue_count=sum(summary.validation_issue_count for summary in sample_summaries),
    )


def build_manifest_outcomes_section(
    passed_entries: Sequence[PassedManifestEntry],
    dropped_entries: Sequence[DroppedManifestEntry],
) -> ManifestOutcomesSection:
    """Build final manifest outcome projection from final manifest rows."""
    passed_summary = summarize_passed_manifest(passed_entries)
    dropped_summary = summarize_dropped_manifest(dropped_entries)
    return ManifestOutcomesSection(
        passed_count=passed_summary.entry_count,
        dropped_count=dropped_summary.entry_count,
        passed_validation_issue_count=passed_summary.validation_issue_count,
        dropped_validation_issue_count=dropped_summary.validation_issue_count,
    )


def build_pose_health_section(payloads: Sequence[PreparedSample]) -> PoseHealthSection:
    """Build pose health projection from PreparedSample payloads."""
    sample_summaries = tuple(summarize_prepared_sample_payload(payload) for payload in payloads)
    return PoseHealthSection(
        prepared_payload_count=len(payloads),
        total_frame_count=sum(payload.pose.frame_count for payload in payloads),
        total_valid_frame_count=sum(
            int(np.count_nonzero(payload.pose.valid_frame_mask)) for payload in payloads
        ),
        pose_complete_count=sum(1 for summary in sample_summaries if summary.pose_complete),
    )


def build_gate_outcomes_section(
    bundles: Sequence[GateDecisionBundle],
) -> GateOutcomesSection:
    """Build gate outcome projection from gate admission decisions."""
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
    handoff = summarize_checkpoint_handoff(payloads, passed_entries, dropped_entries)
    return CheckpointIntegritySection(
        payload_count=handoff.payload_count,
        passed_manifest_count=handoff.passed_count,
        dropped_manifest_count=handoff.dropped_count,
        coherent_passed_count=handoff.coherent_passed_count,
        coherence_issue_count=handoff.coherence_issue_count,
    )


def build_dropped_debug_payload_section(
    dropped_entries: Sequence[DroppedManifestEntry],
    *,
    materialize_dropped_debug_payloads: bool,
    dropped_debug_payload_written_count: int,
) -> DroppedDebugPayloadSection:
    """Build dropped manifest/debug payload projection for report clarity."""
    pose_or_source_count = sum(
        1
        for entry in dropped_entries
        if entry.drop_stage in {GateDropStage.SOURCE, GateDropStage.POSE}
    )
    debug_ref_count = sum(1 for entry in dropped_entries if entry.debug_ref is not None)
    return DroppedDebugPayloadSection(
        materialize_dropped_debug_payloads=materialize_dropped_debug_payloads,
        dropped_total_count=len(dropped_entries),
        pose_or_source_dropped_without_prepared_payload_count=pose_or_source_count,
        gate_dropped_prepared_sample_count=sum(
            1 for entry in dropped_entries if entry.drop_stage == GateDropStage.GATES
        ),
        dropped_debug_payload_written_count=dropped_debug_payload_written_count,
        dropped_manifest_entries_with_debug_ref_count=debug_ref_count,
        dropped_manifest_entries_without_debug_ref_count=len(dropped_entries)
        - debug_ref_count,
    )


__all__ = [
    "build_checkpoint_integrity_section",
    "build_dropped_debug_payload_section",
    "build_gate_outcomes_section",
    "build_manifest_outcomes_section",
    "build_pose_health_section",
    "build_source_coverage_section",
]
