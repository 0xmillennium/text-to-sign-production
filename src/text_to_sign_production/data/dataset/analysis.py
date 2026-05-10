"""Read-only samples checkpoint observation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.dataset.validate import (
    validate_manifest_entry,
    validate_payload_manifest_coherence,
    validate_prepared_sample,
)


@dataclass(frozen=True, slots=True)
class PreparedSampleSummary:
    """Compact observation of a PreparedSample payload."""

    sample_id: str
    split: str
    frame_count: int
    valid_frame_count: int
    source_complete: bool
    pose_complete: bool
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class PassedManifestSummary:
    """Compact observation of passed manifest rows."""

    entry_count: int
    canonical_normalized_text_present_count: int
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class DroppedManifestSummary:
    """Compact observation of dropped manifest rows."""

    entry_count: int
    entries_with_issue_codes_count: int
    validation_issue_count: int


@dataclass(frozen=True, slots=True)
class CheckpointHandoffSummary:
    """Compact payload/manifest handoff integrity observation."""

    payload_count: int
    passed_count: int
    dropped_count: int
    coherent_passed_count: int
    coherence_issue_count: int
    passed_canonical_normalized_text_complete: bool


def summarize_prepared_sample_payload(sample: PreparedSample) -> PreparedSampleSummary:
    """Summarize PreparedSample payload completeness."""
    issues = validate_prepared_sample(sample)
    return PreparedSampleSummary(
        sample_id=sample.source.sample_id,
        split=sample.source.split.value,
        frame_count=sample.pose.frame_count,
        valid_frame_count=int(np.count_nonzero(sample.pose.valid_frame_mask)),
        source_complete=bool(
            sample.source.sample_id
            and sample.source.text
            and sample.source.canonical_normalized_text
            and sample.source.source_video_id
            and sample.source.source_sentence_id
            and sample.source.source_sentence_name
        ),
        pose_complete=sample.pose.frame_count > 0
        and all(
            tensor.shape[0] == sample.pose.frame_count
            for tensor in (
                sample.pose.body_xyc,
                sample.pose.face_xyc,
                sample.pose.left_hand_xyc,
                sample.pose.right_hand_xyc,
            )
        ),
        validation_issue_count=len(issues),
    )


def summarize_passed_manifest(
    entries: Sequence[PassedManifestEntry],
) -> PassedManifestSummary:
    """Summarize passed manifest completeness."""
    issue_count = sum(len(validate_manifest_entry(entry)) for entry in entries)
    return PassedManifestSummary(
        entry_count=len(entries),
        canonical_normalized_text_present_count=sum(
            1 for entry in entries if bool(entry.canonical_normalized_text.strip())
        ),
        validation_issue_count=issue_count,
    )


def summarize_dropped_manifest(
    entries: Sequence[DroppedManifestEntry],
) -> DroppedManifestSummary:
    """Summarize dropped manifest completeness."""
    issue_count = sum(len(validate_manifest_entry(entry)) for entry in entries)
    return DroppedManifestSummary(
        entry_count=len(entries),
        entries_with_issue_codes_count=sum(1 for entry in entries if entry.issue_codes),
        validation_issue_count=issue_count,
    )


def summarize_checkpoint_handoff(
    payloads: Sequence[PreparedSample],
    passed_entries: Sequence[PassedManifestEntry],
    dropped_entries: Sequence[DroppedManifestEntry],
) -> CheckpointHandoffSummary:
    """Summarize checkpoint handoff integrity between payloads and manifests."""
    payload_by_id = {payload.source.sample_id: payload for payload in payloads}
    coherent = 0
    coherence_issue_count = 0
    for entry in passed_entries:
        payload = payload_by_id.get(entry.sample_id)
        if payload is None:
            coherence_issue_count += 1
            continue
        issues = validate_payload_manifest_coherence(payload, entry)
        coherence_issue_count += len(issues)
        if not issues:
            coherent += 1
    return CheckpointHandoffSummary(
        payload_count=len(payloads),
        passed_count=len(passed_entries),
        dropped_count=len(dropped_entries),
        coherent_passed_count=coherent,
        coherence_issue_count=coherence_issue_count,
        passed_canonical_normalized_text_complete=all(
            bool(entry.canonical_normalized_text.strip()) for entry in passed_entries
        ),
    )


__all__ = [
    "CheckpointHandoffSummary",
    "DroppedManifestSummary",
    "PassedManifestSummary",
    "PreparedSampleSummary",
    "summarize_checkpoint_handoff",
    "summarize_dropped_manifest",
    "summarize_passed_manifest",
    "summarize_prepared_sample_payload",
]
