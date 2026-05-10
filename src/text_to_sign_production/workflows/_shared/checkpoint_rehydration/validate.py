"""Checkpoint rehydration validation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.tier._shared.types import (
    QualityPoseTruth,
    QualitySourceTruth,
)
from text_to_sign_production.workflows._shared.checkpoint_rehydration.types import (
    CheckpointRehydrationBundle,
    CheckpointRehydrationInput,
    CheckpointRehydrationIssue,
    CheckpointRehydrationIssueCode,
)


def validate_checkpoint_input(
    input: CheckpointRehydrationInput,
) -> tuple[CheckpointRehydrationIssue, ...]:
    """Validate checkpoint input coherence between manifest and payload."""
    issues: list[CheckpointRehydrationIssue] = []
    manifest = input.passed_manifest
    payload = input.payload

    if manifest.sample_id != payload.sample_id:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.MANIFEST_PAYLOAD_IDENTITY_MISMATCH,
                message="Manifest sample_id does not match payload sample_id.",
                field_path="sample_id",
            )
        )
    if manifest.split is not payload.split:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.MANIFEST_PAYLOAD_SPLIT_MISMATCH,
                message="Manifest split does not match payload split.",
                field_path="split",
            )
        )
    if manifest.text != payload.text:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.MANIFEST_PAYLOAD_TEXT_MISMATCH,
                message="Manifest text does not match payload text.",
                field_path="text",
            )
        )
    if manifest.frame_count != payload.frame.frame_count:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.MANIFEST_PAYLOAD_FRAME_COUNT_MISMATCH,
                message="Manifest frame_count does not match payload frame count.",
                field_path="frame_count",
            )
        )
    if manifest.canonical_normalized_text is None:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.NORMALIZED_TEXT_HANDOFF_MISSING,
                message=(
                    "Manifest does not carry canonical_normalized_text; "
                    "downstream quality layers cannot use checkpoint-authoritative "
                    "normalized text."
                ),
                field_path="canonical_normalized_text",
            )
        )

    return tuple(issues)


def validate_rehydrated_source_truth(
    source_truth: QualitySourceTruth,
    input: CheckpointRehydrationInput,
) -> tuple[CheckpointRehydrationIssue, ...]:
    """Validate rehydrated source truth coherence against checkpoint."""
    issues: list[CheckpointRehydrationIssue] = []

    if source_truth.sample_id != input.payload.sample_id:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.SOURCE_TRUTH_INCOHERENT,
                message="Rehydrated source truth sample_id does not match checkpoint.",
                field_path="source_truth.sample_id",
            )
        )
    if source_truth.text != input.payload.text:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.SOURCE_TRUTH_INCOHERENT,
                message="Rehydrated source truth text does not match checkpoint.",
                field_path="source_truth.text",
            )
        )
    if (
        input.passed_manifest.canonical_normalized_text is not None
        and source_truth.canonical_normalized_text
        != input.passed_manifest.canonical_normalized_text
    ):
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.SOURCE_TRUTH_INCOHERENT,
                message=(
                    "Rehydrated canonical_normalized_text does not match "
                    "manifest checkpoint authority."
                ),
                field_path="source_truth.canonical_normalized_text",
            )
        )

    return tuple(issues)


def validate_rehydrated_pose_truth(
    pose_truth: QualityPoseTruth,
    input: CheckpointRehydrationInput,
) -> tuple[CheckpointRehydrationIssue, ...]:
    """Validate rehydrated pose truth coherence against checkpoint."""
    issues: list[CheckpointRehydrationIssue] = []

    if pose_truth.frame_count != input.payload.frame.frame_count:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.POSE_TRUTH_FRAME_COUNT_MISMATCH,
                message="Rehydrated pose truth frame count does not match checkpoint.",
                field_path="pose_truth.frame_count",
            )
        )
    if pose_truth.valid_frame_count != int(np.count_nonzero(input.payload.frame.frame_valid_mask)):
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.POSE_TRUTH_INCOHERENT,
                message="Rehydrated valid frame count does not match checkpoint.",
                field_path="pose_truth.valid_frame_count",
            )
        )

    return tuple(issues)


def validate_checkpoint_bundle(
    bundle: CheckpointRehydrationBundle,
) -> tuple[CheckpointRehydrationIssue, ...]:
    """Validate final rehydrated bundle coherence."""
    issues: list[CheckpointRehydrationIssue] = []

    if bundle.source_truth is None:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.BUNDLE_SOURCE_TRUTH_MISSING,
                message="Rehydrated bundle is missing source truth.",
                field_path="source_truth",
            )
        )
    if bundle.pose_truth is None:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.BUNDLE_POSE_TRUTH_MISSING,
                message="Rehydrated bundle is missing pose truth.",
                field_path="pose_truth",
            )
        )
    if bundle.facts is None:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.BUNDLE_FACTS_MISSING,
                message="Rehydrated bundle is missing quality facts.",
                field_path="facts",
            )
        )
    if bundle.context is None:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.BUNDLE_CONTEXT_MISSING,
                message="Rehydrated bundle is missing quality context.",
                field_path="context",
            )
        )
    if bundle.metrics is None:
        issues.append(
            CheckpointRehydrationIssue(
                code=CheckpointRehydrationIssueCode.BUNDLE_METRICS_MISSING,
                message="Rehydrated bundle is missing quality metrics.",
                field_path="metrics",
            )
        )

    # Cross-bundle coherence: facts source should match source truth
    if bundle.source_truth is not None and bundle.facts is not None:
        if bundle.facts.source.sample_id != bundle.source_truth.sample_id:
            issues.append(
                CheckpointRehydrationIssue(
                    code=CheckpointRehydrationIssueCode.SOURCE_TRUTH_INCOHERENT,
                    message="Bundle facts sample_id does not match source truth.",
                    field_path="facts.source.sample_id",
                )
            )

    # Cross-bundle coherence: facts frame count should match pose truth
    if bundle.pose_truth is not None and bundle.facts is not None:
        if bundle.facts.frame.frame_count != bundle.pose_truth.frame_count:
            issues.append(
                CheckpointRehydrationIssue(
                    code=CheckpointRehydrationIssueCode.POSE_TRUTH_FRAME_COUNT_MISMATCH,
                    message="Bundle facts frame count does not match pose truth.",
                    field_path="facts.frame.frame_count",
                )
            )

    return tuple(issues)


__all__ = [
    "validate_checkpoint_bundle",
    "validate_checkpoint_input",
    "validate_rehydrated_pose_truth",
    "validate_rehydrated_source_truth",
]
