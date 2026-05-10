"""Orchestrator for full checkpoint rehydration.

This module provides the main entry point for rehydrating a complete
downstream quality bundle from a gate checkpoint. The orchestration
follows the pattern:

    validate input → build source truth → build pose truth →
    build facts → build context → build metrics → validate bundle
"""

from __future__ import annotations

from text_to_sign_production.workflows._shared.checkpoint_rehydration.pose import (
    build_quality_pose_truth,
)
from text_to_sign_production.workflows._shared.checkpoint_rehydration.quality import (
    build_quality_context_from_checkpoint,
    build_quality_facts_from_checkpoint,
    build_quality_metrics_from_checkpoint,
)
from text_to_sign_production.workflows._shared.checkpoint_rehydration.source import (
    build_quality_source_truth,
)
from text_to_sign_production.workflows._shared.checkpoint_rehydration.types import (
    CheckpointRehydrationBundle,
    CheckpointRehydrationInput,
    CheckpointRehydrationIssue,
)
from text_to_sign_production.workflows._shared.checkpoint_rehydration.validate import (
    validate_checkpoint_bundle,
    validate_checkpoint_input,
    validate_rehydrated_pose_truth,
    validate_rehydrated_source_truth,
)


class CheckpointRehydrationError(Exception):
    """Raised when checkpoint rehydration fails validation."""

    def __init__(
        self,
        message: str,
        issues: tuple[CheckpointRehydrationIssue, ...],
    ) -> None:
        super().__init__(message)
        self.issues = issues


def rehydrate_checkpoint(
    input: CheckpointRehydrationInput,
    *,
    strict: bool = True,
) -> CheckpointRehydrationBundle:
    """Rehydrate a complete downstream quality bundle from checkpoint.

    This is the main entry point for the checkpoint rehydration bridge.
    The orchestration delegates to each owning bounded context's builder:

    1. Validate input coherence between manifest and payload
    2. Build source truth from checkpoint
    3. Build pose truth from checkpoint
    4. Build facts from neutral truth (delegated to facts bounded context)
    5. Build context from neutral truth (delegated to context bounded context)
    6. Build metrics from neutral truth (delegated to families bounded context)
    7. Validate final bundle coherence

    Parameters
    ----------
    input : CheckpointRehydrationInput
        The checkpoint authority (manifest + payload).
    strict : bool
        If True (default), raise ``CheckpointRehydrationError`` on any
        validation issue. If False, proceed despite non-blocking issues
        (missing canonical text warning is always non-blocking).

    Returns
    -------
    CheckpointRehydrationBundle
        The fully rehydrated quality bundle.

    Raises
    ------
    CheckpointRehydrationError
        If any blocking validation issue is detected (strict mode).
    """
    # Step 1: Validate input coherence
    input_issues = validate_checkpoint_input(input)
    if strict and input_issues:
        raise CheckpointRehydrationError(
            f"Checkpoint input validation failed with {len(input_issues)} issue(s).",
            input_issues,
        )

    # Step 2: Build source truth
    source_truth = build_quality_source_truth(input)

    # Step 3: Build pose truth
    pose_truth = build_quality_pose_truth(input)

    # Step 4: Validate rehydrated truth
    source_issues = validate_rehydrated_source_truth(source_truth, input)
    pose_issues = validate_rehydrated_pose_truth(pose_truth, input)
    if strict and (source_issues or pose_issues):
        raise CheckpointRehydrationError(
            "Rehydrated truth validation failed.",
            source_issues + pose_issues,
        )

    # Step 5: Build facts
    facts = build_quality_facts_from_checkpoint(source_truth, pose_truth)

    # Step 6: Build context
    context = build_quality_context_from_checkpoint(source_truth, pose_truth, facts)

    # Step 7: Build metrics
    metrics = build_quality_metrics_from_checkpoint(source_truth, pose_truth, facts, context)

    # Step 8: Assemble bundle
    bundle = CheckpointRehydrationBundle(
        source_truth=source_truth,
        pose_truth=pose_truth,
        facts=facts,
        context=context,
        metrics=metrics,
    )

    # Step 9: Validate bundle coherence
    bundle_issues = validate_checkpoint_bundle(bundle)
    if strict and bundle_issues:
        raise CheckpointRehydrationError(
            f"Bundle validation failed with {len(bundle_issues)} issue(s).",
            bundle_issues,
        )

    return bundle


__all__ = ["CheckpointRehydrationError", "rehydrate_checkpoint"]
