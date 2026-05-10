"""Read-only inspection and audit surface for checkpoint rehydration."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.workflows._shared.checkpoint_rehydration.types import (
    CheckpointRehydrationBundle,
)


@dataclass(frozen=True, slots=True)
class CheckpointRehydrationSummary:
    """High-level summary of a rehydrated checkpoint."""

    sample_id: str
    source_truth_ready: bool
    pose_truth_ready: bool
    facts_ready: bool
    context_ready: bool
    metrics_ready: bool
    canonical_normalized_text_present: bool
    source_frame_count: int
    pose_frame_count: int
    active_frame_count: int | None
    metric_family_count: int


def summarize_checkpoint_rehydration(
    bundle: CheckpointRehydrationBundle,
) -> CheckpointRehydrationSummary:
    """Summarize the completeness and readiness of a rehydrated bundle."""
    return CheckpointRehydrationSummary(
        sample_id=bundle.source_truth.sample_id,
        source_truth_ready=True,  # Built bundle guarantees these exist
        pose_truth_ready=True,
        facts_ready=bundle.facts is not None,
        context_ready=bundle.context is not None,
        metrics_ready=bundle.metrics is not None,
        canonical_normalized_text_present=bool(bundle.source_truth.canonical_normalized_text),
        source_frame_count=bundle.source_truth.source_frame_count,
        pose_frame_count=bundle.pose_truth.frame_count,
        active_frame_count=(
            bundle.context.active_span.active_frame_count if bundle.context else None
        ),
        metric_family_count=12 if bundle.metrics else 0,  # Based on 12 explicit metric fields
    )


def audit_checkpoint_rehydration_surface(
    bundle: CheckpointRehydrationBundle,
) -> tuple[str, ...]:
    """Audit a rehydrated bundle for expected semantic continuation readiness.

    Returns a list of human-readable audit flags/warnings, or an empty tuple
    if the bundle is perfectly aligned.
    """
    flags: list[str] = []
    summary = summarize_checkpoint_rehydration(bundle)

    if not summary.canonical_normalized_text_present:
        flags.append("Missing canonical_normalized_text from source truth.")

    if summary.source_frame_count != summary.pose_frame_count:
        flags.append(
            f"Frame count mismatch: source={summary.source_frame_count}, "
            f"pose={summary.pose_frame_count}"
        )

    if not summary.facts_ready:
        flags.append("Quality facts are not available.")

    if not summary.context_ready:
        flags.append("Quality context is not available.")

    if not summary.metrics_ready:
        flags.append("Quality metric bundle is not available.")

    if summary.active_frame_count is not None and summary.active_frame_count == 0:
        flags.append("Active span contains zero active frames.")

    return tuple(flags)


__all__ = [
    "CheckpointRehydrationSummary",
    "audit_checkpoint_rehydration_surface",
    "summarize_checkpoint_rehydration",
]
