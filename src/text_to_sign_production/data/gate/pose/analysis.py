"""Lightweight pose-level inspection helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from text_to_sign_production.data.gate.pose.types import (
    ParsedFrame,
    ParsedFrameSummary,
    PersonTrackingResult,
    PoseBuildOutput,
    PoseDiagnosticCode,
    PoseTensorOutput,
    TensorAvailabilitySummary,
    TrackingSummary,
)


def summarize_parsed_frames(frames: Sequence[ParsedFrame]) -> ParsedFrameSummary:
    """Summarize parsed frame availability without producing pose truth."""
    people_counts = [len(frame.people) for frame in frames]
    return ParsedFrameSummary(
        frame_count=len(frames),
        valid_frame_count=sum(1 for frame in frames if frame.frame_valid),
        people_count_total=sum(people_counts),
        max_people_per_frame=max(people_counts, default=0),
    )


def summarize_tracking(tracking: PersonTrackingResult) -> TrackingSummary:
    """Summarize frame-wise tracking output."""
    return TrackingSummary(
        frame_count=len(tracking.frame_selections),
        missing_frame_count=tracking.target_missing_frame_count,
        continuity_break_count=tracking.continuity_break_count,
        reanchor_count=tracking.reanchor_count,
    )


def summarize_tensor_availability(output: PoseTensorOutput) -> TensorAvailabilitySummary:
    """Summarize tensor availability without computing quality metrics."""
    return TensorAvailabilitySummary(
        frame_count=len(output.frame_valid_mask),
        valid_frame_count=int(output.frame_valid_mask.sum()),
        channel_nonzero_frame_counts=dict(output.channel_nonzero_frame_counts),
    )


def diagnostic_code_counts(
    output: PoseBuildOutput,
) -> tuple[tuple[PoseDiagnosticCode, int], ...]:
    """Count diagnostic codes for debugging."""
    counter = Counter(diagnostic.code for diagnostic in output.diagnostics.diagnostics)
    return tuple(
        (code, count) for code, count in sorted(counter.items(), key=lambda item: item[0].value)
    )


__all__ = [
    "diagnostic_code_counts",
    "summarize_parsed_frames",
    "summarize_tensor_availability",
    "summarize_tracking",
]
