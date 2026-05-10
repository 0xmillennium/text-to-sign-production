"""Text diagnostic family metrics."""

from __future__ import annotations

from text_to_sign_production.data.tier.families.types import (
    QualityMetricBuildInput,
    TextDiagnosticMetrics,
)


def compute_text_diagnostic_metrics(input: QualityMetricBuildInput) -> TextDiagnosticMetrics:
    """Compute text-side diagnostic densities from persisted facts."""
    text = input.quality_facts.text_length
    frame_count = input.quality_facts.frame.frame_count
    return TextDiagnosticMetrics(
        token_count=text.text_token_count,
        character_count=text.text_character_count,
        non_whitespace_character_count=text.text_non_whitespace_character_count,
        frames_per_token=_density(frame_count, text.text_token_count),
        frames_per_character=_density(frame_count, text.text_character_count),
        frames_per_non_whitespace_character=_density(
            frame_count,
            text.text_non_whitespace_character_count,
        ),
    )


def _density(frame_count: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return frame_count / denominator


__all__ = ["compute_text_diagnostic_metrics"]
