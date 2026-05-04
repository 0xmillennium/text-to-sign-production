"""Text metrics computation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import TextMetrics
from text_to_sign_production.data.samples.types import ProcessedSamplePayload


def compute_text_metrics(payload: ProcessedSamplePayload) -> TextMetrics:
    """Compute text metrics."""
    normalized_text = " ".join(payload.text.split())

    character_count = len(normalized_text)
    token_count = 0 if normalized_text == "" else len(normalized_text.split(" "))

    return TextMetrics(
        normalized_text=normalized_text,
        character_count=character_count,
        token_count=token_count,
    )
