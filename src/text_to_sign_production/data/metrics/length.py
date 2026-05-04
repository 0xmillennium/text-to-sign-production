"""Length and duration metrics computation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import LengthMetrics, TextMetrics
from text_to_sign_production.data.samples.types import ProcessedSamplePayload


def compute_length_metrics(
    payload: ProcessedSamplePayload, text_metrics: TextMetrics
) -> LengthMetrics:
    """Compute length metrics."""
    num_frames = payload.num_frames
    fps = payload.fps

    duration_seconds = None
    if fps is not None:
        if fps <= 0:
            raise ValueError(f"Invalid fps ({fps}) for length metrics.")
        duration_seconds = num_frames / fps

    if text_metrics.token_count <= 0:
        raise ValueError(f"Invalid token_count ({text_metrics.token_count}) for length metrics.")
    frames_per_token = num_frames / text_metrics.token_count

    if text_metrics.character_count <= 0:
        raise ValueError(
            f"Invalid character_count ({text_metrics.character_count}) for length metrics."
        )
    frames_per_character = num_frames / text_metrics.character_count

    return LengthMetrics(
        num_frames=num_frames,
        fps=fps,
        duration_seconds=duration_seconds,
        frames_per_token=frames_per_token,
        frames_per_character=frames_per_character,
    )
