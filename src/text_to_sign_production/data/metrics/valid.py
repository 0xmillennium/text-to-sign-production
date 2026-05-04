"""Valid frame metrics computation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import ValidMetrics
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_valid_metrics(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> ValidMetrics:
    """Compute valid frame metrics."""
    valid_frame_count = manifest.frame_quality.valid_frame_count
    invalid_frame_count = manifest.frame_quality.invalid_frame_count

    zeroed_canonical_joint_frame_count = (
        manifest.frame_quality.frames_with_any_zeroed_canonical_joint
    )

    num_frames = payload.num_frames

    if valid_frame_count + invalid_frame_count != num_frames:
        raise ValueError(
            f"Valid frame consistency error: valid ({valid_frame_count}) + "
            f"invalid ({invalid_frame_count}) != num_frames ({num_frames})"
        )

    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for valid metrics.")

    valid_ratio = valid_frame_count / num_frames
    invalid_ratio = invalid_frame_count / num_frames
    zeroed_ratio = zeroed_canonical_joint_frame_count / num_frames

    return ValidMetrics(
        valid_frame_count=valid_frame_count,
        invalid_frame_count=invalid_frame_count,
        valid_frame_ratio=valid_ratio,
        invalid_frame_ratio=invalid_ratio,
        zeroed_canonical_joint_frame_count=zeroed_canonical_joint_frame_count,
        zeroed_canonical_joint_frame_ratio=zeroed_ratio,
    )
