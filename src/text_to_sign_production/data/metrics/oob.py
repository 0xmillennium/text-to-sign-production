"""Out-of-bounds metrics computation."""

from __future__ import annotations

from text_to_sign_production.data.metrics.types import OobMetrics
from text_to_sign_production.data.samples.schema import (
    CANONICAL_POSE_CHANNELS,
    POSE_CHANNEL_JOINT_COUNTS,
    POSE_COORDINATE_DIMENSIONS,
)
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_oob_metrics(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> OobMetrics:
    """Compute out-of-bounds metrics."""
    out_of_bounds_coordinate_count = manifest.frame_quality.out_of_bounds_coordinate_count

    total_joint_count = sum(
        POSE_CHANNEL_JOINT_COUNTS[channel] for channel in CANONICAL_POSE_CHANNELS
    )
    coordinate_dimensions = POSE_COORDINATE_DIMENSIONS

    total_coordinate_slots = payload.num_frames * total_joint_count * coordinate_dimensions

    if total_coordinate_slots <= 0:
        raise ValueError(
            f"Invalid total_coordinate_slots ({total_coordinate_slots}) for OOB metrics."
        )

    out_of_bounds_ratio = out_of_bounds_coordinate_count / total_coordinate_slots

    return OobMetrics(
        out_of_bounds_coordinate_count=out_of_bounds_coordinate_count,
        total_coordinate_slots=total_coordinate_slots,
        out_of_bounds_ratio=out_of_bounds_ratio,
    )
