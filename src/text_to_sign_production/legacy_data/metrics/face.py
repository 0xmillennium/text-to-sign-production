"""Face metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.legacy_data.metrics.types import (
    ActiveSigningSpanMetrics,
    NonManualVisibilityMetrics,
)
from text_to_sign_production.legacy_data.samples.types import (
    PassedManifestEntry,
    ProcessedSamplePayload,
)


def compute_face_metrics(
    payload: ProcessedSamplePayload,
    manifest: PassedManifestEntry,
    active_signing_span: ActiveSigningSpanMetrics,
) -> NonManualVisibilityMetrics:
    """Compute temporal non-manual visibility metrics."""
    face_conf = np.asarray(payload.pose.face.confidence)
    face_frame_available = np.any(face_conf > 0.0, axis=1)

    face_available_frame_count = int(np.count_nonzero(face_frame_available))
    face_unavailable_frame_count = manifest.frame_quality.face_missing_frame_count

    num_frames = payload.num_frames

    if face_available_frame_count + face_unavailable_frame_count != num_frames:
        raise ValueError(
            f"Face frame consistency error: available ({face_available_frame_count}) + "
            f"unavailable ({face_unavailable_frame_count}) != num_frames ({num_frames})"
        )

    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames ({num_frames}) for face metrics.")

    face_available_ratio = face_available_frame_count / num_frames
    face_unavailable_ratio = face_unavailable_frame_count / num_frames
    active_face_available = face_frame_available[
        active_signing_span.start_frame_index : active_signing_span.end_frame_index_exclusive
    ]
    if active_face_available.size != active_signing_span.frame_count:
        raise ValueError("Active signing span frame count does not match face confidence array.")

    active_span_face_available_frame_count = int(np.count_nonzero(active_face_available))
    max_active_face_unavailable_run = _longest_false_run(active_face_available)

    return NonManualVisibilityMetrics(
        face_available_frame_count=face_available_frame_count,
        face_unavailable_frame_count=face_unavailable_frame_count,
        face_available_frame_ratio=face_available_ratio,
        face_unavailable_frame_ratio=face_unavailable_ratio,
        active_span_face_available_frame_count=active_span_face_available_frame_count,
        active_span_face_available_frame_ratio=(
            active_span_face_available_frame_count / active_signing_span.frame_count
        ),
        max_active_span_face_unavailable_run_count=max_active_face_unavailable_run,
        max_active_span_face_unavailable_run_ratio=(
            max_active_face_unavailable_run / active_signing_span.frame_count
        ),
    )


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest
