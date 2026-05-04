"""Face metrics computation."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.data.metrics.types import FaceMetrics
from text_to_sign_production.data.samples.types import PassedManifestEntry, ProcessedSamplePayload


def compute_face_metrics(
    payload: ProcessedSamplePayload, manifest: PassedManifestEntry
) -> FaceMetrics:
    """Compute temporal face availability metrics."""
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

    return FaceMetrics(
        face_available_frame_count=face_available_frame_count,
        face_unavailable_frame_count=face_unavailable_frame_count,
        face_available_frame_ratio=face_available_ratio,
        face_unavailable_frame_ratio=face_unavailable_ratio,
    )
