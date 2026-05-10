"""Face-region context builder."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.core.models import PreparedSample
from text_to_sign_production.data.tier.context.types import (
    ActiveSpanContext,
    FaceRegionContext,
    FaceRegionFrameContext,
    RepresentativeHandContext,
)

FACE_UPPER_REGION_INDICES: tuple[int, ...] = tuple(range(17, 48)) + (68, 69)
FACE_LOWER_REGION_INDICES: tuple[int, ...] = tuple(range(0, 17)) + tuple(range(48, 68))
_WHOLE_FACE_POINT_COUNT = 70


def build_face_region_context(
    sample: PreparedSample,
    active_span: ActiveSpanContext,
    hand_context: RepresentativeHandContext,
) -> FaceRegionContext:
    """Build face-region support context aligned to active-span context."""
    face_conf = sample.pose.face_xyc[..., 2]
    frames: list[FaceRegionFrameContext] = []
    active_face: list[bool] = []
    active_upper: list[bool] = []
    active_lower: list[bool] = []
    overlap: list[bool] = []
    for frame_index in range(active_span.frame_count):
        whole_count = int(np.count_nonzero(face_conf[frame_index] > 0.0))
        upper_count = int(
            np.count_nonzero(face_conf[frame_index, list(FACE_UPPER_REGION_INDICES)] > 0.0)
        )
        lower_count = int(
            np.count_nonzero(face_conf[frame_index, list(FACE_LOWER_REGION_INDICES)] > 0.0)
        )
        active = active_span.active_frame_mask[frame_index]
        manual_available = hand_context.frames[frame_index].available
        whole_available = whole_count > 0
        upper_supported = upper_count > 0
        lower_supported = lower_count > 0
        ready = active and manual_available and whole_available
        active_face.append(active and whole_available)
        active_upper.append(active and upper_supported)
        active_lower.append(active and lower_supported)
        overlap.append(ready)
        frames.append(
            FaceRegionFrameContext(
                frame_index=frame_index,
                active=active,
                whole_face_present_count=whole_count,
                whole_face_coverage_fraction=_fraction(whole_count, _WHOLE_FACE_POINT_COUNT),
                upper_face_present_count=upper_count,
                upper_face_coverage_fraction=_fraction(upper_count, len(FACE_UPPER_REGION_INDICES)),
                lower_face_present_count=lower_count,
                lower_face_coverage_fraction=_fraction(lower_count, len(FACE_LOWER_REGION_INDICES)),
                whole_face_available=whole_available,
                upper_face_supported=upper_supported,
                lower_face_supported=lower_supported,
                manual_available=manual_available,
                manual_face_overlap_ready=ready,
            )
        )
    return FaceRegionContext(
        frames=tuple(frames),
        active_face_available_mask=tuple(active_face),
        active_upper_face_supported_mask=tuple(active_upper),
        active_lower_face_supported_mask=tuple(active_lower),
        manual_face_overlap_ready_mask=tuple(overlap),
    )


def _fraction(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


__all__ = ["build_face_region_context"]
