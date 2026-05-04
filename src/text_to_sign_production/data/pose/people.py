"""Selected-person logic for multi-person scenarios."""

from __future__ import annotations

from text_to_sign_production.data.pose.types import ParsedFrameResult
from text_to_sign_production.data.samples.types import SelectedPersonMetadata


def resolve_target_person_index(frames: list[ParsedFrameResult]) -> int:
    """Resolve which person index should be parsed from frames.

    Currently we always pick index 0 (the primary signer detected by OpenPose).
    """
    return 0


def build_person_metadata(
    *, target_index: int, multi_person_frame_count: int, max_people_per_frame: int
) -> SelectedPersonMetadata:
    """Build the final person selection facts."""
    return SelectedPersonMetadata(
        index=target_index,
        multi_person_frame_count=multi_person_frame_count,
        max_people_per_frame=max_people_per_frame,
    )
