"""Keypoint directory resolution and source-level facts."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.sources.types import KeypointSourceFacts


def resolve_keypoint_source(directory: Path) -> KeypointSourceFacts:
    """Resolve facts about a keypoint source directory.

    This function checks existence and counts frame JSON files, but it does
    not parse frame content.
    """
    if not directory.is_dir():
        return KeypointSourceFacts(
            directory=directory,
            exists=False,
            frame_count=0,
        )

    # We only count JSON files.
    # We do not list them all out into memory here beyond what's needed for the count,
    # though glob() does load them. Using len(list(glob)) is acceptable at source level.
    # Downstream `pose.files` will enumerate and sort them.
    frame_count = sum(1 for _ in directory.glob("*.json"))

    return KeypointSourceFacts(
        directory=directory,
        exists=True,
        frame_count=frame_count,
    )
