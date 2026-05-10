"""Keypoint-source truth construction."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.gate.sources.types import KeypointSourceRecord, keypoint_identity


def build_keypoint_record(
    directory: Path,
    *,
    sample_id: str | None = None,
    frame_pattern: str = "*.json",
) -> KeypointSourceRecord:
    """Build keypoint source truth without parsing frame content."""
    source_sample_id = sample_id.strip() if sample_id is not None else directory.name
    identity = keypoint_identity(source_sample_id) if sample_id is not None else None
    if not directory.is_dir():
        return KeypointSourceRecord(
            sample_id=source_sample_id,
            directory=directory,
            exists=False,
            frame_count=0,
            identity=identity,
        )

    frame_count = sum(1 for _ in directory.glob(frame_pattern))
    return KeypointSourceRecord(
        sample_id=source_sample_id,
        directory=directory,
        exists=True,
        frame_count=frame_count,
        identity=identity,
    )


__all__ = ["build_keypoint_record"]
