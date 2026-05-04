"""Deterministic frame file discovery."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.pose.types import FrameFileListing


def discover_frame_files(directory: Path) -> FrameFileListing:
    """Discover and order frame JSON files deterministically."""

    if not directory.exists():
        return FrameFileListing(directory=directory, files=(), missing=True)

    if not directory.is_dir():
        return FrameFileListing(directory=directory, files=(), missing=True)

    files = sorted(directory.glob("*.json"), key=lambda p: p.name)
    return FrameFileListing(
        directory=directory,
        files=tuple(files),
        missing=False,
    )
