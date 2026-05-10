"""Pose-file access helpers sourced from data.sources.SourceCandidate."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.gate.pose.types import (
    FrameFileListing,
    PoseDiagnostic,
    PoseDiagnosticCode,
    PoseDiagnosticSeverity,
)
from text_to_sign_production.data.gate.sources import SourceCandidate


def _diagnostic(code: PoseDiagnosticCode, message: str) -> PoseDiagnostic:
    return PoseDiagnostic(
        code=code,
        severity=PoseDiagnosticSeverity.ERROR,
        message=message,
    )


def frame_directory_from_candidate(candidate: SourceCandidate) -> Path:
    """Return the authoritative pose frame directory from source truth."""
    return candidate.keypoints_dir


def discover_frame_files(candidate: SourceCandidate) -> FrameFileListing:
    """Discover frame JSON files for a source candidate deterministically."""
    directory = frame_directory_from_candidate(candidate)
    diagnostics: list[PoseDiagnostic] = []
    if not directory.exists():
        diagnostics.append(
            _diagnostic(
                PoseDiagnosticCode.FRAME_FILE_DIRECTORY_MISSING,
                f"Pose frame directory does not exist: {directory}.",
            )
        )
        return FrameFileListing(
            candidate=candidate,
            directory=directory,
            files=(),
            missing=True,
            diagnostics=tuple(diagnostics),
        )
    if not directory.is_dir():
        diagnostics.append(
            _diagnostic(
                PoseDiagnosticCode.FRAME_FILE_DIRECTORY_INVALID,
                f"Pose frame path is not a directory: {directory}.",
            )
        )
        return FrameFileListing(
            candidate=candidate,
            directory=directory,
            files=(),
            missing=True,
            diagnostics=tuple(diagnostics),
        )

    files = tuple(sorted(directory.glob("*.json"), key=lambda path: path.name))
    if len(files) != candidate.frame_count:
        diagnostics.append(
            PoseDiagnostic(
                code=PoseDiagnosticCode.FRAME_FILE_COUNT_MISMATCH,
                severity=PoseDiagnosticSeverity.WARNING,
                message=(
                    "Discovered frame file count does not match "
                    f"SourceCandidate frame_count: {len(files)} != {candidate.frame_count}."
                ),
            )
        )
    return FrameFileListing(
        candidate=candidate,
        directory=directory,
        files=files,
        missing=False,
        diagnostics=tuple(diagnostics),
    )


__all__ = ["discover_frame_files", "frame_directory_from_candidate"]
