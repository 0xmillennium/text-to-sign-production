"""Shared generated-artifact value objects and errors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.data.schemas import ProcessedManifestEntry

_CANDIDATE_LIMIT = 8


class ProcessedSampleDataError(ValueError):
    """Raised when processed sample artifact inputs violate the expected contract."""


class SampleSelectionError(ProcessedSampleDataError):
    """Raised when a requested processed sample cannot be selected."""


class DuplicateSampleIdError(SampleSelectionError):
    """Raised when a sample id is present in more than one loaded manifest record."""

    def __init__(self, sample_id: str, candidates: Iterable[ManifestSampleRecord]) -> None:
        self.sample_id = sample_id
        self.candidates = tuple(candidates)
        rendered = "\n".join(
            f"- split={record.split} sample_id={record.sample_id} sample_path={record.sample_path}"
            for record in self.candidates[:_CANDIDATE_LIMIT]
        )
        suffix = ""
        if len(self.candidates) > _CANDIDATE_LIMIT:
            suffix = f"\n... {len(self.candidates) - _CANDIDATE_LIMIT} more candidate(s)."
        super().__init__(
            f"Sample id {sample_id!r} appears in {len(self.candidates)} loaded processed "
            f"manifest records; select a unique sample id or restrict --split.\n{rendered}{suffix}"
        )


@dataclass(frozen=True, slots=True)
class TimingMetadata:
    """Timing and source-video metadata joined from interim manifests."""

    start_time: float | None
    end_time: float | None
    fps: float | None
    num_frames: int | None
    source_video_path: str | None
    video_width: int | None
    video_height: int | None
    source_manifest: str | None
    join_key: str | None


@dataclass(frozen=True, slots=True)
class ManifestSampleRecord:
    """One canonical processed manifest record resolved to a local `.npz` path."""

    entry: ProcessedManifestEntry
    sample_path: Path

    @property
    def sample_id(self) -> str:
        return self.entry.sample_id

    @property
    def split(self) -> str:
        return self.entry.split

    @property
    def text(self) -> str:
        return self.entry.text

    @property
    def fps(self) -> float | None:
        return self.entry.fps

    @property
    def num_frames(self) -> int:
        return self.entry.num_frames

    @property
    def sample_path_value(self) -> str:
        return self.entry.sample_path

    @property
    def source_video_path(self) -> str | None:
        return self.entry.source_video_path

    @property
    def video_width(self) -> int | None:
        return self.entry.video_width

    @property
    def video_height(self) -> int | None:
        return self.entry.video_height


__all__ = [
    "DuplicateSampleIdError",
    "ManifestSampleRecord",
    "ProcessedSampleDataError",
    "SampleSelectionError",
    "TimingMetadata",
]
