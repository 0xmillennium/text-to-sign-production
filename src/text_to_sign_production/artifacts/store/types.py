"""Typed physical path values for artifact storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from text_to_sign_production.data._shared.identities import SampleSplit


class SplitName(StrEnum):
    """Supported dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def split_name_from_sample_split(split: SplitName | SampleSplit | str) -> SplitName:
    """Canonicalize sample split identity at the artifact-store boundary."""
    return SplitName(str(split))


class SampleStatus(StrEnum):
    """Physical sample status buckets."""

    PASSED = "passed"
    DROPPED = "dropped"


class TierName(StrEnum):
    """Tiered manifest names."""

    LOOSE = "loose"
    CLEAN = "clean"
    TIGHT = "tight"


class TierMembership(StrEnum):
    """Tiered manifest membership buckets."""

    INCLUDED = "included"
    EXCLUDED = "excluded"


@dataclass(frozen=True, slots=True)
class ArtifactPathRef:
    """Physical artifact path reference."""

    path: Path


@dataclass(frozen=True, slots=True)
class ManifestPathRef:
    """Physical manifest path reference."""

    path: Path


@dataclass(frozen=True, slots=True)
class SamplePathRef:
    """Physical sample path reference."""

    path: Path


@dataclass(frozen=True, slots=True)
class ArchivePathRef:
    """Physical archive path reference."""

    path: Path


@dataclass(frozen=True, slots=True)
class ArchiveMemberPathRef:
    """Path reference for one member inside an archive."""

    path: PurePosixPath


__all__ = [
    "ArchiveMemberPathRef",
    "ArchivePathRef",
    "ArtifactPathRef",
    "ManifestPathRef",
    "SamplePathRef",
    "SampleStatus",
    "SplitName",
    "TierMembership",
    "TierName",
    "split_name_from_sample_split",
]
