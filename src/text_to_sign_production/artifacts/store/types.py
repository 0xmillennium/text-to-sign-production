"""Typed physical path values for artifact storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from text_to_sign_production.core.ids import (
    SampleSplit,
    SampleStatus,
    TierMembership,
    TierName,
)


def sample_split_from_value(split: SampleSplit | str) -> SampleSplit:
    """Canonicalize sample split identity at the artifact-store boundary."""
    return SampleSplit(str(split))


def sample_status_from_value(status: SampleStatus | str) -> SampleStatus:
    """Canonicalize physical sample status at the artifact-store boundary."""
    return SampleStatus(str(status))


def tier_name_from_value(tier: TierName | str) -> TierName:
    """Canonicalize physical tier identity at the artifact-store boundary."""
    return TierName(str(tier))


def tier_membership_from_value(membership: TierMembership | str) -> TierMembership:
    """Canonicalize physical tier membership at the artifact-store boundary."""
    return TierMembership(str(membership))


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
class ReportPathRef:
    """Physical report path reference."""

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
    "ReportPathRef",
    "SamplePathRef",
    "sample_split_from_value",
    "sample_status_from_value",
    "tier_membership_from_value",
    "tier_name_from_value",
]
