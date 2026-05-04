"""Logical catalog models for sample artifact lookup."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.artifacts.store.types import (
    ArchiveMemberPathRef,
    ArchivePathRef,
    SamplePathRef,
    SampleStatus,
    SplitName,
    TierMembership,
    TierName,
)
from text_to_sign_production.data.samples.types import (
    DroppedManifestEntry,
    PassedManifestEntry,
)


@dataclass(frozen=True, slots=True)
class SampleRef:
    """Logical sample identity inside catalog surfaces."""

    split: SplitName
    sample_id: str


@dataclass(frozen=True, slots=True)
class SampleHandle:
    """Logical handle for a passed or dropped sample manifest item."""

    ref: SampleRef
    status: SampleStatus
    manifest_entry: PassedManifestEntry | DroppedManifestEntry
    runtime_sample: SamplePathRef | None
    drive_archive: ArchivePathRef | None
    drive_archive_member: ArchiveMemberPathRef | None
    timing_metadata: object | None = None
    source_metadata: object | None = None


@dataclass(frozen=True, slots=True)
class TieredSampleHandle:
    """Logical handle for a tiered manifest item backed by a passed sample."""

    ref: SampleRef
    tier: TierName
    membership: TierMembership
    manifest_entry: PassedManifestEntry
    runtime_sample: SamplePathRef
    drive_archive: ArchivePathRef
    drive_archive_member: ArchiveMemberPathRef


@dataclass(frozen=True, slots=True)
class SamplesCatalog:
    """Logical catalog for one sample status surface."""

    status: SampleStatus
    items: dict[SampleRef, SampleHandle]


@dataclass(frozen=True, slots=True)
class TieredCatalog:
    """Logical catalog for one tier and membership surface."""

    tier: TierName
    membership: TierMembership
    items: dict[SampleRef, TieredSampleHandle]


__all__ = [
    "SampleHandle",
    "SampleRef",
    "SamplesCatalog",
    "TieredCatalog",
    "TieredSampleHandle",
]
