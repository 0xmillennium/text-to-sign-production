"""Logical catalog models for sample artifact lookup."""

from __future__ import annotations

from dataclasses import dataclass

from text_to_sign_production.artifacts.store.types import (
    ArchiveMemberPathRef,
    ArchivePathRef,
    SamplePathRef,
)
from text_to_sign_production.core.ids import (
    SampleSplit,
    SampleStatus,
    TierMembership,
    TierName,
)


@dataclass(frozen=True, slots=True)
class SampleRef:
    """Logical sample identity inside catalog surfaces."""

    split: SampleSplit
    sample_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(str(self.split)))


@dataclass(frozen=True, slots=True)
class SampleManifestProjection:
    """Artifact-owned projection of manifest facts needed for physical lookup."""

    sample_id: str
    split: SampleSplit
    status: SampleStatus
    sample_path: str | None
    payload_declared_present: bool
    archive_publishable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(str(self.split)))
        object.__setattr__(self, "status", SampleStatus(str(self.status)))


@dataclass(frozen=True, slots=True)
class SampleHandle:
    """Logical handle for a passed or dropped sample manifest item."""

    ref: SampleRef
    status: SampleStatus
    manifest: SampleManifestProjection
    runtime_sample: SamplePathRef | None
    drive_archive: ArchivePathRef | None
    drive_archive_member: ArchiveMemberPathRef | None
    timing_metadata: object | None = None
    source_metadata: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SampleStatus(str(self.status)))


@dataclass(frozen=True, slots=True)
class TieredSampleHandle:
    """Logical handle for a tiered manifest item backed by a passed sample."""

    ref: SampleRef
    tier: TierName
    membership: TierMembership
    manifest: SampleManifestProjection
    runtime_sample: SamplePathRef
    drive_archive: ArchivePathRef
    drive_archive_member: ArchiveMemberPathRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", TierName(str(self.tier)))
        object.__setattr__(self, "membership", TierMembership(str(self.membership)))


@dataclass(frozen=True, slots=True)
class SamplesCatalog:
    """Logical catalog for one sample status surface."""

    status: SampleStatus
    items: dict[SampleRef, SampleHandle]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SampleStatus(str(self.status)))


@dataclass(frozen=True, slots=True)
class TieredCatalog:
    """Logical catalog for one tier and membership surface."""

    tier: TierName
    membership: TierMembership
    items: dict[SampleRef, TieredSampleHandle]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", TierName(str(self.tier)))
        object.__setattr__(self, "membership", TierMembership(str(self.membership)))


__all__ = [
    "SampleHandle",
    "SampleManifestProjection",
    "SampleRef",
    "SamplesCatalog",
    "TieredCatalog",
    "TieredSampleHandle",
]
