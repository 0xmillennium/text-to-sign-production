"""Operational catalog models for physical prepared-sample artifact lookup."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

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

CatalogMetadataValue: TypeAlias = str | int | float | bool | None
CatalogMetadata: TypeAlias = Mapping[str, CatalogMetadataValue]
"""Flat operational metadata attached to catalog handles, not semantic sample truth."""


@dataclass(frozen=True, slots=True)
class SampleRef:
    """Logical identity key used to find a physical prepared-sample artifact."""

    split: SampleSplit
    sample_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(str(self.split)))


@dataclass(frozen=True, slots=True)
class SampleManifestProjection:
    """Artifact-owned projection of manifest facts needed for physical lookup.

    ``projected_status`` is derived by the projection loader from the manifest
    surface being read. It is not a required field in the manifest row contract,
    and this projection is not a semantic replacement for core manifest models.
    """

    sample_id: str
    split: SampleSplit
    projected_status: SampleStatus
    payload_ref: str | None
    payload_declared_present: bool
    archive_publishable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(str(self.split)))
        object.__setattr__(
            self,
            "projected_status",
            SampleStatus(str(self.projected_status)),
        )


@dataclass(frozen=True, slots=True)
class SampleHandle:
    """Logical handle for one physical prepared-sample artifact catalog row."""

    ref: SampleRef
    status: SampleStatus
    manifest: SampleManifestProjection
    runtime_sample: SamplePathRef | None
    drive_archive: ArchivePathRef | None
    drive_archive_member: ArchiveMemberPathRef | None
    timing_metadata: CatalogMetadata | None = None
    source_metadata: CatalogMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SampleStatus(str(self.status)))


@dataclass(frozen=True, slots=True)
class TieredSampleHandle:
    """Logical handle for a tiered physical artifact backed by a passed row projection."""

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
    """Logical catalog for one physical sample artifact status surface.

    ``Samples`` is artifact terminology here: the catalog indexes prepared
    sample payload files. It is not a gate-stage ownership surface.
    """

    status: SampleStatus
    items: dict[SampleRef, SampleHandle]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SampleStatus(str(self.status)))


@dataclass(frozen=True, slots=True)
class TieredCatalog:
    """Logical catalog for one physical tier/membership projection surface."""

    tier: TierName
    membership: TierMembership
    items: dict[SampleRef, TieredSampleHandle]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", TierName(str(self.tier)))
        object.__setattr__(self, "membership", TierMembership(str(self.membership)))


__all__ = [
    "CatalogMetadata",
    "CatalogMetadataValue",
    "SampleHandle",
    "SampleManifestProjection",
    "SampleRef",
    "SamplesCatalog",
    "TieredCatalog",
    "TieredSampleHandle",
]
