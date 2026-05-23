"""Pairing-key contracts for reference and generated pose surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.core.models import PassedManifestEntry
from text_to_sign_production.modeling.data.errors import ModelingDataError

if TYPE_CHECKING:
    from text_to_sign_production.modeling.artifacts.generated_pose_manifest import (
        GeneratedPoseManifestEntry,
    )


@dataclass(frozen=True, slots=True, order=True)
class ReferencePairingKey:
    """Unique key for one reference prepared sample."""

    split: SampleSplit
    sample_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        _require_text(self.sample_id, "sample_id")


@dataclass(frozen=True, slots=True, order=True)
class GeneratedPairingKey:
    """Unique key for one generated candidate sample."""

    split: SampleSplit
    sample_id: str
    generation_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        _require_text(self.sample_id, "sample_id")
        if self.generation_index < 0:
            raise ModelingDataError("generation_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Source sentence identity shared by reference and generated entries."""

    source_video_id: str
    source_sentence_id: str
    source_sentence_name: str

    def __post_init__(self) -> None:
        _require_text(self.source_video_id, "source_video_id")
        _require_text(self.source_sentence_id, "source_sentence_id")
        _require_text(self.source_sentence_name, "source_sentence_name")


def reference_pairing_key(entry: PassedManifestEntry) -> ReferencePairingKey:
    """Return the reference pairing key for a passed manifest entry."""

    return ReferencePairingKey(split=entry.split, sample_id=entry.sample_id)


def generated_pairing_key(entry: GeneratedPoseManifestEntry) -> GeneratedPairingKey:
    """Return the generated pairing key for a generated manifest entry."""

    return GeneratedPairingKey(
        split=entry.split,
        sample_id=entry.sample_id,
        generation_index=entry.generation_index,
    )


def source_identity_from_reference(entry: PassedManifestEntry) -> SourceIdentity:
    """Return source identity from a reference manifest entry."""

    return SourceIdentity(
        source_video_id=entry.source_video_id,
        source_sentence_id=entry.source_sentence_id,
        source_sentence_name=entry.source_sentence_name,
    )


def source_identity_from_generated(entry: GeneratedPoseManifestEntry) -> SourceIdentity:
    """Return source identity from a generated manifest entry."""

    return SourceIdentity(
        source_video_id=entry.source_video_id,
        source_sentence_id=entry.source_sentence_id,
        source_sentence_name=entry.source_sentence_name,
    )


def build_reference_index(
    entries: Iterable[PassedManifestEntry],
) -> Mapping[ReferencePairingKey, PassedManifestEntry]:
    """Build a duplicate-checked reference entry index."""

    index: dict[ReferencePairingKey, PassedManifestEntry] = {}
    for entry in entries:
        key = reference_pairing_key(entry)
        if key in index:
            raise ModelingDataError(f"duplicate reference pairing key: {key}")
        index[key] = entry
    return MappingProxyType(index)


def build_generated_index(
    entries: Iterable[GeneratedPoseManifestEntry],
) -> Mapping[GeneratedPairingKey, GeneratedPoseManifestEntry]:
    """Build a duplicate-checked generated entry index."""

    index: dict[GeneratedPairingKey, Any] = {}
    for entry in entries:
        key = generated_pairing_key(entry)
        if key in index:
            raise ModelingDataError(f"duplicate generated pairing key: {key}")
        index[key] = entry
    return MappingProxyType(index)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelingDataError(f"{field_name} must be non-empty.")


__all__ = [
    "GeneratedPairingKey",
    "ReferencePairingKey",
    "SourceIdentity",
    "build_generated_index",
    "build_reference_index",
    "generated_pairing_key",
    "reference_pairing_key",
    "source_identity_from_generated",
    "source_identity_from_reference",
]
