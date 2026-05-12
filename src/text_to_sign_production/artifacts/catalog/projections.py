"""Artifact-owned parsing for minimal manifest projections."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from text_to_sign_production.artifacts.catalog.types import SampleManifestProjection
from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import DroppedManifestEntry, PassedManifestEntry
from text_to_sign_production.data.dataset.manifests import (
    dropped_entry_from_record,
    passed_entry_from_record,
)


def passed_manifest_projection_from_record(
    record: Mapping[str, Any],
    path: Path,
) -> SampleManifestProjection:
    """Parse artifact-relevant fields from a passed manifest record.

    The passed status is derived from the manifest surface. The row contract is
    the root ``PassedManifestEntry`` shape and does not need to carry status.
    """
    entry = passed_entry_from_record(record)
    return passed_manifest_projection_from_entry(entry)


def passed_manifest_projection_from_entry(
    entry: PassedManifestEntry,
) -> SampleManifestProjection:
    """Project artifact lookup facts from a dataset-owned passed manifest entry."""
    return SampleManifestProjection(
        sample_id=entry.sample_id,
        split=entry.split,
        projected_status=SampleStatus.PASSED,
        payload_ref=entry.payload_ref,
        payload_declared_present=True,
        archive_publishable=True,
    )


def dropped_manifest_projection_from_record(
    record: Mapping[str, Any],
    path: Path,
) -> SampleManifestProjection:
    """Parse artifact-relevant fields from a dropped manifest record.

    The dropped status is derived from the manifest surface. It is projection
    metadata only, not manifest row authority.
    """
    entry = dropped_entry_from_record(record)
    return dropped_manifest_projection_from_entry(entry)


def dropped_manifest_projection_from_entry(
    entry: DroppedManifestEntry,
) -> SampleManifestProjection:
    """Project artifact lookup facts from a dataset-owned dropped manifest entry."""
    return SampleManifestProjection(
        sample_id=entry.sample_id,
        split=entry.split,
        projected_status=SampleStatus.DROPPED,
        payload_ref=entry.debug_ref,
        payload_declared_present=entry.debug_ref is not None,
        archive_publishable=entry.debug_ref is not None,
    )


def tiered_manifest_projection_from_record(
    record: Mapping[str, Any],
    path: Path,
) -> SampleManifestProjection:
    """Parse a tiered manifest row as passed-row semantics plus tier context."""
    return passed_manifest_projection_from_record(record, path)


__all__ = [
    "dropped_manifest_projection_from_record",
    "dropped_manifest_projection_from_entry",
    "passed_manifest_projection_from_entry",
    "passed_manifest_projection_from_record",
    "tiered_manifest_projection_from_record",
]
