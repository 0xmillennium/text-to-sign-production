"""Artifact-owned projections for minimal manifest catalog facts."""

from __future__ import annotations

from text_to_sign_production.artifacts.catalog.types import SampleManifestProjection
from text_to_sign_production.core.ids import SampleStatus
from text_to_sign_production.core.models import DroppedManifestEntry, PassedManifestEntry


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


def dropped_manifest_projection_from_entry(
    entry: DroppedManifestEntry,
) -> SampleManifestProjection:
    """Project artifact lookup facts from a dataset-owned dropped manifest entry."""
    return SampleManifestProjection(
        sample_id=entry.sample_id,
        split=entry.split,
        projected_status=SampleStatus.DROPPED,
        payload_ref=entry.dropped_sample_ref,
        payload_declared_present=True,
        archive_publishable=True,
    )


__all__ = [
    "dropped_manifest_projection_from_entry",
    "passed_manifest_projection_from_entry",
]
