"""Artifact-owned parsing for minimal manifest projections."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from text_to_sign_production.artifacts.catalog.types import SampleManifestProjection
from text_to_sign_production.core.ids import SampleSplit, SampleStatus


def passed_manifest_projection_from_record(
    record: Mapping[str, Any],
    path: Path,
) -> SampleManifestProjection:
    """Parse artifact-relevant fields from a passed manifest record.

    The passed status is derived from the manifest surface. The row contract is
    the root ``PassedManifestEntry`` shape and does not need to carry status.
    """
    payload_ref = _required_text(record, "payload_ref", path)
    return SampleManifestProjection(
        sample_id=_required_text(record, "sample_id", path),
        split=_split_from_record(record, path),
        projected_status=SampleStatus.PASSED,
        payload_ref=payload_ref,
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
    return SampleManifestProjection(
        sample_id=_required_text(record, "sample_id", path),
        split=_split_from_record(record, path),
        projected_status=SampleStatus.DROPPED,
        payload_ref=_optional_text(record, "debug_ref", path),
        payload_declared_present=record.get("debug_ref") is not None,
        archive_publishable=record.get("debug_ref") is not None,
    )


def tiered_manifest_projection_from_record(
    record: Mapping[str, Any],
    path: Path,
) -> SampleManifestProjection:
    """Parse a tiered manifest row as passed-row semantics plus tier context."""
    return passed_manifest_projection_from_record(record, path)


def _split_from_record(record: Mapping[str, Any], path: Path) -> SampleSplit:
    value = _required_text(record, "split", path)
    try:
        return SampleSplit(value)
    except ValueError as exc:
        raise ValueError(f"Manifest {path} has invalid split {value!r}.") from exc


def _required_text(record: Mapping[str, Any], key: str, path: Path) -> str:
    if key not in record:
        raise KeyError(f"Manifest {path} record is missing {key!r}.")
    value = record[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"Manifest {path} field {key!r} must be a non-empty string.")
    return value


def _optional_text(record: Mapping[str, Any], key: str, path: Path) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Manifest {path} field {key!r} must be a non-empty string or null.")
    return value


__all__ = [
    "dropped_manifest_projection_from_record",
    "passed_manifest_projection_from_record",
    "tiered_manifest_projection_from_record",
]
