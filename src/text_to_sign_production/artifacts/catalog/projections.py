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
    """Parse artifact-relevant fields from a passed manifest record."""
    status = _status_from_record(record, path)
    if status is not SampleStatus.PASSED:
        raise TypeError(f"Expected passed manifest record in {path}, got {status.value!r}.")

    sample_path = _required_text(record, "sample_path", path)
    return SampleManifestProjection(
        sample_id=_required_text(record, "sample_id", path),
        split=_split_from_record(record, path),
        status=SampleStatus.PASSED,
        sample_path=sample_path,
        payload_declared_present=True,
        archive_publishable=True,
    )


def dropped_manifest_projection_from_record(
    record: Mapping[str, Any],
    path: Path,
) -> SampleManifestProjection:
    """Parse artifact-relevant fields from a dropped manifest record."""
    status = _status_from_record(record, path)
    if status is not SampleStatus.DROPPED:
        raise TypeError(f"Expected dropped manifest record in {path}, got {status.value!r}.")

    materialization = _required_mapping(record, "materialization", path)
    return SampleManifestProjection(
        sample_id=_required_text(record, "sample_id", path),
        split=_split_from_record(record, path),
        status=SampleStatus.DROPPED,
        sample_path=_optional_text(materialization, "payload_path", path),
        payload_declared_present=_required_bool(materialization, "payload_exists", path),
        archive_publishable=_required_bool(materialization, "archive_publishable", path),
    )


def _status_from_record(record: Mapping[str, Any], path: Path) -> SampleStatus:
    value = _required_text(record, "status", path)
    try:
        return SampleStatus(value)
    except ValueError as exc:
        raise ValueError(f"Manifest {path} has invalid status {value!r}.") from exc


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


def _required_bool(record: Mapping[str, Any], key: str, path: Path) -> bool:
    if key not in record:
        raise KeyError(f"Manifest {path} record is missing {key!r}.")
    value = record[key]
    if not isinstance(value, bool):
        raise TypeError(f"Manifest {path} field {key!r} must be a boolean.")
    return value


def _required_mapping(record: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    if key not in record:
        raise KeyError(f"Manifest {path} record is missing {key!r}.")
    value = record[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"Manifest {path} field {key!r} must be an object.")
    if any(not isinstance(item_key, str) for item_key in value):
        raise TypeError(f"Manifest {path} field {key!r} must contain string keys.")
    return value


__all__ = [
    "dropped_manifest_projection_from_record",
    "passed_manifest_projection_from_record",
]
