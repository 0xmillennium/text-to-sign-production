"""Manifest validation helpers for raw, normalized, and processed datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from .constants import PROCESSED_SCHEMA_VERSION, SPLITS
from .jsonl import iter_jsonl
from .schemas import (
    NormalizedManifestEntry,
    ProcessedManifestEntry,
    RawManifestEntry,
    ValidationIssue,
)
from .utils import resolve_repo_path

RAW_REQUIRED_FIELDS = frozenset(
    {
        "sample_id",
        "source_split",
        "video_id",
        "video_name",
        "sentence_id",
        "sentence_name",
        "text",
        "start_time",
        "end_time",
        "keypoints_dir",
        "source_metadata_path",
        "has_face",
        "num_frames",
    }
)
PROCESSED_REQUIRED_FIELDS = frozenset(
    field.name for field in dataclass_fields(ProcessedManifestEntry)
)


def _sample_id_from_record(record: dict[str, Any]) -> str | None:
    sample_id = str(record.get("sample_id", "")).strip()
    return sample_id or None


def validate_manifest(path: Path, kind: str) -> list[ValidationIssue]:
    """Validate a manifest and return structural errors plus auditable warnings."""

    if kind == "raw":
        return validate_raw_records(path, iter_jsonl(path))
    if kind == "normalized":
        return validate_normalized_records(path, iter_jsonl(path))
    if kind == "processed":
        return validate_processed_records(path, iter_jsonl(path))
    raise ValueError(f"Unsupported manifest kind: {kind}")


def _missing_required_fields(record: dict[str, Any], required_fields: Iterable[str]) -> list[str]:
    return sorted(set(required_fields).difference(record.keys()))


def validate_raw_records(path: Path, records: Iterable[dict[str, Any]]) -> list[ValidationIssue]:
    """Validate raw-manifest structure plus expected data-quality warnings."""

    issues: list[ValidationIssue] = []
    seen_sample_ids: set[str] = set()
    for index, record in enumerate(records):
        sample_id = _sample_id_from_record(record)
        missing = _missing_required_fields(record, RAW_REQUIRED_FIELDS)
        if missing:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_required_fields",
                    message=f"Record {index} is missing required fields: {missing}",
                    sample_id=sample_id,
                    path=path.as_posix(),
                )
            )

        if sample_id is not None and sample_id in seen_sample_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_sample_id",
                    message=f"Duplicate raw sample_id detected: {sample_id}",
                    sample_id=sample_id,
                    split=str(record.get("source_split")) if record.get("source_split") else None,
                    path=path.as_posix(),
                )
            )
        elif sample_id is not None:
            seen_sample_ids.add(sample_id)

        try:
            raw_entry = RawManifestEntry.from_record(record)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="raw_record_parse_error",
                    message=f"Could not parse raw record: {exc}",
                    sample_id=sample_id or None,
                    path=path.as_posix(),
                )
            )
            continue

        if raw_entry.source_split not in SPLITS:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_split",
                    message=f"Unexpected raw split: {raw_entry.source_split}",
                    sample_id=raw_entry.sample_id,
                    split=raw_entry.source_split,
                    path=path.as_posix(),
                )
            )
        if raw_entry.end_time <= raw_entry.start_time:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="invalid_time_range",
                    message="Raw sample has a non-positive time range.",
                    sample_id=raw_entry.sample_id,
                    split=raw_entry.source_split,
                    path=path.as_posix(),
                )
            )
        if raw_entry.keypoints_dir is None:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_keypoints_dir",
                    message="Raw sample has no matched keypoints directory.",
                    sample_id=raw_entry.sample_id,
                    split=raw_entry.source_split,
                    path=path.as_posix(),
                )
            )
        elif raw_entry.num_frames <= 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_frame_json_files",
                    message="Matched raw sample has no frame JSON files.",
                    sample_id=raw_entry.sample_id,
                    split=raw_entry.source_split,
                    path=path.as_posix(),
                )
            )

    return issues


def validate_normalized_records(
    path: Path, records: Iterable[dict[str, Any]]
) -> list[ValidationIssue]:
    """Validate normalized candidate manifests."""

    issues: list[ValidationIssue] = []
    seen_sample_ids: set[str] = set()
    for record in records:
        sample_id = _sample_id_from_record(record)
        if sample_id is not None and sample_id in seen_sample_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_sample_id",
                    message=f"Duplicate normalized sample_id detected: {sample_id}",
                    sample_id=sample_id,
                    path=path.as_posix(),
                )
            )
        elif sample_id is not None:
            seen_sample_ids.add(sample_id)

        try:
            entry = NormalizedManifestEntry.from_record(record)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="normalized_record_parse_error",
                    message=f"Could not parse normalized record: {exc}",
                    sample_id=sample_id or None,
                    path=path.as_posix(),
                )
            )
            continue
        if entry.processed_schema_version != PROCESSED_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unexpected_processed_schema_version",
                    message=f"Normalized entry uses schema {entry.processed_schema_version}.",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
        if (
            entry.sample_path is None
            and entry.source_keypoints_dir is not None
            and entry.sample_parse_error is None
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_sample_file",
                    message=(
                        "Normalized entry is missing a sample file despite having keypoints input."
                    ),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
    return issues


def validate_processed_records(
    path: Path, records: Iterable[dict[str, Any]]
) -> list[ValidationIssue]:
    """Validate final processed manifests."""

    issues: list[ValidationIssue] = []
    seen_sample_ids: set[str] = set()
    for index, record in enumerate(records):
        sample_id = _sample_id_from_record(record)
        missing = _missing_required_fields(record, PROCESSED_REQUIRED_FIELDS)
        if missing:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_required_fields",
                    message=f"Record {index} is missing required fields: {missing}",
                    sample_id=sample_id,
                    path=path.as_posix(),
                )
            )

        if sample_id is not None and sample_id in seen_sample_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="duplicate_sample_id",
                    message=f"Duplicate processed sample_id detected: {sample_id}",
                    sample_id=sample_id,
                    path=path.as_posix(),
                )
            )
        elif sample_id is not None:
            seen_sample_ids.add(sample_id)

        try:
            entry = ProcessedManifestEntry.from_record(record)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="processed_record_parse_error",
                    message=f"Could not parse processed record: {exc}",
                    sample_id=sample_id or None,
                    path=path.as_posix(),
                )
            )
            continue

        if entry.processed_schema_version != PROCESSED_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unexpected_processed_schema_version",
                    message=f"Processed entry uses schema {entry.processed_schema_version}.",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
        if entry.split not in SPLITS:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_split",
                    message=f"Unexpected processed split: {entry.split}",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
        if entry.selected_person_index != 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unexpected_selected_person_index",
                    message="Processed entry selected a non-zero person index.",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )

        raw_sample_path = Path(entry.sample_path)
        if raw_sample_path.is_absolute():
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="absolute_sample_path",
                    message=f"Processed sample_path must be repo-relative: {entry.sample_path}",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
            continue

        try:
            sample_path = resolve_repo_path(entry.sample_path)
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_sample_path",
                    message=str(exc),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
            continue
        if not sample_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_sample_file",
                    message=f"Processed sample file does not exist: {entry.sample_path}",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
    return issues
