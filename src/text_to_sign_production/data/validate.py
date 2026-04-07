"""Manifest validation helpers for raw, normalized, and processed datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from .constants import OPENPOSE_CHANNEL_SPECS, PROCESSED_SCHEMA_VERSION, SPLITS
from .schemas import (
    NormalizedManifestEntry,
    ProcessedManifestEntry,
    RawManifestEntry,
    ValidationIssue,
)
from .utils import validate_processed_sample_path

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
PROCESSED_SAMPLE_REQUIRED_KEYS = frozenset(
    {
        "processed_schema_version",
        "body",
        "body_confidence",
        "left_hand",
        "left_hand_confidence",
        "right_hand",
        "right_hand_confidence",
        "face",
        "face_confidence",
        "people_per_frame",
        "selected_person_index",
        "frame_valid_mask",
    }
)


def _expected_processed_sample_shapes(num_frames: int) -> dict[str, tuple[int, ...]]:
    return {
        "body": (num_frames, OPENPOSE_CHANNEL_SPECS["body"][1], 2),
        "body_confidence": (num_frames, OPENPOSE_CHANNEL_SPECS["body"][1]),
        "left_hand": (num_frames, OPENPOSE_CHANNEL_SPECS["left_hand"][1], 2),
        "left_hand_confidence": (num_frames, OPENPOSE_CHANNEL_SPECS["left_hand"][1]),
        "right_hand": (num_frames, OPENPOSE_CHANNEL_SPECS["right_hand"][1], 2),
        "right_hand_confidence": (num_frames, OPENPOSE_CHANNEL_SPECS["right_hand"][1]),
        "face": (num_frames, OPENPOSE_CHANNEL_SPECS["face"][1], 2),
        "face_confidence": (num_frames, OPENPOSE_CHANNEL_SPECS["face"][1]),
        "people_per_frame": (num_frames,),
        "frame_valid_mask": (num_frames,),
    }


def _validate_processed_sample_payload(
    entry: ProcessedManifestEntry,
    sample: Any,
    *,
    path: Path,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    processed_schema_version = sample["processed_schema_version"]
    if tuple(processed_schema_version.shape) != ():
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_sample_array_shape",
                message=(
                    "Processed sample array 'processed_schema_version' has shape "
                    f"{tuple(processed_schema_version.shape)}; expected ()."
                ),
                sample_id=entry.sample_id,
                split=entry.split,
                path=path.as_posix(),
            )
        )
    else:
        payload_schema_version = str(processed_schema_version.item())
        if payload_schema_version != PROCESSED_SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unexpected_sample_schema_version",
                    message=f"Processed sample payload uses schema {payload_schema_version}.",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )

    selected_person_index = sample["selected_person_index"]
    if tuple(selected_person_index.shape) != ():
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_sample_array_shape",
                message=(
                    "Processed sample array 'selected_person_index' has shape "
                    f"{tuple(selected_person_index.shape)}; expected ()."
                ),
                sample_id=entry.sample_id,
                split=entry.split,
                path=path.as_posix(),
            )
        )
    else:
        payload_selected_person_index = int(selected_person_index.item())
        if payload_selected_person_index != entry.selected_person_index:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unexpected_sample_selected_person_index",
                    message=(
                        "Processed sample payload selected_person_index does not match the "
                        f"manifest expectation: payload={payload_selected_person_index} "
                        f"manifest={entry.selected_person_index}."
                    ),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )

    for key, expected_shape in _expected_processed_sample_shapes(entry.num_frames).items():
        observed_shape = tuple(sample[key].shape)
        if observed_shape != expected_shape:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_sample_array_shape",
                    message=(
                        f"Processed sample array {key!r} has shape {observed_shape}; "
                        f"expected {expected_shape}."
                    ),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )

    frame_valid_mask = sample["frame_valid_mask"]
    if tuple(frame_valid_mask.shape) == (entry.num_frames,):
        frame_valid_count = int(np.count_nonzero(frame_valid_mask))
        if frame_valid_count != entry.frame_valid_count:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="frame_valid_count_mismatch",
                    message=(
                        "Processed sample frame_valid_mask count does not match the manifest: "
                        f"payload={frame_valid_count} manifest={entry.frame_valid_count}."
                    ),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
        frame_invalid_count = int(frame_valid_mask.shape[0] - frame_valid_count)
        if frame_invalid_count != entry.frame_invalid_count:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="frame_invalid_count_mismatch",
                    message=(
                        "Processed sample frame_valid_mask invalid count does not match the "
                        f"manifest: payload={frame_invalid_count} "
                        f"manifest={entry.frame_invalid_count}."
                    ),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )

    return issues


def _sample_id_from_record(record: dict[str, Any]) -> str | None:
    sample_id = str(record.get("sample_id", "")).strip()
    return sample_id or None


def _iter_validation_records(
    path: Path,
) -> Iterator[tuple[dict[str, Any] | None, ValidationIssue | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield (
                    None,
                    ValidationIssue(
                        severity="error",
                        code="invalid_jsonl_line",
                        message=f"Line {line_number} is not valid JSON: {exc.msg}",
                        path=path.as_posix(),
                    ),
                )
                continue

            if not isinstance(payload, dict):
                yield (
                    None,
                    ValidationIssue(
                        severity="error",
                        code="non_object_record",
                        message=(
                            f"Line {line_number} must contain a JSON object, "
                            f"got {type(payload).__name__}."
                        ),
                        path=path.as_posix(),
                    ),
                )
                continue

            yield payload, None


def validate_manifest(path: Path, kind: str) -> list[ValidationIssue]:
    """Validate a manifest and return structural errors plus auditable warnings."""

    issues: list[ValidationIssue] = []

    def _records() -> Iterator[dict[str, Any]]:
        for record, issue in _iter_validation_records(path):
            if issue is not None:
                issues.append(issue)
                continue
            assert record is not None
            yield record

    if kind == "raw":
        issues.extend(validate_raw_records(path, _records()))
        return issues
    if kind == "normalized":
        issues.extend(validate_normalized_records(path, _records()))
        return issues
    if kind == "processed":
        issues.extend(validate_processed_records(path, _records()))
        return issues
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

        sample_path_value = str(entry.sample_path)
        if Path(sample_path_value.strip()).is_absolute():
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="absolute_sample_path",
                    message=(
                        f"Processed sample_path must be repo-relative: {sample_path_value.strip()}"
                    ),
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
            continue

        try:
            sample_path = validate_processed_sample_path(
                sample_path_value,
                split=entry.split,
                sample_id=entry.sample_id,
            )
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
            continue

        try:
            with np.load(sample_path, allow_pickle=False) as sample:
                missing_keys = sorted(PROCESSED_SAMPLE_REQUIRED_KEYS.difference(sample.files))
                if missing_keys:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="missing_sample_arrays",
                            message=(
                                f"Processed sample file is missing required arrays: {missing_keys}"
                            ),
                            sample_id=entry.sample_id,
                            split=entry.split,
                            path=path.as_posix(),
                        )
                    )
                    continue

                issues.extend(_validate_processed_sample_payload(entry, sample, path=path))
        except (BadZipFile, EOFError, OSError, ValueError) as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_sample_file",
                    message=f"Processed sample file could not be read as .npz: {exc}",
                    sample_id=entry.sample_id,
                    split=entry.split,
                    path=path.as_posix(),
                )
            )
            continue
    return issues
