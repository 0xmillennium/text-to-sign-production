"""Manifest row construction and JSONL IO for dataset-owned surfaces.

Passed and dropped manifests are separate dataset surfaces. Row status is
derived by the reader from the surface or row shape; it is not serialized as a
manifest field.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np

from text_to_sign_production.core.ids import SampleSplit, SampleStatus
from text_to_sign_production.core.models import (
    DroppedManifestEntry,
    GateDecisionBundle,
    GateDropIssueCode,
    GateDropStage,
    PassedManifestEntry,
    PreparedSample,
)
from text_to_sign_production.data.dataset.validate import validate_manifest_entry

ManifestEntry: TypeAlias = PassedManifestEntry | DroppedManifestEntry
GATE_MANIFEST_SCHEMA_VERSION = "gate.manifest.v3"

_PASSED_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "split",
        "payload_ref",
        "text",
        "fps",
        "frame_count",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "valid_frame_count",
        "body_nonzero_frame_count",
        "face_nonzero_frame_count",
        "left_hand_nonzero_frame_count",
        "right_hand_nonzero_frame_count",
    }
)
_DROPPED_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "split",
        "text",
        "source_video_id",
        "source_sentence_id",
        "source_sentence_name",
        "drop_stage",
        "issue_codes",
        "debug_ref",
    }
)
_PROJECTION_ONLY_KEYS = frozenset(
    {"status", "projected_status", "payload_declared_present", "archive_publishable"}
)


def build_passed_entry(
    sample: PreparedSample,
    gate: GateDecisionBundle,
    payload_ref: str,
    *,
    schema_version: str,
) -> PassedManifestEntry:
    """Build a passed manifest row from prepared-sample handoff truth."""
    _schema_version(schema_version)
    if gate.final_status is not SampleStatus.PASSED:
        raise ValueError("Passed manifest entries require a passed gate bundle.")
    entry = PassedManifestEntry(
        schema_version=schema_version,
        sample_id=sample.source.sample_id,
        split=sample.source.split,
        payload_ref=payload_ref,
        text=sample.source.text,
        fps=sample.source.fps,
        frame_count=sample.pose.frame_count,
        source_video_id=sample.source.source_video_id,
        source_sentence_id=sample.source.source_sentence_id,
        source_sentence_name=sample.source.source_sentence_name,
        valid_frame_count=int(np.count_nonzero(sample.pose.valid_frame_mask)),
        body_nonzero_frame_count=sample.pose.body_nonzero_frame_count,
        face_nonzero_frame_count=sample.pose.face_nonzero_frame_count,
        left_hand_nonzero_frame_count=sample.pose.left_hand_nonzero_frame_count,
        right_hand_nonzero_frame_count=sample.pose.right_hand_nonzero_frame_count,
    )
    issues = validate_manifest_entry(entry)
    if issues:
        raise ValueError(f"Invalid passed manifest entry: {issues}")
    return entry


def build_dropped_entry(
    *,
    schema_version: str,
    sample_id: str,
    split: SampleSplit | str,
    drop_stage: GateDropStage | str,
    issue_codes: Iterable[GateDropIssueCode | str],
    text: str | None = None,
    source_video_id: str | None = None,
    source_sentence_id: str | None = None,
    source_sentence_name: str | None = None,
    debug_ref: str | None = None,
) -> DroppedManifestEntry:
    """Build a dropped manifest row from explicit drop facts."""
    _schema_version(schema_version)
    entry = DroppedManifestEntry(
        schema_version=schema_version,
        sample_id=sample_id,
        split=SampleSplit(split),
        text=text,
        source_video_id=source_video_id,
        source_sentence_id=source_sentence_id,
        source_sentence_name=source_sentence_name,
        drop_stage=GateDropStage(drop_stage),
        issue_codes=tuple(GateDropIssueCode(code) for code in issue_codes),
        debug_ref=debug_ref,
    )
    issues = validate_manifest_entry(entry)
    if issues:
        raise ValueError(f"Invalid dropped manifest entry: {issues}")
    return entry


def write_passed_manifest_jsonl(
    path: str | Path,
    entries: Iterable[PassedManifestEntry],
) -> None:
    """Write passed manifest entries as deterministic JSONL records."""
    _write_jsonl(path, entries)


def read_passed_manifest_jsonl(path: str | Path) -> tuple[PassedManifestEntry, ...]:
    """Read passed manifest entries from deterministic JSONL records."""
    return tuple(passed_entry_from_record(record) for record in _read_jsonl(path))


def write_dropped_manifest_jsonl(
    path: str | Path,
    entries: Iterable[DroppedManifestEntry],
) -> None:
    """Write dropped manifest entries as deterministic JSONL records."""
    _write_jsonl(path, entries)


def read_dropped_manifest_jsonl(path: str | Path) -> tuple[DroppedManifestEntry, ...]:
    """Read dropped manifest entries from deterministic JSONL records."""
    return tuple(dropped_entry_from_record(record) for record in _read_jsonl(path))


def write_manifest_jsonl(path: str | Path, entries: Iterable[ManifestEntry]) -> None:
    """Write mixed manifest entries as deterministic JSONL records."""
    _write_jsonl(path, entries)


def read_manifest_jsonl(path: str | Path) -> tuple[ManifestEntry, ...]:
    """Read mixed manifest entries by inferring row kind from row shape."""
    entries: list[ManifestEntry] = []
    for record in _read_jsonl(path):
        if _looks_like_passed_record(record):
            entries.append(passed_entry_from_record(record))
        elif _looks_like_dropped_record(record):
            entries.append(dropped_entry_from_record(record))
        else:
            raise ValueError("Manifest record shape is neither passed nor dropped.")
    return tuple(entries)


def _write_jsonl(path: str | Path, entries: Iterable[ManifestEntry]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            issues = validate_manifest_entry(entry)
            if issues:
                raise ValueError(f"Invalid manifest entry: {issues}")
            handle.write(json.dumps(_entry_to_record(entry), sort_keys=True) + "\n")


def _read_jsonl(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    manifest_path = Path(path)
    records: list[Mapping[str, Any]] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    loaded = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed manifest JSON in {manifest_path} at line {line_number}: "
                        f"{exc.msg}"
                    ) from exc
                records.append(_require_mapping(loaded, f"line {line_number}"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}") from exc
    return tuple(records)


def _entry_to_record(entry: ManifestEntry) -> dict[str, Any]:
    if isinstance(entry, PassedManifestEntry):
        return {
            "schema_version": entry.schema_version,
            "sample_id": entry.sample_id,
            "split": entry.split.value,
            "payload_ref": entry.payload_ref,
            "text": entry.text,
            "fps": entry.fps,
            "frame_count": entry.frame_count,
            "source_video_id": entry.source_video_id,
            "source_sentence_id": entry.source_sentence_id,
            "source_sentence_name": entry.source_sentence_name,
            "valid_frame_count": entry.valid_frame_count,
            "body_nonzero_frame_count": entry.body_nonzero_frame_count,
            "face_nonzero_frame_count": entry.face_nonzero_frame_count,
            "left_hand_nonzero_frame_count": entry.left_hand_nonzero_frame_count,
            "right_hand_nonzero_frame_count": entry.right_hand_nonzero_frame_count,
        }
    return {
        "schema_version": entry.schema_version,
        "sample_id": entry.sample_id,
        "split": entry.split.value,
        "text": entry.text,
        "source_video_id": entry.source_video_id,
        "source_sentence_id": entry.source_sentence_id,
        "source_sentence_name": entry.source_sentence_name,
        "drop_stage": entry.drop_stage.value,
        "issue_codes": [code.value for code in entry.issue_codes],
        "debug_ref": entry.debug_ref,
    }


def _looks_like_passed_record(record: Mapping[str, Any]) -> bool:
    return "payload_ref" in record and "drop_stage" not in record


def _looks_like_dropped_record(record: Mapping[str, Any]) -> bool:
    return "drop_stage" in record and "issue_codes" in record


def passed_entry_from_record(record: Mapping[str, Any]) -> PassedManifestEntry:
    """Parse one passed manifest record using the dataset-owned row contract."""
    _require_exact_record_keys(record, _PASSED_RECORD_KEYS, row_kind="passed")
    entry = PassedManifestEntry(
        schema_version=_schema_version(record["schema_version"]),
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        payload_ref=_text(record["payload_ref"], "payload_ref"),
        text=_text(record["text"], "text"),
        fps=_float(record["fps"], "fps"),
        frame_count=_int(record["frame_count"], "frame_count"),
        source_video_id=_text(record["source_video_id"], "source_video_id"),
        source_sentence_id=_text(record["source_sentence_id"], "source_sentence_id"),
        source_sentence_name=_text(record["source_sentence_name"], "source_sentence_name"),
        valid_frame_count=_int(record["valid_frame_count"], "valid_frame_count"),
        body_nonzero_frame_count=_int(
            record["body_nonzero_frame_count"],
            "body_nonzero_frame_count",
        ),
        face_nonzero_frame_count=_int(
            record["face_nonzero_frame_count"],
            "face_nonzero_frame_count",
        ),
        left_hand_nonzero_frame_count=_int(
            record["left_hand_nonzero_frame_count"],
            "left_hand_nonzero_frame_count",
        ),
        right_hand_nonzero_frame_count=_int(
            record["right_hand_nonzero_frame_count"],
            "right_hand_nonzero_frame_count",
        ),
    )
    issues = validate_manifest_entry(entry)
    if issues:
        raise ValueError(f"Invalid passed manifest entry: {issues}")
    return entry


def dropped_entry_from_record(record: Mapping[str, Any]) -> DroppedManifestEntry:
    """Parse one dropped manifest record using the dataset-owned row contract."""
    _require_exact_record_keys(record, _DROPPED_RECORD_KEYS, row_kind="dropped")
    entry = DroppedManifestEntry(
        schema_version=_schema_version(record["schema_version"]),
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        text=_optional_text(record.get("text"), "text"),
        source_video_id=_optional_text(record.get("source_video_id"), "source_video_id"),
        source_sentence_id=_optional_text(
            record.get("source_sentence_id"),
            "source_sentence_id",
        ),
        source_sentence_name=_optional_text(
            record.get("source_sentence_name"),
            "source_sentence_name",
        ),
        drop_stage=GateDropStage(_text(record["drop_stage"], "drop_stage")),
        issue_codes=tuple(
            GateDropIssueCode(_text(value, "issue_codes"))
            for value in _list(record["issue_codes"], "issue_codes")
        ),
        debug_ref=_optional_text(record.get("debug_ref"), "debug_ref"),
    )
    issues = validate_manifest_entry(entry)
    if issues:
        raise ValueError(f"Invalid dropped manifest entry: {issues}")
    return entry


def _require_exact_record_keys(
    record: Mapping[str, Any],
    expected_keys: frozenset[str],
    *,
    row_kind: str,
) -> None:
    projection_keys = sorted(set(record).intersection(_PROJECTION_ONLY_KEYS))
    if projection_keys:
        raise ValueError(
            "Manifest row projection metadata is derived by readers, not serialized: "
            f"{projection_keys}"
        )
    observed_keys = frozenset(record)
    missing = sorted(expected_keys.difference(observed_keys))
    extra = sorted(observed_keys.difference(expected_keys))
    if missing or extra:
        raise ValueError(
            f"{row_kind} manifest row keys do not match the serialized contract: "
            f"missing={missing}, extra={extra}"
        )


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings.")
    return cast(Mapping[str, Any], value)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    return value


def _schema_version(value: object) -> str:
    schema_version = _text(value, "schema_version")
    if schema_version != GATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Gate manifest schema_version is unsupported: "
            f"expected {GATE_MANIFEST_SCHEMA_VERSION!r}, observed {schema_version!r}."
        )
    return schema_version


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    return value


def _float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


__all__ = [
    "GATE_MANIFEST_SCHEMA_VERSION",
    "ManifestEntry",
    "build_dropped_entry",
    "build_passed_entry",
    "dropped_entry_from_record",
    "passed_entry_from_record",
    "read_dropped_manifest_jsonl",
    "read_manifest_jsonl",
    "read_passed_manifest_jsonl",
    "write_dropped_manifest_jsonl",
    "write_manifest_jsonl",
    "write_passed_manifest_jsonl",
]
