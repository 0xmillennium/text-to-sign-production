"""Manifest document construction and JSON IO for dataset-owned surfaces."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np

from text_to_sign_production.core.ids import SampleSplit, SampleStatus, TierMembership, TierName
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
GATE_MANIFEST_SCHEMA_VERSION = "gate.manifest.v4"
TIER_MANIFEST_SCHEMA_VERSION = "tier.manifest.v1"

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
        "dropped_sample_ref",
    }
)
_PASSED_DOCUMENT_KEYS = frozenset(
    {"schema_version", "manifest_kind", "split", "entry_count", "entries"}
)
_DROPPED_DOCUMENT_KEYS = _PASSED_DOCUMENT_KEYS
_TIER_DOCUMENT_KEYS = frozenset(
    {"schema_version", "manifest_kind", "tier", "membership", "split", "entry_count", "entries"}
)
_SAMPLE_DROP_STAGES = frozenset({GateDropStage.SOURCE, GateDropStage.POSE, GateDropStage.GATES})


def build_passed_entry(
    sample: PreparedSample,
    gate: GateDecisionBundle,
    payload_ref: str,
    *,
    schema_version: str,
) -> PassedManifestEntry:
    """Build a passed manifest row from prepared-sample handoff truth."""
    _require_gate_schema_version(schema_version)
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
    _raise_if_invalid(entry, "passed")
    return entry


def build_dropped_entry(
    *,
    schema_version: str,
    sample_id: str,
    split: SampleSplit | str,
    drop_stage: GateDropStage | str,
    issue_codes: Iterable[GateDropIssueCode | str],
    dropped_sample_ref: str,
    text: str | None = None,
    source_video_id: str | None = None,
    source_sentence_id: str | None = None,
    source_sentence_name: str | None = None,
) -> DroppedManifestEntry:
    """Build a dropped manifest row from explicit dropped-sample payload facts."""
    _require_gate_schema_version(schema_version)
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
        dropped_sample_ref=dropped_sample_ref,
    )
    _raise_if_invalid(entry, "dropped")
    return entry


def write_passed_manifest_json(
    path: str | Path,
    entries: Iterable[PassedManifestEntry],
    *,
    split: SampleSplit | str,
) -> None:
    """Write a passed manifest JSON document."""
    split = SampleSplit(split)
    materialized = tuple(entries)
    document = _manifest_document(
        schema_version=GATE_MANIFEST_SCHEMA_VERSION,
        manifest_kind="passed",
        split=split,
        entries=tuple(_passed_entry_to_record(entry) for entry in materialized),
    )
    _validate_passed_document(materialized, split, GATE_MANIFEST_SCHEMA_VERSION)
    _write_json_document(path, document)


def read_passed_manifest_json(path: str | Path) -> tuple[PassedManifestEntry, ...]:
    """Read a passed manifest JSON document."""
    document = _read_document(path, _PASSED_DOCUMENT_KEYS, "passed manifest")
    if _text(document["manifest_kind"], "manifest_kind") != "passed":
        raise ValueError("Passed manifest document manifest_kind must be passed.")
    split = SampleSplit(_text(document["split"], "split"))
    entries = tuple(
        passed_entry_from_record(_mapping(record, "passed manifest entry"))
        for record in _list(document["entries"], "entries")
    )
    _validate_document_count(document, len(entries))
    _validate_passed_document(entries, split, _text(document["schema_version"], "schema_version"))
    return entries


def write_dropped_manifest_json(
    path: str | Path,
    entries: Iterable[DroppedManifestEntry],
    *,
    split: SampleSplit | str,
) -> None:
    """Write a dropped manifest JSON document."""
    split = SampleSplit(split)
    materialized = tuple(entries)
    document = _manifest_document(
        schema_version=GATE_MANIFEST_SCHEMA_VERSION,
        manifest_kind="dropped",
        split=split,
        entries=tuple(_dropped_entry_to_record(entry) for entry in materialized),
    )
    _validate_dropped_document(materialized, split, GATE_MANIFEST_SCHEMA_VERSION)
    _write_json_document(path, document)


def read_dropped_manifest_json(path: str | Path) -> tuple[DroppedManifestEntry, ...]:
    """Read a dropped manifest JSON document."""
    document = _read_document(path, _DROPPED_DOCUMENT_KEYS, "dropped manifest")
    if _text(document["manifest_kind"], "manifest_kind") != "dropped":
        raise ValueError("Dropped manifest document manifest_kind must be dropped.")
    split = SampleSplit(_text(document["split"], "split"))
    entries = tuple(
        dropped_entry_from_record(_mapping(record, "dropped manifest entry"))
        for record in _list(document["entries"], "entries")
    )
    _validate_document_count(document, len(entries))
    _validate_dropped_document(entries, split, _text(document["schema_version"], "schema_version"))
    return entries


def write_tier_manifest_json(
    path: str | Path,
    entries: Iterable[PassedManifestEntry],
    *,
    tier: TierName | str,
    membership: TierMembership | str,
    split: SampleSplit | str,
) -> None:
    """Write a tier manifest JSON document containing passed manifest entries."""
    tier = TierName(tier)
    membership = TierMembership(membership)
    split = SampleSplit(split)
    materialized = tuple(entries)
    document = {
        "schema_version": TIER_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "tiered",
        "tier": tier.value,
        "membership": membership.value,
        "split": split.value,
        "entry_count": len(materialized),
        "entries": tuple(_passed_entry_to_record(entry) for entry in materialized),
    }
    _validate_passed_document(materialized, split, GATE_MANIFEST_SCHEMA_VERSION)
    _write_json_document(path, document)


def read_tier_manifest_json(path: str | Path) -> tuple[PassedManifestEntry, ...]:
    """Read a tier manifest JSON document."""
    document = _read_document(path, _TIER_DOCUMENT_KEYS, "tier manifest")
    if _text(document["schema_version"], "schema_version") != TIER_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Tier manifest schema_version is unsupported.")
    if _text(document["manifest_kind"], "manifest_kind") != "tiered":
        raise ValueError("Tier manifest document manifest_kind must be tiered.")
    TierName(_text(document["tier"], "tier"))
    TierMembership(_text(document["membership"], "membership"))
    split = SampleSplit(_text(document["split"], "split"))
    entries = tuple(
        passed_entry_from_record(_mapping(record, "tier manifest entry"))
        for record in _list(document["entries"], "entries")
    )
    _validate_document_count(document, len(entries))
    _validate_passed_document(entries, split, GATE_MANIFEST_SCHEMA_VERSION)
    return entries


def passed_entry_from_record(record: Mapping[str, Any]) -> PassedManifestEntry:
    """Parse one passed manifest entry record."""
    _require_exact_record_keys(record, _PASSED_RECORD_KEYS, row_kind="passed")
    entry = PassedManifestEntry(
        schema_version=_require_gate_schema_version(record["schema_version"]),
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
            record["face_nonzero_frame_count"], "face_nonzero_frame_count"
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
    _raise_if_invalid(entry, "passed")
    return entry


def dropped_entry_from_record(record: Mapping[str, Any]) -> DroppedManifestEntry:
    """Parse one dropped manifest entry record."""
    _require_exact_record_keys(record, _DROPPED_RECORD_KEYS, row_kind="dropped")
    entry = DroppedManifestEntry(
        schema_version=_require_gate_schema_version(record["schema_version"]),
        sample_id=_text(record["sample_id"], "sample_id"),
        split=SampleSplit(_text(record["split"], "split")),
        text=_optional_text(record["text"], "text"),
        source_video_id=_optional_text(record["source_video_id"], "source_video_id"),
        source_sentence_id=_optional_text(record["source_sentence_id"], "source_sentence_id"),
        source_sentence_name=_optional_text(record["source_sentence_name"], "source_sentence_name"),
        drop_stage=GateDropStage(_text(record["drop_stage"], "drop_stage")),
        issue_codes=tuple(
            GateDropIssueCode(_text(value, "issue_codes"))
            for value in _list(record["issue_codes"], "issue_codes")
        ),
        dropped_sample_ref=_text(record["dropped_sample_ref"], "dropped_sample_ref"),
    )
    _raise_if_invalid(entry, "dropped")
    return entry


def _manifest_document(
    *,
    schema_version: str,
    manifest_kind: Literal["passed", "dropped"],
    split: SampleSplit,
    entries: tuple[Mapping[str, Any], ...],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "manifest_kind": manifest_kind,
        "split": split.value,
        "entry_count": len(entries),
        "entries": entries,
    }


def _passed_entry_to_record(entry: PassedManifestEntry) -> dict[str, Any]:
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


def _dropped_entry_to_record(entry: DroppedManifestEntry) -> dict[str, Any]:
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
        "dropped_sample_ref": entry.dropped_sample_ref,
    }


def _validate_passed_document(
    entries: tuple[PassedManifestEntry, ...],
    split: SampleSplit,
    schema_version: str,
) -> None:
    _require_gate_schema_version(schema_version)
    for entry in entries:
        if entry.split is not split:
            raise ValueError("Manifest entry split must match document split.")
        if entry.schema_version != schema_version:
            raise ValueError("Manifest entry schema_version must match document schema_version.")
        if not entry.payload_ref.endswith(".npz"):
            raise ValueError("Passed manifest payload_ref must point to a .npz payload.")


def _validate_dropped_document(
    entries: tuple[DroppedManifestEntry, ...],
    split: SampleSplit,
    schema_version: str,
) -> None:
    _require_gate_schema_version(schema_version)
    for entry in entries:
        if entry.split is not split:
            raise ValueError("Manifest entry split must match document split.")
        if entry.schema_version != schema_version:
            raise ValueError("Manifest entry schema_version must match document schema_version.")
        if entry.drop_stage not in _SAMPLE_DROP_STAGES:
            raise ValueError("Dropped manifest drop_stage must be source, pose, or gates.")
        if not entry.dropped_sample_ref.endswith(".json"):
            raise ValueError("Dropped manifest dropped_sample_ref must point to a .json payload.")


def _write_json_document(path: str | Path, payload: Mapping[str, object]) -> None:
    manifest_path = Path(path)
    if manifest_path.suffix != ".json":
        raise ValueError("Manifest path must end with .json.")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_document(
    path: str | Path,
    expected_keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".json":
        raise ValueError("Manifest path must end with .json.")
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed manifest JSON in {manifest_path}: {exc.msg}") from exc
    document = _mapping(loaded, label)
    _require_exact_record_keys(document, expected_keys, row_kind=label)
    return document


def _validate_document_count(document: Mapping[str, Any], actual_count: int) -> None:
    if _int(document["entry_count"], "entry_count") != actual_count:
        raise ValueError("Manifest entry_count must equal entries length.")


def _raise_if_invalid(entry: ManifestEntry, kind: str) -> None:
    issues = validate_manifest_entry(entry)
    if issues:
        raise ValueError(f"Invalid {kind} manifest entry: {issues}")


def _require_exact_record_keys(
    record: Mapping[str, Any],
    expected_keys: frozenset[str],
    *,
    row_kind: str,
) -> None:
    observed_keys = frozenset(record)
    missing = sorted(expected_keys.difference(observed_keys))
    extra = sorted(observed_keys.difference(expected_keys))
    if missing or extra:
        raise ValueError(
            f"{row_kind} keys do not match the serialized contract: "
            f"missing={missing}, extra={extra}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
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


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _require_gate_schema_version(value: object) -> str:
    schema_version = _text(value, "schema_version")
    if schema_version != GATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Gate manifest schema_version is unsupported: "
            f"expected {GATE_MANIFEST_SCHEMA_VERSION!r}, observed {schema_version!r}."
        )
    return schema_version


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
    "TIER_MANIFEST_SCHEMA_VERSION",
    "ManifestEntry",
    "build_dropped_entry",
    "build_passed_entry",
    "dropped_entry_from_record",
    "passed_entry_from_record",
    "read_dropped_manifest_json",
    "read_passed_manifest_json",
    "read_tier_manifest_json",
    "write_dropped_manifest_json",
    "write_passed_manifest_json",
    "write_tier_manifest_json",
]
