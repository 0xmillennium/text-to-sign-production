"""Construction, record conversions, and JSONL IO for manifest entries."""

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.data._shared.identities import SampleSplit
from text_to_sign_production.data.samples._shared.parsing import (
    bool_from_record,
    frame_quality_from_record,
    optional_mapping,
    require_mapping,
    sample_split_from_record,
    selected_person_from_record,
    string_tuple_from_sequence,
)
from text_to_sign_production.data.samples.schema import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.data.samples.types import (
    DroppedManifestEntry,
    FrameQualitySummary,
    JsonValue,
    ManifestEntry,
    PassedManifestEntry,
    SampleStatus,
    SelectedPersonMetadata,
)

ManifestWriteProgressCallback = Callable[..., None]


def build_passed_entry(
    sample_id: str,
    text: str,
    split: SampleSplit | str,
    num_frames: int,
    fps: float | None,
    sample_path: str,
    source_video_id: str,
    source_sentence_id: str,
    source_sentence_name: str,
    selected_person: SelectedPersonMetadata,
    frame_quality: FrameQualitySummary,
) -> PassedManifestEntry:
    """Build a semantically valid passed manifest entry."""
    return PassedManifestEntry(
        sample_id=sample_id,
        schema_version=PROCESSED_SCHEMA_VERSION,
        status=SampleStatus.PASSED,
        text=text,
        split=SampleSplit(split),
        num_frames=num_frames,
        fps=fps,
        sample_path=sample_path,
        source_video_id=source_video_id,
        source_sentence_id=source_sentence_id,
        source_sentence_name=source_sentence_name,
        selected_person=selected_person,
        frame_quality=frame_quality,
    )


def build_dropped_entry(
    sample_id: str,
    split: SampleSplit | str,
    drop_stage: str,
    drop_reasons: tuple[str, ...],
    debug_only: bool = False,
    sample_path: str | None = None,
    drop_details: dict[str, JsonValue] | None = None,
    text: str | None = None,
    num_frames: int | None = None,
    fps: float | None = None,
    selected_person: SelectedPersonMetadata | None = None,
    frame_quality: FrameQualitySummary | None = None,
) -> DroppedManifestEntry:
    """Build a semantically valid dropped manifest entry."""
    return DroppedManifestEntry(
        sample_id=sample_id,
        schema_version=PROCESSED_SCHEMA_VERSION,
        status=SampleStatus.DROPPED,
        split=SampleSplit(split),
        drop_stage=drop_stage,
        drop_reasons=drop_reasons,
        debug_only=debug_only,
        sample_path=sample_path,
        drop_details=drop_details or {},
        text=text,
        num_frames=num_frames,
        fps=fps,
        selected_person=selected_person,
        frame_quality=frame_quality,
    )


def manifest_entry_from_record(record: Mapping[str, Any]) -> ManifestEntry:
    """Parse a manifest entry record dictionary into its typed model."""
    status = record.get("status")

    if status == SampleStatus.PASSED.value:
        selected_person_record = require_mapping(
            record["selected_person"], "selected_person", surface="Manifest"
        )
        frame_quality_record = require_mapping(
            record["frame_quality"], "frame_quality", surface="Manifest"
        )
        return PassedManifestEntry(
            sample_id=str(record["sample_id"]),
            schema_version=str(record["schema_version"]),
            status=SampleStatus.PASSED,
            text=str(record["text"]),
            split=sample_split_from_record(record["split"], "Passed manifest split"),
            num_frames=int(record["num_frames"]),
            fps=float(record["fps"]) if record.get("fps") is not None else None,
            sample_path=str(record["sample_path"]),
            source_video_id=str(record["source_video_id"]),
            source_sentence_id=str(record["source_sentence_id"]),
            source_sentence_name=str(record["source_sentence_name"]),
            selected_person=selected_person_from_record(selected_person_record),
            frame_quality=frame_quality_from_record(frame_quality_record),
        )
    if status == SampleStatus.DROPPED.value:
        dropped_selected_person_record = optional_mapping(
            record.get("selected_person"), "selected_person", surface="Manifest"
        )
        dropped_frame_quality_record = optional_mapping(
            record.get("frame_quality"), "frame_quality", surface="Manifest"
        )
        return DroppedManifestEntry(
            sample_id=str(record["sample_id"]),
            schema_version=str(record["schema_version"]),
            status=SampleStatus.DROPPED,
            split=sample_split_from_record(record["split"], "Dropped manifest split"),
            drop_stage=str(record["drop_stage"]),
            drop_reasons=string_tuple_from_sequence(
                record["drop_reasons"], "Manifest drop_reasons"
            ),
            debug_only=bool_from_record(record["debug_only"], "Manifest debug_only"),
            sample_path=(
                str(record["sample_path"]) if record.get("sample_path") is not None else None
            ),
            drop_details=cast(dict[str, JsonValue], dict(record.get("drop_details", {}))),
            text=str(record["text"]) if "text" in record else None,
            num_frames=int(record["num_frames"]) if record.get("num_frames") is not None else None,
            fps=float(record["fps"]) if record.get("fps") is not None else None,
            selected_person=(
                selected_person_from_record(dropped_selected_person_record)
                if dropped_selected_person_record is not None
                else None
            ),
            frame_quality=(
                frame_quality_from_record(dropped_frame_quality_record)
                if dropped_frame_quality_record is not None
                else None
            ),
        )
    raise ValueError(f"Unknown or missing status '{status}' in record.")


def write_manifest_jsonl(
    path: Path,
    entries: Iterable[ManifestEntry],
    *,
    progress_callback: ManifestWriteProgressCallback | None = None,
) -> None:
    """Write canonical manifest entries as deterministic JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        completed = 0
        for entry in entries:
            handle.write(json.dumps(entry.to_record(), sort_keys=True) + "\n")
            completed += 1
            if progress_callback is not None:
                progress_callback(path=path, completed=completed, entry=entry)
