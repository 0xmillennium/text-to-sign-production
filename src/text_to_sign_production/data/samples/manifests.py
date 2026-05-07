"""Construction, record conversions, and JSONL IO for manifest entries."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.samples._shared.parsing import (
    bool_from_record,
    frame_quality_from_record,
    int_from_record,
    optional_float_from_record,
    optional_int_from_record,
    optional_mapping,
    optional_text_from_record,
    require_mapping,
    require_record_keys,
    reject_record_keys,
    sample_split_from_record,
    sample_status_from_record,
    selected_person_from_record,
    string_tuple_from_sequence,
    text_from_record,
)
from text_to_sign_production.data.samples.schema import (
    DROPPED_ONLY_MANIFEST_KEYS,
    PASSED_ONLY_MANIFEST_KEYS,
    PROCESSED_SCHEMA_VERSION,
    REQUIRED_DROPPED_MANIFEST_KEYS,
    REQUIRED_DROPPED_MATERIALIZATION_KEYS,
    REQUIRED_PASSED_MANIFEST_KEYS,
)
from text_to_sign_production.data.samples.types import (
    DroppedDebugMaterializationOutcome,
    DroppedManifestEntry,
    DroppedMaterializationLifecycle,
    FrameQualitySummary,
    JsonValue,
    ManifestEntry,
    PassedManifestEntry,
    SampleStatus,
    SelectedPersonMetadata,
)

@dataclass(frozen=True, slots=True)
class ManifestWriteProgressEvent:
    path: Path
    completed: int
    entry: ManifestEntry


class ManifestWriteProgressSink(Protocol):
    def update_manifest_write_progress(self, event: ManifestWriteProgressEvent) -> None: ...


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
    materialization: DroppedMaterializationLifecycle,
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
        materialization=materialization,
        drop_details=drop_details or {},
        text=text,
        num_frames=num_frames,
        fps=fps,
        selected_person=selected_person,
        frame_quality=frame_quality,
    )


def build_dropped_materialization_lifecycle(
    *,
    debug_materialization_eligible: bool,
    debug_materialization_attempted: bool,
    debug_materialization_outcome: DroppedDebugMaterializationOutcome | str,
    payload_path: str | None = None,
    payload_exists: bool = False,
    archive_publishable: bool = False,
    failure_reason: str | None = None,
) -> DroppedMaterializationLifecycle:
    """Build explicit lifecycle facts for a dropped debug payload."""
    return DroppedMaterializationLifecycle(
        debug_materialization_eligible=debug_materialization_eligible,
        debug_materialization_attempted=debug_materialization_attempted,
        debug_materialization_outcome=DroppedDebugMaterializationOutcome(
            debug_materialization_outcome
        ),
        payload_path=payload_path,
        payload_exists=payload_exists,
        archive_publishable=archive_publishable,
        failure_reason=failure_reason,
    )


def manifest_entry_from_record(record: Mapping[str, Any]) -> ManifestEntry:
    """Parse a manifest entry record dictionary into its typed model."""
    status = sample_status_from_record(record.get("status"), "Manifest status")

    if status is SampleStatus.PASSED:
        require_record_keys(record, REQUIRED_PASSED_MANIFEST_KEYS, surface="Passed manifest entry")
        reject_record_keys(record, DROPPED_ONLY_MANIFEST_KEYS, surface="Passed manifest entry")
        selected_person_record = require_mapping(
            record["selected_person"], "selected_person", surface="Manifest"
        )
        frame_quality_record = require_mapping(
            record["frame_quality"], "frame_quality", surface="Manifest"
        )
        return PassedManifestEntry(
            sample_id=text_from_record(record["sample_id"], "Passed manifest sample_id"),
            schema_version=text_from_record(
                record["schema_version"],
                "Passed manifest schema_version",
            ),
            status=SampleStatus.PASSED,
            text=text_from_record(record["text"], "Passed manifest text"),
            split=sample_split_from_record(record["split"], "Passed manifest split"),
            num_frames=int_from_record(record["num_frames"], "Passed manifest num_frames"),
            fps=optional_float_from_record(record.get("fps"), "Passed manifest fps"),
            sample_path=text_from_record(record["sample_path"], "Passed manifest sample_path"),
            source_video_id=text_from_record(
                record["source_video_id"],
                "Passed manifest source_video_id",
            ),
            source_sentence_id=text_from_record(
                record["source_sentence_id"],
                "Passed manifest source_sentence_id",
            ),
            source_sentence_name=text_from_record(
                record["source_sentence_name"],
                "Passed manifest source_sentence_name",
            ),
            selected_person=selected_person_from_record(selected_person_record),
            frame_quality=frame_quality_from_record(frame_quality_record),
        )
    if status is SampleStatus.DROPPED:
        require_record_keys(record, REQUIRED_DROPPED_MANIFEST_KEYS, surface="Dropped manifest entry")
        reject_record_keys(record, PASSED_ONLY_MANIFEST_KEYS, surface="Dropped manifest entry")
        dropped_selected_person_record = optional_mapping(
            record.get("selected_person"), "selected_person", surface="Manifest"
        )
        dropped_frame_quality_record = optional_mapping(
            record.get("frame_quality"), "frame_quality", surface="Manifest"
        )
        drop_details_record = require_mapping(
            record.get("drop_details", {}),
            "drop_details",
            surface="Manifest",
        )
        materialization_record = require_mapping(
            record["materialization"],
            "materialization",
            surface="Manifest",
        )
        return DroppedManifestEntry(
            sample_id=text_from_record(record["sample_id"], "Dropped manifest sample_id"),
            schema_version=text_from_record(
                record["schema_version"],
                "Dropped manifest schema_version",
            ),
            status=SampleStatus.DROPPED,
            split=sample_split_from_record(record["split"], "Dropped manifest split"),
            drop_stage=text_from_record(record["drop_stage"], "Dropped manifest drop_stage"),
            drop_reasons=string_tuple_from_sequence(
                record["drop_reasons"], "Manifest drop_reasons"
            ),
            materialization=_materialization_from_record(materialization_record),
            drop_details=cast(dict[str, JsonValue], dict(drop_details_record)),
            text=optional_text_from_record(record.get("text"), "Dropped manifest text"),
            num_frames=optional_int_from_record(
                record.get("num_frames"),
                "Dropped manifest num_frames",
            ),
            fps=optional_float_from_record(record.get("fps"), "Dropped manifest fps"),
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
    raise ValueError(f"Unknown manifest status: {status!r}.")


def _materialization_from_record(
    record: Mapping[str, Any],
) -> DroppedMaterializationLifecycle:
    require_record_keys(
        record,
        REQUIRED_DROPPED_MATERIALIZATION_KEYS,
        surface="Dropped materialization lifecycle",
    )
    return DroppedMaterializationLifecycle(
        debug_materialization_eligible=bool_from_record(
            record["debug_materialization_eligible"],
            "Dropped materialization debug_materialization_eligible",
        ),
        debug_materialization_attempted=bool_from_record(
            record["debug_materialization_attempted"],
            "Dropped materialization debug_materialization_attempted",
        ),
        debug_materialization_outcome=DroppedDebugMaterializationOutcome(
            text_from_record(
                record["debug_materialization_outcome"],
                "Dropped materialization debug_materialization_outcome",
            )
        ),
        payload_path=optional_text_from_record(
            record.get("payload_path"),
            "Dropped materialization payload_path",
        ),
        payload_exists=bool_from_record(
            record["payload_exists"],
            "Dropped materialization payload_exists",
        ),
        archive_publishable=bool_from_record(
            record["archive_publishable"],
            "Dropped materialization archive_publishable",
        ),
        failure_reason=optional_text_from_record(
            record.get("failure_reason"),
            "Dropped materialization failure_reason",
        ),
    )


def write_manifest_jsonl(
    path: Path,
    entries: Iterable[ManifestEntry],
    *,
    progress_sink: ManifestWriteProgressSink | None = None,
) -> None:
    """Write canonical manifest entries as deterministic JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        completed = 0
        for entry in entries:
            handle.write(json.dumps(entry.to_record(), sort_keys=True) + "\n")
            completed += 1
            if progress_sink is not None:
                progress_sink.update_manifest_write_progress(
                    ManifestWriteProgressEvent(path=path, completed=completed, entry=entry)
                )


def read_manifest_jsonl(path: str | Path) -> tuple[ManifestEntry, ...]:
    """Read canonical manifest entries from a JSONL manifest file."""
    manifest_path = Path(path)
    entries: list[ManifestEntry] = []
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw: object = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in manifest {manifest_path} at line "
                        f"{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise TypeError(
                        f"Manifest {manifest_path} line {line_number} must contain "
                        "a JSON object."
                    )
                if any(not isinstance(key, str) for key in raw):
                    raise TypeError(
                        f"Manifest {manifest_path} line {line_number} must contain "
                        "string keys."
                    )
                entries.append(manifest_entry_from_record(cast(Mapping[str, Any], raw)))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}") from exc
    return tuple(entries)
