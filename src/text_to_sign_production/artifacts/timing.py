"""Timing metadata joins for generated processed-sample artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from text_to_sign_production.artifacts.refs import (
    ManifestSampleRecord,
    ProcessedSampleDataError,
    TimingMetadata,
)
from text_to_sign_production.data.jsonl import iter_jsonl_with_line_numbers
from text_to_sign_production.data.schemas import NormalizedManifestEntry

if TYPE_CHECKING:
    from text_to_sign_production.artifacts.index import SampleIndex


def fallback_timing_metadata(record: ManifestSampleRecord) -> TimingMetadata:
    """Return timing metadata from the processed manifest record itself."""

    return TimingMetadata(
        start_time=None,
        end_time=None,
        fps=record.fps,
        num_frames=record.num_frames,
        source_video_path=record.source_video_path,
        video_width=record.video_width,
        video_height=record.video_height,
        source_manifest=None,
        join_key=None,
    )


def find_timing_metadata(
    *,
    index: SampleIndex,
    record: ManifestSampleRecord,
) -> TimingMetadata | None:
    """Join timing metadata from the filtered manifest for the selected split."""

    for source_name, path, timing_records in _load_timing_sources(
        index=index,
        split=record.split,
    ):
        match, join_key = _find_timing_match(record, timing_records, path)
        if match is None:
            continue
        return _timing_metadata_from_record(
            match,
            source_manifest=path.as_posix(),
            join_key=f"{source_name}:{join_key}",
        )
    return None


def _load_timing_sources(
    *,
    index: SampleIndex,
    split: str,
) -> list[tuple[str, Path, tuple[NormalizedManifestEntry, ...]]]:
    path = index.filtered_manifests_root / f"filtered_{split}.jsonl"
    if not path.is_file():
        return []

    parsed_records: list[NormalizedManifestEntry] = []
    for line_number, record in iter_jsonl_with_line_numbers(path):
        try:
            parsed_records.append(NormalizedManifestEntry.from_record(record))
        except Exception as exc:
            raise ProcessedSampleDataError(
                f"Could not parse filtered timing manifest {path} line {line_number}: {exc}"
            ) from exc
    return [("filtered", path, tuple(parsed_records))]


def _find_timing_match(
    record: ManifestSampleRecord,
    timing_records: tuple[NormalizedManifestEntry, ...],
    manifest_path: Path,
) -> tuple[NormalizedManifestEntry | None, str | None]:
    key_specs = (
        ("sample_id", record.sample_id, ("sample_id",)),
        (
            "source_sentence_name",
            record.entry.source_sentence_name,
            ("source_sentence_name", "sentence_name"),
        ),
        (
            "source_sentence_id",
            record.entry.source_sentence_id,
            ("source_sentence_id", "sentence_id"),
        ),
    )
    for join_name, expected_value, field_names in key_specs:
        if not expected_value:
            continue
        matches = [
            candidate
            for candidate in timing_records
            if any(
                _timing_record_field(candidate, field_name) == expected_value
                for field_name in field_names
            )
        ]
        if len(matches) == 1:
            return matches[0], join_name
        if len(matches) > 1:
            raise ProcessedSampleDataError(
                f"Timing join for sample_id={record.sample_id!r} is ambiguous in "
                f"{manifest_path} using {join_name}={expected_value!r}."
            )
    return None, None


def _timing_record_field(record: NormalizedManifestEntry, field_name: str) -> str | None:
    value = getattr(record, field_name, None)
    if value is None:
        return None
    return str(value)


def _timing_metadata_from_record(
    record: NormalizedManifestEntry,
    *,
    source_manifest: str,
    join_key: str,
) -> TimingMetadata:
    return TimingMetadata(
        start_time=record.start_time,
        end_time=record.end_time,
        fps=record.fps,
        num_frames=record.num_frames,
        source_video_path=record.source_video_path,
        video_width=record.video_width,
        video_height=record.video_height,
        source_manifest=source_manifest,
        join_key=join_key,
    )


__all__ = ["fallback_timing_metadata", "find_timing_metadata"]
