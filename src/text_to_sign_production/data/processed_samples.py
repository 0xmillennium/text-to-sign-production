"""Processed manifest sample indexing and timing metadata joins."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from text_to_sign_production.core import paths as core_paths
from text_to_sign_production.core.files import require_file
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.data.constants import (
    PROCESSED_SCHEMA_VERSION,
    SPLITS,
)
from text_to_sign_production.data.jsonl import iter_jsonl_with_line_numbers
from text_to_sign_production.data.schemas import (
    NormalizedManifestEntry,
    ProcessedManifestEntry,
    RawManifestEntry,
)
from text_to_sign_production.data.validate import validate_processed_sample_path

_CANDIDATE_LIMIT = 8
_TimingRecord: TypeAlias = NormalizedManifestEntry | RawManifestEntry


class ProcessedSampleDataError(ValueError):
    """Raised when processed sample inputs violate the expected contract."""


class SampleSelectionError(ProcessedSampleDataError):
    """Raised when a requested processed sample cannot be selected."""


class DuplicateSampleIdError(SampleSelectionError):
    """Raised when a sample id is present in more than one loaded manifest record."""

    def __init__(self, sample_id: str, candidates: Iterable[ManifestSampleRecord]) -> None:
        self.sample_id = sample_id
        self.candidates = tuple(candidates)
        rendered = "\n".join(
            f"- split={record.split} sample_id={record.sample_id} sample_path={record.sample_path}"
            for record in self.candidates[:_CANDIDATE_LIMIT]
        )
        suffix = ""
        if len(self.candidates) > _CANDIDATE_LIMIT:
            suffix = f"\n... {len(self.candidates) - _CANDIDATE_LIMIT} more candidate(s)."
        super().__init__(
            f"Sample id {sample_id!r} appears in {len(self.candidates)} loaded processed "
            f"manifest records; select a unique sample id or restrict --split.\n{rendered}{suffix}"
        )


def _layout_from_data_root(data_root: Path | str | None = None) -> ProjectLayout:
    if data_root is None:
        return ProjectLayout(core_paths.DEFAULT_REPO_ROOT)
    resolved_data_root = Path(data_root).expanduser().resolve()
    if resolved_data_root.name != "data":
        raise ValueError(
            f"data_root must point at the project data directory: {resolved_data_root}"
        )
    return ProjectLayout(resolved_data_root.parent)


@dataclass(frozen=True, slots=True)
class TimingMetadata:
    """Timing and source-video metadata joined from interim manifests."""

    start_time: float | None
    end_time: float | None
    fps: float | None
    num_frames: int | None
    source_video_path: str | None
    video_width: int | None
    video_height: int | None
    source_manifest: str | None
    join_key: str | None


@dataclass(frozen=True, slots=True)
class ManifestSampleRecord:
    """One canonical processed manifest record resolved to a local `.npz` path."""

    entry: ProcessedManifestEntry
    sample_path: Path

    @property
    def sample_id(self) -> str:
        return self.entry.sample_id

    @property
    def split(self) -> str:
        return self.entry.split

    @property
    def text(self) -> str:
        return self.entry.text

    @property
    def fps(self) -> float | None:
        return self.entry.fps

    @property
    def num_frames(self) -> int:
        return self.entry.num_frames

    @property
    def sample_path_value(self) -> str:
        return self.entry.sample_path

    @property
    def source_video_path(self) -> str | None:
        return self.entry.source_video_path

    @property
    def video_width(self) -> int | None:
        return self.entry.video_width

    @property
    def video_height(self) -> int | None:
        return self.entry.video_height


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """Processed manifest records across one or more splits."""

    data_root: Path
    manifest_paths: Mapping[str, Path]
    records: tuple[ManifestSampleRecord, ...]
    duplicate_sample_ids: Mapping[str, tuple[ManifestSampleRecord, ...]]
    filtered_manifests_root: Path
    normalized_manifests_root: Path
    raw_manifests_root: Path

    @property
    def splits(self) -> tuple[str, ...]:
        """Return loaded splits in stable repository order."""

        observed = {record.split for record in self.records}
        return tuple(split for split in SPLITS if split in observed)

    def records_for_sample_id(self, sample_id: str) -> tuple[ManifestSampleRecord, ...]:
        """Return all loaded records with the exact sample id."""

        return tuple(record for record in self.records if record.sample_id == sample_id)

    def timing_for(self, record: ManifestSampleRecord) -> TimingMetadata:
        """Return joined timing metadata, falling back to processed manifest fields."""

        timing = _find_timing_metadata(index=self, record=record)
        return timing if timing is not None else _fallback_timing_metadata(record)


def build_sample_index(
    data_root: Path | str,
    *,
    splits: Sequence[str] | None = None,
    require_sample_files: bool = True,
    processed_manifests_root: Path | str | None = None,
    processed_samples_root: Path | str | None = None,
    filtered_manifests_root: Path | str | None = None,
    normalized_manifests_root: Path | str | None = None,
    raw_manifests_root: Path | str | None = None,
) -> SampleIndex:
    """Build an index from processed manifests, optionally requiring `.npz` files."""

    resolved_data_root = Path(data_root).expanduser().resolve()
    layout = _layout_from_data_root(resolved_data_root)
    resolved_processed_manifests_root = (
        layout.processed_v1_manifests_root
        if processed_manifests_root is None
        else Path(processed_manifests_root)
    )
    resolved_processed_samples_root = (
        layout.processed_v1_samples_root
        if processed_samples_root is None
        else Path(processed_samples_root)
    )
    resolved_filtered_manifests_root = (
        layout.filtered_manifests_root
        if filtered_manifests_root is None
        else Path(filtered_manifests_root)
    )
    resolved_normalized_manifests_root = (
        layout.normalized_manifests_root
        if normalized_manifests_root is None
        else Path(normalized_manifests_root)
    )
    resolved_raw_manifests_root = (
        layout.raw_manifests_root if raw_manifests_root is None else Path(raw_manifests_root)
    )
    requested_splits = _resolved_splits(splits)
    records: list[ManifestSampleRecord] = []
    manifest_paths: dict[str, Path] = {}

    for split in requested_splits:
        manifest_path = resolved_processed_manifests_root / f"{split}.jsonl"
        samples_split_root = resolved_processed_samples_root / split
        if not manifest_path.is_file():
            if splits is None:
                continue
            raise FileNotFoundError(f"Processed manifest not found for {split}: {manifest_path}")
        if splits is None and require_sample_files and not samples_split_root.is_dir():
            continue

        split_records = _load_processed_manifest_records(
            manifest_path,
            data_root=resolved_data_root,
            processed_samples_root=resolved_processed_samples_root,
            expected_split=split,
            require_sample_files=require_sample_files,
        )
        if split_records:
            manifest_paths[split] = manifest_path
            records.extend(split_records)

    if not records:
        requested = ", ".join(requested_splits)
        raise FileNotFoundError(
            f"No processed manifests with records were found under {resolved_data_root} "
            f"for split scope: {requested}."
        )

    records_tuple = tuple(records)
    duplicate_sample_ids = _duplicate_sample_ids(records_tuple)
    return SampleIndex(
        data_root=resolved_data_root,
        manifest_paths=manifest_paths,
        records=records_tuple,
        duplicate_sample_ids=duplicate_sample_ids,
        filtered_manifests_root=resolved_filtered_manifests_root,
        normalized_manifests_root=resolved_normalized_manifests_root,
        raw_manifests_root=resolved_raw_manifests_root,
    )


def select_sample(
    index: SampleIndex,
    *,
    sample_id: str | None = None,
    random_selection: bool = False,
    seed: int | None = None,
) -> ManifestSampleRecord:
    """Select one sample by exact id or random choice with optional deterministic seeding."""

    modes = [sample_id is not None, random_selection]
    if sum(modes) != 1:
        raise SampleSelectionError("Exactly one selection mode is required: sample_id or random.")

    if sample_id is not None:
        normalized_sample_id = sample_id.strip()
        if not normalized_sample_id:
            raise SampleSelectionError("Sample id must not be blank.")
        matches = index.records_for_sample_id(normalized_sample_id)
        if not matches:
            splits = ", ".join(index.splits) or "<none>"
            raise SampleSelectionError(
                f"Sample id {normalized_sample_id!r} was not found in loaded processed "
                f"manifest splits: {splits}."
            )
        if len(matches) > 1:
            raise DuplicateSampleIdError(normalized_sample_id, matches)
        return matches[0]

    if index.duplicate_sample_ids:
        duplicate_id, candidates = next(iter(index.duplicate_sample_ids.items()))
        raise DuplicateSampleIdError(duplicate_id, candidates)
    return random.Random(seed).choice(list(index.records))


def _resolved_splits(splits: Sequence[str] | None) -> tuple[str, ...]:
    if splits is None:
        return SPLITS
    if not splits:
        raise ProcessedSampleDataError("At least one split must be requested.")
    invalid = [split for split in splits if split not in SPLITS]
    if invalid:
        expected = ", ".join(SPLITS)
        raise ProcessedSampleDataError(
            f"Unsupported split(s) {invalid!r}; expected values from: {expected}."
        )
    return tuple(split for split in SPLITS if split in set(splits))


def _load_processed_manifest_records(
    manifest_path: Path,
    *,
    data_root: Path,
    processed_samples_root: Path,
    expected_split: str,
    require_sample_files: bool,
) -> tuple[ManifestSampleRecord, ...]:
    records: list[ManifestSampleRecord] = []
    for line_number, raw_record in iter_jsonl_with_line_numbers(manifest_path):
        entry = _processed_entry_from_record(raw_record, manifest_path, line_number)
        records.append(
            _manifest_sample_record_from_entry(
                entry,
                data_root=data_root,
                processed_samples_root=processed_samples_root,
                manifest_path=manifest_path,
                line_number=line_number,
                expected_split=expected_split,
                require_sample_files=require_sample_files,
            )
        )
    return tuple(records)


def _processed_entry_from_record(
    record: Mapping[str, Any],
    manifest_path: Path,
    line_number: int,
) -> ProcessedManifestEntry:
    try:
        entry = ProcessedManifestEntry.from_record(record)
    except Exception as exc:
        raise ProcessedSampleDataError(
            f"Could not parse processed manifest {manifest_path} line {line_number}: {exc}"
        ) from exc
    if entry.processed_schema_version != PROCESSED_SCHEMA_VERSION:
        raise ProcessedSampleDataError(
            f"Processed manifest {manifest_path} line {line_number} uses schema "
            f"{entry.processed_schema_version!r}; expected {PROCESSED_SCHEMA_VERSION!r}."
        )
    return entry


def _manifest_sample_record_from_entry(
    entry: ProcessedManifestEntry,
    *,
    data_root: Path,
    processed_samples_root: Path,
    manifest_path: Path,
    line_number: int,
    expected_split: str,
    require_sample_files: bool,
) -> ManifestSampleRecord:
    if entry.split != expected_split:
        raise ProcessedSampleDataError(
            f"Processed manifest {manifest_path} line {line_number} has split {entry.split!r}; "
            f"expected {expected_split!r}."
        )
    try:
        sample_path = validate_processed_sample_path(
            entry.sample_path,
            data_root=data_root,
            processed_samples_root=processed_samples_root,
            split=entry.split,
            sample_id=entry.sample_id,
        )
        if require_sample_files:
            sample_path = require_file(sample_path, label="Processed sample file")
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        raise ProcessedSampleDataError(
            f"Processed manifest {manifest_path} line {line_number} has invalid sample_path: {exc}"
        ) from exc

    return ManifestSampleRecord(entry=entry, sample_path=sample_path)


def _duplicate_sample_ids(
    records: tuple[ManifestSampleRecord, ...],
) -> dict[str, tuple[ManifestSampleRecord, ...]]:
    records_by_sample_id: dict[str, list[ManifestSampleRecord]] = {}
    for record in records:
        records_by_sample_id.setdefault(record.sample_id, []).append(record)
    return {
        sample_id: tuple(sample_records)
        for sample_id, sample_records in records_by_sample_id.items()
        if len(sample_records) > 1
    }


def _fallback_timing_metadata(record: ManifestSampleRecord) -> TimingMetadata:
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


def _find_timing_metadata(
    *,
    index: SampleIndex,
    record: ManifestSampleRecord,
) -> TimingMetadata | None:
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
) -> list[tuple[str, Path, tuple[_TimingRecord, ...]]]:
    source_specs: tuple[
        tuple[str, Path, Callable[[Mapping[str, Any]], _TimingRecord]],
        ...,
    ] = (
        (
            "filtered",
            index.filtered_manifests_root / f"filtered_{split}.jsonl",
            NormalizedManifestEntry.from_record,
        ),
        (
            "normalized",
            index.normalized_manifests_root / f"normalized_{split}.jsonl",
            NormalizedManifestEntry.from_record,
        ),
        (
            "raw",
            index.raw_manifests_root / f"raw_{split}.jsonl",
            RawManifestEntry.from_record,
        ),
    )
    loaded: list[tuple[str, Path, tuple[_TimingRecord, ...]]] = []
    for source_name, path, parser in source_specs:
        if not path.is_file():
            continue
        parsed_records: list[_TimingRecord] = []
        for line_number, record in iter_jsonl_with_line_numbers(path):
            try:
                parsed_records.append(parser(record))
            except Exception as exc:
                raise ProcessedSampleDataError(
                    f"Could not parse {source_name} timing manifest {path} "
                    f"line {line_number}: {exc}"
                ) from exc
        loaded.append((source_name, path, tuple(parsed_records)))
    return loaded


def _find_timing_match(
    record: ManifestSampleRecord,
    timing_records: tuple[_TimingRecord, ...],
    manifest_path: Path,
) -> tuple[_TimingRecord | None, str | None]:
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


def _timing_record_field(record: _TimingRecord, field_name: str) -> str | None:
    value = getattr(record, field_name, None)
    if value is None:
        return None
    return str(value)


def _timing_metadata_from_record(
    record: _TimingRecord,
    *,
    source_manifest: str,
    join_key: str,
) -> TimingMetadata:
    if isinstance(record, RawManifestEntry):
        fps = record.video_fps
    else:
        fps = record.fps
    return TimingMetadata(
        start_time=record.start_time,
        end_time=record.end_time,
        fps=fps,
        num_frames=record.num_frames,
        source_video_path=record.source_video_path,
        video_width=record.video_width,
        video_height=record.video_height,
        source_manifest=source_manifest,
        join_key=join_key,
    )
