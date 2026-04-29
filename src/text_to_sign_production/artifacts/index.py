"""Processed manifest indexing for generated sample artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_to_sign_production.artifacts.refs import (
    ManifestSampleRecord,
    ProcessedSampleDataError,
    TimingMetadata,
)
from text_to_sign_production.artifacts.timing import (
    fallback_timing_metadata,
    find_timing_metadata,
)
from text_to_sign_production.data.constants import (
    PROCESSED_SCHEMA_VERSION,
    SPLITS,
)
from text_to_sign_production.data.jsonl import iter_jsonl_with_line_numbers
from text_to_sign_production.data.schemas import ProcessedManifestEntry
from text_to_sign_production.data.validate import validate_processed_sample_path


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """Processed manifest records across one or more splits."""

    data_root: Path
    manifest_paths: Mapping[str, Path]
    records: tuple[ManifestSampleRecord, ...]
    duplicate_sample_ids: Mapping[str, tuple[ManifestSampleRecord, ...]]
    filtered_manifests_root: Path

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

        timing = find_timing_metadata(index=self, record=record)
        return timing if timing is not None else fallback_timing_metadata(record)


def build_sample_index(
    data_root: Path | str,
    *,
    splits: Sequence[str] | None = None,
    require_sample_files: bool = True,
    processed_manifests_root: Path | str | None = None,
    processed_samples_root: Path | str | None = None,
    filtered_manifests_root: Path | str | None = None,
) -> SampleIndex:
    """Build an index from processed manifests, optionally requiring `.npz` files."""

    resolved_data_root = _resolve_root(data_root)
    resolved_processed_manifests_root = (
        resolved_data_root / "processed" / "v1" / "manifests"
        if processed_manifests_root is None
        else _resolve_root(processed_manifests_root)
    )
    resolved_processed_samples_root = (
        resolved_data_root / "processed" / "v1" / "samples"
        if processed_samples_root is None
        else _resolve_root(processed_samples_root)
    )
    resolved_filtered_manifests_root = (
        resolved_data_root / "interim" / "filtered_manifests"
        if filtered_manifests_root is None
        else _resolve_root(filtered_manifests_root)
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
    )


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
            sample_path = _require_file(sample_path, label="Processed sample file")
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


def _resolve_root(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _require_file(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if resolved.is_dir():
        raise IsADirectoryError(f"{label} is a directory, expected a file: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved.resolve()


__all__ = ["SampleIndex", "build_sample_index"]
