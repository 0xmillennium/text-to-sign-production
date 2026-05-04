"""Logical catalog loading and lookup for sample artifact manifests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.artifacts.catalog.types import (
    SampleHandle,
    SampleRef,
    SamplesCatalog,
    TieredCatalog,
    TieredSampleHandle,
)
from text_to_sign_production.artifacts.catalog.validate import (
    validate_samples_catalog,
    validate_tiered_catalog,
)
from text_to_sign_production.artifacts.store import (
    ArchiveMemberPathRef,
    ArchivePathRef,
    ArtifactStores,
    SamplePathRef,
    SampleStatus,
    SplitName,
    TierMembership,
    TierName,
    resolve_samples_relative,
    split_name_from_sample_split,
)
from text_to_sign_production.data.samples.manifests import manifest_entry_from_record
from text_to_sign_production.data.samples.types import (
    DroppedManifestEntry,
    PassedManifestEntry,
)

_CATALOG_SPLITS = (SplitName.TRAIN, SplitName.VAL, SplitName.TEST)
_SPLIT_ORDER = {split: index for index, split in enumerate(_CATALOG_SPLITS)}
CatalogProgressCallback = Callable[..., None]


def load_passed_samples_catalog(
    stores: ArtifactStores,
    splits: Iterable[SplitName | str] | None = None,
    *,
    progress_callback: CatalogProgressCallback | None = None,
) -> SamplesCatalog:
    """Load passed sample manifests into a logical catalog."""

    items: dict[SampleRef, SampleHandle] = {}
    selected_splits = _catalog_splits(splits)
    for split in selected_splits:
        manifest_path = stores.runtime.manifests.untiered_passed_manifest(split).path
        split_count = 0
        for record in _read_jsonl_records(manifest_path):
            entry = _parse_passed_entry(record, manifest_path)
            ref = SampleRef(split=split, sample_id=entry.sample_id)
            _require_new_ref(items, ref, manifest_path)
            items[ref] = SampleHandle(
                ref=ref,
                status=SampleStatus.PASSED,
                manifest_entry=entry,
                runtime_sample=resolve_samples_relative(
                    stores.runtime,
                    entry.sample_path,
                ),
                drive_archive=stores.drive.samples.split_archive(
                    SampleStatus.PASSED,
                    split,
                ),
                drive_archive_member=stores.drive.samples.archive_member(
                    split,
                    entry.sample_id,
                ),
            )
            split_count += 1
        if progress_callback is not None:
            progress_callback(manifest=split.value, records=split_count)
    catalog = SamplesCatalog(status=SampleStatus.PASSED, items=items)
    _require_valid_samples_catalog(catalog, stores, "passed samples catalog")
    return catalog


def load_dropped_samples_catalog(
    stores: ArtifactStores,
    splits: Iterable[SplitName | str] | None = None,
) -> SamplesCatalog:
    """Load dropped sample manifests into a logical catalog."""

    items: dict[SampleRef, SampleHandle] = {}
    for split in _catalog_splits(splits):
        manifest_path = stores.runtime.manifests.untiered_dropped_manifest(split).path
        for record in _read_jsonl_records(manifest_path):
            entry = _parse_dropped_entry(record, manifest_path)
            ref = SampleRef(split=split, sample_id=entry.sample_id)
            _require_new_ref(items, ref, manifest_path)
            runtime_sample: SamplePathRef | None = None
            drive_archive: ArchivePathRef | None = None
            drive_archive_member: ArchiveMemberPathRef | None = None
            if entry.sample_path is not None:
                runtime_sample = resolve_samples_relative(
                    stores.runtime,
                    entry.sample_path,
                )
                drive_archive = stores.drive.samples.split_archive(
                    SampleStatus.DROPPED,
                    split,
                )
                drive_archive_member = stores.drive.samples.archive_member(
                    split,
                    entry.sample_id,
                )
            items[ref] = SampleHandle(
                ref=ref,
                status=SampleStatus.DROPPED,
                manifest_entry=entry,
                runtime_sample=runtime_sample,
                drive_archive=drive_archive,
                drive_archive_member=drive_archive_member,
            )
    catalog = SamplesCatalog(status=SampleStatus.DROPPED, items=items)
    _require_valid_samples_catalog(catalog, stores, "dropped samples catalog")
    return catalog


def load_tiered_catalog(
    stores: ArtifactStores,
    tier: TierName,
    membership: TierMembership,
    splits: Iterable[SplitName | str] | None = None,
) -> TieredCatalog:
    """Load tiered manifests into a logical catalog."""

    items: dict[SampleRef, TieredSampleHandle] = {}
    for split in _catalog_splits(splits):
        manifest_path = stores.runtime.manifests.tiered_manifest(tier, membership, split).path
        for record in _read_jsonl_records(manifest_path):
            entry = _parse_passed_entry(record, manifest_path)
            ref = SampleRef(split=split, sample_id=entry.sample_id)
            _require_new_ref(items, ref, manifest_path)
            items[ref] = TieredSampleHandle(
                ref=ref,
                tier=tier,
                membership=membership,
                manifest_entry=entry,
                runtime_sample=resolve_samples_relative(
                    stores.runtime,
                    entry.sample_path,
                ),
                drive_archive=stores.drive.samples.split_archive(
                    SampleStatus.PASSED,
                    split,
                ),
                drive_archive_member=stores.drive.samples.archive_member(
                    split,
                    entry.sample_id,
                ),
            )
    catalog = TieredCatalog(tier=tier, membership=membership, items=items)
    _require_valid_tiered_catalog(catalog, stores, "tiered catalog")
    return catalog


def get_sample(catalog: SamplesCatalog, ref: SampleRef) -> SampleHandle:
    """Return one sample handle by logical identity."""

    try:
        return catalog.items[ref]
    except KeyError as exc:
        raise KeyError(
            f"Sample ref not found: split={ref.split.value}, sample_id={ref.sample_id}"
        ) from exc


def get_tiered_sample(catalog: TieredCatalog, ref: SampleRef) -> TieredSampleHandle:
    """Return one tiered sample handle by logical identity."""

    try:
        return catalog.items[ref]
    except KeyError as exc:
        raise KeyError(
            f"Tiered sample ref not found: split={ref.split.value}, sample_id={ref.sample_id}"
        ) from exc


def iter_samples(catalog: SamplesCatalog) -> Iterator[SampleHandle]:
    """Iterate sample handles in deterministic logical order."""

    for ref in _sorted_refs(catalog.items):
        yield catalog.items[ref]


def iter_samples_split(catalog: SamplesCatalog, split: SplitName) -> Iterator[SampleHandle]:
    """Iterate sample handles for one split in deterministic logical order."""

    for ref in _sorted_refs(catalog.items):
        if ref.split is split:
            yield catalog.items[ref]


def iter_tiered(catalog: TieredCatalog) -> Iterator[TieredSampleHandle]:
    """Iterate tiered sample handles in deterministic logical order."""

    for ref in _sorted_refs(catalog.items):
        yield catalog.items[ref]


def iter_tiered_split(
    catalog: TieredCatalog,
    split: SplitName,
) -> Iterator[TieredSampleHandle]:
    """Iterate tiered sample handles for one split in deterministic logical order."""

    for ref in _sorted_refs(catalog.items):
        if ref.split is split:
            yield catalog.items[ref]


def _read_jsonl_records(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw: object = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in manifest {path} at line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise TypeError(
                        f"Manifest {path} line {line_number} must contain a JSON object."
                    )
                if any(not isinstance(key, str) for key in raw):
                    raise TypeError(f"Manifest {path} line {line_number} must contain string keys.")
                yield cast(Mapping[str, Any], raw)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manifest file not found: {path}") from exc


def _parse_passed_entry(record: Mapping[str, Any], path: Path) -> PassedManifestEntry:
    entry = manifest_entry_from_record(record)
    if not isinstance(entry, PassedManifestEntry):
        raise TypeError(f"Expected passed manifest entry in {path}, got {type(entry).__name__}.")
    return entry


def _parse_dropped_entry(record: Mapping[str, Any], path: Path) -> DroppedManifestEntry:
    entry = manifest_entry_from_record(record)
    if not isinstance(entry, DroppedManifestEntry):
        raise TypeError(f"Expected dropped manifest entry in {path}, got {type(entry).__name__}.")
    return entry


def _require_new_ref(
    items: Mapping[SampleRef, object],
    ref: SampleRef,
    path: Path,
) -> None:
    if ref in items:
        raise ValueError(
            f"Duplicate sample ref in manifest input: "
            f"split={ref.split.value}, sample_id={ref.sample_id}, path={path}"
        )


def _sorted_refs(items: Mapping[SampleRef, object]) -> list[SampleRef]:
    return sorted(items, key=lambda ref: (_SPLIT_ORDER[ref.split], ref.sample_id))


def _catalog_splits(splits: Iterable[SplitName | str] | None) -> tuple[SplitName, ...]:
    if splits is None:
        return _CATALOG_SPLITS
    selected = tuple(split_name_from_sample_split(split) for split in splits)
    if len(set(selected)) != len(selected):
        raise ValueError("Catalog splits must be unique.")
    return selected


def _require_valid_samples_catalog(
    catalog: SamplesCatalog,
    stores: ArtifactStores,
    label: str,
) -> None:
    errors = validate_samples_catalog(catalog, stores)
    if errors:
        raise ValueError(f"Invalid {label}: {'; '.join(errors)}")


def _require_valid_tiered_catalog(
    catalog: TieredCatalog,
    stores: ArtifactStores,
    label: str,
) -> None:
    errors = validate_tiered_catalog(catalog, stores)
    if errors:
        raise ValueError(f"Invalid {label}: {'; '.join(errors)}")


__all__ = [
    "get_sample",
    "get_tiered_sample",
    "iter_samples",
    "iter_samples_split",
    "iter_tiered",
    "iter_tiered_split",
    "load_dropped_samples_catalog",
    "load_passed_samples_catalog",
    "load_tiered_catalog",
]
