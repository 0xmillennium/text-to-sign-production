"""Lazy reader for provider-neutral tensor data surfaces."""

from __future__ import annotations

import json
from collections.abc import Iterator, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from text_to_sign_production.modeling.data_surfaces.cache_keys import sha256_file
from text_to_sign_production.modeling.data_surfaces.contracts import (
    ModelDataShardRecord,
    ModelDataSurface,
    ModelDataSurfaceError,
    metadata_from_dict,
    shard_record_from_dict,
)


class ModelDataSurfaceReader:
    """Load metadata up front, then load and release one shard at a time."""

    def __init__(self, surface: ModelDataSurface | Path) -> None:
        if isinstance(surface, ModelDataSurface):
            self.surface = surface
        else:
            self.surface = load_model_data_surface(Path(surface))
        self._records: tuple[ModelDataShardRecord, ...] | None = None

    @property
    def metadata(self):
        return self.surface.metadata

    def shard_records(self) -> tuple[ModelDataShardRecord, ...]:
        if self._records is None:
            records = []
            with self.surface.manifest_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ModelDataSurfaceError(
                            f"malformed surface manifest line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(raw, dict):
                        raise ModelDataSurfaceError("surface manifest rows must be objects.")
                    records.append(shard_record_from_dict(raw))
            if len(records) != self.surface.metadata.shard_count:
                raise ModelDataSurfaceError("surface manifest shard_count does not match metadata.")
            self._records = tuple(records)
        return self._records

    def iter_shards(self, *, verify_sha256: bool = True) -> Iterator[dict[str, Any]]:
        for record in self.shard_records():
            path = self.surface.root / record.relative_path
            if verify_sha256 and sha256_file(path) != record.sha256:
                raise ModelDataSurfaceError(f"surface shard checksum mismatch: {path}")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ModelDataSurfaceError(f"surface shard could not be loaded: {path}") from exc
            if not isinstance(payload, dict):
                raise ModelDataSurfaceError("surface shard payload must be a dict.")
            _validate_tensor_shard_payload(payload, path=path)
            yield payload
            del payload

    def iter_batches(
        self,
        *,
        batch_size: int,
        shuffle_shards: bool = False,
        shuffle_units: bool = False,
        verify_sha256: bool = True,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        runtime_trace: MutableMapping[str, object] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if batch_size <= 0:
            raise ModelDataSurfaceError("batch_size must be positive.")
        _validate_worker_options(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        if runtime_trace is not None:
            runtime_trace["surface_reader_num_workers_used"] = num_workers
            runtime_trace["surface_reader_worker_mode"] = (
                "multiprocess" if num_workers > 0 else "single_process"
            )
            runtime_trace["surface_reader_prefetch_factor_used"] = (
                prefetch_factor if num_workers > 0 else None
            )
            runtime_trace["surface_reader_persistent_workers_used"] = (
                persistent_workers if num_workers > 0 else False
            )
        records = list(self.shard_records())
        if shuffle_shards:
            import random

            random.shuffle(records)
        if num_workers > 0:
            dataset = _SurfaceShardDataset(
                records,
                surface_root=self.surface.root,
                verify_sha256=verify_sha256,
            )
            shard_iter = DataLoader(
                dataset,
                batch_size=None,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                pin_memory=False,
                collate_fn=_identity_collate,
            )
        else:
            shard_iter = _iter_shard_records_single_process(
                records,
                surface_root=self.surface.root,
                verify_sha256=verify_sha256,
            )
        for record, payload in shard_iter:
            unit_count = int(record.unit_count)
            order = torch.arange(unit_count)
            if shuffle_units and unit_count:
                order = order[torch.randperm(unit_count)]
            for start in range(0, unit_count, batch_size):
                index = order[start : start + batch_size]
                yield _slice_payload(payload, index)
            del payload


class _SurfaceShardDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ModelDataShardRecord],
        *,
        surface_root: Path,
        verify_sha256: bool,
    ) -> None:
        self.records = tuple(records)
        self.surface_root = Path(surface_root)
        self.verify_sha256 = bool(verify_sha256)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[ModelDataShardRecord, dict[str, Any]]:
        record = self.records[index]
        return record, _load_surface_shard_payload(
            record,
            surface_root=self.surface_root,
            verify_sha256=self.verify_sha256,
        )


def _identity_collate(item):
    return item


def _iter_shard_records_single_process(
    records: Sequence[ModelDataShardRecord],
    *,
    surface_root: Path,
    verify_sha256: bool,
) -> Iterator[tuple[ModelDataShardRecord, dict[str, Any]]]:
    for record in records:
        yield record, _load_surface_shard_payload(
            record,
            surface_root=surface_root,
            verify_sha256=verify_sha256,
        )


def _load_surface_shard_payload(
    record: ModelDataShardRecord,
    *,
    surface_root: Path,
    verify_sha256: bool,
) -> dict[str, Any]:
    path = Path(surface_root) / record.relative_path
    if verify_sha256 and sha256_file(path) != record.sha256:
        raise ModelDataSurfaceError(f"surface shard checksum mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelDataSurfaceError(f"surface shard could not be loaded: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelDataSurfaceError("surface shard payload must be a dict.")
    _validate_tensor_shard_payload(payload, path=path)
    return payload


def _validate_worker_options(
    *,
    num_workers: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> None:
    if not isinstance(num_workers, int) or isinstance(num_workers, bool) or num_workers < 0:
        raise ModelDataSurfaceError("num_workers must be a non-negative integer.")
    if not isinstance(persistent_workers, bool):
        raise ModelDataSurfaceError("persistent_workers must be a boolean.")
    if persistent_workers and num_workers <= 0:
        raise ModelDataSurfaceError("persistent_workers=True requires num_workers > 0.")
    if prefetch_factor is not None:
        if not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool) or prefetch_factor <= 0:
            raise ModelDataSurfaceError("prefetch_factor must be a positive integer when provided.")
        if num_workers <= 0:
            raise ModelDataSurfaceError("prefetch_factor is only valid when num_workers > 0.")


def load_model_data_surface(root: Path) -> ModelDataSurface:
    surface_root = Path(root)
    metadata_path = surface_root / "metadata.json"
    manifest_path = surface_root / "manifest.jsonl"
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelDataSurfaceError(f"malformed surface metadata JSON: {metadata_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelDataSurfaceError("surface metadata JSON must be an object.")
    metadata = metadata_from_dict(raw)
    return ModelDataSurface(
        root=surface_root,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        shard_root=surface_root / "shards",
        metadata=metadata,
    )


def _validate_tensor_shard_payload(payload: dict[str, Any], *, path: Path) -> None:
    if "metadata" not in payload or not isinstance(payload["metadata"], dict):
        raise ModelDataSurfaceError(f"surface shard is missing metadata dict: {path}")
    tensor_field_count = 0
    for key, value in payload.items():
        if key == "metadata" or value is None:
            continue
        if isinstance(value, torch.Tensor):
            tensor_field_count += 1
            continue
        raise ModelDataSurfaceError(
            f"surface shard field {key!r} is not tensor/array-heavy: {type(value).__name__}"
        )
    if tensor_field_count <= 0:
        raise ModelDataSurfaceError(f"surface shard contains no tensor fields: {path}")


def _slice_payload(payload: dict[str, Any], index: torch.Tensor) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "metadata" or value is None:
            batch[key] = value
        elif isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] >= int(index.max().item()) + 1:
            batch[key] = value.index_select(0, index)
        else:
            batch[key] = value
    return batch


__all__ = ["ModelDataSurfaceReader", "load_model_data_surface"]
