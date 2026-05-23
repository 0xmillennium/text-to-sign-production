"""Bounded-memory writer for provider-neutral tensor data surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from text_to_sign_production.modeling.data_surfaces.cache_keys import sha256_file
from text_to_sign_production.modeling.data_surfaces.contracts import (
    MODEL_DATA_SURFACE_SCHEMA_VERSION,
    ModelDataShardRecord,
    ModelDataSurface,
    ModelDataSurfaceError,
    ModelDataSurfaceMetadata,
)


class ModelDataSurfaceWriter:
    """Write tensor shards while keeping only the current shard in memory."""

    def __init__(
        self,
        *,
        root: Path,
        provider_key: str,
        surface_kind: str,
        split: str,
        manifest_family: str,
        source_manifest_path: Path,
        source_manifest_sha256: str,
        provider_config_sha256: str,
        cache_key: str,
        manifest_entry_count: int,
        run_mode: str,
        data_version: str,
        max_units_per_shard: int = 50_000,
        max_source_samples_per_shard: int = 512,
    ) -> None:
        if max_units_per_shard <= 0 or max_source_samples_per_shard <= 0:
            raise ModelDataSurfaceError("shard limits must be positive.")
        self.root = Path(root)
        self.shard_root = self.root / "shards"
        self.metadata_path = self.root / "metadata.json"
        self.manifest_path = self.root / "manifest.jsonl"
        self.provider_key = provider_key
        self.surface_kind = surface_kind
        self.split = split
        self.manifest_family = manifest_family
        self.source_manifest_path = Path(source_manifest_path)
        self.source_manifest_sha256 = source_manifest_sha256
        self.provider_config_sha256 = provider_config_sha256
        self.cache_key = cache_key
        self.manifest_entry_count = int(manifest_entry_count)
        self.run_mode = run_mode
        self.data_version = data_version
        self.max_units_per_shard = int(max_units_per_shard)
        self.max_source_samples_per_shard = int(max_source_samples_per_shard)
        self._fields: dict[str, list[Any]] = {}
        self._sample_count = 0
        self._unit_count = 0
        self._loaded_sample_count = 0
        self._total_units = 0
        self._total_frames = 0
        self._feature_dim: int | None = None
        self._records: list[ModelDataShardRecord] = []
        self._closed = False
        self.shard_root.mkdir(parents=True, exist_ok=True)

    def append_units(
        self,
        fields: Mapping[str, Any],
        *,
        sample_count: int = 0,
        frame_count: int | None = None,
    ) -> None:
        """Append one source/sample worth of tensor-like fields."""

        if self._closed:
            raise ModelDataSurfaceError("cannot append to a closed surface writer.")
        normalized = {key: _numpy_or_scalar(value) for key, value in fields.items()}
        unit_count = _unit_count(normalized)
        if unit_count <= 0:
            return
        feature_dim = _feature_dim(normalized)
        if feature_dim is not None:
            if self._feature_dim is None:
                self._feature_dim = feature_dim
            elif self._feature_dim != feature_dim:
                raise ModelDataSurfaceError("all appended values must share feature_dim.")
        if self._unit_count and (
            self._unit_count + unit_count > self.max_units_per_shard
            or self._sample_count + sample_count > self.max_source_samples_per_shard
        ):
            self.flush()
        for key, value in normalized.items():
            self._fields.setdefault(key, []).append(value)
        self._unit_count += unit_count
        self._sample_count += int(sample_count)
        self._loaded_sample_count += int(sample_count)
        self._total_units += unit_count
        if frame_count is not None:
            self._total_frames += int(frame_count)

    def flush(self) -> None:
        if not self._fields:
            return
        shard_id = len(self._records)
        relative_path = f"shards/shard_{shard_id:05d}.pt"
        shard_path = self.root / relative_path
        payload = {key: _to_tensor_or_object(values) for key, values in self._fields.items()}
        payload.setdefault("metadata", {})
        if isinstance(payload["metadata"], dict):
            payload["metadata"].update(
                {
                    "schema_version": MODEL_DATA_SURFACE_SCHEMA_VERSION,
                    "provider_key": self.provider_key,
                    "surface_kind": self.surface_kind,
                    "split": self.split,
                    "shard_id": shard_id,
                    "sample_count": self._sample_count,
                    "unit_count": self._unit_count,
                }
            )
        torch.save(payload, shard_path)
        record = ModelDataShardRecord(
            shard_id=shard_id,
            relative_path=relative_path,
            sample_count=self._sample_count,
            unit_count=self._unit_count,
            frame_count=self._total_frames if self._total_frames else None,
            byte_count=shard_path.stat().st_size,
            sha256=sha256_file(shard_path),
        )
        self._records.append(record)
        self._fields.clear()
        self._sample_count = 0
        self._unit_count = 0
        self._total_frames = 0

    def close(self) -> ModelDataSurface:
        self.flush()
        metadata = ModelDataSurfaceMetadata(
            schema_version=MODEL_DATA_SURFACE_SCHEMA_VERSION,
            provider_key=self.provider_key,
            surface_kind=self.surface_kind,
            split=self.split,
            manifest_family=self.manifest_family,
            source_manifest_path=str(self.source_manifest_path),
            source_manifest_sha256=self.source_manifest_sha256,
            provider_config_sha256=self.provider_config_sha256,
            cache_key=self.cache_key,
            manifest_entry_count=self.manifest_entry_count,
            loaded_sample_count=self._loaded_sample_count,
            unit_count=self._total_units,
            shard_count=len(self._records),
            feature_dim=self._feature_dim,
            created_at=datetime.now(UTC).isoformat(),
            run_mode=self.run_mode,
            data_version=self.data_version,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(metadata.to_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(
                    json.dumps(
                        record.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        self._closed = True
        return ModelDataSurface(
            root=self.root,
            metadata_path=self.metadata_path,
            manifest_path=self.manifest_path,
            shard_root=self.shard_root,
            metadata=metadata,
        )


def _numpy_or_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list | tuple):
        return np.asarray(value)
    if isinstance(value, int | float | bool | str):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise ModelDataSurfaceError(f"surface field is not tensor/array-like: {type(value).__name__}")


def _unit_count(fields: Mapping[str, Any]) -> int:
    for key, value in fields.items():
        if key == "metadata" or value is None or isinstance(value, Mapping):
            continue
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if array.ndim >= 1:
            return int(array.shape[0])
    raise ModelDataSurfaceError("surface append requires at least one non-empty array field.")


def _feature_dim(fields: Mapping[str, Any]) -> int | None:
    for key in ("values", "latent_values", "target_values"):
        value = fields.get(key)
        if value is None:
            continue
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if array.ndim >= 2:
            return int(array.shape[-1])
    return None


def _to_tensor_or_object(values: list[Any]) -> Any:
    first = values[0]
    if first is None:
        return None
    if isinstance(first, Mapping):
        merged: dict[str, Any] = {}
        for value in values:
            merged.update(dict(value))
        return merged
    if isinstance(first, str):
        return list(values)
    if isinstance(first, torch.Tensor):
        return torch.cat([value.detach().cpu() for value in values], dim=0)
    arrays = [np.asarray(value) for value in values]
    if arrays[0].ndim == 0:
        return torch.as_tensor(np.asarray(arrays))
    return torch.as_tensor(np.concatenate(arrays, axis=0))


__all__ = ["ModelDataSurfaceWriter"]
