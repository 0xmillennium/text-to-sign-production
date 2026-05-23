"""Contracts for provider-neutral tensor data surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_DATA_SURFACE_SCHEMA_VERSION = "model.data_surface.v1"


class ModelDataSurfaceError(ValueError):
    """Raised when a model data surface violates its persistence contract."""


@dataclass(frozen=True, slots=True)
class ModelDataSurfaceMetadata:
    schema_version: str
    provider_key: str
    surface_kind: str
    split: str
    manifest_family: str
    source_manifest_path: str
    source_manifest_sha256: str
    provider_config_sha256: str
    cache_key: str
    manifest_entry_count: int
    loaded_sample_count: int
    unit_count: int
    shard_count: int
    feature_dim: int | None
    created_at: str
    run_mode: str
    data_version: str

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_DATA_SURFACE_SCHEMA_VERSION:
            raise ModelDataSurfaceError("model data surface schema_version is unsupported.")
        for name in (
            "provider_key",
            "surface_kind",
            "split",
            "manifest_family",
            "source_manifest_path",
            "source_manifest_sha256",
            "provider_config_sha256",
            "cache_key",
            "created_at",
            "run_mode",
            "data_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ModelDataSurfaceError(f"{name} must be non-empty text.")
        for name in (
            "manifest_entry_count",
            "loaded_sample_count",
            "unit_count",
            "shard_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelDataSurfaceError(f"{name} must be a non-negative integer.")
        if self.feature_dim is not None and (
            not isinstance(self.feature_dim, int)
            or isinstance(self.feature_dim, bool)
            or self.feature_dim <= 0
        ):
            raise ModelDataSurfaceError("feature_dim must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelDataShardRecord:
    shard_id: int
    relative_path: str
    sample_count: int
    unit_count: int
    frame_count: int | None
    byte_count: int | None
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.shard_id, int) or isinstance(self.shard_id, bool) or self.shard_id < 0:
            raise ModelDataSurfaceError("shard_id must be a non-negative integer.")
        if not isinstance(self.relative_path, str) or not self.relative_path.strip():
            raise ModelDataSurfaceError("relative_path must be non-empty text.")
        for name in ("sample_count", "unit_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelDataSurfaceError(f"{name} must be a non-negative integer.")
        for name in ("frame_count", "byte_count"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ModelDataSurfaceError(f"{name} must be non-negative when provided.")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ModelDataSurfaceError("sha256 must be a hex digest.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelDataSurface:
    root: Path
    metadata_path: Path
    manifest_path: Path
    shard_root: Path
    metadata: ModelDataSurfaceMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "metadata_path", Path(self.metadata_path))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "shard_root", Path(self.shard_root))
        if not isinstance(self.metadata, ModelDataSurfaceMetadata):
            raise ModelDataSurfaceError("metadata must be ModelDataSurfaceMetadata.")


def metadata_from_dict(record: dict[str, Any]) -> ModelDataSurfaceMetadata:
    return ModelDataSurfaceMetadata(**record)


def shard_record_from_dict(record: dict[str, Any]) -> ModelDataShardRecord:
    return ModelDataShardRecord(**record)


__all__ = [
    "MODEL_DATA_SURFACE_SCHEMA_VERSION",
    "ModelDataShardRecord",
    "ModelDataSurface",
    "ModelDataSurfaceError",
    "ModelDataSurfaceMetadata",
    "metadata_from_dict",
    "shard_record_from_dict",
]
