"""Provider-neutral sharded tensor data surfaces."""

from __future__ import annotations

from text_to_sign_production.modeling.data_surfaces.cache_keys import (
    model_data_surface_cache_key,
    sha256_file,
    sha256_json_payload,
)
from text_to_sign_production.modeling.data_surfaces.contracts import (
    MODEL_DATA_SURFACE_SCHEMA_VERSION,
    ModelDataShardRecord,
    ModelDataSurface,
    ModelDataSurfaceError,
    ModelDataSurfaceMetadata,
)
from text_to_sign_production.modeling.data_surfaces.progress import (
    resolve_effective_manifest_count,
    surface_batch_count,
)
from text_to_sign_production.modeling.data_surfaces.reader import (
    ModelDataSurfaceReader,
    load_model_data_surface,
)
from text_to_sign_production.modeling.data_surfaces.telemetry import (
    TensorShardCacheTelemetry,
    cache_read_dominates_runtime,
)
from text_to_sign_production.modeling.data_surfaces.writer import ModelDataSurfaceWriter

__all__ = [
    "MODEL_DATA_SURFACE_SCHEMA_VERSION",
    "ModelDataShardRecord",
    "ModelDataSurface",
    "ModelDataSurfaceError",
    "ModelDataSurfaceMetadata",
    "ModelDataSurfaceReader",
    "ModelDataSurfaceWriter",
    "TensorShardCacheTelemetry",
    "cache_read_dominates_runtime",
    "load_model_data_surface",
    "model_data_surface_cache_key",
    "resolve_effective_manifest_count",
    "sha256_file",
    "sha256_json_payload",
    "surface_batch_count",
]
