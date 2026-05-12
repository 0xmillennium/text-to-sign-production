"""Public surface for logical artifact catalogs."""

from __future__ import annotations

from text_to_sign_production.artifacts.catalog.catalog import (
    get_sample,
    get_tiered_sample,
    iter_samples,
    iter_samples_split,
    iter_tiered,
    iter_tiered_split,
    load_dropped_samples_catalog,
    load_passed_samples_catalog,
    load_tiered_catalog,
)
from text_to_sign_production.artifacts.catalog.types import (
    CatalogMetadata,
    CatalogMetadataValue,
    SampleHandle,
    SampleManifestProjection,
    SampleRef,
    SamplesCatalog,
    TieredCatalog,
    TieredSampleHandle,
)

__all__ = [
    "CatalogMetadata",
    "CatalogMetadataValue",
    "SampleHandle",
    "SampleManifestProjection",
    "SampleRef",
    "SamplesCatalog",
    "TieredCatalog",
    "TieredSampleHandle",
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
