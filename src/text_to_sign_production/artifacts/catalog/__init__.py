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
from text_to_sign_production.artifacts.catalog.enrich import (
    attach_source_metadata,
    attach_timing_metadata,
)
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

__all__ = [
    "SampleHandle",
    "SampleRef",
    "SamplesCatalog",
    "TieredCatalog",
    "TieredSampleHandle",
    "attach_source_metadata",
    "attach_timing_metadata",
    "get_sample",
    "get_tiered_sample",
    "iter_samples",
    "iter_samples_split",
    "iter_tiered",
    "iter_tiered_split",
    "load_dropped_samples_catalog",
    "load_passed_samples_catalog",
    "load_tiered_catalog",
    "validate_samples_catalog",
    "validate_tiered_catalog",
]
