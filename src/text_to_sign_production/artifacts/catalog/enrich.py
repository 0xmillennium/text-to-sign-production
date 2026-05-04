"""Optional enrichment hooks for sample catalogs."""

from __future__ import annotations

from dataclasses import replace

from text_to_sign_production.artifacts.catalog.types import (
    SampleHandle,
    SampleRef,
    SamplesCatalog,
)


def attach_timing_metadata(
    catalog: SamplesCatalog,
    timing_by_ref: dict[SampleRef, object],
) -> SamplesCatalog:
    """Return a new sample catalog with externally supplied timing metadata attached."""

    items: dict[SampleRef, SampleHandle] = {}
    for ref, handle in catalog.items.items():
        items[ref] = (
            replace(handle, timing_metadata=timing_by_ref[ref]) if ref in timing_by_ref else handle
        )
    return SamplesCatalog(status=catalog.status, items=items)


def attach_source_metadata(
    catalog: SamplesCatalog,
    source_by_ref: dict[SampleRef, object],
) -> SamplesCatalog:
    """Return a new sample catalog with externally supplied source metadata attached."""

    items: dict[SampleRef, SampleHandle] = {}
    for ref, handle in catalog.items.items():
        items[ref] = (
            replace(handle, source_metadata=source_by_ref[ref]) if ref in source_by_ref else handle
        )
    return SamplesCatalog(status=catalog.status, items=items)


__all__ = [
    "attach_source_metadata",
    "attach_timing_metadata",
]
