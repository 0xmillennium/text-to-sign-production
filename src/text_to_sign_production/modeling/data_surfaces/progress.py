"""Progress helpers for tensor data surfaces."""

from __future__ import annotations

from pathlib import Path


def resolve_effective_manifest_count(
    *,
    manifest_path: Path,
    limit_samples: int | None,
) -> int:
    """Return the exact manifest total after an optional run-mode limit."""

    count = 0
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    if limit_samples is None:
        return count
    if not isinstance(limit_samples, int) or isinstance(limit_samples, bool) or limit_samples <= 0:
        raise ValueError("limit_samples must be positive when provided.")
    return min(limit_samples, count)


def surface_batch_count(*, unit_count: int, batch_size: int) -> int:
    if not isinstance(unit_count, int) or isinstance(unit_count, bool) or unit_count < 0:
        raise ValueError("unit_count must be a non-negative integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    return (unit_count + batch_size - 1) // batch_size


__all__ = ["resolve_effective_manifest_count", "surface_batch_count"]
