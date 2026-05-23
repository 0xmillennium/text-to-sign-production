"""Telemetry helpers for tensor shard cache reads and writes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorShardCacheTelemetry:
    cache_enabled: bool
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_read_seconds: float = 0.0
    cache_write_seconds: float = 0.0
    cache_read_units: int = 0
    cache_write_units: int = 0
    cache_read_bytes: int = 0
    cache_write_bytes: int = 0

    def to_metadata(self) -> dict[str, object]:
        return {
            "cache_enabled": self.cache_enabled,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_read_seconds": self.cache_read_seconds,
            "cache_write_seconds": self.cache_write_seconds,
            "cache_read_units_per_second": _rate(self.cache_read_units, self.cache_read_seconds),
            "cache_write_units_per_second": _rate(self.cache_write_units, self.cache_write_seconds),
            "cache_read_bytes_per_second": _rate(self.cache_read_bytes, self.cache_read_seconds),
            "cache_write_bytes_per_second": _rate(self.cache_write_bytes, self.cache_write_seconds),
        }


def cache_read_dominates_runtime(
    *,
    cache_read_seconds: float,
    stage_elapsed_seconds: float,
    threshold: float = 0.50,
) -> bool:
    if stage_elapsed_seconds <= 0:
        return False
    return cache_read_seconds / stage_elapsed_seconds > threshold


def _rate(numerator: int, seconds: float) -> float | None:
    if seconds <= 0.0:
        return None
    return float(numerator) / float(seconds)


__all__ = ["TensorShardCacheTelemetry", "cache_read_dominates_runtime"]
