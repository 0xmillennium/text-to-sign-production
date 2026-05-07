"""Shared primitive analysis helpers for data-layer surfaces."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Domain-neutral numeric summary over a finite value sequence."""

    sample_count: int
    missing_count: int
    unique_value_count: int
    minimum: float | None
    p5: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


def finite_values(values: Iterable[float | int | None]) -> tuple[float, ...]:
    """Return finite numeric values as floats, preserving input order."""
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
    return tuple(finite)


def quantile(sorted_values: Sequence[float], probability: float) -> float | None:
    """Return a linearly interpolated quantile from sorted finite values."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Quantile probability must be within [0, 1], got {probability!r}.")
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def numeric_summary(values: Sequence[float | int | None]) -> NumericSummary:
    """Build a stable numeric summary, treating non-finite and missing values as missing."""
    finite = sorted(finite_values(values))
    missing_count = len(values) - len(finite)
    return NumericSummary(
        sample_count=len(values),
        missing_count=missing_count,
        unique_value_count=len(set(finite)),
        minimum=finite[0] if finite else None,
        p5=quantile(finite, 0.05),
        p25=quantile(finite, 0.25),
        p50=quantile(finite, 0.50),
        p75=quantile(finite, 0.75),
        p95=quantile(finite, 0.95),
        p99=quantile(finite, 0.99),
        maximum=finite[-1] if finite else None,
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a ratio, using 0.0 for an empty denominator."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def threshold_distance(actual: float | int | None, expected: float | int) -> float:
    """Return absolute distance from a threshold, or infinity when actual is missing."""
    if actual is None:
        return math.inf
    return abs(float(actual) - float(expected))


def comparison_passes(actual: float | int | None, expected: float | int, comparison: str) -> bool:
    """Evaluate a simple threshold comparison for analytical pass/fail counts."""
    if actual is None:
        return False
    if comparison == ">=":
        return actual >= expected
    if comparison == "<=":
        return actual <= expected
    raise ValueError(f"Unsupported comparison {comparison!r}.")


def frequency_counts(values: Iterable[K]) -> tuple[tuple[K, int], ...]:
    """Return deterministic frequency counts for naturally sortable keys."""
    counter = Counter(values)
    return tuple((key, counter[key]) for key in sorted(counter))


def grouped(values: Iterable[T], key_fn: Callable[[T], K]) -> tuple[tuple[K, tuple[T, ...]], ...]:
    """Group values by key with deterministic key ordering."""
    groups: dict[K, list[T]] = defaultdict(list)
    for value in values:
        groups[key_fn(value)].append(value)
    return tuple((key, tuple(groups[key])) for key in sorted(groups))


def cooccurrence_counts(
    groups: Iterable[Iterable[K]],
    *,
    include_self: bool = False,
) -> tuple[tuple[K, K, int], ...]:
    """Count deterministic ordered co-occurrences inside each key group."""
    counter: Counter[tuple[K, K]] = Counter()
    for group in groups:
        keys = tuple(sorted(set(group)))
        for left in keys:
            for right in keys:
                if not include_self and left == right:
                    continue
                counter[(left, right)] += 1
    return tuple((left, right, count) for (left, right), count in sorted(counter.items()))


def sorted_mapping_items(mapping: Mapping[K, T]) -> tuple[tuple[K, T], ...]:
    """Return mapping items sorted by key for stable analytical output."""
    return tuple((key, mapping[key]) for key in sorted(mapping))
