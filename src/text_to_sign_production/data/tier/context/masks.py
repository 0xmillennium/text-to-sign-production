"""Pure mask helpers for staged active-span derivation."""

from __future__ import annotations

import numpy as np


def remove_short_true_runs(mask: np.ndarray, min_run_length: int) -> np.ndarray:
    """Remove true-runs shorter than ``min_run_length``."""
    source = np.asarray(mask, dtype=np.bool_)
    result = np.array(source, copy=True)
    if min_run_length <= 1:
        return result
    for start, end in _runs(source, value=True):
        if end - start < min_run_length:
            result[start:end] = False
    return result


def bridge_short_false_gaps(mask: np.ndarray, max_gap_length: int) -> np.ndarray:
    """Fill false-gaps between true-runs when the gap is short enough."""
    source = np.asarray(mask, dtype=np.bool_)
    result = np.array(source, copy=True)
    if max_gap_length < 1 or source.size == 0:
        return result
    for start, end in _runs(source, value=False):
        if start == 0 or end == source.size:
            continue
        if bool(source[start - 1]) and bool(source[end]) and end - start <= max_gap_length:
            result[start:end] = True
    return result


def pad_true_runs(mask: np.ndarray, pad: int, support_mask: np.ndarray) -> np.ndarray:
    """Expand true-runs by ``pad`` frames without leaving ``support_mask``."""
    source = np.asarray(mask, dtype=np.bool_)
    support = np.asarray(support_mask, dtype=np.bool_)
    result = np.array(source, copy=True)
    if source.shape != support.shape:
        raise ValueError("mask and support_mask must have matching shapes.")
    if pad < 1:
        return result & support
    result[:] = False
    for start, end in _runs(source, value=True):
        padded_start = max(0, start - pad)
        padded_end = min(source.size, end + pad)
        result[padded_start:padded_end] = True
    return result & support


def mask_bounds(mask: np.ndarray) -> tuple[int, int]:
    """Return inclusive-exclusive bounds for true values, or ``(0, 0)``."""
    indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if indices.size == 0:
        return (0, 0)
    return (int(indices[0]), int(indices[-1]) + 1)


def max_true_run(mask: np.ndarray) -> int:
    """Return the longest true-run length."""
    longest = 0
    for start, end in _runs(np.asarray(mask, dtype=np.bool_), value=True):
        longest = max(longest, end - start)
    return longest


def count_true_runs(mask: np.ndarray) -> int:
    """Return the number of true-runs."""
    return sum(1 for _ in _runs(np.asarray(mask, dtype=np.bool_), value=True))


def _runs(mask: np.ndarray, *, value: bool) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(mask):
        if bool(item) is value:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return tuple(runs)


__all__ = [
    "bridge_short_false_gaps",
    "count_true_runs",
    "mask_bounds",
    "max_true_run",
    "pad_true_runs",
    "remove_short_true_runs",
]
