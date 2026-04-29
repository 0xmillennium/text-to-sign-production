"""Small data-pipeline helpers that are not path/root resolution policy."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median


def remove_stale_split_files(
    root: Path,
    *,
    filename_template: str,
    requested_splits: tuple[str, ...],
    all_splits: tuple[str, ...],
) -> None:
    """Remove split-specific files for unrequested splits after a successful subset run."""

    for split in all_splits:
        if split in requested_splits:
            continue
        path = root / filename_template.format(split=split)
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"Expected split output path to be a file: {path}")
        path.unlink()


def ensure_directory(path: Path) -> None:
    """Create a directory tree if it does not exist yet."""

    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for reports."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def summarize_numbers(values: Iterable[int | float]) -> dict[str, float | int | None]:
    """Return a compact numeric summary for reporting."""

    collected = [float(value) for value in values]
    if not collected:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

    return {
        "count": len(collected),
        "min": min(collected),
        "max": max(collected),
        "mean": mean(collected),
        "median": median(collected),
    }
