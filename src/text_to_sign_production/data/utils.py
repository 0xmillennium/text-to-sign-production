"""Small generic helpers shared across the Sprint 2 pipeline modules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from .constants import REPO_ROOT


def repo_relative_path(path: Path | None) -> str | None:
    """Return a stable repo-relative POSIX path string.

    Raises:
        ValueError: If the resolved path is outside ``REPO_ROOT``.
    """

    if path is None:
        return None
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def resolve_repo_path(path: Path | str) -> Path:
    """Resolve an absolute path or a repo-relative path against the repo root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()

    repo_root = REPO_ROOT.resolve()
    resolved = (repo_root / candidate).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"Repo-relative path escapes repo root: {path}")
    return resolved


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


def write_json(path: Path, payload: Mapping[str, object] | list[object]) -> None:
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
