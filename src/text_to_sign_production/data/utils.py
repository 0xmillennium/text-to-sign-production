"""Small helpers shared across the Sprint 2 pipeline modules."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

from .constants import REPO_ROOT


def repo_relative_path(path: Path | None) -> str | None:
    """Return a stable repo-relative POSIX path string."""

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


def resolve_processed_sample_path(path: Path | str) -> Path:
    """Resolve and validate a processed sample path stored in a manifest."""

    raw_value = str(path)
    normalized_value = raw_value.strip()
    if not normalized_value:
        raise ValueError("Processed sample_path must be a non-empty repo-relative .npz path.")

    candidate = Path(normalized_value)
    if candidate.is_absolute():
        raise ValueError(f"Processed sample_path must be repo-relative: {normalized_value}")
    if candidate.suffix != ".npz":
        raise ValueError(f"Processed sample_path must end with .npz: {normalized_value}")

    resolved = resolve_repo_path(candidate)
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"Processed sample_path must resolve to a file: {normalized_value}")
    return resolved


def ensure_directory(path: Path) -> None:
    """Create a directory tree if it does not exist yet."""

    path.mkdir(parents=True, exist_ok=True)


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
