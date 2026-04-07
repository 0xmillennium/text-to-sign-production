"""Small helpers shared across the Sprint 2 pipeline modules."""

from __future__ import annotations

from collections.abc import Iterable
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

    processed_samples_root = (
        REPO_ROOT.resolve() / "data" / "processed" / "v1" / "samples"
    ).resolve()
    resolved = resolve_repo_path(candidate)
    if not resolved.is_relative_to(processed_samples_root):
        raise ValueError(
            f"Processed sample_path must stay under data/processed/v1/samples: {normalized_value}"
        )

    relative_path = resolved.relative_to(processed_samples_root)
    if len(relative_path.parts) != 2:
        raise ValueError(
            "Processed sample_path must follow "
            "data/processed/v1/samples/<split>/<sample_id>.npz: "
            f"{normalized_value}"
        )

    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"Processed sample_path must resolve to a file: {normalized_value}")
    return resolved


def validate_processed_sample_path(
    path: Path | str,
    *,
    split: str,
    sample_id: str,
) -> Path:
    """Resolve and validate a processed sample path against its manifest identity."""

    resolved = resolve_processed_sample_path(path)
    relative_path = resolved.relative_to(
        (REPO_ROOT.resolve() / "data" / "processed" / "v1" / "samples").resolve()
    )
    relative_split, filename = relative_path.parts

    if relative_split != split:
        raise ValueError(
            f"Processed sample_path split does not match manifest split {split}: {path}"
        )

    expected_filename = f"{sample_id}.npz"
    if filename != expected_filename:
        raise ValueError(
            f"Processed sample_path filename does not match manifest sample_id {sample_id}: {path}"
        )

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
