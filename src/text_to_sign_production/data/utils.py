"""Small helpers local to the Dataset data domain."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import TypeVar

from text_to_sign_production.core.progress import iter_with_progress as _core_iter_with_progress

T = TypeVar("T")


def ensure_directory(path: Path) -> None:
    """Create a directory tree if it does not exist yet."""

    path.mkdir(parents=True, exist_ok=True)


def require_file(path: Path | str, *, label: str) -> Path:
    """Require an existing regular file and return its resolved path."""

    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if resolved.is_dir():
        raise IsADirectoryError(f"{label} is a directory, expected a file: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved.resolve()


def resolve_data_root(data_root: Path | str) -> Path:
    """Resolve the dataset data root without depending on core layout policy."""

    resolved = Path(data_root).expanduser().resolve()
    if resolved.name != "data":
        raise ValueError(f"data_root must point at the project data directory: {resolved}")
    return resolved


def repo_relative_path(path: Path | str, *, data_root: Path | str) -> str:
    """Return a project-root-relative POSIX path for a path under ``data_root.parent``."""

    root = resolve_data_root(data_root).parent
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path must stay under {root}: {resolved}")
    return resolved.relative_to(root).as_posix()


def data_root_relative_path(
    path: Path | str,
    *,
    data_root: Path | str,
    allow_repo_prefix: bool = True,
) -> Path:
    """Return a path relative to the runtime data root."""

    root = resolve_data_root(data_root)
    return resolve_data_root_path(
        path,
        data_root=root,
        allow_repo_prefix=allow_repo_prefix,
    ).relative_to(root)


def manifest_data_path(path: Path, *, data_root: Path | str) -> str:
    """Return the repo-relative ``data/...`` value stored in dataset manifests."""

    return (Path("data") / data_root_relative_path(path, data_root=data_root)).as_posix()


def resolve_data_root_path(
    path: Path | str,
    *,
    data_root: Path | str,
    allow_repo_prefix: bool = True,
) -> Path:
    """Resolve a relative data path under the runtime data root."""

    root = resolve_data_root(data_root)
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"Data path must not contain parent traversal: {path}")
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.is_relative_to(root):
            return resolved
        raise ValueError(f"Data path must stay under {root}: {path}")

    relative = _without_data_prefix(candidate, allow_repo_prefix=allow_repo_prefix)
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Data path must stay under {root}: {path}")
    return resolved


def resolve_manifest_path(
    value: Path | str,
    *,
    data_root: Path | str,
    allowed_root: Path | str | None = None,
    allow_absolute: bool = False,
    allow_repo_prefix: bool = True,
) -> Path:
    """Resolve a manifest path value into a runtime absolute path."""

    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("Manifest path must be a non-empty relative path.")

    root = resolve_data_root(data_root)
    candidate = Path(raw_value).expanduser()
    if ".." in candidate.parts:
        raise ValueError(f"Manifest path must not contain parent traversal: {raw_value}")

    if candidate.is_absolute():
        if not allow_absolute:
            raise ValueError(f"Manifest path must be relative: {raw_value}")
        resolved = candidate.resolve()
    else:
        resolved = resolve_data_root_path(
            candidate,
            data_root=root,
            allow_repo_prefix=allow_repo_prefix,
        )

    if allowed_root is not None:
        boundary = _resolve_allowed_root(allowed_root, data_root=root)
        if not resolved.is_relative_to(boundary):
            raise ValueError(f"Manifest path must stay under {boundary}: {resolved}")
    return resolved


def iter_with_progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "",
    unit: str = "items",
) -> Iterator[T]:
    """Yield an iterable through the project-standard progress API."""

    yield from _core_iter_with_progress(iterable, total=total, desc=desc, unit=unit)


def _without_data_prefix(path: Path, *, allow_repo_prefix: bool) -> Path:
    if not path.parts or path.parts[0] != "data":
        return path
    if not allow_repo_prefix:
        raise ValueError(f"Repo-prefixed data path is not allowed: {path}")
    if len(path.parts) == 1:
        return Path(".")
    return Path(*path.parts[1:])


def _resolve_allowed_root(allowed_root: Path | str, *, data_root: Path) -> Path:
    boundary = Path(allowed_root).expanduser()
    if boundary.is_absolute():
        resolved = boundary.resolve()
        if resolved.is_relative_to(data_root):
            return resolved
        return resolved
    return resolve_data_root_path(boundary, data_root=data_root)


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
