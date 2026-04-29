"""Path formatting helpers for M0 inference artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath


def portable_path(path: Path, *, path_formatter: Callable[[Path], str]) -> str:
    """Return a caller-formatted artifact metadata path."""

    return path_formatter(path)


def portable_optional_string_path(
    value: object,
    *,
    path_formatter: Callable[[Path], str],
) -> object:
    """Normalize optional string paths from JSON while preserving other metadata."""

    if not isinstance(value, str):
        return value

    if value == "":
        return value

    stripped_value = value.strip()
    if not stripped_value:
        raise ValueError("Artifact metadata path strings must not be blank after stripping.")

    path = Path(stripped_value)
    if path.is_absolute():
        return portable_path(path, path_formatter=path_formatter)
    return _portable_relative_metadata_path(stripped_value)


def _portable_relative_metadata_path(value: str) -> str:
    if "\\" in value:
        raise ValueError(f"Artifact metadata paths must use POSIX '/' separators, got: {value!r}")

    relative_path = PurePosixPath(value)
    normalized = relative_path.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"Artifact metadata path must identify a relative file path: {value!r}")
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            "Artifact metadata paths must be project-root-relative POSIX paths "
            f"without parent traversal, got: {value!r}"
        )
    return normalized
