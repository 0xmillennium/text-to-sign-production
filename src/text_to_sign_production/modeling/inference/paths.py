"""Path formatting helpers for Sprint 3 inference artifacts."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.paths import repo_relative_path


def portable_path(path: Path) -> str:
    """Return a repo-relative POSIX path when possible, otherwise an absolute path."""

    try:
        relative_path = repo_relative_path(path)
    except ValueError:
        return path.resolve().as_posix()

    if relative_path is None:
        raise ValueError("portable_path requires a non-null path.")
    return relative_path


def portable_optional_string_path(value: object) -> object:
    """Normalize optional string paths from JSON while preserving other metadata."""

    if not isinstance(value, str) or not value.strip():
        return value

    path = Path(value)
    if path.is_absolute():
        return portable_path(path)
    return path.as_posix()
