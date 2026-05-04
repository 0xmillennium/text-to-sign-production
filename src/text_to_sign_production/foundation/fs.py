"""Generic filesystem primitives."""

from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path


class OutputExistsPolicy(StrEnum):
    """Policy for preparing output paths that may already exist."""

    ERROR = "ERROR"
    OVERWRITE = "OVERWRITE"
    SKIP = "SKIP"


def require_file(path: Path) -> Path:
    """Return ``path`` if it exists as a file."""

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file but found a directory: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    return path


def require_dir(path: Path) -> Path:
    """Return ``path`` if it exists as a directory."""

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if needed and return it."""

    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_file(
    path: Path,
    policy: OutputExistsPolicy = OutputExistsPolicy.ERROR,
) -> Path:
    """Prepare a file output path according to an existence policy."""

    if path.exists():
        if path.is_dir():
            raise IsADirectoryError(f"Output file path is a directory: {path}")
        if policy == OutputExistsPolicy.ERROR:
            raise FileExistsError(f"Output file already exists: {path}")
        if policy == OutputExistsPolicy.SKIP:
            ensure_dir(path.parent)
            return path
        if policy == OutputExistsPolicy.OVERWRITE:
            path.unlink()
        else:
            raise ValueError(f"Unsupported output exists policy: {policy!r}")

    ensure_dir(path.parent)
    return path


def prepare_output_dir(
    path: Path,
    policy: OutputExistsPolicy = OutputExistsPolicy.ERROR,
) -> Path:
    """Prepare a directory output path according to an existence policy."""

    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output directory path exists and is not a directory: {path}")
        if policy == OutputExistsPolicy.ERROR:
            raise FileExistsError(f"Output directory already exists: {path}")
        if policy == OutputExistsPolicy.SKIP:
            return path
        if policy == OutputExistsPolicy.OVERWRITE:
            shutil.rmtree(path)
        else:
            raise ValueError(f"Unsupported output exists policy: {policy!r}")

    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_output_file(path: Path) -> Path:
    """Return ``path`` if a producing step created a file."""

    return require_file(path)


def verify_output_dir(path: Path) -> Path:
    """Return ``path`` if a producing step created a directory."""

    return require_dir(path)
