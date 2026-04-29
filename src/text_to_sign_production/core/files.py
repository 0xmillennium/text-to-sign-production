"""Generic filesystem preconditions, output conflict policy, and file hashes."""

from __future__ import annotations

import hashlib
import json
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any

_HASH_CHUNK_SIZE = 8 * 1024 * 1024


class OutputExistsPolicy(StrEnum):
    """Policy for preparing output paths that may already exist."""

    FAIL = "fail"
    OVERWRITE = "overwrite"
    SKIP = "skip"
    RESUME = "resume"


def _path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _policy(policy: OutputExistsPolicy | str) -> OutputExistsPolicy:
    return policy if isinstance(policy, OutputExistsPolicy) else OutputExistsPolicy(policy)


def require_file(path: str | Path, *, label: str) -> Path:
    """Require an existing file and return its normalized path."""

    resolved = _path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if resolved.is_dir():
        raise IsADirectoryError(f"{label} is a directory, expected a file: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved.resolve()


def require_non_empty_file(path: str | Path, *, label: str) -> Path:
    """Require an existing non-empty file and return its normalized path."""

    resolved = require_file(path, label=label)
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {resolved}")
    return resolved


def require_dir(path: str | Path, *, label: str) -> Path:
    """Require an existing directory and return its normalized path."""

    resolved = _path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved.resolve()


def require_dir_contains(path: str | Path, pattern: str, *, label: str) -> Path:
    """Require a directory to contain at least one path matching ``pattern``."""

    resolved = require_dir(path, label=label)
    if not any(resolved.glob(pattern)):
        raise FileNotFoundError(f"{label} contains no matches for {pattern!r}: {resolved}")
    return resolved


def ensure_dir(path: str | Path, *, label: str) -> Path:
    """Create a directory, including parents, and return its normalized path."""

    resolved = _path(path)
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"{label} exists and is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def prepare_output_file(
    path: str | Path,
    *,
    policy: OutputExistsPolicy = OutputExistsPolicy.FAIL,
    label: str,
) -> Path:
    """Prepare an output file path according to the requested conflict policy."""

    resolved = _path(path)
    resolved_policy = _policy(policy)
    if resolved.exists():
        if resolved.is_dir():
            raise IsADirectoryError(f"{label} exists as a directory: {resolved}")
        if resolved_policy == OutputExistsPolicy.FAIL:
            raise FileExistsError(f"{label} already exists: {resolved}")
        if resolved_policy == OutputExistsPolicy.OVERWRITE:
            resolved.unlink()
        elif resolved_policy in {OutputExistsPolicy.SKIP, OutputExistsPolicy.RESUME}:
            return resolved.resolve()
    ensure_dir(resolved.parent, label=f"{label} parent directory")
    return resolved.resolve()


def prepare_output_dir(
    path: str | Path,
    *,
    policy: OutputExistsPolicy = OutputExistsPolicy.FAIL,
    label: str,
) -> Path:
    """Prepare an output directory according to the requested conflict policy."""

    resolved = _path(path)
    resolved_policy = _policy(policy)
    if resolved.exists():
        if not resolved.is_dir():
            raise FileExistsError(f"{label} exists and is not a directory: {resolved}")
        if resolved_policy == OutputExistsPolicy.FAIL:
            raise FileExistsError(f"{label} already exists: {resolved}")
        if resolved_policy == OutputExistsPolicy.OVERWRITE:
            shutil.rmtree(resolved)
        elif resolved_policy in {OutputExistsPolicy.SKIP, OutputExistsPolicy.RESUME}:
            return resolved.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def verify_output_file(
    path: str | Path,
    *,
    label: str,
    non_empty: bool = True,
) -> Path:
    """Verify that an output file was produced."""

    if non_empty:
        return require_non_empty_file(path, label=label)
    return require_file(path, label=label)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a regular file."""

    resolved = require_file(path, label="SHA-256 input file")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-serializable content."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_output_dir(path: str | Path, *, label: str) -> Path:
    """Verify that an output directory was produced."""

    return require_dir(path, label=label)


__all__ = [
    "OutputExistsPolicy",
    "ensure_dir",
    "prepare_output_dir",
    "prepare_output_file",
    "require_dir",
    "require_dir_contains",
    "require_file",
    "require_non_empty_file",
    "sha256_file",
    "sha256_json",
    "verify_output_dir",
    "verify_output_file",
]
