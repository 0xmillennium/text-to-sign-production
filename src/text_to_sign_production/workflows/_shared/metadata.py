"""Domain-free portable metadata validation helpers for workflow artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from json import JSONDecodeError
from pathlib import Path
from typing import Any, cast

from text_to_sign_production.core.paths import repo_relative_path


def read_json_object(
    path: Path,
    *,
    error_factory: Callable[[str], Exception] = RuntimeError,
) -> dict[str, Any]:
    """Read a JSON object from disk, failing closed for non-object payloads."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise error_factory(f"Expected JSON object in {path}.")
    return cast(dict[str, Any], payload)


def write_json_object(path: Path, payload: object) -> None:
    """Write a JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL records without parsing the file."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object records from a JSONL file."""

    yield from _iter_jsonl(path, error_factory=ValueError)


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read a whole JSONL file into memory."""

    return list(iter_jsonl_records(path))


def portable_repo_path(
    path: Path,
    *,
    project_root: Path,
    error_factory: Callable[[str], Exception] = ValueError,
    message: str | None = None,
) -> str:
    """Return a project-root-relative path or raise without falling back to CWD."""

    try:
        return repo_relative_path(path, repo_root=project_root)
    except ValueError as exc:
        raise error_factory(message or f"Path must stay under project_root: {path}") from exc


def verify_portable_json_file(
    path: Path,
    *,
    project_root: Path,
    error_factory: Callable[[str], Exception] = RuntimeError,
) -> None:
    """Verify a JSON metadata file contains no runtime absolute path strings."""

    assert_no_absolute_metadata_path_strings(
        read_json_object(path, error_factory=error_factory),
        artifact_path=path,
        project_root=project_root,
        error_factory=error_factory,
    )


def verify_portable_jsonl_file(
    path: Path,
    *,
    project_root: Path,
    error_factory: Callable[[str], Exception] = RuntimeError,
) -> None:
    """Verify a JSONL metadata file contains no runtime absolute path strings."""

    for record in _iter_jsonl(path, error_factory=error_factory):
        assert_no_absolute_metadata_path_strings(
            record,
            artifact_path=path,
            project_root=project_root,
            error_factory=error_factory,
        )


def verify_portable_text_file(
    path: Path,
    *,
    project_root: Path,
    error_factory: Callable[[str], Exception] = RuntimeError,
) -> None:
    """Verify a text metadata file contains no obvious runtime absolute paths."""

    text = path.read_text(encoding="utf-8")
    project_root_text = project_root.as_posix()
    if "/content/" in text or project_root_text in text:
        raise error_factory(f"Metadata text contains a runtime absolute path: {path}")


def assert_no_absolute_metadata_path_strings(
    value: Any,
    *,
    artifact_path: Path,
    project_root: Path,
    error_factory: Callable[[str], Exception] = RuntimeError,
) -> None:
    """Fail if nested metadata contains absolute runtime path strings."""

    if isinstance(value, dict):
        for item in value.values():
            assert_no_absolute_metadata_path_strings(
                item,
                artifact_path=artifact_path,
                project_root=project_root,
                error_factory=error_factory,
            )
        return
    if isinstance(value, list):
        for item in value:
            assert_no_absolute_metadata_path_strings(
                item,
                artifact_path=artifact_path,
                project_root=project_root,
                error_factory=error_factory,
            )
        return
    if isinstance(value, str) and _looks_like_forbidden_absolute_path(
        value,
        project_root=project_root,
    ):
        raise error_factory(
            f"Metadata file contains a runtime absolute path string: {artifact_path}: {value!r}"
        )


def _looks_like_forbidden_absolute_path(value: str, *, project_root: Path) -> bool:
    if "/content/" in value or project_root.as_posix() in value:
        return True
    if Path(value).is_absolute():
        return True
    return len(value) >= 3 and value[1:3] in {":\\", ":/"}


def _iter_jsonl(
    path: Path,
    *,
    error_factory: Callable[[str], Exception],
) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except JSONDecodeError as exc:
                raise error_factory(
                    f"JSONL record in {path} line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise error_factory(
                    f"Expected object record in {path} line {line_number}, "
                    f"got {type(payload).__name__}."
                )
            yield cast(dict[str, Any], payload)


__all__ = [
    "assert_no_absolute_metadata_path_strings",
    "count_jsonl_records",
    "iter_jsonl_records",
    "portable_repo_path",
    "read_jsonl_records",
    "read_json_object",
    "verify_portable_json_file",
    "verify_portable_jsonl_file",
    "verify_portable_text_file",
    "write_json_object",
]
