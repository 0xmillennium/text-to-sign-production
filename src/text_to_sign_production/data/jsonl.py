"""JSONL reading and writing helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .utils import ensure_directory


def _serialize_record(record: Any) -> str:
    if hasattr(record, "to_record"):
        payload = record.to_record()
    else:
        payload = record
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, records: list[Any]) -> None:
    """Write a complete JSONL file."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_serialize_record(record))
            handle.write("\n")


def count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL records without parsing the file."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl_with_line_numbers(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield JSON object records with their 1-based source line numbers."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL record in {path} line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected object record in {path} line {line_number}, "
                    f"got {type(payload).__name__}."
                )
            yield line_number, payload


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    for _, payload in iter_jsonl_with_line_numbers(path):
        yield payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a whole JSONL file into memory."""

    return list(iter_jsonl(path))
