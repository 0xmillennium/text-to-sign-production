"""Helpers for reading and writing JSONL manifests."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
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


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object record in {path}, got {type(payload).__name__}.")
            yield payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a whole JSONL file into memory."""

    return list(iter_jsonl(path))


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Write a JSON file with stable formatting."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
