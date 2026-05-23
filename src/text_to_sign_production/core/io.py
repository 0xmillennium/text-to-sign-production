"""Generic text and JSON file writing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Path | tuple["JsonValue", ...] | list["JsonValue"] | Mapping[
    str,
    "JsonValue",
]


def jsonable(value: JsonValue) -> JsonScalar | list[object] | dict[str, object]:
    if isinstance(value, Mapping):
        return {key: jsonable(mapping_value) for key, mapping_value in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_text(path: Path, content: str) -> None:
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    _ensure_path_parent(path)
    path.write_text(content, encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    write_text(path, content)


def write_json(path: Path, value: JsonValue) -> None:
    _ensure_path_parent(path)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ensure_path_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "JsonScalar",
    "JsonValue",
    "jsonable",
    "write_json",
    "write_markdown",
    "write_text",
]
