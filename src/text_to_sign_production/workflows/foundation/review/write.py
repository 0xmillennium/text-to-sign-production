from __future__ import annotations

import json
from pathlib import Path

from text_to_sign_production.workflows.foundation.review.markdown import JsonValue, jsonable


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
