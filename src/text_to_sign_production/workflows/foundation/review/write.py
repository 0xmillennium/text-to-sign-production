from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from text_to_sign_production.core.progress import ProgressTaskHandle
from text_to_sign_production.workflows.foundation.review.markdown import jsonable


def write_text(path: Path, content: str) -> None:
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    _ensure_path_parent(path)
    path.write_text(content, encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    write_text(path, content)


def write_json(path: Path, value: object) -> None:
    _ensure_path_parent(path)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    progress: ProgressTaskHandle | None = None,
) -> None:
    _ensure_path_parent(path)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True))
            file.write("\n")
            if progress is not None:
                progress.advance()


def _ensure_path_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
