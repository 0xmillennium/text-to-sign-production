"""Deterministic IO helpers for evaluation protocol artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def write_json(path: str | Path, document: Mapping[str, object]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(document), sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")
    return output_path


def write_text(path: str | Path, text: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


__all__ = ["write_json", "write_jsonl", "write_text"]
