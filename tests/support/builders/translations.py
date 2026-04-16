"""Translation TSV builders for tiny How2Sign-like test inputs."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.data.constants import EXPECTED_TRANSLATION_COLUMNS


def translation_row(
    *,
    split: str = "train",
    sentence_name: str | None = None,
    sentence_index: int = 0,
    text: str | None = None,
    start_time: str = "0.0",
    end_time: str = "0.5",
) -> dict[str, str]:
    resolved_sentence_name = sentence_name or f"{split}_sample_0-1-rgb_front"
    return {
        "VIDEO_ID": f"{split}_video",
        "VIDEO_NAME": f"{split}_video-1-rgb_front",
        "SENTENCE_ID": f"{split}_video_{sentence_index}",
        "SENTENCE_NAME": resolved_sentence_name,
        "START_REALIGNED": start_time,
        "END_REALIGNED": end_time,
        "SENTENCE": text if text is not None else f"{split} text",
    }


def write_translation_file(path: Path, rows: list[dict[str, str]]) -> None:
    header = "\t".join(EXPECTED_TRANSLATION_COLUMNS)
    lines = [header]
    for row in rows:
        lines.append("\t".join(row[column] for column in EXPECTED_TRANSLATION_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
