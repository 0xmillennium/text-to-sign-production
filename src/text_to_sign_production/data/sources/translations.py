"""Translation row loading, validation, and normalization into typed records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

from text_to_sign_production.data.sources.types import TranslationRow

EXPECTED_TRANSLATION_COLUMNS: Final[tuple[str, ...]] = (
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
)


def load_translation_rows(path: Path) -> list[TranslationRow]:
    """Read and validate a tab-delimited translation file.

    Validates expected columns and converts raw strings into strongly typed
    ``TranslationRow`` records.
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")
        if first_line == "":
            raise ValueError(f"Translation file {path} is empty.")
        if "\t" not in first_line:
            raise ValueError(f"Translation file {path} is not tab-delimited.")

        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Translation file {path} is empty.")

        observed_columns = tuple(reader.fieldnames)
        if observed_columns != EXPECTED_TRANSLATION_COLUMNS:
            raise ValueError(f"Unexpected translation columns in {path}: {observed_columns}")

        rows: list[TranslationRow] = []
        for row in reader:
            rows.append(
                TranslationRow(
                    video_id=row["VIDEO_ID"],
                    video_name=row["VIDEO_NAME"],
                    sentence_id=row["SENTENCE_ID"],
                    sentence_name=row["SENTENCE_NAME"],
                    start_time=float(row["START_REALIGNED"]),
                    end_time=float(row["END_REALIGNED"]),
                    text=row["SENTENCE"],
                )
            )
        return rows
