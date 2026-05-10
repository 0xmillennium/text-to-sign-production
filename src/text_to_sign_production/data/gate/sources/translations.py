"""Translation-source truth construction."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from text_to_sign_production.data.gate.sources.types import TranslationSourceRecord

EXPECTED_TRANSLATION_COLUMNS: Final[tuple[str, ...]] = (
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
)


def build_translation_record(
    record: Mapping[str, str],
    *,
    canonical_text_column: str,
) -> TranslationSourceRecord:
    """Build a translation source record from a structurally complete mapping."""
    canonical_text = _required_column_text(
        record,
        canonical_text_column,
        "canonical normalized text",
    )
    return TranslationSourceRecord(
        video_id=record["VIDEO_ID"].strip(),
        video_name=record["VIDEO_NAME"].strip(),
        sentence_id=record["SENTENCE_ID"].strip(),
        sentence_name=record["SENTENCE_NAME"].strip(),
        start_time=float(record["START_REALIGNED"]),
        end_time=float(record["END_REALIGNED"]),
        text=record["SENTENCE"].strip(),
        canonical_normalized_text=canonical_text,
    )


def load_translation_records(
    path: Path,
    *,
    canonical_text_column: str,
) -> tuple[TranslationSourceRecord, ...]:
    """Load tab-delimited translation source records."""
    canonical_text_column = _canonical_column_name(canonical_text_column)
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
        missing_columns = tuple(
            column
            for column in (*EXPECTED_TRANSLATION_COLUMNS, canonical_text_column)
            if column not in observed_columns
        )
        if missing_columns:
            raise ValueError(
                f"Translation file {path} is missing required columns: {missing_columns}."
            )

        rows: list[TranslationSourceRecord] = []
        for row in reader:
            required_columns = (*EXPECTED_TRANSLATION_COLUMNS, canonical_text_column)
            if any(row[column] is None for column in required_columns):
                raise ValueError(f"Translation file {path} contains an incomplete row.")
            rows.append(
                build_translation_record(
                    {key: str(value) for key, value in row.items()},
                    canonical_text_column=canonical_text_column,
                )
            )
        return tuple(rows)


def validate_translation_columns(
    path: Path,
    *,
    canonical_text_column: str,
) -> None:
    """Validate translation TSV structure before expensive samples processing begins."""
    canonical_text_column = _canonical_column_name(canonical_text_column)
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")
        if first_line == "":
            raise ValueError(f"Translation file {path} is empty.")
        if "\t" not in first_line:
            raise ValueError(f"Translation file {path} is not tab-delimited.")
        observed_columns = tuple(first_line.split("\t"))
        missing_columns = tuple(
            column
            for column in (*EXPECTED_TRANSLATION_COLUMNS, canonical_text_column)
            if column not in observed_columns
        )
        if missing_columns:
            raise ValueError(
                f"Translation file {path} is missing required columns: {missing_columns}."
            )

        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=2):
            required_columns = (*EXPECTED_TRANSLATION_COLUMNS, canonical_text_column)
            if any(row[column] is None for column in required_columns):
                raise ValueError(
                    f"Translation file {path} contains an incomplete row at line {line_number}."
                )
            canonical_text = row[canonical_text_column]
            if not isinstance(canonical_text, str) or not canonical_text.strip():
                raise ValueError(
                    f"Translation file {path} has blank canonical text in column "
                    f"{canonical_text_column!r} at line {line_number}."
                )


def _canonical_column_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("canonical_text_column must be a non-blank string.")
    return value.strip()


def _required_column_text(
    record: Mapping[str, str],
    column: str,
    label: str,
) -> str:
    value = record[column]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} column {column!r} must be non-blank.")
    return value


__all__ = [
    "EXPECTED_TRANSLATION_COLUMNS",
    "build_translation_record",
    "load_translation_records",
    "validate_translation_columns",
]
