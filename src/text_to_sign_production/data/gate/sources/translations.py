"""Translation-source truth construction."""

from __future__ import annotations

import csv
from collections import Counter
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
) -> TranslationSourceRecord:
    """Build a translation source record from a structurally complete mapping."""
    text = _required_column_text(record, "SENTENCE", "sentence")
    return TranslationSourceRecord(
        video_id=record["VIDEO_ID"].strip(),
        video_name=record["VIDEO_NAME"].strip(),
        sentence_id=record["SENTENCE_ID"].strip(),
        sentence_name=record["SENTENCE_NAME"].strip(),
        start_time=float(record["START_REALIGNED"]),
        end_time=float(record["END_REALIGNED"]),
        text=text,
    )


def load_translation_records(path: Path) -> tuple[TranslationSourceRecord, ...]:
    """Load tab-delimited translation source records."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")
        if first_line == "":
            raise ValueError(f"Translation file {path} is empty.")
        delimiter = _detect_delimiter(path, first_line)

        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Translation file {path} is empty.")
        observed_columns = tuple(reader.fieldnames)
        _validate_observed_columns(path, observed_columns)

        rows: list[TranslationSourceRecord] = []
        for line_number, row in enumerate(reader, start=2):
            _validate_row(path, line_number, row)
            rows.append(
                build_translation_record(
                    {column: str(row[column]) for column in EXPECTED_TRANSLATION_COLUMNS}
                )
            )
        _validate_unique_physical_sample_ids(path, tuple(rows))
        return tuple(rows)


def validate_translation_columns(path: Path) -> None:
    """Validate translation TSV structure before expensive samples processing begins."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().rstrip("\r\n")
        if first_line == "":
            raise ValueError(f"Translation file {path} is empty.")
        delimiter = _detect_delimiter(path, first_line)
        observed_columns = tuple(next(csv.reader([first_line], delimiter=delimiter)))
        _validate_observed_columns(path, observed_columns)

        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        for line_number, row in enumerate(reader, start=2):
            _validate_row(path, line_number, row)


def _detect_delimiter(path: Path, first_line: str) -> str:
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    raise ValueError(f"Translation file {path} is not comma- or tab-delimited.")


def _validate_observed_columns(path: Path, observed_columns: tuple[str, ...]) -> None:
    if observed_columns != EXPECTED_TRANSLATION_COLUMNS:
        missing = sorted(set(EXPECTED_TRANSLATION_COLUMNS).difference(observed_columns))
        extra = sorted(set(observed_columns).difference(EXPECTED_TRANSLATION_COLUMNS))
        raise ValueError(
            f"Translation file {path} columns must match the source contract exactly: "
            f"expected={EXPECTED_TRANSLATION_COLUMNS}, observed={observed_columns}, "
            f"missing={missing}, extra={extra}."
        )


def _validate_row(
    path: Path,
    line_number: int,
    row: Mapping[str, str | None],
) -> None:
    if None in row:
        raise ValueError(
            f"Translation file {path} contains an extra field at line {line_number}."
        )
    if any(row[column] is None for column in EXPECTED_TRANSLATION_COLUMNS):
        raise ValueError(
            f"Translation file {path} contains an incomplete row at line {line_number}."
        )
    sentence = row["SENTENCE"]
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError(
            f"Translation file {path} has blank sentence text at line {line_number}."
        )
    start = _parse_time(row["START_REALIGNED"], "START_REALIGNED", path, line_number)
    end = _parse_time(row["END_REALIGNED"], "END_REALIGNED", path, line_number)
    if start > end:
        raise ValueError(
            f"Translation file {path} has START_REALIGNED after END_REALIGNED "
            f"at line {line_number}."
        )


def _parse_time(value: str | None, column: str, path: Path, line_number: int) -> float:
    if value is None:
        raise ValueError(
            f"Translation file {path} has missing {column} at line {line_number}."
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Translation file {path} has unparseable {column} at line {line_number}."
        ) from exc


def _validate_unique_physical_sample_ids(
    path: Path,
    records: tuple[TranslationSourceRecord, ...],
) -> None:
    counts = Counter(record.sentence_name for record in records)
    duplicates = tuple(
        sentence_name for sentence_name, count in sorted(counts.items()) if count > 1
    )
    if duplicates:
        raise ValueError(
            f"Translation file {path} has duplicate physical sample ids "
            f"in SENTENCE_NAME: {duplicates}."
        )


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
