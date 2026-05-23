"""Strict IO helpers for learned pose-token artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.tokens import (
    POSE_TOKEN_MANIFEST_SCHEMA_VERSION,
    POSE_TOKEN_SCHEMA_VERSION,
    PoseTokenManifestEntry,
    PoseTokenSequence,
)

_TOKEN_NPZ_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "source_sentence_name",
        "split",
        "token_ids",
        "frame_count",
        "token_count",
        "codebook_size",
        "temporal_granularity",
        "window_size",
        "stride",
    }
)


def write_pose_token_sequence_npz(path: Path, sequence: PoseTokenSequence) -> None:
    """Write one pose token sequence as deterministic compressed NPZ."""

    if not isinstance(sequence, PoseTokenSequence):
        raise LearnedPoseTokenError("sequence must be a PoseTokenSequence.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(
        output_path,
        {
            "schema_version": np.asarray(sequence.schema_version),
            "sample_id": np.asarray(sequence.sample_id),
            "source_sentence_name": np.asarray(sequence.source_sentence_name),
            "split": np.asarray(sequence.split.value),
            "token_ids": np.asarray(sequence.token_ids, dtype=np.int64),
            "frame_count": np.asarray(sequence.frame_count, dtype=np.int64),
            "token_count": np.asarray(sequence.token_count, dtype=np.int64),
            "codebook_size": np.asarray(sequence.codebook_size, dtype=np.int64),
            "temporal_granularity": np.asarray(sequence.temporal_granularity),
            "window_size": np.asarray(sequence.window_size, dtype=np.int64),
            "stride": np.asarray(sequence.stride, dtype=np.int64),
        },
    )


def read_pose_token_sequence_npz(path: Path) -> PoseTokenSequence:
    """Read and validate one pose token sequence NPZ."""

    input_path = Path(path)
    try:
        with np.load(input_path, allow_pickle=False) as loaded:
            keys = frozenset(loaded.files)
            if keys != _TOKEN_NPZ_KEYS:
                raise LearnedPoseTokenError(
                    f"pose token NPZ keys mismatch: expected={sorted(_TOKEN_NPZ_KEYS)}, "
                    f"observed={sorted(keys)}."
                )
            schema_version = _scalar_text(loaded["schema_version"], "schema_version")
            if schema_version != POSE_TOKEN_SCHEMA_VERSION:
                raise LearnedPoseTokenError("pose token sequence schema_version is unsupported.")
            return PoseTokenSequence(
                schema_version=schema_version,
                sample_id=_scalar_text(loaded["sample_id"], "sample_id"),
                source_sentence_name=_scalar_text(
                    loaded["source_sentence_name"],
                    "source_sentence_name",
                ),
                split=SampleSplit(_scalar_text(loaded["split"], "split")),
                token_ids=np.asarray(loaded["token_ids"], dtype=np.int64),
                frame_count=_scalar_int(loaded["frame_count"], "frame_count"),
                token_count=_scalar_int(loaded["token_count"], "token_count"),
                codebook_size=_scalar_int(loaded["codebook_size"], "codebook_size"),
                temporal_granularity=_scalar_text(
                    loaded["temporal_granularity"],
                    "temporal_granularity",
                ),
                window_size=_scalar_int(loaded["window_size"], "window_size"),
                stride=_scalar_int(loaded["stride"], "stride"),
            )
    except LearnedPoseTokenError:
        raise
    except (BadZipFile, EOFError, OSError, KeyError, TypeError, ValueError) as exc:
        raise LearnedPoseTokenError(
            f"pose token sequence could not be loaded from {input_path}: {exc}"
        ) from exc


def write_pose_token_manifest_jsonl(
    path: Path,
    entries: Iterable[PoseTokenManifestEntry],
) -> None:
    """Write deterministic UTF-8 JSONL token manifest entries."""

    materialized = tuple(entries)
    if any(not isinstance(entry, PoseTokenManifestEntry) for entry in materialized):
        raise LearnedPoseTokenError("entries must contain PoseTokenManifestEntry values.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in materialized:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_pose_token_manifest_jsonl(path: Path) -> tuple[PoseTokenManifestEntry, ...]:
    """Read strict pose-token manifest JSONL entries."""

    entries: list[PoseTokenManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LearnedPoseTokenError(
                    f"malformed pose-token manifest JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, Mapping):
                raise LearnedPoseTokenError(
                    f"pose-token manifest record at line {line_number} must be an object."
                )
            entries.append(_manifest_entry_from_record(record, line_number=line_number))
    return tuple(entries)


def validate_token_manifest_entry_matches_sequence(
    entry: PoseTokenManifestEntry,
    sequence: PoseTokenSequence,
) -> None:
    """Validate manifest and sequence identity fields match exactly."""

    if entry.sample_id != sequence.sample_id:
        raise LearnedPoseTokenError("token manifest sample_id does not match sequence.")
    if entry.source_sentence_name != sequence.source_sentence_name:
        raise LearnedPoseTokenError(
            "token manifest source_sentence_name does not match sequence."
        )
    if entry.split is not sequence.split:
        raise LearnedPoseTokenError("token manifest split does not match sequence.")
    for field_name in (
        "token_count",
        "frame_count",
        "codebook_size",
        "temporal_granularity",
        "window_size",
        "stride",
    ):
        if getattr(entry, field_name) != getattr(sequence, field_name):
            raise LearnedPoseTokenError(
                f"token manifest {field_name} does not match sequence."
            )


def _manifest_entry_from_record(
    record: Mapping[str, object],
    *,
    line_number: int,
) -> PoseTokenManifestEntry:
    expected = {
        "schema_version",
        "sample_id",
        "source_sentence_name",
        "split",
        "token_path",
        "token_count",
        "frame_count",
        "codebook_size",
        "temporal_granularity",
        "window_size",
        "stride",
        "issues",
    }
    if set(record) != expected:
        raise LearnedPoseTokenError(
            f"pose-token manifest record keys mismatch at line {line_number}."
        )
    if record["schema_version"] != POSE_TOKEN_MANIFEST_SCHEMA_VERSION:
        raise LearnedPoseTokenError(
            f"pose-token manifest schema_version is unsupported at line {line_number}."
        )
    issues = record["issues"]
    if not isinstance(issues, list):
        raise LearnedPoseTokenError(f"issues must be a list at line {line_number}.")
    return PoseTokenManifestEntry(
        schema_version=_text(record["schema_version"], "schema_version"),
        sample_id=_text(record["sample_id"], "sample_id"),
        source_sentence_name=_text(record["source_sentence_name"], "source_sentence_name"),
        split=SampleSplit(_text(record["split"], "split")),
        token_path=Path(_text(record["token_path"], "token_path")),
        token_count=_int(record["token_count"], "token_count"),
        frame_count=_int(record["frame_count"], "frame_count"),
        codebook_size=_int(record["codebook_size"], "codebook_size"),
        temporal_granularity=_text(record["temporal_granularity"], "temporal_granularity"),
        window_size=_int(record["window_size"], "window_size"),
        stride=_int(record["stride"], "stride"),
        issues=tuple(_text(issue, "issue") for issue in issues),
    )


def _write_deterministic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            info = ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, _npy_bytes(np.asarray(arrays[key])))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _scalar_text(value: np.ndarray, name: str) -> str:
    return _text(np.asarray(value).item(), name)


def _scalar_int(value: np.ndarray, name: str) -> int:
    return _int(np.asarray(value).item(), name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"{name} must be non-empty.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool):
        raise LearnedPoseTokenError(f"{name} must be an integer.")
    return int(value)


__all__ = [
    "read_pose_token_manifest_jsonl",
    "read_pose_token_sequence_npz",
    "validate_token_manifest_entry_matches_sequence",
    "write_pose_token_manifest_jsonl",
    "write_pose_token_sequence_npz",
]
