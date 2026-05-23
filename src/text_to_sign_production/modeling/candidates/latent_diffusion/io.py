"""Strict deterministic IO for latent_diffusion foundation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    bfh_tensor_layout_from_dict,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_MANIFEST_SCHEMA_VERSION,
    LATENT_MANIFEST_SCHEMA_VERSION_V1,
    LATENT_SEQUENCE_SCHEMA_VERSION,
    LATENT_SEQUENCE_SCHEMA_VERSION_V1,
    LATENT_TARGET_SPEC_SCHEMA_VERSION,
    LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
    LatentManifestEntry,
    LatentSequence,
    LatentTargetSpec,
)

_LATENT_SEQUENCE_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "source_sentence_name",
        "split",
        "values",
        "validity_mask",
        "frame_count",
        "latent_count",
        "latent_dim",
        "target_spec_json",
    }
)

_LATENT_SEQUENCE_KEYS_V1 = _LATENT_SEQUENCE_KEYS - {"latent_count"}


def write_latent_sequence_npz(path: Path, sequence: LatentSequence) -> None:
    """Write one latent sequence as a deterministic compressed NPZ."""

    if not isinstance(sequence, LatentSequence):
        raise LatentDiffusionError("sequence must be a LatentSequence.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_deterministic_npz(
        output_path,
        {
            "schema_version": np.asarray(sequence.schema_version),
            "sample_id": np.asarray(sequence.sample_id),
            "source_sentence_name": np.asarray(sequence.source_sentence_name),
            "split": np.asarray(sequence.split.value),
            "values": np.asarray(sequence.values, dtype=np.float32),
            "validity_mask": np.asarray(sequence.validity_mask, dtype=np.bool_),
            "frame_count": np.asarray(sequence.frame_count, dtype=np.int64),
            "latent_count": np.asarray(sequence.latent_count, dtype=np.int64),
            "latent_dim": np.asarray(sequence.latent_dim, dtype=np.int64),
            "target_spec_json": np.asarray(_json_dumps(sequence.target_spec.to_dict())),
        },
    )


def read_latent_sequence_npz(path: Path) -> LatentSequence:
    """Read and validate one latent sequence NPZ."""

    input_path = Path(path)
    try:
        with np.load(input_path, allow_pickle=False) as loaded:
            keys = frozenset(loaded.files)
            if keys not in {_LATENT_SEQUENCE_KEYS, _LATENT_SEQUENCE_KEYS_V1}:
                raise LatentDiffusionError(
                    f"latent sequence NPZ keys mismatch: expected={sorted(_LATENT_SEQUENCE_KEYS)}, "
                    f"observed={sorted(keys)}."
                )
            schema_version = _scalar_text(loaded["schema_version"], "schema_version")
            if schema_version not in {
                LATENT_SEQUENCE_SCHEMA_VERSION,
                LATENT_SEQUENCE_SCHEMA_VERSION_V1,
            }:
                raise LatentDiffusionError("latent sequence schema_version is unsupported.")
            target_spec = _target_spec_from_record(
                _json_object(
                    _scalar_text(loaded["target_spec_json"], "target_spec_json"),
                    "target_spec_json",
                )
            )
            return LatentSequence(
                schema_version=schema_version,
                sample_id=_scalar_text(loaded["sample_id"], "sample_id"),
                source_sentence_name=_scalar_text(
                    loaded["source_sentence_name"],
                    "source_sentence_name",
                ),
                split=SampleSplit(_scalar_text(loaded["split"], "split")),
                values=np.asarray(loaded["values"], dtype=np.float32),
                validity_mask=np.asarray(loaded["validity_mask"], dtype=np.bool_),
                frame_count=_scalar_int(loaded["frame_count"], "frame_count"),
                latent_count=(
                    _scalar_int(loaded["latent_count"], "latent_count")
                    if "latent_count" in keys
                    else _scalar_int(loaded["frame_count"], "frame_count")
                ),
                latent_dim=_scalar_int(loaded["latent_dim"], "latent_dim"),
                target_spec=target_spec,
            )
    except LatentDiffusionError:
        raise
    except (BadZipFile, EOFError, OSError, KeyError, TypeError, ValueError) as exc:
        raise LatentDiffusionError(
            f"latent sequence could not be loaded from {input_path}: {exc}"
        ) from exc


def write_latent_manifest_jsonl(
    path: Path,
    entries: Iterable[LatentManifestEntry],
) -> None:
    """Write deterministic UTF-8 JSONL latent manifest entries."""

    materialized = tuple(entries)
    if any(not isinstance(entry, LatentManifestEntry) for entry in materialized):
        raise LatentDiffusionError("entries must contain LatentManifestEntry values.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in materialized:
            handle.write(_json_dumps(entry.to_dict()))
            handle.write("\n")


def read_latent_manifest_jsonl(path: Path) -> tuple[LatentManifestEntry, ...]:
    """Read strict latent manifest JSONL entries."""

    entries: list[LatentManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LatentDiffusionError(
                    f"malformed latent manifest JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, Mapping):
                raise LatentDiffusionError(
                    f"latent manifest record at line {line_number} must be an object."
                )
            entries.append(_manifest_entry_from_record(record, line_number=line_number))
    return tuple(entries)


def write_latent_target_spec_json(path: Path, spec: LatentTargetSpec) -> None:
    """Write a deterministic UTF-8 latent target spec JSON document."""

    if not isinstance(spec, LatentTargetSpec):
        raise LatentDiffusionError("spec must be a LatentTargetSpec.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_dumps(spec.to_dict()) + "\n", encoding="utf-8")


def read_latent_target_spec_json(path: Path) -> LatentTargetSpec:
    """Read and validate a latent target spec JSON document."""

    record = _json_object(Path(path).read_text(encoding="utf-8"), "latent target spec")
    return _target_spec_from_record(record)


def _target_spec_from_record(record: Mapping[str, object]) -> LatentTargetSpec:
    expected = {
        "schema_version",
        "target_type",
        "layout",
        "coordinate_mode",
        "confidence_policy",
        "temporal_granularity",
        "window_size",
        "stride",
    }
    optional_v2 = {"latent_dim", "base_feature_dim"}
    if not expected <= set(record) <= expected | optional_v2:
        raise LatentDiffusionError("latent target spec keys do not match the schema.")
    if record["schema_version"] != LATENT_TARGET_SPEC_SCHEMA_VERSION:
        raise LatentDiffusionError("latent target spec schema_version is unsupported.")
    layout = bfh_tensor_layout_from_dict(_mapping(record["layout"], "layout"))
    target_type = _text(record["target_type"], "target_type")
    latent_dim = (
        _int(record["latent_dim"], "latent_dim")
        if "latent_dim" in record
        else layout.total_feature_dim
    )
    base_feature_dim = (
        _int(record["base_feature_dim"], "base_feature_dim")
        if "base_feature_dim" in record
        else layout.total_feature_dim
    )
    if "latent_dim" not in record and target_type != LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
        raise LatentDiffusionError("latent target spec latent_dim is required for non-legacy targets.")
    return LatentTargetSpec(
        schema_version=_text(record["schema_version"], "schema_version"),
        target_type=target_type,
        layout=layout,
        coordinate_mode=_text(record["coordinate_mode"], "coordinate_mode"),
        confidence_policy=_text(record["confidence_policy"], "confidence_policy"),
        temporal_granularity=_text(record["temporal_granularity"], "temporal_granularity"),
        window_size=_int(record["window_size"], "window_size"),
        stride=_int(record["stride"], "stride"),
        latent_dim=latent_dim,
        base_feature_dim=base_feature_dim,
    )


def _manifest_entry_from_record(
    record: Mapping[str, object],
    *,
    line_number: int,
) -> LatentManifestEntry:
    expected = {
        "schema_version",
        "sample_id",
        "source_sentence_name",
        "split",
        "latent_path",
        "frame_count",
        "latent_count",
        "latent_dim",
        "target_type",
        "temporal_granularity",
        "window_size",
        "stride",
        "issues",
    }
    expected_v1 = expected - {
        "latent_count",
        "temporal_granularity",
        "window_size",
        "stride",
    }
    if set(record) not in {expected, expected_v1}:
        raise LatentDiffusionError(
            f"latent manifest record keys mismatch at line {line_number}."
        )
    if record["schema_version"] not in {
        LATENT_MANIFEST_SCHEMA_VERSION,
        LATENT_MANIFEST_SCHEMA_VERSION_V1,
    }:
        raise LatentDiffusionError(
            f"latent manifest schema_version is unsupported at line {line_number}."
        )
    issues = record["issues"]
    if not isinstance(issues, list):
        raise LatentDiffusionError(f"issues must be a list at line {line_number}.")
    return LatentManifestEntry(
        schema_version=_text(record["schema_version"], "schema_version"),
        sample_id=_text(record["sample_id"], "sample_id"),
        source_sentence_name=_text(record["source_sentence_name"], "source_sentence_name"),
        split=SampleSplit(_text(record["split"], "split")),
        latent_path=Path(_text(record["latent_path"], "latent_path")),
        frame_count=_int(record["frame_count"], "frame_count"),
        latent_count=(
            _int(record["latent_count"], "latent_count")
            if "latent_count" in record
            else _int(record["frame_count"], "frame_count")
        ),
        latent_dim=_int(record["latent_dim"], "latent_dim"),
        target_type=_text(record["target_type"], "target_type"),
        temporal_granularity=(
            _text(record["temporal_granularity"], "temporal_granularity")
            if "temporal_granularity" in record
            else "frame"
        ),
        window_size=(
            _int(record["window_size"], "window_size")
            if "window_size" in record
            else 1
        ),
        stride=(
            _int(record["stride"], "stride")
            if "stride" in record
            else 1
        ),
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


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_object(value: str, name: str) -> Mapping[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LatentDiffusionError(f"malformed {name} JSON: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise LatentDiffusionError(f"{name} JSON must contain an object.")
    return loaded


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LatentDiffusionError(f"{name} must be a JSON object.")
    return value


def _scalar_text(value: np.ndarray, name: str) -> str:
    return _text(np.asarray(value).item(), name)


def _scalar_int(value: np.ndarray, name: str) -> int:
    return _int(np.asarray(value).item(), name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LatentDiffusionError(f"{name} must be non-empty.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool):
        raise LatentDiffusionError(f"{name} must be an integer.")
    return int(value)


__all__ = [
    "read_latent_manifest_jsonl",
    "read_latent_sequence_npz",
    "read_latent_target_spec_json",
    "write_latent_manifest_jsonl",
    "write_latent_sequence_npz",
    "write_latent_target_spec_json",
]
