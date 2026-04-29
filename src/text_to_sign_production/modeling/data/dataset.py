"""Processed-manifest and `.npz` loading for Sprint 3 baseline modeling."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.constants import (
    OPENPOSE_CHANNEL_SPECS,
    PROCESSED_SCHEMA_VERSION,
    SPLITS,
)
from text_to_sign_production.data.schemas import ProcessedManifestEntry
from text_to_sign_production.data.validate import validate_processed_sample_path

from .schemas import (
    SPRINT3_TARGET_CHANNELS,
    MaskArray,
    PoseArray,
    ProcessedModelingManifestRecord,
    ProcessedPoseItem,
    ProcessedPoseSample,
)


class ProcessedModelingDataError(ValueError):
    """Raised when processed modeling data violates the Sprint 3 contract."""


def _validate_split(split: str, *, context: str) -> None:
    if split not in SPLITS:
        expected = ", ".join(SPLITS)
        raise ProcessedModelingDataError(
            f"{context} uses unknown split {split!r}; expected one of: {expected}."
        )


def _processed_manifest_record_from_entry(
    entry: ProcessedManifestEntry,
    *,
    manifest_path: Path,
    expected_split: str | None,
    data_root: Path,
) -> ProcessedModelingManifestRecord:
    sample_id = entry.sample_id.strip()
    if not sample_id:
        raise ProcessedModelingDataError(
            f"Processed manifest record in {manifest_path} has a blank sample_id."
        )
    if sample_id != entry.sample_id:
        raise ProcessedModelingDataError(
            f"Processed manifest record {entry.sample_id!r} in {manifest_path} has leading "
            "or trailing whitespace in sample_id."
        )
    if entry.processed_schema_version != PROCESSED_SCHEMA_VERSION:
        raise ProcessedModelingDataError(
            "Processed manifest record "
            f"{sample_id!r} uses schema {entry.processed_schema_version!r}; "
            f"expected {PROCESSED_SCHEMA_VERSION!r}."
        )
    _validate_split(entry.split, context=f"Processed manifest record {sample_id!r}")
    if expected_split is not None and entry.split != expected_split:
        raise ProcessedModelingDataError(
            f"Processed manifest record {sample_id!r} has split {entry.split!r}; "
            f"expected {expected_split!r}."
        )
    if entry.num_frames < 0:
        raise ProcessedModelingDataError(
            f"Processed manifest record {sample_id!r} has negative num_frames."
        )

    try:
        resolved_sample_path = validate_processed_sample_path(
            entry.sample_path,
            split=entry.split,
            sample_id=sample_id,
            data_root=data_root,
        )
    except ValueError as exc:
        raise ProcessedModelingDataError(
            f"Processed manifest record {sample_id!r} has invalid sample_path: {exc}"
        ) from exc
    if not resolved_sample_path.is_file():
        raise FileNotFoundError(
            "Processed sample file referenced by manifest record "
            f"{sample_id!r} does not exist: {entry.sample_path}"
        )

    return ProcessedModelingManifestRecord(
        sample_id=sample_id,
        split=entry.split,
        text=entry.text,
        fps=entry.fps,
        num_frames=entry.num_frames,
        sample_path=resolved_sample_path,
        sample_path_value=entry.sample_path,
        processed_schema_version=entry.processed_schema_version,
        selected_person_index=entry.selected_person_index,
        frame_valid_count=entry.frame_valid_count,
        frame_invalid_count=entry.frame_invalid_count,
    )


def _parse_processed_manifest_entry(
    record: Mapping[str, Any],
    *,
    manifest_path: Path,
    line_number: int,
) -> ProcessedManifestEntry:
    try:
        return ProcessedManifestEntry.from_record(record)
    except Exception as exc:
        sample_id = str(record.get("sample_id", "<unknown>"))
        raise ProcessedModelingDataError(
            f"Could not parse processed manifest record on line {line_number} "
            f"({sample_id!r}) in {manifest_path}: {exc}"
        ) from exc


def _iter_processed_manifest_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ProcessedModelingDataError(
                    f"Processed manifest {path} line {line_number} is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ProcessedModelingDataError(
                    f"Processed manifest {path} line {line_number} must contain a JSON object, "
                    f"got {type(payload).__name__}."
                )
            yield line_number, payload


def read_processed_modeling_manifest(
    manifest_path: Path | str,
    *,
    split: str | None = None,
    data_root: Path | str | None = None,
) -> list[ProcessedModelingManifestRecord]:
    """Read and validate a processed manifest for Sprint 3 modeling use."""

    path = Path(manifest_path)
    if split is not None:
        _validate_split(split, context="Requested modeling manifest split")
    resolved_data_root = (
        _infer_data_root_from_manifest(path)
        if data_root is None
        else Path(data_root).expanduser().resolve()
    )

    records: list[ProcessedModelingManifestRecord] = []
    seen_sample_ids: set[str] = set()
    for line_number, raw_record in _iter_processed_manifest_records(path):
        entry = _parse_processed_manifest_entry(
            raw_record,
            manifest_path=path,
            line_number=line_number,
        )
        manifest_record = _processed_manifest_record_from_entry(
            entry,
            manifest_path=path,
            expected_split=split,
            data_root=resolved_data_root,
        )
        if manifest_record.sample_id in seen_sample_ids:
            raise ProcessedModelingDataError(
                f"Duplicate processed sample_id in {path}: {manifest_record.sample_id}"
            )
        seen_sample_ids.add(manifest_record.sample_id)
        records.append(manifest_record)

    return records


def _infer_data_root_from_manifest(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for parent in resolved.parents:
        if parent.name == "data":
            return parent
    raise ProcessedModelingDataError(
        f"Could not infer project data root from processed manifest path: {path}"
    )


def _sample_array(
    sample: Any,
    key: str,
    *,
    sample_path: Path,
) -> npt.NDArray[Any]:
    if key not in sample.files:
        raise ProcessedModelingDataError(
            f"Processed sample {sample_path} is missing required array {key!r}."
        )
    return cast(npt.NDArray[Any], sample[key])


def _require_scalar_value(sample: Any, key: str, *, sample_path: Path) -> Any:
    array = _sample_array(sample, key, sample_path=sample_path)
    if tuple(array.shape) != ():
        raise ProcessedModelingDataError(
            f"Processed sample array {key!r} in {sample_path} has shape "
            f"{tuple(array.shape)}; expected scalar ()."
        )
    return array.item()


def _load_pose_channel(
    sample: Any,
    channel: str,
    *,
    record: ProcessedModelingManifestRecord,
) -> PoseArray:
    array = _sample_array(sample, channel, sample_path=record.sample_path)
    expected_shape = (record.num_frames, OPENPOSE_CHANNEL_SPECS[channel][1], 2)
    observed_shape = tuple(array.shape)
    if observed_shape != expected_shape:
        raise ProcessedModelingDataError(
            f"Processed sample array {channel!r} for {record.sample_id!r} has shape "
            f"{observed_shape}; expected {expected_shape}."
        )
    return cast(PoseArray, np.asarray(array, dtype=np.float32))


def load_processed_pose_sample(
    record: ProcessedModelingManifestRecord,
) -> ProcessedPoseSample:
    """Load the processed `.npz` referenced by one modeling manifest record."""

    try:
        with np.load(record.sample_path, allow_pickle=False) as sample:
            payload_schema_version = str(
                _require_scalar_value(
                    sample,
                    "processed_schema_version",
                    sample_path=record.sample_path,
                )
            )
            if payload_schema_version != PROCESSED_SCHEMA_VERSION:
                raise ProcessedModelingDataError(
                    f"Processed sample {record.sample_path} uses schema "
                    f"{payload_schema_version!r}; expected {PROCESSED_SCHEMA_VERSION!r}."
                )

            payload_selected_person_index = int(
                _require_scalar_value(
                    sample,
                    "selected_person_index",
                    sample_path=record.sample_path,
                )
            )
            if payload_selected_person_index != record.selected_person_index:
                raise ProcessedModelingDataError(
                    "Processed sample selected_person_index does not match manifest record "
                    f"{record.sample_id!r}: payload={payload_selected_person_index} "
                    f"manifest={record.selected_person_index}."
                )

            channel_arrays = {
                channel: _load_pose_channel(sample, channel, record=record)
                for channel in SPRINT3_TARGET_CHANNELS
            }

            frame_valid_mask = _sample_array(
                sample,
                "frame_valid_mask",
                sample_path=record.sample_path,
            )
            expected_mask_shape = (record.num_frames,)
            if tuple(frame_valid_mask.shape) != expected_mask_shape:
                raise ProcessedModelingDataError(
                    f"Processed sample frame_valid_mask for {record.sample_id!r} has shape "
                    f"{tuple(frame_valid_mask.shape)}; expected {expected_mask_shape}."
                )
            if frame_valid_mask.dtype != np.dtype(np.bool_):
                raise ProcessedModelingDataError(
                    f"Processed sample frame_valid_mask for {record.sample_id!r} "
                    f"has dtype {frame_valid_mask.dtype}; expected bool."
                )
            frame_valid_array = cast(MaskArray, np.asarray(frame_valid_mask, dtype=np.bool_))
            frame_valid_count = int(np.count_nonzero(frame_valid_array))
            if frame_valid_count != record.frame_valid_count:
                raise ProcessedModelingDataError(
                    "Processed sample frame_valid_mask valid count does not match manifest "
                    f"record {record.sample_id!r}: payload={frame_valid_count} "
                    f"manifest={record.frame_valid_count}."
                )
            frame_invalid_count = int(frame_valid_array.shape[0] - frame_valid_count)
            if frame_invalid_count != record.frame_invalid_count:
                raise ProcessedModelingDataError(
                    "Processed sample frame_valid_mask invalid count does not match manifest "
                    f"record {record.sample_id!r}: payload={frame_invalid_count} "
                    f"manifest={record.frame_invalid_count}."
                )

    except ProcessedModelingDataError:
        raise
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        raise ProcessedModelingDataError(
            f"Processed sample file could not be read as .npz: {record.sample_path}: {exc}"
        ) from exc

    return ProcessedPoseSample(
        body=channel_arrays["body"],
        left_hand=channel_arrays["left_hand"],
        right_hand=channel_arrays["right_hand"],
        frame_valid_mask=frame_valid_array,
    )


class ProcessedPoseDataset:
    """Dataset over processed manifest records and their processed `.npz` payloads."""

    def __init__(
        self,
        manifest_path: Path | str,
        *,
        split: str | None = None,
        data_root: Path | str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = tuple(
            read_processed_modeling_manifest(
                self.manifest_path,
                split=split,
                data_root=data_root,
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ProcessedPoseItem:
        record = self.records[index]
        return ProcessedPoseItem.from_manifest_and_sample(
            record,
            load_processed_pose_sample(record),
        )

    def __iter__(self) -> Iterator[ProcessedPoseItem]:
        for index in range(len(self)):
            yield self[index]
