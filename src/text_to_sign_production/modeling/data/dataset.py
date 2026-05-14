"""Legacy M0 adapter over canonical dataset manifests and PreparedSample payloads.

This module is not a semantic owner for prepared samples or manifest rows. It
adapts the current data/dataset contracts into the M0 modeling batch shape.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from text_to_sign_production.core.ids import VALID_SAMPLE_SPLITS as SPLITS
from text_to_sign_production.core.models import PassedManifestEntry, PreparedSample
from text_to_sign_production.data.dataset import PREPARED_SAMPLE_SCHEMA_VERSION
from text_to_sign_production.data.dataset.manifests import read_passed_manifest_json
from text_to_sign_production.data.dataset.payloads import load_prepared_sample_payload

from .schemas import (
    ConfidenceArray,
    IntegerArray,
    MaskArray,
    PoseArray,
    ProcessedModelingManifestRecord,
    ProcessedPoseItem,
    ProcessedPoseSample,
)


class ProcessedModelingDataError(ValueError):
    """Raised when processed modeling data violates the M0 full-BFH contract."""


def _validate_split(split: str, *, context: str) -> None:
    if split not in SPLITS:
        expected = ", ".join(SPLITS)
        raise ProcessedModelingDataError(
            f"{context} uses unknown split {split!r}; expected one of: {expected}."
        )


def _processed_manifest_record_from_entry(
    entry: PassedManifestEntry,
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
    split = entry.split.value
    _validate_split(split, context=f"Processed manifest record {sample_id!r}")
    if expected_split is not None and split != expected_split:
        raise ProcessedModelingDataError(
            f"Processed manifest record {sample_id!r} has split {split!r}; "
            f"expected {expected_split!r}."
        )
    if entry.frame_count < 0:
        raise ProcessedModelingDataError(
            f"Processed manifest record {sample_id!r} has negative num_frames."
        )

    try:
        resolved_sample_path = _validate_processed_sample_path(
            entry.payload_ref,
            split=split,
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
            f"{sample_id!r} does not exist: {entry.payload_ref}"
        )

    return ProcessedModelingManifestRecord(
        sample_id=sample_id,
        split=split,
        text=entry.text,
        fps=entry.fps,
        num_frames=entry.frame_count,
        sample_path=resolved_sample_path,
        sample_path_value=entry.payload_ref,
        processed_schema_version=entry.schema_version,
        selected_person_index=-1,
        multi_person_frame_count=0,
        max_people_per_frame=1,
        frame_valid_count=entry.valid_frame_count,
        frame_invalid_count=entry.frame_count - entry.valid_frame_count,
    )


def read_processed_modeling_manifest(
    manifest_path: Path | str,
    *,
    split: str | None = None,
    data_root: Path | str | None = None,
) -> list[ProcessedModelingManifestRecord]:
    """Read and validate a processed manifest for M0 modeling use."""

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
    for entry in read_passed_manifest_json(path):
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
        if parent.name == "manifests":
            return parent.parent / "samples"
    raise ProcessedModelingDataError(
        f"Could not infer samples root from processed manifest path: {path}"
    )


def _validate_processed_sample_path(
    sample_path: str,
    *,
    split: str,
    sample_id: str,
    data_root: Path,
) -> Path:
    path = Path(sample_path)
    if not path.parts or path.parts[0] not in {"passed", "dropped"}:
        path = Path("passed") / path
    resolved = path if path.is_absolute() else data_root / path
    resolved = resolved.expanduser().resolve()
    if resolved.suffix != ".npz":
        raise ValueError(f"sample_path must point to an .npz file: {sample_path!r}.")
    if sample_id not in resolved.stem:
        raise ValueError(
            f"sample_path {sample_path!r} does not appear to reference sample {sample_id!r}."
        )
    if split not in resolved.parts:
        raise ValueError(f"sample_path {sample_path!r} does not include split {split!r}.")
    return resolved


def load_processed_pose_sample(
    record: ProcessedModelingManifestRecord,
) -> ProcessedPoseSample:
    """Load the processed `.npz` referenced by one modeling manifest record."""
    prepared = _load_aligned_prepared_sample(record)
    payload_schema_version = prepared.schema_version
    channel_arrays = _prepared_coordinate_arrays(prepared)
    confidence_arrays = _prepared_confidence_arrays(prepared)
    people_per_frame = _people_per_frame(prepared)
    frame_valid_array = cast(
        MaskArray,
        np.asarray(prepared.pose.valid_frame_mask, dtype=np.bool_),
    )
    frame_valid_count = int(np.count_nonzero(frame_valid_array))
    if frame_valid_count != record.frame_valid_count:
        raise ProcessedModelingDataError(
            "PreparedSample frame_valid_mask valid count does not match manifest "
            f"record {record.sample_id!r}: payload={frame_valid_count} "
            f"manifest={record.frame_valid_count}."
        )
    frame_invalid_count = int(frame_valid_array.shape[0] - frame_valid_count)
    if frame_invalid_count != record.frame_invalid_count:
        raise ProcessedModelingDataError(
            "PreparedSample frame_valid_mask invalid count does not match manifest "
            f"record {record.sample_id!r}: payload={frame_invalid_count} "
            f"manifest={record.frame_invalid_count}."
        )

    return ProcessedPoseSample(
        processed_schema_version=payload_schema_version,
        body=channel_arrays["body"],
        body_confidence=confidence_arrays["body"],
        left_hand=channel_arrays["left_hand"],
        left_hand_confidence=confidence_arrays["left_hand"],
        right_hand=channel_arrays["right_hand"],
        right_hand_confidence=confidence_arrays["right_hand"],
        face=channel_arrays["face"],
        face_confidence=confidence_arrays["face"],
        frame_valid_mask=frame_valid_array,
        people_per_frame=people_per_frame,
        selected_person_index=_representative_selected_person_index(prepared),
    )


def _load_aligned_prepared_sample(record: ProcessedModelingManifestRecord) -> PreparedSample:
    try:
        sample = load_prepared_sample_payload(record.sample_path)
    except (OSError, ValueError) as exc:
        raise ProcessedModelingDataError(
            f"PreparedSample payload could not be loaded for modeling: {record.sample_path}: {exc}"
        ) from exc
    if sample.schema_version != PREPARED_SAMPLE_SCHEMA_VERSION:
        raise ProcessedModelingDataError(
            f"PreparedSample {record.sample_path} uses schema {sample.schema_version!r}; "
            f"expected {PREPARED_SAMPLE_SCHEMA_VERSION!r}."
        )
    if sample.source.sample_id != record.sample_id or sample.source.split.value != record.split:
        raise ProcessedModelingDataError(
            "PreparedSample identity does not match modeling manifest record "
            f"{record.split}/{record.sample_id}."
        )
    if sample.pose.frame_count != record.num_frames:
        raise ProcessedModelingDataError(
            "PreparedSample frame_count does not match modeling manifest record "
            f"{record.sample_id!r}: payload={sample.pose.frame_count} "
            f"manifest={record.num_frames}."
        )
    return sample


def _prepared_coordinate_arrays(sample: PreparedSample) -> dict[str, PoseArray]:
    return {
        "body": _coordinates(sample.pose.body_xyc),
        "left_hand": _coordinates(sample.pose.left_hand_xyc),
        "right_hand": _coordinates(sample.pose.right_hand_xyc),
        "face": _coordinates(sample.pose.face_xyc),
    }


def _prepared_confidence_arrays(sample: PreparedSample) -> dict[str, ConfidenceArray]:
    return {
        "body": _confidence(sample.pose.body_xyc),
        "left_hand": _confidence(sample.pose.left_hand_xyc),
        "right_hand": _confidence(sample.pose.right_hand_xyc),
        "face": _confidence(sample.pose.face_xyc),
    }


def _coordinates(xyc: npt.NDArray[np.float32]) -> PoseArray:
    return cast(PoseArray, np.asarray(xyc[..., :2], dtype=np.float32))


def _confidence(xyc: npt.NDArray[np.float32]) -> ConfidenceArray:
    return cast(ConfidenceArray, np.asarray(xyc[..., 2], dtype=np.float32))


def _people_per_frame(sample: PreparedSample) -> IntegerArray:
    values = [0 if index is None else 1 for index in sample.pose.selected_person_indices]
    return cast(IntegerArray, np.asarray(values, dtype=np.int16))


def _representative_selected_person_index(sample: PreparedSample) -> int:
    for index in sample.pose.selected_person_indices:
        if index is not None:
            return index
    return -1


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
