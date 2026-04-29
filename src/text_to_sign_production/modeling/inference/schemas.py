"""Prediction sample and manifest schemas for M0 full-BFH baseline outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION
from text_to_sign_production.modeling.contracts import (
    CONFIDENCE_POLICY,
    LENGTH_POLICY,
    PEOPLE_PER_FRAME_POLICY,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    SELECTED_PERSON_INDEX_POLICY,
)
from text_to_sign_production.modeling.data import (
    M0_CHANNEL_POLICY,
    M0_TARGET_CHANNEL_SHAPES,
    M0_TARGET_CHANNELS,
)

PREDICTION_SAMPLE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "body",
    "body_confidence",
    "left_hand",
    "left_hand_confidence",
    "right_hand",
    "right_hand_confidence",
    "face",
    "face_confidence",
    "frame_valid_mask",
    "people_per_frame",
    "selected_person_index",
    "prediction_schema_version",
    "source_processed_schema_version",
)


class BaselinePredictionSchemaError(ValueError):
    """Raised when an M0 prediction artifact violates the Phase 4A schema contract."""


@dataclass(frozen=True, slots=True)
class PredictionManifestRow:
    """Split-scoped prediction manifest row sufficient for evaluation pairing."""

    run_name: str
    split: str
    sample_id: str
    text: str
    reference_sample_path: str
    prediction_sample_path: str
    reference_num_frames: int
    prediction_num_frames: int
    checkpoint_path: str
    schema_version: str = PREDICTION_MANIFEST_SCHEMA_VERSION
    length_policy: str = LENGTH_POLICY
    channel_policy: str = M0_CHANNEL_POLICY
    confidence_policy: str = CONFIDENCE_POLICY

    def to_record(self) -> dict[str, object]:
        """Return the manifest row as JSONL-ready data."""

        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "split": self.split,
            "sample_id": self.sample_id,
            "text": self.text,
            "reference_sample_path": self.reference_sample_path,
            "prediction_sample_path": self.prediction_sample_path,
            "reference_num_frames": self.reference_num_frames,
            "prediction_num_frames": self.prediction_num_frames,
            "length_policy": self.length_policy,
            "channel_policy": self.channel_policy,
            "confidence_policy": self.confidence_policy,
            "checkpoint_path": self.checkpoint_path,
        }


def build_prediction_sample_payload(
    channel_arrays: Mapping[str, npt.NDArray[Any]],
    *,
    frame_valid_mask: npt.NDArray[Any],
    selected_person_index: int,
    source_processed_schema_version: str = PROCESSED_SCHEMA_VERSION,
) -> dict[str, npt.NDArray[Any]]:
    """Build a schema-compatible M0 prediction `.npz` payload.

    Generated confidence arrays are synthetic validity masks: 1.0 for generated points on valid
    frames and 0.0 for invalid frames. They are not copied from reference confidence and are not
    model uncertainty estimates.
    """

    coordinates: dict[str, npt.NDArray[np.float32]] = {}
    expected_num_frames: int | None = None
    for channel in M0_TARGET_CHANNELS:
        array = _coordinate_array(channel_arrays, channel=channel)
        if expected_num_frames is None:
            expected_num_frames = int(array.shape[0])
        elif int(array.shape[0]) != expected_num_frames:
            raise BaselinePredictionSchemaError(
                "All prediction channels must have the same frame count."
            )
        coordinates[channel] = array

    if expected_num_frames is None:
        raise BaselinePredictionSchemaError("Prediction payload has no configured channels.")
    valid_mask = _frame_valid_mask(frame_valid_mask, expected_num_frames=expected_num_frames)
    if source_processed_schema_version != PROCESSED_SCHEMA_VERSION:
        raise BaselinePredictionSchemaError(
            "source_processed_schema_version must preserve processed-v1 provenance: "
            f"expected {PROCESSED_SCHEMA_VERSION!r}, got {source_processed_schema_version!r}."
        )

    payload: dict[str, npt.NDArray[Any]] = {
        "prediction_schema_version": np.asarray(PREDICTION_SCHEMA_VERSION),
        "source_processed_schema_version": np.asarray(source_processed_schema_version),
        "frame_valid_mask": valid_mask,
        "people_per_frame": _synthetic_people_per_frame(valid_mask),
        "selected_person_index": np.asarray(selected_person_index, dtype=np.int16),
    }
    for channel in M0_TARGET_CHANNELS:
        payload[channel] = coordinates[channel]
        payload[f"{channel}_confidence"] = _synthetic_confidence(
            valid_mask,
            keypoints=M0_TARGET_CHANNEL_SHAPES[channel][0],
        )

    validate_prediction_sample_payload(payload)
    return payload


def validate_prediction_sample_payload(payload: Mapping[str, npt.NDArray[Any]]) -> None:
    """Validate a prediction sample payload against the Phase 4A M0 schema."""

    missing_keys = sorted(set(PREDICTION_SAMPLE_REQUIRED_KEYS).difference(payload))
    if missing_keys:
        formatted = ", ".join(missing_keys)
        raise BaselinePredictionSchemaError(
            f"Prediction sample is missing required key(s): {formatted}."
        )

    prediction_schema_version = _scalar_string(payload, "prediction_schema_version")
    if prediction_schema_version != PREDICTION_SCHEMA_VERSION:
        raise BaselinePredictionSchemaError(
            "Prediction sample schema version mismatch: "
            f"expected {PREDICTION_SCHEMA_VERSION!r}, got {prediction_schema_version!r}."
        )
    source_schema_version = _scalar_string(payload, "source_processed_schema_version")
    if source_schema_version != PROCESSED_SCHEMA_VERSION:
        raise BaselinePredictionSchemaError(
            "Prediction sample source processed schema mismatch: "
            f"expected {PROCESSED_SCHEMA_VERSION!r}, got {source_schema_version!r}."
        )

    num_frames: int | None = None
    for channel in M0_TARGET_CHANNELS:
        coordinates = _coordinate_array(payload, channel=channel)
        if num_frames is None:
            num_frames = int(coordinates.shape[0])
        elif int(coordinates.shape[0]) != num_frames:
            raise BaselinePredictionSchemaError(
                "Prediction sample coordinate channels must share a frame count."
            )
        _confidence_array(
            payload,
            channel=channel,
            expected_shape=coordinates.shape[:2],
        )

    if num_frames is None:
        raise BaselinePredictionSchemaError("Prediction sample has no configured channels.")
    _frame_valid_mask(payload["frame_valid_mask"], expected_num_frames=num_frames)
    people_per_frame = np.asarray(payload["people_per_frame"])
    if tuple(people_per_frame.shape) != (num_frames,):
        raise BaselinePredictionSchemaError(
            "Prediction sample people_per_frame shape mismatch: "
            f"expected {(num_frames,)}, got {tuple(people_per_frame.shape)}."
        )
    if not np.issubdtype(people_per_frame.dtype, np.integer):
        raise BaselinePredictionSchemaError("Prediction sample people_per_frame must be integer.")

    selected_person_index = np.asarray(payload["selected_person_index"])
    if tuple(selected_person_index.shape) != ():
        raise BaselinePredictionSchemaError(
            "Prediction sample selected_person_index must be a scalar array."
        )


def build_prediction_manifest_row(
    *,
    run_name: str,
    split: str,
    sample_id: str,
    text: str,
    reference_sample_path: Path | str,
    prediction_sample_path: Path | str,
    reference_num_frames: int,
    prediction_num_frames: int,
    checkpoint_path: Path | str,
) -> PredictionManifestRow:
    """Build one split-scoped M0 prediction manifest row."""

    if reference_num_frames < 0:
        raise BaselinePredictionSchemaError("reference_num_frames must not be negative.")
    if prediction_num_frames < 0:
        raise BaselinePredictionSchemaError("prediction_num_frames must not be negative.")
    if reference_num_frames != prediction_num_frames:
        raise BaselinePredictionSchemaError(
            f"M0 prediction_num_frames must equal reference_num_frames under {LENGTH_POLICY!r}."
        )
    for field_name, value in {
        "run_name": run_name,
        "split": split,
        "sample_id": sample_id,
        "text": text,
    }.items():
        if not value.strip():
            raise BaselinePredictionSchemaError(f"{field_name} must not be blank.")

    return PredictionManifestRow(
        run_name=run_name,
        split=split,
        sample_id=sample_id,
        text=text,
        reference_sample_path=_path_string(reference_sample_path),
        prediction_sample_path=_path_string(prediction_sample_path),
        reference_num_frames=reference_num_frames,
        prediction_num_frames=prediction_num_frames,
        checkpoint_path=_path_string(checkpoint_path),
    )


def prediction_sample_policy_metadata() -> dict[str, str]:
    """Return policy metadata that describes generated prediction sample side channels."""

    return {
        "length_policy": LENGTH_POLICY,
        "channel_policy": M0_CHANNEL_POLICY,
        "confidence_policy": CONFIDENCE_POLICY,
        "people_per_frame_policy": PEOPLE_PER_FRAME_POLICY,
        "selected_person_index_policy": SELECTED_PERSON_INDEX_POLICY,
    }


def _coordinate_array(
    payload: Mapping[str, npt.NDArray[Any]],
    *,
    channel: str,
) -> npt.NDArray[np.float32]:
    if channel not in payload:
        raise BaselinePredictionSchemaError(f"Prediction sample is missing channel {channel!r}.")
    array = np.asarray(payload[channel], dtype=np.float32)
    expected_shape_suffix = M0_TARGET_CHANNEL_SHAPES[channel]
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_shape_suffix:
        raise BaselinePredictionSchemaError(
            f"Prediction channel {channel!r} has shape {tuple(array.shape)}; "
            f"expected [T, {expected_shape_suffix[0]}, {expected_shape_suffix[1]}]."
        )
    return array


def _confidence_array(
    payload: Mapping[str, npt.NDArray[Any]],
    *,
    channel: str,
    expected_shape: tuple[int, int],
) -> npt.NDArray[np.float32]:
    key = f"{channel}_confidence"
    if key not in payload:
        raise BaselinePredictionSchemaError(f"Prediction sample is missing confidence {key!r}.")
    array = np.asarray(payload[key], dtype=np.float32)
    if tuple(array.shape) != expected_shape:
        raise BaselinePredictionSchemaError(
            f"Prediction confidence {key!r} has shape {tuple(array.shape)}; "
            f"expected {expected_shape}."
        )
    return array


def _frame_valid_mask(
    frame_valid_mask: npt.NDArray[Any],
    *,
    expected_num_frames: int,
) -> npt.NDArray[np.bool_]:
    array = np.asarray(frame_valid_mask, dtype=np.bool_)
    if tuple(array.shape) != (expected_num_frames,):
        raise BaselinePredictionSchemaError(
            f"Prediction frame_valid_mask has shape {tuple(array.shape)}; "
            f"expected {(expected_num_frames,)}."
        )
    return array


def _synthetic_confidence(
    frame_valid_mask: npt.NDArray[np.bool_],
    *,
    keypoints: int,
) -> npt.NDArray[np.float32]:
    values = frame_valid_mask.astype(np.float32, copy=False)[:, np.newaxis]
    return cast(npt.NDArray[np.float32], np.repeat(values, keypoints, axis=1))


def _synthetic_people_per_frame(
    frame_valid_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.int16]:
    return cast(npt.NDArray[np.int16], frame_valid_mask.astype(np.int16, copy=False))


def _scalar_string(payload: Mapping[str, npt.NDArray[Any]], key: str) -> str:
    value = np.asarray(payload[key])
    if tuple(value.shape) != ():
        raise BaselinePredictionSchemaError(
            f"Prediction sample {key!r} must be a scalar array; got {tuple(value.shape)}."
        )
    return str(value.item())


def _path_string(path: Path | str) -> str:
    if isinstance(path, Path):
        return path.as_posix()
    return path


__all__ = [
    "BaselinePredictionSchemaError",
    "PREDICTION_SAMPLE_REQUIRED_KEYS",
    "PredictionManifestRow",
    "build_prediction_manifest_row",
    "build_prediction_sample_payload",
    "prediction_sample_policy_metadata",
    "validate_prediction_sample_payload",
]
