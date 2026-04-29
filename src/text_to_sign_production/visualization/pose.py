"""Processed-v1-compatible pose array loading for rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from zipfile import BadZipFile

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION


class PoseSampleError(ValueError):
    """Raised when a processed pose sample cannot be loaded for rendering."""


@dataclass(frozen=True, slots=True)
class PoseSample:
    """Processed-v1-compatible pose arrays loaded from a `.npz` file."""

    path: Path
    processed_schema_version: str
    body: npt.NDArray[np.float32]
    body_confidence: npt.NDArray[np.float32]
    left_hand: npt.NDArray[np.float32]
    left_hand_confidence: npt.NDArray[np.float32]
    right_hand: npt.NDArray[np.float32]
    right_hand_confidence: npt.NDArray[np.float32]
    face: npt.NDArray[np.float32]
    face_confidence: npt.NDArray[np.float32]
    people_per_frame: npt.NDArray[np.integer[Any]]
    selected_person_index: int
    frame_valid_mask: npt.NDArray[np.bool_]

    @property
    def num_frames(self) -> int:
        """Number of frames in the processed pose sample."""

        return int(self.body.shape[0])


def load_pose_sample(path: Path | str) -> PoseSample:
    """Load and validate a processed-v1-compatible `.npz` pose sample."""

    sample_path = Path(path).expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Processed pose sample not found: {sample_path}")

    try:
        with np.load(sample_path, allow_pickle=False) as sample:
            _require_keys(sample.files, sample_path=sample_path)
            schema_version = _scalar_string(sample, "processed_schema_version", sample_path)
            if schema_version != PROCESSED_SCHEMA_VERSION:
                raise PoseSampleError(
                    f"Processed pose sample {sample_path} uses schema {schema_version!r}; "
                    f"expected {PROCESSED_SCHEMA_VERSION!r}."
                )
            selected_person_index = int(_scalar_value(sample, "selected_person_index", sample_path))
            body = _pose_array(sample, "body", expected_joints=25, sample_path=sample_path)
            body_confidence = _confidence_array(
                sample,
                "body_confidence",
                expected_shape=body.shape[:2],
                sample_path=sample_path,
            )
            left_hand = _pose_array(
                sample,
                "left_hand",
                expected_joints=21,
                sample_path=sample_path,
                expected_frames=body.shape[0],
            )
            left_hand_confidence = _confidence_array(
                sample,
                "left_hand_confidence",
                expected_shape=left_hand.shape[:2],
                sample_path=sample_path,
            )
            right_hand = _pose_array(
                sample,
                "right_hand",
                expected_joints=21,
                sample_path=sample_path,
                expected_frames=body.shape[0],
            )
            right_hand_confidence = _confidence_array(
                sample,
                "right_hand_confidence",
                expected_shape=right_hand.shape[:2],
                sample_path=sample_path,
            )
            face = _pose_array(
                sample,
                "face",
                expected_joints=70,
                sample_path=sample_path,
                expected_frames=body.shape[0],
            )
            face_confidence = _confidence_array(
                sample,
                "face_confidence",
                expected_shape=face.shape[:2],
                sample_path=sample_path,
            )
            people_per_frame = _one_dimensional_array(
                sample,
                "people_per_frame",
                expected_frames=body.shape[0],
                sample_path=sample_path,
            )
            frame_valid_mask = _one_dimensional_array(
                sample,
                "frame_valid_mask",
                expected_frames=body.shape[0],
                sample_path=sample_path,
            ).astype(np.bool_)
    except PoseSampleError:
        raise
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        raise PoseSampleError(
            f"Processed pose sample could not be read as .npz: {sample_path}: {exc}"
        ) from exc

    return PoseSample(
        path=sample_path,
        processed_schema_version=schema_version,
        body=body,
        body_confidence=body_confidence,
        left_hand=left_hand,
        left_hand_confidence=left_hand_confidence,
        right_hand=right_hand,
        right_hand_confidence=right_hand_confidence,
        face=face,
        face_confidence=face_confidence,
        people_per_frame=cast(npt.NDArray[np.integer[Any]], people_per_frame),
        selected_person_index=selected_person_index,
        frame_valid_mask=frame_valid_mask,
    )


def _require_keys(keys: list[str], *, sample_path: Path) -> None:
    observed = set(keys)
    required = {
        "processed_schema_version",
        "body",
        "body_confidence",
        "left_hand",
        "left_hand_confidence",
        "right_hand",
        "right_hand_confidence",
        "face",
        "face_confidence",
        "people_per_frame",
        "selected_person_index",
        "frame_valid_mask",
    }
    missing = sorted(required.difference(observed))
    if missing:
        raise PoseSampleError(
            f"Processed pose sample {sample_path} is missing required arrays: {missing}"
        )


def _scalar_value(sample: Any, key: str, sample_path: Path) -> Any:
    array = sample[key]
    if tuple(array.shape) != ():
        raise PoseSampleError(
            f"Processed pose sample array {key!r} in {sample_path} has shape "
            f"{tuple(array.shape)}; expected scalar ()."
        )
    return array.item()


def _scalar_string(sample: Any, key: str, sample_path: Path) -> str:
    return str(_scalar_value(sample, key, sample_path))


def _pose_array(
    sample: Any,
    key: str,
    *,
    expected_joints: int,
    sample_path: Path,
    expected_frames: int | None = None,
) -> npt.NDArray[np.float32]:
    array = np.asarray(sample[key], dtype=np.float32)
    expected_suffix = (expected_joints, 2)
    if len(array.shape) != 3 or tuple(array.shape[1:]) != expected_suffix:
        raise PoseSampleError(
            f"Processed pose sample array {key!r} in {sample_path} has shape "
            f"{tuple(array.shape)}; expected (frames, {expected_joints}, 2)."
        )
    if expected_frames is not None and int(array.shape[0]) != expected_frames:
        raise PoseSampleError(
            f"Processed pose sample array {key!r} in {sample_path} has "
            f"{array.shape[0]} frames; expected {expected_frames}."
        )
    return array


def _confidence_array(
    sample: Any,
    key: str,
    *,
    expected_shape: tuple[int, int],
    sample_path: Path,
) -> npt.NDArray[np.float32]:
    array = np.asarray(sample[key], dtype=np.float32)
    if tuple(array.shape) != expected_shape:
        raise PoseSampleError(
            f"Processed pose sample array {key!r} in {sample_path} has shape "
            f"{tuple(array.shape)}; expected {expected_shape}."
        )
    return array


def _one_dimensional_array(
    sample: Any,
    key: str,
    *,
    expected_frames: int,
    sample_path: Path,
) -> npt.NDArray[Any]:
    array = np.asarray(sample[key])
    if tuple(array.shape) != (expected_frames,):
        raise PoseSampleError(
            f"Processed pose sample array {key!r} in {sample_path} has shape "
            f"{tuple(array.shape)}; expected ({expected_frames},)."
        )
    return array
