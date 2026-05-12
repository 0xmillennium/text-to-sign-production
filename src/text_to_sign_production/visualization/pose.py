"""PreparedSample pose array loading for rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.dataset import PREPARED_SAMPLE_SCHEMA_VERSION
from text_to_sign_production.data.dataset.payloads import load_prepared_sample_payload


class PoseSampleError(ValueError):
    """Raised when a prepared pose sample cannot be loaded for rendering."""


@dataclass(frozen=True, slots=True)
class PoseSample:
    """PreparedSample-compatible pose arrays loaded from a `.npz` file."""

    path: Path
    schema_version: str
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
        """Number of frames in the prepared pose sample."""

        return int(self.body.shape[0])


def load_pose_sample(path: Path | str) -> PoseSample:
    """Load and validate a PreparedSample-compatible `.npz` pose sample."""

    sample_path = Path(path).expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Prepared pose sample not found: {sample_path}")
    try:
        sample = load_prepared_sample_payload(sample_path)
    except (OSError, ValueError) as exc:
        raise PoseSampleError(
            f"Prepared pose sample could not be read as .npz: {sample_path}: {exc}"
        ) from exc
    if sample.schema_version != PREPARED_SAMPLE_SCHEMA_VERSION:
        raise PoseSampleError(
            f"Prepared pose sample {sample_path} uses schema {sample.schema_version!r}; "
            f"expected {PREPARED_SAMPLE_SCHEMA_VERSION!r}."
        )

    return PoseSample(
        path=sample_path,
        schema_version=sample.schema_version,
        body=_coordinates(sample.pose.body_xyc),
        body_confidence=_confidence(sample.pose.body_xyc),
        left_hand=_coordinates(sample.pose.left_hand_xyc),
        left_hand_confidence=_confidence(sample.pose.left_hand_xyc),
        right_hand=_coordinates(sample.pose.right_hand_xyc),
        right_hand_confidence=_confidence(sample.pose.right_hand_xyc),
        face=_coordinates(sample.pose.face_xyc),
        face_confidence=_confidence(sample.pose.face_xyc),
        people_per_frame=_people_per_frame(sample.pose.selected_person_indices),
        selected_person_index=_representative_selected_person_index(
            sample.pose.selected_person_indices
        ),
        frame_valid_mask=cast(
            npt.NDArray[np.bool_],
            np.asarray(sample.pose.valid_frame_mask, dtype=np.bool_),
        ),
    )


def _coordinates(xyc: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return cast(npt.NDArray[np.float32], np.asarray(xyc[..., :2], dtype=np.float32))


def _confidence(xyc: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return cast(npt.NDArray[np.float32], np.asarray(xyc[..., 2], dtype=np.float32))


def _people_per_frame(indices: tuple[int | None, ...]) -> npt.NDArray[np.integer[Any]]:
    values = [0 if index is None else 1 for index in indices]
    return cast(npt.NDArray[np.integer[Any]], np.asarray(values, dtype=np.int16))


def _representative_selected_person_index(indices: tuple[int | None, ...]) -> int:
    for index in indices:
        if index is not None:
            return index
    return -1
