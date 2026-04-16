"""Processed `.npz` sample builders for validation and workflow tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.constants import PROCESSED_SCHEMA_VERSION


def processed_sample_payload(
    *,
    num_frames: int = 2,
    schema_version: str = PROCESSED_SCHEMA_VERSION,
    selected_person_index: int = 0,
    frame_valid_mask: npt.NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    resolved_frame_valid_mask = (
        np.ones((num_frames,), dtype=np.bool_)
        if frame_valid_mask is None
        else frame_valid_mask.astype(np.bool_)
    )
    return {
        "processed_schema_version": np.asarray(schema_version),
        "body": np.zeros((num_frames, 25, 2), dtype=np.float32),
        "body_confidence": np.ones((num_frames, 25), dtype=np.float32),
        "left_hand": np.zeros((num_frames, 21, 2), dtype=np.float32),
        "left_hand_confidence": np.ones((num_frames, 21), dtype=np.float32),
        "right_hand": np.zeros((num_frames, 21, 2), dtype=np.float32),
        "right_hand_confidence": np.ones((num_frames, 21), dtype=np.float32),
        "face": np.zeros((num_frames, 70, 2), dtype=np.float32),
        "face_confidence": np.zeros((num_frames, 70), dtype=np.float32),
        "people_per_frame": np.ones((num_frames,), dtype=np.int16),
        "selected_person_index": np.asarray(selected_person_index, dtype=np.int16),
        "frame_valid_mask": resolved_frame_valid_mask,
    }


def write_processed_sample_npz(
    path: Path,
    *,
    num_frames: int = 2,
    drop_keys: tuple[str, ...] = (),
    overrides: dict[str, Any] | None = None,
) -> None:
    payload = processed_sample_payload(num_frames=num_frames)
    if overrides is not None:
        payload.update(overrides)
    for key in drop_keys:
        payload.pop(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
