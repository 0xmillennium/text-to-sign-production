"""Processed-data contracts for Sprint 3 baseline modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.constants import OPENPOSE_CHANNEL_SPECS

SPRINT3_TARGET_CHANNELS: Final[tuple[str, str, str]] = (
    "body",
    "left_hand",
    "right_hand",
)
SPRINT3_TARGET_CHANNEL_SHAPES: Final[dict[str, tuple[int, int]]] = {
    channel: (OPENPOSE_CHANNEL_SPECS[channel][1], 2) for channel in SPRINT3_TARGET_CHANNELS
}

PoseArray: TypeAlias = npt.NDArray[np.float32]
MaskArray: TypeAlias = npt.NDArray[np.bool_]
TorchTensor: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class ProcessedModelingManifestRecord:
    """Manifest metadata needed to build one Sprint 3 modeling item."""

    sample_id: str
    split: str
    text: str
    fps: float | None
    num_frames: int
    sample_path: Path
    sample_path_value: str
    processed_schema_version: str
    selected_person_index: int
    frame_valid_count: int
    frame_invalid_count: int


@dataclass(frozen=True, slots=True)
class ProcessedPoseSample:
    """Channel-separated pose arrays loaded from one processed `.npz` sample."""

    body: PoseArray
    left_hand: PoseArray
    right_hand: PoseArray
    frame_valid_mask: MaskArray

    @property
    def num_frames(self) -> int:
        """Number of timesteps in the pose sample."""

        return int(self.body.shape[0])


@dataclass(frozen=True, slots=True)
class ProcessedPoseItem:
    """One Sprint 3 modeling item from one manifest record plus one `.npz` sample."""

    sample_id: str
    split: str
    text: str
    fps: float | None
    num_frames: int
    sample_path: Path
    sample_path_value: str
    body: PoseArray
    left_hand: PoseArray
    right_hand: PoseArray
    frame_valid_mask: MaskArray

    @classmethod
    def from_manifest_and_sample(
        cls,
        manifest_record: ProcessedModelingManifestRecord,
        pose_sample: ProcessedPoseSample,
    ) -> ProcessedPoseItem:
        """Combine processed manifest metadata with loaded pose arrays."""

        return cls(
            sample_id=manifest_record.sample_id,
            split=manifest_record.split,
            text=manifest_record.text,
            fps=manifest_record.fps,
            num_frames=manifest_record.num_frames,
            sample_path=manifest_record.sample_path,
            sample_path_value=manifest_record.sample_path_value,
            body=pose_sample.body,
            left_hand=pose_sample.left_hand,
            right_hand=pose_sample.right_hand,
            frame_valid_mask=pose_sample.frame_valid_mask,
        )

    @property
    def length(self) -> int:
        """Number of real timesteps before batching/padding."""

        return int(self.body.shape[0])


@dataclass(frozen=True, slots=True)
class ProcessedPoseBatch:
    """Variable-length batch contract for future Sprint 3 model training."""

    texts: list[str]
    sample_ids: list[str]
    splits: list[str]
    lengths: TorchTensor
    body: TorchTensor
    left_hand: TorchTensor
    right_hand: TorchTensor
    padding_mask: TorchTensor
    frame_valid_mask: TorchTensor
    fps: list[float | None]
    num_frames: list[int]
