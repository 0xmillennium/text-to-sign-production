"""Processed-v1 BFH data contracts for the M0 baseline modeling layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias

import numpy as np
import numpy.typing as npt

from text_to_sign_production.data.pose.schema import OPENPOSE_CHANNEL_SPECS

M0_TARGET_CHANNELS: Final[tuple[str, str, str, str]] = (
    "body",
    "left_hand",
    "right_hand",
    "face",
)
M0_TARGET_CHANNEL_SHAPES: Final[dict[str, tuple[int, int]]] = {
    channel: (OPENPOSE_CHANNEL_SPECS[channel][1], 2) for channel in M0_TARGET_CHANNELS
}
M0_CONFIDENCE_CHANNELS: Final[tuple[str, str, str, str]] = (
    "body_confidence",
    "left_hand_confidence",
    "right_hand_confidence",
    "face_confidence",
)
M0_CHANNEL_POLICY: Final[str] = "full_bfh"

PoseArray: TypeAlias = npt.NDArray[np.float32]
ConfidenceArray: TypeAlias = npt.NDArray[np.float32]
MaskArray: TypeAlias = npt.NDArray[np.bool_]
IntegerArray: TypeAlias = npt.NDArray[np.integer[Any]]
TorchTensor: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class ProcessedModelingManifestRecord:
    """Manifest metadata needed to build one M0 modeling item."""

    sample_id: str
    split: str
    text: str
    fps: float | None
    num_frames: int
    sample_path: Path
    sample_path_value: str
    processed_schema_version: str
    selected_person_index: int
    multi_person_frame_count: int
    max_people_per_frame: int
    frame_valid_count: int
    frame_invalid_count: int


@dataclass(frozen=True, slots=True)
class ProcessedPoseSample:
    """Channel-separated BFH arrays loaded from one processed-v1 `.npz` sample."""

    processed_schema_version: str
    body: PoseArray
    body_confidence: ConfidenceArray
    left_hand: PoseArray
    left_hand_confidence: ConfidenceArray
    right_hand: PoseArray
    right_hand_confidence: ConfidenceArray
    face: PoseArray
    face_confidence: ConfidenceArray
    frame_valid_mask: MaskArray
    people_per_frame: IntegerArray
    selected_person_index: int

    @property
    def num_frames(self) -> int:
        """Number of timesteps in the pose sample."""

        return int(self.body.shape[0])


@dataclass(frozen=True, slots=True)
class ProcessedPoseItem:
    """One M0 modeling item from one manifest record plus one processed-v1 `.npz` sample."""

    sample_id: str
    split: str
    text: str
    fps: float | None
    num_frames: int
    sample_path: Path
    sample_path_value: str
    processed_schema_version: str
    selected_person_index: int
    body: PoseArray
    body_confidence: ConfidenceArray
    left_hand: PoseArray
    left_hand_confidence: ConfidenceArray
    right_hand: PoseArray
    right_hand_confidence: ConfidenceArray
    face: PoseArray
    face_confidence: ConfidenceArray
    frame_valid_mask: MaskArray
    people_per_frame: IntegerArray

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
            processed_schema_version=pose_sample.processed_schema_version,
            selected_person_index=pose_sample.selected_person_index,
            body=pose_sample.body,
            body_confidence=pose_sample.body_confidence,
            left_hand=pose_sample.left_hand,
            left_hand_confidence=pose_sample.left_hand_confidence,
            right_hand=pose_sample.right_hand,
            right_hand_confidence=pose_sample.right_hand_confidence,
            face=pose_sample.face,
            face_confidence=pose_sample.face_confidence,
            frame_valid_mask=pose_sample.frame_valid_mask,
            people_per_frame=pose_sample.people_per_frame,
        )

    @property
    def length(self) -> int:
        """Number of real timesteps before batching/padding."""

        return int(self.body.shape[0])


@dataclass(frozen=True, slots=True)
class ProcessedPoseBatch:
    """Variable-length full-BFH batch contract for M0 model training."""

    texts: list[str]
    sample_ids: list[str]
    splits: list[str]
    lengths: TorchTensor
    body: TorchTensor
    body_confidence: TorchTensor
    left_hand: TorchTensor
    left_hand_confidence: TorchTensor
    right_hand: TorchTensor
    right_hand_confidence: TorchTensor
    face: TorchTensor
    face_confidence: TorchTensor
    padding_mask: TorchTensor
    frame_valid_mask: TorchTensor
    people_per_frame: TorchTensor
    selected_person_indices: list[int]
    processed_schema_versions: list[str]
    fps: list[float | None]
    num_frames: list[int]
