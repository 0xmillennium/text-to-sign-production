"""Variable-length collation for M0 full-BFH processed pose samples."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from .schemas import (
    M0_TARGET_CHANNEL_SHAPES,
    M0_TARGET_CHANNELS,
    ProcessedPoseBatch,
    ProcessedPoseItem,
)


def _load_torch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "M0 processed-pose collation requires torch. "
                "Install the modeling extra or run inside the configured modeling environment."
            ) from exc
        raise


def collate_processed_pose_samples(samples: Sequence[ProcessedPoseItem]) -> ProcessedPoseBatch:
    """Pad variable-length processed pose samples into the M0 full-BFH batch contract."""

    if not samples:
        raise ValueError("Cannot collate an empty processed-pose batch.")

    torch = _load_torch()
    batch_size = len(samples)
    lengths = [sample.length for sample in samples]
    max_length = max(lengths)

    channel_tensors = {
        channel: torch.zeros(
            (
                batch_size,
                max_length,
                M0_TARGET_CHANNEL_SHAPES[channel][0],
                M0_TARGET_CHANNEL_SHAPES[channel][1],
            ),
            dtype=torch.float32,
        )
        for channel in M0_TARGET_CHANNELS
    }
    confidence_tensors = {
        channel: torch.zeros(
            (
                batch_size,
                max_length,
                M0_TARGET_CHANNEL_SHAPES[channel][0],
            ),
            dtype=torch.float32,
        )
        for channel in M0_TARGET_CHANNELS
    }
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    frame_valid_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    people_per_frame = torch.zeros((batch_size, max_length), dtype=torch.int16)

    for batch_index, sample in enumerate(samples):
        length = sample.length
        if length != sample.num_frames:
            raise ValueError(
                f"Sample {sample.sample_id!r} has length {length}, "
                f"but metadata num_frames={sample.num_frames}."
            )
        if length != int(sample.frame_valid_mask.shape[0]):
            raise ValueError(
                f"Sample {sample.sample_id!r} frame_valid_mask length does not match pose length."
            )
        if length != int(sample.people_per_frame.shape[0]):
            raise ValueError(
                f"Sample {sample.sample_id!r} people_per_frame length does not match pose length."
            )
        if length == 0:
            continue

        for channel in M0_TARGET_CHANNELS:
            channel_tensors[channel][batch_index, :length] = torch.as_tensor(
                getattr(sample, channel),
                dtype=torch.float32,
            )
            confidence_tensors[channel][batch_index, :length] = torch.as_tensor(
                getattr(sample, f"{channel}_confidence"),
                dtype=torch.float32,
            )
        padding_mask[batch_index, :length] = False
        frame_valid_mask[batch_index, :length] = torch.as_tensor(
            sample.frame_valid_mask,
            dtype=torch.bool,
        )
        people_per_frame[batch_index, :length] = torch.as_tensor(
            sample.people_per_frame,
            dtype=torch.int16,
        )

    return ProcessedPoseBatch(
        texts=[sample.text for sample in samples],
        sample_ids=[sample.sample_id for sample in samples],
        splits=[sample.split for sample in samples],
        lengths=torch.as_tensor(lengths, dtype=torch.long),
        body=channel_tensors["body"],
        body_confidence=confidence_tensors["body"],
        left_hand=channel_tensors["left_hand"],
        left_hand_confidence=confidence_tensors["left_hand"],
        right_hand=channel_tensors["right_hand"],
        right_hand_confidence=confidence_tensors["right_hand"],
        face=channel_tensors["face"],
        face_confidence=confidence_tensors["face"],
        padding_mask=padding_mask,
        frame_valid_mask=frame_valid_mask,
        people_per_frame=people_per_frame,
        selected_person_indices=[sample.selected_person_index for sample in samples],
        processed_schema_versions=[sample.processed_schema_version for sample in samples],
        fps=[sample.fps for sample in samples],
        num_frames=[sample.num_frames for sample in samples],
    )
