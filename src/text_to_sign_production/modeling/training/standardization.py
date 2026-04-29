"""Target standardization helpers for M0 pose regression."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from text_to_sign_production.modeling.data import M0_TARGET_CHANNELS, ProcessedPoseBatch


@dataclass(frozen=True, slots=True)
class TargetStandardization:
    """Per-channel scalar target transform fitted on supervised train points."""

    mean_by_channel: Mapping[str, float]
    std_by_channel: Mapping[str, float]
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mean_by_channel": dict(self.mean_by_channel),
            "std_by_channel": dict(self.std_by_channel),
            "epsilon": self.epsilon,
            "coordinate_space": "canonical_fixed_canvas_xy",
        }


def standardize_batch_targets(
    batch: ProcessedPoseBatch,
    transform: TargetStandardization | None,
) -> ProcessedPoseBatch:
    """Return a batch with target coordinates standardized, preserving masks."""

    if transform is None:
        return batch
    return ProcessedPoseBatch(
        texts=batch.texts,
        sample_ids=batch.sample_ids,
        splits=batch.splits,
        lengths=batch.lengths,
        body=_standardize_tensor(batch.body, transform, "body"),
        body_confidence=batch.body_confidence,
        left_hand=_standardize_tensor(batch.left_hand, transform, "left_hand"),
        left_hand_confidence=batch.left_hand_confidence,
        right_hand=_standardize_tensor(batch.right_hand, transform, "right_hand"),
        right_hand_confidence=batch.right_hand_confidence,
        face=_standardize_tensor(batch.face, transform, "face"),
        face_confidence=batch.face_confidence,
        padding_mask=batch.padding_mask,
        frame_valid_mask=batch.frame_valid_mask,
        people_per_frame=batch.people_per_frame,
        selected_person_indices=batch.selected_person_indices,
        processed_schema_versions=batch.processed_schema_versions,
        fps=batch.fps,
        num_frames=batch.num_frames,
    )


def inverse_standardize_predictions(
    predictions: Any,
    transform: TargetStandardization | Mapping[str, Any] | None,
) -> Any:
    """Return prediction object in canonical fixed-canvas coordinate space."""

    if transform is None:
        return predictions
    resolved = (
        transform if isinstance(transform, TargetStandardization) else from_mapping(transform)
    )
    if resolved is None:
        return predictions
    from text_to_sign_production.modeling.models import BaselinePoseOutput

    return BaselinePoseOutput(
        body=_inverse_tensor(predictions.body, resolved, "body"),
        left_hand=_inverse_tensor(predictions.left_hand, resolved, "left_hand"),
        right_hand=_inverse_tensor(predictions.right_hand, resolved, "right_hand"),
        face=_inverse_tensor(predictions.face, resolved, "face"),
    )


def from_mapping(payload: Mapping[str, Any]) -> TargetStandardization | None:
    """Parse checkpoint target-standardization metadata."""

    if not payload or payload.get("enabled") is False:
        return None
    mean_by_channel = {
        channel: float(dict(payload["mean_by_channel"])[channel]) for channel in M0_TARGET_CHANNELS
    }
    std_by_channel = {
        channel: float(dict(payload["std_by_channel"])[channel]) for channel in M0_TARGET_CHANNELS
    }
    return TargetStandardization(
        mean_by_channel=mean_by_channel,
        std_by_channel=std_by_channel,
        epsilon=float(payload.get("epsilon", 1e-6)),
    )


def _standardize_tensor(
    value: torch.Tensor,
    transform: TargetStandardization,
    channel: str,
) -> torch.Tensor:
    mean = torch.as_tensor(
        transform.mean_by_channel[channel], device=value.device, dtype=value.dtype
    )
    std = torch.as_tensor(transform.std_by_channel[channel], device=value.device, dtype=value.dtype)
    return (value - mean) / std.clamp_min(transform.epsilon)


def _inverse_tensor(
    value: torch.Tensor,
    transform: TargetStandardization,
    channel: str,
) -> torch.Tensor:
    mean = torch.as_tensor(
        transform.mean_by_channel[channel], device=value.device, dtype=value.dtype
    )
    std = torch.as_tensor(transform.std_by_channel[channel], device=value.device, dtype=value.dtype)
    return value * std.clamp_min(transform.epsilon) + mean


__all__ = [
    "TargetStandardization",
    "from_mapping",
    "inverse_standardize_predictions",
    "standardize_batch_targets",
]
