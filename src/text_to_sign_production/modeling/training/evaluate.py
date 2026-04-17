"""Validation loop utilities for Sprint 3 baseline training."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from text_to_sign_production.modeling.data import ProcessedPoseBatch

from .losses import masked_pose_mse_loss
from .masking import build_effective_frame_mask
from .metrics import masked_average_keypoint_l2_error


@dataclass(frozen=True, slots=True)
class ValidationStepResult:
    """Scalar validation outputs for one batch."""

    loss: float
    metric: float
    valid_frame_count: int


@dataclass(frozen=True, slots=True)
class ValidationEpochResult:
    """Aggregated validation outputs for one epoch."""

    loss: float
    metric: float
    valid_frame_count: int


def move_batch_to_device(batch: ProcessedPoseBatch, device: torch.device) -> ProcessedPoseBatch:
    """Move tensor fields in a processed-pose batch while preserving metadata fields."""

    return ProcessedPoseBatch(
        texts=batch.texts,
        sample_ids=batch.sample_ids,
        splits=batch.splits,
        lengths=batch.lengths.to(device=device),
        body=batch.body.to(device=device),
        left_hand=batch.left_hand.to(device=device),
        right_hand=batch.right_hand.to(device=device),
        padding_mask=batch.padding_mask.to(device=device),
        frame_valid_mask=batch.frame_valid_mask.to(device=device),
        fps=batch.fps,
        num_frames=batch.num_frames,
    )


def count_valid_contributing_frames(batch: ProcessedPoseBatch) -> int:
    """Count real valid frames contributing to losses and metrics."""

    effective_frame_mask = build_effective_frame_mask(
        padding_mask=batch.padding_mask,
        frame_valid_mask=batch.frame_valid_mask,
    )
    return int(effective_frame_mask.sum().item())


def validation_step(
    model: nn.Module,
    batch: ProcessedPoseBatch,
    *,
    valid_frame_count: int | None = None,
) -> ValidationStepResult:
    """Run one validation step using the Phase 4 masked numeric surface."""

    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        loss = masked_pose_mse_loss(
            predictions=predictions,
            targets=batch,
            padding_mask=batch.padding_mask,
            frame_valid_mask=batch.frame_valid_mask,
        )
        metric = masked_average_keypoint_l2_error(
            predictions=predictions,
            targets=batch,
            padding_mask=batch.padding_mask,
            frame_valid_mask=batch.frame_valid_mask,
        )

    return ValidationStepResult(
        loss=float(loss.detach().cpu().item()),
        metric=float(metric.detach().cpu().item()),
        valid_frame_count=(
            count_valid_contributing_frames(batch)
            if valid_frame_count is None
            else valid_frame_count
        ),
    )


def run_validation_epoch(
    model: nn.Module,
    batches: Iterable[ProcessedPoseBatch],
    *,
    device: torch.device,
) -> ValidationEpochResult:
    """Aggregate validation loss and metric over an epoch."""

    total_loss = 0.0
    total_metric = 0.0
    total_valid_frames = 0

    for batch in batches:
        device_batch = move_batch_to_device(batch, device)
        valid_frame_count = count_valid_contributing_frames(device_batch)
        if valid_frame_count == 0:
            continue

        result = validation_step(
            model,
            device_batch,
            valid_frame_count=valid_frame_count,
        )
        total_loss += result.loss * result.valid_frame_count
        total_metric += result.metric * result.valid_frame_count
        total_valid_frames += result.valid_frame_count

    if total_valid_frames == 0:
        raise ValueError("Validation epoch has zero valid contributing frames.")

    return ValidationEpochResult(
        loss=total_loss / total_valid_frames,
        metric=total_metric / total_valid_frames,
        valid_frame_count=total_valid_frames,
    )
