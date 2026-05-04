"""Validation loop utilities for M0 baseline training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from text_to_sign_production.foundation.progress import (
    BatchProgress,
    NoOpProgressReporter,
    ProgressReporter,
)
from text_to_sign_production.modeling.data import M0_TARGET_CHANNELS, ProcessedPoseBatch

from .losses import channel_balanced_masked_pose_mse_loss
from .masking import build_effective_frame_mask
from .metrics import (
    masked_average_keypoint_l2_error,
    masked_pose_metric_tensors,
    metric_tensor_to_float,
)
from .standardization import (
    TargetStandardization,
    inverse_standardize_predictions,
    standardize_batch_targets,
)


@dataclass(frozen=True, slots=True)
class ValidationStepResult:
    """Scalar validation outputs for one batch.

    Carries both the per-batch metric mean (for display/logging) and the raw
    numerator/denominator sums needed for correct epoch-level aggregation.
    """

    loss: float
    metric: float
    metrics: dict[str, float]
    valid_frame_count: int
    channel_losses: dict[str, float]
    channel_valid_point_counts: dict[str, int]
    # Raw numerator/denominator sums for correct epoch aggregation.
    masked_l2_sum: float
    masked_abs_sum: float
    masked_squared_sum: float
    valid_point_count: int
    per_channel_masked_l2_sum: dict[str, float]
    per_channel_masked_abs_sum: dict[str, float]
    per_channel_masked_squared_sum: dict[str, float]
    per_channel_valid_point_count: dict[str, int]


@dataclass(frozen=True, slots=True)
class ValidationEpochResult:
    """Aggregated validation outputs for one epoch."""

    loss: float
    metric: float
    metrics: dict[str, float]
    grouped_metrics: dict[str, dict[str, float]]
    valid_frame_count: int
    channel_losses: dict[str, float]
    channel_valid_point_counts: dict[str, int]


def move_batch_to_device(
    batch: ProcessedPoseBatch,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> ProcessedPoseBatch:
    """Move tensor fields in a processed-pose batch while preserving metadata fields."""

    return ProcessedPoseBatch(
        texts=batch.texts,
        sample_ids=batch.sample_ids,
        splits=batch.splits,
        lengths=batch.lengths.to(device=device, non_blocking=non_blocking),
        body=batch.body.to(device=device, non_blocking=non_blocking),
        body_confidence=batch.body_confidence.to(device=device, non_blocking=non_blocking),
        left_hand=batch.left_hand.to(device=device, non_blocking=non_blocking),
        left_hand_confidence=batch.left_hand_confidence.to(
            device=device,
            non_blocking=non_blocking,
        ),
        right_hand=batch.right_hand.to(device=device, non_blocking=non_blocking),
        right_hand_confidence=batch.right_hand_confidence.to(
            device=device,
            non_blocking=non_blocking,
        ),
        face=batch.face.to(device=device, non_blocking=non_blocking),
        face_confidence=batch.face_confidence.to(device=device, non_blocking=non_blocking),
        padding_mask=batch.padding_mask.to(device=device, non_blocking=non_blocking),
        frame_valid_mask=batch.frame_valid_mask.to(device=device, non_blocking=non_blocking),
        people_per_frame=batch.people_per_frame.to(device=device, non_blocking=non_blocking),
        selected_person_indices=batch.selected_person_indices,
        processed_schema_versions=batch.processed_schema_versions,
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
    channel_weights: Mapping[str, float] | None = None,
    target_standardization: TargetStandardization | None = None,
) -> ValidationStepResult:
    """Run one validation step using the Phase 4 masked numeric surface."""

    model.eval()
    with torch.no_grad():
        raw_predictions = model(batch)
        standardized_targets = standardize_batch_targets(batch, target_standardization)
        loss_result = channel_balanced_masked_pose_mse_loss(
            predictions=raw_predictions,
            targets=standardized_targets,
            padding_mask=batch.padding_mask,
            frame_valid_mask=batch.frame_valid_mask,
            channel_weights=channel_weights,
        )
        predictions = inverse_standardize_predictions(raw_predictions, target_standardization)
        metric = masked_average_keypoint_l2_error(
            predictions=predictions,
            targets=batch,
            padding_mask=batch.padding_mask,
            frame_valid_mask=batch.frame_valid_mask,
        )
        metric_tensors = masked_pose_metric_tensors(
            predictions=predictions,
            targets=batch,
            padding_mask=batch.padding_mask,
            frame_valid_mask=batch.frame_valid_mask,
        )

    resolved_valid_frame_count = (
        count_valid_contributing_frames(batch) if valid_frame_count is None else valid_frame_count
    )
    metrics_float = {key: metric_tensor_to_float(value) for key, value in metric_tensors.items()}

    return ValidationStepResult(
        loss=float(loss_result.total_loss.detach().cpu().item()),
        metric=float(metric.detach().cpu().item()),
        metrics=metrics_float,
        valid_frame_count=resolved_valid_frame_count,
        channel_losses={
            channel: float(loss.detach().cpu().item())
            for channel, loss in loss_result.channel_losses.items()
        },
        channel_valid_point_counts=loss_result.detached_valid_point_counts(),
        masked_l2_sum=metrics_float.get("masked_l2_sum", 0.0),
        masked_abs_sum=metrics_float.get("masked_abs_sum", 0.0),
        masked_squared_sum=metrics_float.get("masked_squared_sum", 0.0),
        valid_point_count=int(metrics_float.get("valid_point_count", 0)),
        per_channel_masked_l2_sum={
            channel: metrics_float.get(f"per_channel_{channel}_masked_l2_sum", 0.0)
            for channel in M0_TARGET_CHANNELS
        },
        per_channel_masked_abs_sum={
            channel: metrics_float.get(f"per_channel_{channel}_masked_abs_sum", 0.0)
            for channel in M0_TARGET_CHANNELS
        },
        per_channel_masked_squared_sum={
            channel: metrics_float.get(f"per_channel_{channel}_masked_squared_sum", 0.0)
            for channel in M0_TARGET_CHANNELS
        },
        per_channel_valid_point_count={
            channel: int(metrics_float.get(f"per_channel_{channel}_valid_point_count", 0))
            for channel in M0_TARGET_CHANNELS
        },
    )


def run_validation_epoch(
    model: nn.Module,
    batches: Iterable[ProcessedPoseBatch],
    *,
    device: torch.device,
    non_blocking: bool = False,
    progress_label: str = "",
    progress_total: int | None = None,
    progress_reporter: ProgressReporter | None = None,
    progress_interval_batches: int = 100,
    channel_weights: Mapping[str, float] | None = None,
    tier_by_sample_id: Mapping[str, str] | None = None,
    target_standardization: TargetStandardization | None = None,
) -> ValidationEpochResult:
    """Aggregate validation loss and metric over an epoch.

    Epoch-level masked metrics use numerator/denominator aggregation
    (sum of raw sums / sum of valid point counts) instead of frame-count
    weighted per-batch means.  This ensures correct confidence-masked
    metric aggregation.
    """

    total_loss = 0.0
    # Raw numerator/denominator accumulators for correct aggregation.
    total_masked_l2_sum = 0.0
    total_masked_abs_sum = 0.0
    total_masked_squared_sum = 0.0
    total_valid_point_count = 0
    per_channel_l2_sums: dict[str, float] = {ch: 0.0 for ch in M0_TARGET_CHANNELS}
    per_channel_abs_sums: dict[str, float] = {ch: 0.0 for ch in M0_TARGET_CHANNELS}
    per_channel_sq_sums: dict[str, float] = {ch: 0.0 for ch in M0_TARGET_CHANNELS}
    per_channel_point_counts: dict[str, int] = {ch: 0 for ch in M0_TARGET_CHANNELS}
    grouped_l2_sums: dict[str, float] = {}
    grouped_point_counts: dict[str, int] = {}
    total_valid_frames = 0
    channel_loss_sums = {channel: 0.0 for channel in M0_TARGET_CHANNELS}
    channel_valid_point_counts = {channel: 0 for channel in M0_TARGET_CHANNELS}

    progress = _progress(
        progress_label,
        total=progress_total,
        reporter=progress_reporter,
        interval=progress_interval_batches,
    )
    for _batch_index, batch in enumerate(batches, start=1):
        device_batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
        valid_frame_count = count_valid_contributing_frames(device_batch)
        if valid_frame_count == 0:
            progress.advance()
            continue

        result = validation_step(
            model,
            device_batch,
            valid_frame_count=valid_frame_count,
            channel_weights=channel_weights,
            target_standardization=target_standardization,
        )
        total_loss += result.loss * result.valid_frame_count
        total_valid_frames += result.valid_frame_count

        # Aggregate raw numerator/denominator sums.
        total_masked_l2_sum += result.masked_l2_sum
        total_masked_abs_sum += result.masked_abs_sum
        total_masked_squared_sum += result.masked_squared_sum
        total_valid_point_count += result.valid_point_count
        for ch in M0_TARGET_CHANNELS:
            per_channel_l2_sums[ch] += result.per_channel_masked_l2_sum[ch]
            per_channel_abs_sums[ch] += result.per_channel_masked_abs_sum[ch]
            per_channel_sq_sums[ch] += result.per_channel_masked_squared_sum[ch]
            per_channel_point_counts[ch] += result.per_channel_valid_point_count[ch]
        _accumulate_grouped_metrics(
            grouped_l2_sums=grouped_l2_sums,
            grouped_point_counts=grouped_point_counts,
            result=result,
            batch=batch,
            tier_by_sample_id=tier_by_sample_id,
        )
        for channel in M0_TARGET_CHANNELS:
            valid_point_count = result.channel_valid_point_counts[channel]
            channel_valid_point_counts[channel] += valid_point_count
            channel_loss_sums[channel] += result.channel_losses[channel] * valid_point_count
        progress.advance(
            loss=f"{total_loss / max(1, total_valid_frames):.6g}",
        )

    if total_valid_frames == 0:
        raise ValueError("Validation epoch has zero valid contributing frames.")
    if total_valid_point_count == 0:
        raise ValueError("Validation epoch has zero valid contributing points.")

    # Correctly aggregated masked metrics using numerator/denominator sums.
    epoch_masked_l2 = total_masked_l2_sum / total_valid_point_count
    epoch_masked_mae = total_masked_abs_sum / total_valid_point_count
    epoch_masked_rmse = math.sqrt(total_masked_squared_sum / total_valid_point_count)

    epoch_metrics: dict[str, float] = {
        "validation_masked_l2_mean": epoch_masked_l2,
        "validation_masked_mae": epoch_masked_mae,
        "validation_masked_rmse": epoch_masked_rmse,
        "validation_metric": epoch_masked_l2,
        "valid_point_count": float(total_valid_point_count),
        "masked_l2_sum": total_masked_l2_sum,
        "masked_abs_sum": total_masked_abs_sum,
        "masked_squared_sum": total_masked_squared_sum,
    }
    for ch in M0_TARGET_CHANNELS:
        ch_count = per_channel_point_counts[ch]
        epoch_metrics[f"per_channel_{ch}_valid_point_count"] = float(ch_count)
        if ch_count > 0:
            epoch_metrics[f"per_channel_{ch}_masked_l2"] = per_channel_l2_sums[ch] / ch_count
            epoch_metrics[f"per_channel_{ch}_masked_mae"] = per_channel_abs_sums[ch] / ch_count
        else:
            epoch_metrics[f"per_channel_{ch}_masked_l2"] = 0.0
            epoch_metrics[f"per_channel_{ch}_masked_mae"] = 0.0

    # Grouped tier metrics using valid point denominators.
    grouped_metrics: dict[str, dict[str, float]] = {}
    for tier, l2_sum in sorted(grouped_l2_sums.items()):
        point_count = grouped_point_counts.get(tier, 0)
        if point_count > 0:
            grouped_metrics[tier] = {
                "validation_masked_l2_mean": l2_sum / point_count,
                "valid_point_count": float(point_count),
            }

    progress.finish(loss=f"{total_loss / total_valid_frames:.6g}")
    return ValidationEpochResult(
        loss=total_loss / total_valid_frames,
        metric=epoch_masked_l2,
        metrics=epoch_metrics,
        grouped_metrics=grouped_metrics,
        valid_frame_count=total_valid_frames,
        channel_losses=_averaged_channel_losses(
            channel_loss_sums,
            channel_valid_point_counts,
        ),
        channel_valid_point_counts=channel_valid_point_counts,
    )


def _averaged_channel_losses(
    channel_loss_sums: Mapping[str, float],
    channel_valid_point_counts: Mapping[str, int],
) -> dict[str, float]:
    return {
        channel: (
            channel_loss_sums[channel] / channel_valid_point_counts[channel]
            if channel_valid_point_counts[channel] > 0
            else 0.0
        )
        for channel in M0_TARGET_CHANNELS
    }


def _progress(
    label: str,
    *,
    total: int | None,
    reporter: ProgressReporter | None,
    interval: int,
) -> BatchProgress:
    return BatchProgress(
        label=label or "validation batch",
        total=total,
        unit="batches",
        reporter=reporter if reporter is not None else NoOpProgressReporter(),
    )


def _accumulate_grouped_metrics(
    *,
    grouped_l2_sums: dict[str, float],
    grouped_point_counts: dict[str, int],
    result: ValidationStepResult,
    batch: ProcessedPoseBatch,
    tier_by_sample_id: Mapping[str, str] | None,
) -> None:
    """Accumulate per-tier raw L2 sums and point counts for grouped metrics."""

    if tier_by_sample_id is None:
        return
    tiers = {tier_by_sample_id.get(sample_id, "broad") for sample_id in batch.sample_ids}
    tier = tiers.pop() if len(tiers) == 1 else "mixed"
    grouped_l2_sums[tier] = grouped_l2_sums.get(tier, 0.0) + result.masked_l2_sum
    grouped_point_counts[tier] = grouped_point_counts.get(tier, 0) + result.valid_point_count
