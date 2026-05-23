"""Automatic full-BFH validation metrics using explicit prefix alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    FULL_BFH_CHANNELS,
    BfhPoseArrays,
)
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import ValidationMetricKey
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationMetricResult,
    ValidationPairingKey,
)


@dataclass(frozen=True, slots=True)
class _ChannelObservations:
    frame_count: int
    joint_count: int
    coordinate_errors: np.ndarray
    distance_errors: np.ndarray
    velocity_coordinate_errors: np.ndarray
    velocity_distance_errors: np.ndarray
    generated_valid_joint_count: int
    generated_expected_joint_count: int


def compute_validation_metric_results(
    *,
    key: ValidationPairingKey,
    reference: BfhPoseArrays,
    generated: BfhPoseArrays,
) -> tuple[ValidationMetricResult, ...]:
    """Compute full-BFH pose errors on aligned valid observations."""

    _require_metric_inputs(key, reference, generated)
    observations = tuple(
        _channel_observations(channel, reference, generated) for channel in FULL_BFH_CHANNELS
    )
    frame_count = min(reference.frame_count, generated.frame_count)
    joint_count = sum(item.joint_count for item in observations)
    coordinate_errors = _concat(item.coordinate_errors for item in observations)
    distance_errors = _concat(item.distance_errors for item in observations)
    velocity_coordinate_errors = _concat(
        item.velocity_coordinate_errors for item in observations
    )
    velocity_distance_errors = _concat(item.velocity_distance_errors for item in observations)
    issues: list[str] = []
    results: list[ValidationMetricResult] = []
    _append_observed_metric(
        results,
        issues,
        key=key,
        metric_key=ValidationMetricKey.MASKED_L1_MEAN,
        values=coordinate_errors,
        frame_count=frame_count,
        joint_count=joint_count,
        unavailable_issue="masked_l1_mean has no aligned valid coordinate observations.",
    )
    _append_observed_metric(
        results,
        issues,
        key=key,
        metric_key=ValidationMetricKey.MASKED_L2_MEAN,
        values=distance_errors,
        frame_count=frame_count,
        joint_count=joint_count,
        unavailable_issue="masked_l2_mean has no aligned valid joint observations.",
    )
    _append_observed_metric(
        results,
        issues,
        key=key,
        metric_key=ValidationMetricKey.VELOCITY_L1_MEAN,
        values=velocity_coordinate_errors,
        frame_count=max(frame_count - 1, 0),
        joint_count=joint_count,
        unavailable_issue="velocity_l1_mean has no consecutive aligned valid observations.",
    )
    _append_observed_metric(
        results,
        issues,
        key=key,
        metric_key=ValidationMetricKey.VELOCITY_L2_MEAN,
        values=velocity_distance_errors,
        frame_count=max(frame_count - 1, 0),
        joint_count=joint_count,
        unavailable_issue="velocity_l2_mean has no consecutive aligned valid observations.",
    )
    results.append(
        ValidationMetricResult(
            VALIDATION_SCHEMA_VERSION,
            ValidationMetricKey.SEQUENCE_LENGTH_ABSOLUTE_ERROR,
            key.split,
            key.sample_id,
            key.generation_index,
            float(abs(generated.frame_count - reference.frame_count)),
            frame_count,
            joint_count,
            1,
            tuple(issues),
        )
    )
    valid_joint_count = sum(item.generated_valid_joint_count for item in observations)
    expected_joint_count = sum(item.generated_expected_joint_count for item in observations)
    if expected_joint_count <= 0:
        raise ModelValidationError("valid_joint_coverage has no expected generated joints.")
    results.append(
        ValidationMetricResult(
            VALIDATION_SCHEMA_VERSION,
            ValidationMetricKey.VALID_JOINT_COVERAGE,
            key.split,
            key.sample_id,
            key.generation_index,
            float(valid_joint_count / expected_joint_count),
            generated.frame_count,
            joint_count,
            expected_joint_count,
            (),
        )
    )
    return tuple(results)


def _channel_observations(
    channel: PoseChannel,
    reference: BfhPoseArrays,
    generated: BfhPoseArrays,
) -> _ChannelObservations:
    frame_count = min(reference.frame_count, generated.frame_count)
    reference_xy = reference.coordinates(channel)[:frame_count]
    generated_xy = generated.coordinates(channel)[:frame_count]
    valid = (
        reference.valid_frame_mask[:frame_count, None]
        & generated.valid_frame_mask[:frame_count, None]
        & (reference.confidence(channel)[:frame_count] > 0.0)
        & (generated.confidence(channel)[:frame_count] > 0.0)
        & np.all(np.isfinite(reference_xy), axis=-1)
        & np.all(np.isfinite(generated_xy), axis=-1)
    )
    delta = generated_xy - reference_xy
    coordinate_errors = np.abs(delta[valid]).reshape(-1)
    distance_errors = np.linalg.norm(delta[valid], axis=-1)
    velocity_valid = valid[1:] & valid[:-1] if frame_count >= 2 else np.zeros((0, valid.shape[1]), dtype=np.bool_)
    if frame_count >= 2:
        velocity_delta = np.diff(generated_xy, axis=0) - np.diff(reference_xy, axis=0)
        velocity_coordinate_errors = np.abs(velocity_delta[velocity_valid]).reshape(-1)
        velocity_distance_errors = np.linalg.norm(velocity_delta[velocity_valid], axis=-1)
    else:
        velocity_coordinate_errors = np.asarray([], dtype=np.float32)
        velocity_distance_errors = np.asarray([], dtype=np.float32)
    generated_xy_full = generated.coordinates(channel)
    generated_valid = (
        generated.valid_frame_mask[:, None]
        & (generated.confidence(channel) > 0.0)
        & np.all(np.isfinite(generated_xy_full), axis=-1)
    )
    joint_count = BFH_CHANNEL_SPECS[channel].joint_count
    return _ChannelObservations(
        frame_count=frame_count,
        joint_count=joint_count,
        coordinate_errors=coordinate_errors,
        distance_errors=distance_errors,
        velocity_coordinate_errors=velocity_coordinate_errors,
        velocity_distance_errors=velocity_distance_errors,
        generated_valid_joint_count=int(np.count_nonzero(generated_valid)),
        generated_expected_joint_count=generated.frame_count * joint_count,
    )


def _append_observed_metric(
    results: list[ValidationMetricResult],
    issues: list[str],
    *,
    key: ValidationPairingKey,
    metric_key: ValidationMetricKey,
    values: np.ndarray,
    frame_count: int,
    joint_count: int,
    unavailable_issue: str,
) -> None:
    if values.size == 0:
        issues.append(unavailable_issue)
        return
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise ModelValidationError(f"{metric_key.value} computation produced a non-finite value.")
    results.append(
        ValidationMetricResult(
            VALIDATION_SCHEMA_VERSION,
            metric_key,
            key.split,
            key.sample_id,
            key.generation_index,
            value,
            frame_count,
            joint_count,
            int(values.size),
            (),
        )
    )


def _concat(values) -> np.ndarray:
    materialized = tuple(value for value in values if value.size)
    if not materialized:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(materialized)


def _require_metric_inputs(
    key: ValidationPairingKey,
    reference: BfhPoseArrays,
    generated: BfhPoseArrays,
) -> None:
    if not isinstance(key, ValidationPairingKey):
        raise ModelValidationError("metric key must be a ValidationPairingKey.")
    if not isinstance(reference, BfhPoseArrays) or not isinstance(generated, BfhPoseArrays):
        raise ModelValidationError("validation metrics require BfhPoseArrays inputs.")


__all__ = ["compute_validation_metric_results"]
