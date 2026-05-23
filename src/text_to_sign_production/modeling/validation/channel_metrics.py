"""Per-channel BFH validation metrics."""

from __future__ import annotations

import numpy as np

from text_to_sign_production.modeling.data.bfh_schema import FULL_BFH_CHANNELS, BfhPoseArrays
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import ValidationChannelMetricKey
from text_to_sign_production.modeling.validation.metrics import (
    _channel_observations,
    _require_metric_inputs,
)
from text_to_sign_production.modeling.validation.records import (
    VALIDATION_SCHEMA_VERSION,
    ValidationChannelMetricResult,
    ValidationPairingKey,
)


def compute_validation_channel_metric_results(
    *,
    key: ValidationPairingKey,
    reference: BfhPoseArrays,
    generated: BfhPoseArrays,
) -> tuple[ValidationChannelMetricResult, ...]:
    """Compute errors independently for body, hands, and face channels."""

    _require_metric_inputs(key, reference, generated)
    results: list[ValidationChannelMetricResult] = []
    for channel in FULL_BFH_CHANNELS:
        observation = _channel_observations(channel, reference, generated)
        issues: list[str] = []
        for metric_key, values, unavailable_issue, metric_frames in (
            (
                ValidationChannelMetricKey.CHANNEL_MASKED_L1_MEAN,
                observation.coordinate_errors,
                "channel_masked_l1_mean has no aligned valid coordinate observations.",
                observation.frame_count,
            ),
            (
                ValidationChannelMetricKey.CHANNEL_MASKED_L2_MEAN,
                observation.distance_errors,
                "channel_masked_l2_mean has no aligned valid joint observations.",
                observation.frame_count,
            ),
            (
                ValidationChannelMetricKey.CHANNEL_VELOCITY_L1_MEAN,
                observation.velocity_coordinate_errors,
                "channel_velocity_l1_mean has no consecutive aligned valid observations.",
                max(observation.frame_count - 1, 0),
            ),
            (
                ValidationChannelMetricKey.CHANNEL_VELOCITY_L2_MEAN,
                observation.velocity_distance_errors,
                "channel_velocity_l2_mean has no consecutive aligned valid observations.",
                max(observation.frame_count - 1, 0),
            ),
        ):
            if values.size == 0:
                issues.append(unavailable_issue)
                continue
            value = float(np.mean(values))
            if not np.isfinite(value):
                raise ModelValidationError(
                    f"{metric_key.value}:{channel.value} computation produced a non-finite value."
                )
            results.append(
                ValidationChannelMetricResult(
                    VALIDATION_SCHEMA_VERSION,
                    metric_key,
                    channel,
                    key.split,
                    key.sample_id,
                    key.generation_index,
                    value,
                    metric_frames,
                    observation.joint_count,
                    int(values.size),
                    (),
                )
            )
        if observation.generated_expected_joint_count <= 0:
            raise ModelValidationError(
                f"channel_valid_joint_coverage:{channel.value} has no expected generated joints."
            )
        results.append(
            ValidationChannelMetricResult(
                VALIDATION_SCHEMA_VERSION,
                ValidationChannelMetricKey.CHANNEL_VALID_JOINT_COVERAGE,
                channel,
                key.split,
                key.sample_id,
                key.generation_index,
                float(
                    observation.generated_valid_joint_count
                    / observation.generated_expected_joint_count
                ),
                generated.frame_count,
                observation.joint_count,
                observation.generated_expected_joint_count,
                tuple(issues),
            )
        )
    return tuple(results)


__all__ = ["compute_validation_channel_metric_results"]
