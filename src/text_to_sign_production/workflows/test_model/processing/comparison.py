"""Single-sample reference-vs-generated pose comparison for test_model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.progress import ProgressSession
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data.bfh_schema import (
    BFH_CHANNEL_SPECS,
    FULL_BFH_CHANNELS,
    BfhPoseArrays,
)
from text_to_sign_production.workflows.test_model.constants import (
    TEST_MODEL_STAGE_REFERENCE_COMPARE,
)
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelChannelComparisonMetric,
    TestModelComparisonMetric,
    TestModelInferenceResult,
    TestModelReferenceComparisonResult,
    TestModelTargetResolution,
)
from text_to_sign_production.workflows.test_model.layout import TestModelLayout
from text_to_sign_production.workflows.test_model.progress import test_model_progress_stage

REFERENCE_COMPARISON_SCHEMA_VERSION = "test_model_reference_comparison.v1"


@dataclass(frozen=True, slots=True)
class _ChannelObservations:
    channel: PoseChannel
    frame_count: int
    generated_frame_count: int
    joint_count: int
    coordinate_errors: np.ndarray
    distance_errors: np.ndarray
    velocity_coordinate_errors: np.ndarray
    velocity_distance_errors: np.ndarray
    generated_valid_joint_count: int
    generated_expected_joint_count: int


def compare_reference_and_generated(
    layout: TestModelLayout,
    *,
    target: TestModelTargetResolution,
    inference: TestModelInferenceResult,
    progress_session: ProgressSession | None = None,
) -> TestModelReferenceComparisonResult:
    del layout
    if progress_session is None:
        return _compare_reference_and_generated(target=target, inference=inference)
    with progress_session.task(
        test_model_progress_stage(
            stage_id=TEST_MODEL_STAGE_REFERENCE_COMPARE,
            label="test_model reference comparison",
            unit="sample",
            owner_module=__name__,
            operation_kind="reference_comparison",
            total_semantics="single target reference-vs-generated comparison",
            allowed_counters=("compared", "failed", "not_ready"),
        ),
        total=1,
    ) as task:
        result = _compare_reference_and_generated(target=target, inference=inference)
        task.advance(
            1,
            counters={
                "compared": 1 if result.status == "completed" else 0,
                "not_ready": 1 if result.status == "not_ready" else 0,
                "failed": 1 if result.status == "failed" else 0,
            },
        )
        return result


def _compare_reference_and_generated(
    *,
    target: TestModelTargetResolution,
    inference: TestModelInferenceResult,
) -> TestModelReferenceComparisonResult:
    errors = _prerequisite_errors(target=target, inference=inference)
    reference_sample = target.manifest_sample
    provider_result = inference.provider_result
    generated_sample = None if provider_result is None else getattr(
        provider_result, "generated_sample", None
    )
    if errors:
        return _result(
            status="not_ready",
            split=target.split,
            reference_sample_id=None if reference_sample is None else reference_sample.sample_id,
            generated_sample_id=None if generated_sample is None else generated_sample.sample_id,
            errors=tuple(errors),
        )

    try:
        reference_pose = reference_sample.pose  # type: ignore[union-attr]
        generated_pose = generated_sample.pose  # type: ignore[union-attr]
        if not isinstance(reference_pose, BfhPoseArrays):
            raise ValueError("reference pose must be BfhPoseArrays.")
        if not isinstance(generated_pose, BfhPoseArrays):
            raise ValueError("generated pose must be BfhPoseArrays.")
        observations = tuple(
            _channel_observations(channel, reference_pose, generated_pose)
            for channel in FULL_BFH_CHANNELS
        )
        return _completed_result(
            target=target,
            reference=reference_pose,
            generated=generated_pose,
            generated_sample_id=generated_sample.sample_id,  # type: ignore[union-attr]
            observations=observations,
        )
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        return _result(
            status="failed",
            split=target.split,
            reference_sample_id=target.resolved_sample_id,
            generated_sample_id=None if generated_sample is None else generated_sample.sample_id,
            errors=(str(exc),),
        )


def _prerequisite_errors(
    *,
    target: TestModelTargetResolution,
    inference: TestModelInferenceResult,
) -> list[str]:
    errors: list[str] = []
    if not target.succeeded:
        errors.append("target resolution did not succeed")
    if target.manifest_sample is None:
        errors.append("target manifest_sample is missing")
    if not inference.succeeded:
        errors.append("inference did not succeed")
    if inference.provider_result is None:
        errors.append("inference provider_result is missing")
    elif getattr(inference.provider_result, "generated_sample", None) is None:
        errors.append("generated sample is missing")
    return errors


def _completed_result(
    *,
    target: TestModelTargetResolution,
    reference: BfhPoseArrays,
    generated: BfhPoseArrays,
    generated_sample_id: str,
    observations: tuple[_ChannelObservations, ...],
) -> TestModelReferenceComparisonResult:
    aligned_frame_count = min(reference.frame_count, generated.frame_count)
    joint_count = sum(item.joint_count for item in observations)
    coordinate_errors = _concat(item.coordinate_errors for item in observations)
    distance_errors = _concat(item.distance_errors for item in observations)
    velocity_coordinate_errors = _concat(
        item.velocity_coordinate_errors for item in observations
    )
    velocity_distance_errors = _concat(item.velocity_distance_errors for item in observations)
    valid_joint_count = sum(item.generated_valid_joint_count for item in observations)
    expected_joint_count = sum(item.generated_expected_joint_count for item in observations)
    metrics = (
        _observed_metric(
            "masked_l1_mean",
            coordinate_errors,
            frame_count=aligned_frame_count,
            joint_count=joint_count,
        ),
        _observed_metric(
            "masked_l2_mean",
            distance_errors,
            frame_count=aligned_frame_count,
            joint_count=joint_count,
        ),
        _observed_metric(
            "velocity_l1_mean",
            velocity_coordinate_errors,
            frame_count=max(aligned_frame_count - 1, 0),
            joint_count=joint_count,
        ),
        _observed_metric(
            "velocity_l2_mean",
            velocity_distance_errors,
            frame_count=max(aligned_frame_count - 1, 0),
            joint_count=joint_count,
        ),
        TestModelComparisonMetric(
            "sequence_length_absolute_error",
            float(abs(generated.frame_count - reference.frame_count)),
            aligned_frame_count,
            joint_count,
            1,
            (),
        ),
        _coverage_metric(
            "valid_joint_coverage",
            valid_joint_count=valid_joint_count,
            expected_joint_count=expected_joint_count,
            frame_count=generated.frame_count,
            joint_count=joint_count,
        ),
    )
    channel_metrics = tuple(
        metric for observation in observations for metric in _channel_metrics(observation)
    )
    return TestModelReferenceComparisonResult(
        schema_version=REFERENCE_COMPARISON_SCHEMA_VERSION,
        status="completed",
        split=target.split,
        reference_sample_id=target.resolved_sample_id,
        generated_sample_id=generated_sample_id,
        reference_frame_count=reference.frame_count,
        generated_frame_count=generated.frame_count,
        aligned_frame_count=aligned_frame_count,
        metrics=metrics,
        channel_metrics=channel_metrics,
        warnings=(),
        errors=(),
    )


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
    if frame_count >= 2:
        velocity_valid = valid[1:] & valid[:-1]
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
        channel=channel,
        frame_count=frame_count,
        generated_frame_count=generated.frame_count,
        joint_count=joint_count,
        coordinate_errors=coordinate_errors,
        distance_errors=distance_errors,
        velocity_coordinate_errors=velocity_coordinate_errors,
        velocity_distance_errors=velocity_distance_errors,
        generated_valid_joint_count=int(np.count_nonzero(generated_valid)),
        generated_expected_joint_count=generated.frame_count * joint_count,
    )


def _observed_metric(
    metric_key: str,
    values: np.ndarray,
    *,
    frame_count: int,
    joint_count: int,
) -> TestModelComparisonMetric:
    if values.size == 0:
        return TestModelComparisonMetric(
            metric_key,
            None,
            frame_count,
            joint_count,
            0,
            (f"{metric_key} has no aligned valid observations.",),
        )
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise ValueError(f"{metric_key} computation produced a non-finite value.")
    return TestModelComparisonMetric(
        metric_key,
        value,
        frame_count,
        joint_count,
        int(values.size),
        (),
    )


def _coverage_metric(
    metric_key: str,
    *,
    valid_joint_count: int,
    expected_joint_count: int,
    frame_count: int,
    joint_count: int,
) -> TestModelComparisonMetric:
    if expected_joint_count <= 0:
        return TestModelComparisonMetric(
            metric_key,
            None,
            frame_count,
            joint_count,
            0,
            (f"{metric_key} has no expected generated joints.",),
        )
    return TestModelComparisonMetric(
        metric_key,
        float(valid_joint_count / expected_joint_count),
        frame_count,
        joint_count,
        expected_joint_count,
        (),
    )


def _channel_metrics(
    observation: _ChannelObservations,
) -> tuple[TestModelChannelComparisonMetric, ...]:
    return (
        _channel_observed_metric(
            "channel_masked_l1_mean",
            observation,
            observation.coordinate_errors,
            frame_count=observation.frame_count,
        ),
        _channel_observed_metric(
            "channel_masked_l2_mean",
            observation,
            observation.distance_errors,
            frame_count=observation.frame_count,
        ),
        _channel_observed_metric(
            "channel_velocity_l1_mean",
            observation,
            observation.velocity_coordinate_errors,
            frame_count=max(observation.frame_count - 1, 0),
        ),
        _channel_observed_metric(
            "channel_velocity_l2_mean",
            observation,
            observation.velocity_distance_errors,
            frame_count=max(observation.frame_count - 1, 0),
        ),
        _channel_coverage_metric(observation),
    )


def _channel_observed_metric(
    metric_key: str,
    observation: _ChannelObservations,
    values: np.ndarray,
    *,
    frame_count: int,
) -> TestModelChannelComparisonMetric:
    if values.size == 0:
        return TestModelChannelComparisonMetric(
            metric_key,
            observation.channel.value,
            None,
            frame_count,
            observation.joint_count,
            0,
            (f"{metric_key} has no aligned valid observations.",),
        )
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise ValueError(
            f"{metric_key}:{observation.channel.value} computation produced a non-finite value."
        )
    return TestModelChannelComparisonMetric(
        metric_key,
        observation.channel.value,
        value,
        frame_count,
        observation.joint_count,
        int(values.size),
        (),
    )


def _channel_coverage_metric(
    observation: _ChannelObservations,
) -> TestModelChannelComparisonMetric:
    metric_key = "channel_valid_joint_coverage"
    if observation.generated_expected_joint_count <= 0:
        return TestModelChannelComparisonMetric(
            metric_key,
            observation.channel.value,
            None,
            observation.generated_frame_count,
            observation.joint_count,
            0,
            (f"{metric_key} has no expected generated joints.",),
        )
    return TestModelChannelComparisonMetric(
        metric_key,
        observation.channel.value,
        float(
            observation.generated_valid_joint_count
            / observation.generated_expected_joint_count
        ),
        observation.generated_frame_count,
        observation.joint_count,
        observation.generated_expected_joint_count,
        (),
    )


def _result(
    *,
    status: str,
    split,
    reference_sample_id: str | None = None,
    generated_sample_id: str | None = None,
    reference_frame_count: int | None = None,
    generated_frame_count: int | None = None,
    aligned_frame_count: int | None = None,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> TestModelReferenceComparisonResult:
    return TestModelReferenceComparisonResult(
        schema_version=REFERENCE_COMPARISON_SCHEMA_VERSION,
        status=status,
        split=split,
        reference_sample_id=reference_sample_id,
        generated_sample_id=generated_sample_id,
        reference_frame_count=reference_frame_count,
        generated_frame_count=generated_frame_count,
        aligned_frame_count=aligned_frame_count,
        metrics=(),
        channel_metrics=(),
        warnings=warnings,
        errors=errors,
    )


def _concat(values) -> np.ndarray:
    materialized = tuple(value for value in values if value.size)
    if not materialized:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(materialized)


__all__ = [
    "REFERENCE_COMPARISON_SCHEMA_VERSION",
    "compare_reference_and_generated",
]
