from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.data import (
    BFH_CHANNEL_SPECS,
    BfhPoseArrays,
    parse_modeling_manifest_family,
)
from text_to_sign_production.workflows.test_model.contracts import (
    TestModelFinalResult as _TestModelFinalResult,
    TestModelInferenceResult as _TestModelInferenceResult,
    TestModelReferenceComparisonResult as _TestModelReferenceComparisonResult,
    TestModelTargetResolution as _TestModelTargetResolution,
)
from text_to_sign_production.workflows.test_model.processing.comparison import (
    REFERENCE_COMPARISON_SCHEMA_VERSION,
    compare_reference_and_generated,
)


@pytest.mark.unit
def test_reference_comparison_identical_pose_has_zero_error() -> None:
    pose = _pose(frame_count=3)

    result = compare_reference_and_generated(
        _layout(),
        target=_target(pose),
        inference=_inference(pose),
    )

    assert result.status == "completed"
    assert result.succeeded is True
    assert _metric(result, "masked_l1_mean").value == pytest.approx(0.0)
    assert _metric(result, "masked_l2_mean").value == pytest.approx(0.0)
    assert _metric(result, "velocity_l1_mean").value == pytest.approx(0.0)
    assert _metric(result, "velocity_l2_mean").value == pytest.approx(0.0)
    assert _metric(result, "sequence_length_absolute_error").value == pytest.approx(0.0)
    assert _metric(result, "valid_joint_coverage").value == pytest.approx(1.0)


@pytest.mark.unit
def test_reference_comparison_uses_prefix_alignment_for_length_mismatch() -> None:
    result = compare_reference_and_generated(
        _layout(),
        target=_target(_pose(frame_count=3)),
        inference=_inference(_pose(frame_count=2)),
    )

    assert result.status == "completed"
    assert result.aligned_frame_count == 2
    assert _metric(result, "sequence_length_absolute_error").value == pytest.approx(1.0)
    assert _metric(result, "masked_l1_mean").valid_value_count > 0


@pytest.mark.unit
def test_reference_comparison_missing_observations_return_none_metrics() -> None:
    all_valid = compare_reference_and_generated(
        _layout(),
        target=_target(_pose(frame_count=2)),
        inference=_inference(_pose(frame_count=2)),
    )
    missing = compare_reference_and_generated(
        _layout(),
        target=_target(_pose(frame_count=2)),
        inference=_inference(_pose(frame_count=2, confidence=0.0)),
    )

    assert missing.status == "completed"
    assert _metric(missing, "masked_l1_mean").value is None
    assert _metric(missing, "masked_l1_mean").valid_value_count == 0
    assert _metric(all_valid, "masked_l1_mean").valid_value_count > 0
    assert _metric(missing, "masked_l1_mean").issues == (
        "masked_l1_mean has no aligned valid observations.",
    )
    assert _metric(missing, "velocity_l2_mean").value is None
    assert _metric(missing, "valid_joint_coverage").value == pytest.approx(0.0)


@pytest.mark.unit
def test_reference_comparison_not_ready_when_prerequisite_missing() -> None:
    result = compare_reference_and_generated(
        _layout(),
        target=_target(None, status="not_found"),
        inference=_TestModelInferenceResult(
            provider_result=None,
            metadata_path=None,
            receipts=(),
            status="not_ready",
            warnings=(),
            errors=("inference missing",),
        ),
    )

    assert result.status == "not_ready"
    assert result.succeeded is False
    assert result.errors


@pytest.mark.unit
def test_final_result_requires_successful_reference_comparison() -> None:
    successful = _TestModelFinalResult(
        model_run=_succeeded(),
        checkpoint=_succeeded(),
        target=_succeeded(),
        evidence=_succeeded(),
        inference=_succeeded(),
        comparison=_comparison("completed"),
        visualization=SimpleNamespace(status="completed"),
        reports=SimpleNamespace(succeeded=True),
        publish=None,
    )
    missing = _TestModelFinalResult(
        model_run=_succeeded(),
        checkpoint=_succeeded(),
        target=_succeeded(),
        evidence=_succeeded(),
        inference=_succeeded(),
        comparison=None,
        visualization=SimpleNamespace(status="completed"),
        reports=SimpleNamespace(succeeded=True),
        publish=None,
    )
    failed = _TestModelFinalResult(
        model_run=_succeeded(),
        checkpoint=_succeeded(),
        target=_succeeded(),
        evidence=_succeeded(),
        inference=_succeeded(),
        comparison=_comparison("failed", errors=("comparison failed",)),
        visualization=SimpleNamespace(status="completed"),
        reports=SimpleNamespace(succeeded=True),
        publish=None,
    )

    assert successful.succeeded is True
    assert missing.succeeded is False
    assert failed.succeeded is False


def _pose(frame_count: int, *, confidence: float = 1.0) -> BfhPoseArrays:
    return BfhPoseArrays(
        body_xyc=_channel(frame_count, PoseChannel.BODY, confidence=confidence),
        left_hand_xyc=_channel(frame_count, PoseChannel.LEFT_HAND, confidence=confidence),
        right_hand_xyc=_channel(frame_count, PoseChannel.RIGHT_HAND, confidence=confidence),
        face_xyc=_channel(frame_count, PoseChannel.FACE, confidence=confidence),
        valid_frame_mask=np.ones((frame_count,), dtype=np.bool_),
    )


def _channel(frame_count: int, channel: PoseChannel, *, confidence: float) -> np.ndarray:
    joints = BFH_CHANNEL_SPECS[channel].joint_count
    array = np.zeros((frame_count, joints, 3), dtype=np.float32)
    for frame_index in range(frame_count):
        array[frame_index, :, 0] = 0.1 + frame_index * 0.01
        array[frame_index, :, 1] = 0.2 + frame_index * 0.01
    array[..., 2] = confidence
    return array


def _target(
    pose: BfhPoseArrays | None,
    *,
    status: str = "found",
) -> _TestModelTargetResolution:
    manifest_sample = None
    if pose is not None:
        manifest_sample = SimpleNamespace(
            sample_id="reference-sample",
            pose=pose,
            sample=SimpleNamespace(schema_version="test-schema"),
        )
    return _TestModelTargetResolution(
        status=status,
        target_sentence_name="target sentence",
        resolved_sample_id=None if manifest_sample is None else manifest_sample.sample_id,
        source_sentence_name="source sentence",
        manifest_family=parse_modeling_manifest_family("untiered:passed"),
        split=SampleSplit.TEST,
        manifest_path=Path("manifest.jsonl"),
        manifest_entry=None,
        manifest_sample=manifest_sample,
        issues=() if status == "found" else ("target missing",),
        warnings=(),
    )


def _inference(pose: BfhPoseArrays) -> _TestModelInferenceResult:
    generated_sample = SimpleNamespace(sample_id="generated-sample", pose=pose)
    return _TestModelInferenceResult(
        provider_result=SimpleNamespace(generated_sample=generated_sample),
        metadata_path=Path("metadata.json"),
        receipts=(),
        status="completed",
        warnings=(),
        errors=(),
    )


def _layout() -> SimpleNamespace:
    return SimpleNamespace()


def _metric(result: _TestModelReferenceComparisonResult, key: str):
    return next(metric for metric in result.metrics if metric.metric_key == key)


def _comparison(
    status: str,
    *,
    errors: tuple[str, ...] = (),
) -> _TestModelReferenceComparisonResult:
    return _TestModelReferenceComparisonResult(
        schema_version=REFERENCE_COMPARISON_SCHEMA_VERSION,
        status=status,
        split=SampleSplit.TEST,
        reference_sample_id="reference-sample",
        generated_sample_id="generated-sample",
        reference_frame_count=1,
        generated_frame_count=1,
        aligned_frame_count=1,
        metrics=(),
        channel_metrics=(),
        warnings=(),
        errors=errors,
    )


def _succeeded() -> SimpleNamespace:
    return SimpleNamespace(succeeded=True)
