"""Canonical BFH temporal window extraction and reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhTensorLayout,
    BfhVectorizedPose,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError

TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION = "t2sp-temporal-window-spec-v1"
BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION = "t2sp-bfh-pose-window-sequence-v1"

TEMPORAL_GRANULARITY_FRAME = "frame"
TEMPORAL_GRANULARITY_WINDOW = "window"

TEMPORAL_COVERAGE_POLICY_EXACT = "exact"
TEMPORAL_COVERAGE_POLICY_COVER_ALL = "cover_all"

TEMPORAL_PADDING_POLICY_NONE = "none"
TEMPORAL_PADDING_POLICY_RIGHT_ZERO = "right_zero"

WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS = "mean_valid_observations"


@dataclass(frozen=True, slots=True)
class TemporalWindowSpec:
    """Strict temporal granularity/window contract for BFH pose sequences."""

    schema_version: str
    temporal_granularity: str
    window_size: int
    stride: int
    coverage_policy: str
    padding_policy: str

    def __post_init__(self) -> None:
        if self.schema_version != TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION:
            raise ModelingDataError("temporal window spec schema_version is unsupported.")
        if self.temporal_granularity not in {
            TEMPORAL_GRANULARITY_FRAME,
            TEMPORAL_GRANULARITY_WINDOW,
        }:
            raise ModelingDataError(
                f"unsupported temporal_granularity={self.temporal_granularity!r}."
            )
        _require_positive_int(self.window_size, "window_size")
        _require_positive_int(self.stride, "stride")
        if self.coverage_policy not in {
            TEMPORAL_COVERAGE_POLICY_EXACT,
            TEMPORAL_COVERAGE_POLICY_COVER_ALL,
        }:
            raise ModelingDataError(
                f"unsupported temporal coverage_policy={self.coverage_policy!r}"
            )
        if self.padding_policy not in {
            TEMPORAL_PADDING_POLICY_NONE,
            TEMPORAL_PADDING_POLICY_RIGHT_ZERO,
        }:
            raise ModelingDataError(
                f"unsupported temporal padding_policy={self.padding_policy!r}"
            )
        if self.temporal_granularity == TEMPORAL_GRANULARITY_FRAME:
            if self.window_size != 1 or self.stride != 1:
                raise ModelingDataError(
                    "frame temporal granularity requires window_size=1 and stride=1."
                )
            if self.coverage_policy != TEMPORAL_COVERAGE_POLICY_EXACT:
                raise ModelingDataError(
                    "frame temporal granularity requires coverage_policy='exact'."
                )
            if self.padding_policy != TEMPORAL_PADDING_POLICY_NONE:
                raise ModelingDataError(
                    "frame temporal granularity requires padding_policy='none'."
                )
        if self.temporal_granularity == TEMPORAL_GRANULARITY_WINDOW:
            if self.window_size <= 1:
                raise ModelingDataError("window temporal granularity requires window_size > 1.")
            if self.coverage_policy != TEMPORAL_COVERAGE_POLICY_COVER_ALL:
                raise ModelingDataError(
                    "window temporal granularity requires coverage_policy='cover_all'."
                )
            if self.padding_policy != TEMPORAL_PADDING_POLICY_RIGHT_ZERO:
                raise ModelingDataError(
                    "window temporal granularity requires padding_policy='right_zero'."
                )

    @classmethod
    def frame(cls) -> "TemporalWindowSpec":
        return cls(
            schema_version=TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION,
            temporal_granularity=TEMPORAL_GRANULARITY_FRAME,
            window_size=1,
            stride=1,
            coverage_policy=TEMPORAL_COVERAGE_POLICY_EXACT,
            padding_policy=TEMPORAL_PADDING_POLICY_NONE,
        )

    @classmethod
    def window(cls, *, window_size: int, stride: int) -> "TemporalWindowSpec":
        return cls(
            schema_version=TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION,
            temporal_granularity=TEMPORAL_GRANULARITY_WINDOW,
            window_size=window_size,
            stride=stride,
            coverage_policy=TEMPORAL_COVERAGE_POLICY_COVER_ALL,
            padding_policy=TEMPORAL_PADDING_POLICY_RIGHT_ZERO,
        )


@dataclass(frozen=True, slots=True)
class BfhPoseWindowSequence:
    """Mask-aware temporal windows over canonical vectorized BFH pose values."""

    schema_version: str
    spec: TemporalWindowSpec
    layout: BfhTensorLayout
    source_sample_id: str | None
    source_frame_count: int
    values: np.ndarray
    validity_mask: np.ndarray
    confidence_values: np.ndarray
    frame_validity_mask: np.ndarray
    real_frame_mask: np.ndarray
    source_frame_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.schema_version != BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION:
            raise ModelingDataError("BFH pose window sequence schema_version is unsupported.")
        if not isinstance(self.spec, TemporalWindowSpec):
            raise ModelingDataError("spec must be a TemporalWindowSpec.")
        if not isinstance(self.layout, BfhTensorLayout):
            raise ModelingDataError("layout must be a BfhTensorLayout.")
        _require_positive_int(self.source_frame_count, "source_frame_count")
        if self.source_sample_id is not None:
            _require_text(self.source_sample_id, "source_sample_id")

        values = _numeric_array(
            self.values,
            "values",
            (
                -1,
                self.spec.window_size,
                self.layout.total_joint_count,
                self.layout.coordinate_dimensions,
            ),
        )
        num_windows = int(values.shape[0])
        if num_windows <= 0:
            raise ModelingDataError("num_windows must be positive.")
        expected_joints = (num_windows, self.spec.window_size, self.layout.total_joint_count)
        validity_mask = _bool_array(self.validity_mask, "validity_mask", expected_joints)
        confidence_values = _numeric_array(
            self.confidence_values,
            "confidence_values",
            expected_joints,
        )
        frame_shape = (num_windows, self.spec.window_size)
        frame_validity_mask = _bool_array(
            self.frame_validity_mask,
            "frame_validity_mask",
            frame_shape,
        )
        real_frame_mask = _bool_array(self.real_frame_mask, "real_frame_mask", frame_shape)
        source_frame_indices = _int_array(
            self.source_frame_indices,
            "source_frame_indices",
            frame_shape,
        )
        if np.any((confidence_values < 0.0) | (confidence_values > 1.0)):
            raise ModelingDataError("confidence_values must be within [0.0, 1.0].")
        if not np.all(np.isfinite(values)):
            raise ModelingDataError("values must contain only finite values.")
        _validate_real_indices(source_frame_indices, real_frame_mask, self.source_frame_count)
        _validate_padding_invariants(
            values=values,
            validity_mask=validity_mask,
            confidence_values=confidence_values,
            frame_validity_mask=frame_validity_mask,
            real_frame_mask=real_frame_mask,
            source_frame_indices=source_frame_indices,
        )
        expected_validity = (
            real_frame_mask[..., None]
            & frame_validity_mask[..., None]
            & (confidence_values > 0.0)
        )
        if not np.array_equal(validity_mask, expected_validity):
            raise ModelingDataError(
                "validity_mask must exactly match real frames with valid frames and "
                "positive confidence_values."
            )

        object.__setattr__(self, "values", _read_only(values))
        object.__setattr__(self, "validity_mask", _read_only(validity_mask))
        object.__setattr__(self, "confidence_values", _read_only(confidence_values))
        object.__setattr__(self, "frame_validity_mask", _read_only(frame_validity_mask))
        object.__setattr__(self, "real_frame_mask", _read_only(real_frame_mask))
        object.__setattr__(self, "source_frame_indices", _read_only(source_frame_indices))


@dataclass(frozen=True, slots=True)
class BfhPoseWindowMergeResult:
    """Result of reconstructing a vectorized BFH pose from temporal windows."""

    vectorized_pose: BfhVectorizedPose
    contribution_counts: np.ndarray
    merge_policy: str

    def __post_init__(self) -> None:
        if not isinstance(self.vectorized_pose, BfhVectorizedPose):
            raise ModelingDataError("vectorized_pose must be a BfhVectorizedPose.")
        if self.merge_policy != WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS:
            raise ModelingDataError(
                f"unsupported window merge_policy={self.merge_policy!r}; supported: "
                f"{WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS!r}."
            )
        counts = _int_array(
            self.contribution_counts,
            "contribution_counts",
            (
                self.vectorized_pose.frame_count,
                self.vectorized_pose.layout.total_joint_count,
            ),
        )
        if np.any(counts < 0):
            raise ModelingDataError("contribution_counts must be non-negative.")
        object.__setattr__(self, "contribution_counts", _read_only(counts))


def temporal_window_starts(*, frame_count: int, spec: TemporalWindowSpec) -> tuple[int, ...]:
    """Return deterministic frame/window start indices for a temporal spec."""

    _require_positive_int(frame_count, "frame_count")
    if not isinstance(spec, TemporalWindowSpec):
        raise ModelingDataError("spec must be a TemporalWindowSpec.")
    if spec.temporal_granularity == TEMPORAL_GRANULARITY_FRAME:
        return tuple(range(frame_count))
    full_window_stop = max(frame_count - spec.window_size + 1, 1)
    starts = set(range(0, full_window_stop, spec.stride))
    final_start = max(0, frame_count - spec.window_size)
    starts.add(final_start)
    return tuple(sorted(starts))


def extract_bfh_pose_windows(
    vectorized: BfhVectorizedPose,
    *,
    spec: TemporalWindowSpec,
) -> BfhPoseWindowSequence:
    """Extract canonical temporal windows from a vectorized BFH pose."""

    if not isinstance(vectorized, BfhVectorizedPose):
        raise ModelingDataError("vectorized must be a BfhVectorizedPose.")
    if not isinstance(spec, TemporalWindowSpec):
        raise ModelingDataError("spec must be a TemporalWindowSpec.")
    starts = temporal_window_starts(frame_count=vectorized.frame_count, spec=spec)
    shape = (
        len(starts),
        spec.window_size,
        vectorized.layout.total_joint_count,
        vectorized.layout.coordinate_dimensions,
    )
    values = np.zeros(shape, dtype=np.float32)
    validity_mask = np.zeros(shape[:3], dtype=np.bool_)
    confidence_values = np.zeros(shape[:3], dtype=np.float32)
    frame_validity_mask = np.zeros(shape[:2], dtype=np.bool_)
    real_frame_mask = np.zeros(shape[:2], dtype=np.bool_)
    source_frame_indices = np.full(shape[:2], -1, dtype=np.int64)

    for window_index, start in enumerate(starts):
        for offset in range(spec.window_size):
            source_index = start + offset
            if source_index >= vectorized.frame_count:
                continue
            values[window_index, offset] = vectorized.values[source_index]
            validity_mask[window_index, offset] = vectorized.validity_mask[source_index]
            confidence_values[window_index, offset] = vectorized.confidence_values[source_index]
            frame_validity_mask[window_index, offset] = vectorized.frame_validity_mask[
                source_index
            ]
            real_frame_mask[window_index, offset] = True
            source_frame_indices[window_index, offset] = source_index

    return BfhPoseWindowSequence(
        schema_version=BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION,
        spec=spec,
        layout=vectorized.layout,
        source_sample_id=vectorized.source_sample_id,
        source_frame_count=vectorized.frame_count,
        values=values,
        validity_mask=validity_mask,
        confidence_values=confidence_values,
        frame_validity_mask=frame_validity_mask,
        real_frame_mask=real_frame_mask,
        source_frame_indices=source_frame_indices,
    )


def flatten_bfh_pose_windows(sequence: BfhPoseWindowSequence) -> np.ndarray:
    """Return BFH window values flattened to `(windows, window_size, features)`."""

    _require_sequence(sequence)
    flattened = np.asarray(sequence.values, dtype=np.float32).reshape(
        sequence.values.shape[0],
        sequence.spec.window_size,
        sequence.layout.total_feature_dim,
    )
    return _read_only(flattened.copy())


def flatten_bfh_pose_window_validity(sequence: BfhPoseWindowSequence) -> np.ndarray:
    """Repeat joint validity over coordinate dimensions for flattened windows."""

    _require_sequence(sequence)
    flattened = np.repeat(
        sequence.validity_mask,
        sequence.layout.coordinate_dimensions,
        axis=2,
    )
    return _read_only(np.asarray(flattened, dtype=np.bool_).copy())


def merge_bfh_pose_windows(
    sequence: BfhPoseWindowSequence,
    *,
    merge_policy: str = WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS,
) -> BfhPoseWindowMergeResult:
    """Merge temporal windows into a canonical vectorized BFH pose."""

    _require_sequence(sequence)
    if merge_policy != WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS:
        raise ModelingDataError(
            f"unsupported window merge_policy={merge_policy!r}; supported: "
            f"{WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS!r}."
        )
    values_sum = np.zeros(
        (
            sequence.source_frame_count,
            sequence.layout.total_joint_count,
            sequence.layout.coordinate_dimensions,
        ),
        dtype=np.float32,
    )
    contribution_counts = np.zeros(
        (sequence.source_frame_count, sequence.layout.total_joint_count),
        dtype=np.int64,
    )
    confidence_values = np.zeros(
        (sequence.source_frame_count, sequence.layout.total_joint_count),
        dtype=np.float32,
    )
    frame_validity_mask = np.zeros((sequence.source_frame_count,), dtype=np.bool_)

    real_positions = np.argwhere(sequence.real_frame_mask)
    for window_index, offset in real_positions:
        source_index = int(sequence.source_frame_indices[window_index, offset])
        frame_validity_mask[source_index] |= bool(
            sequence.frame_validity_mask[window_index, offset]
        )
        joint_validity = sequence.validity_mask[window_index, offset]
        if not np.any(joint_validity):
            continue
        values_sum[source_index, joint_validity] += sequence.values[
            window_index,
            offset,
            joint_validity,
        ]
        contribution_counts[source_index, joint_validity] += 1
        confidence_values[source_index] = np.maximum(
            confidence_values[source_index],
            np.where(joint_validity, sequence.confidence_values[window_index, offset], 0.0),
        )

    values = np.zeros_like(values_sum)
    positive_counts = contribution_counts > 0
    values[positive_counts] = (
        values_sum[positive_counts] / contribution_counts[positive_counts][:, None]
    )
    validity_mask = frame_validity_mask[:, None] & (confidence_values > 0.0)
    vectorized = BfhVectorizedPose(
        layout=sequence.layout,
        values=values,
        validity_mask=validity_mask,
        frame_validity_mask=frame_validity_mask,
        confidence_values=confidence_values,
        frame_count=sequence.source_frame_count,
        source_sample_id=sequence.source_sample_id,
    )
    return BfhPoseWindowMergeResult(
        vectorized_pose=vectorized,
        contribution_counts=contribution_counts,
        merge_policy=merge_policy,
    )


def _require_sequence(sequence: BfhPoseWindowSequence) -> None:
    if not isinstance(sequence, BfhPoseWindowSequence):
        raise ModelingDataError("sequence must be a BfhPoseWindowSequence.")


def _require_positive_int(value: object, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        if field_name == "frame_count":
            raise ModelingDataError("frame_count must be positive.")
        raise ModelingDataError(f"{field_name} must be a positive integer.")


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelingDataError(f"{field_name} must be non-empty.")


def _numeric_array(value: np.ndarray, field_name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"f", "i", "u", "b"}:
        raise ModelingDataError(f"{field_name} must be numeric.")
    if len(raw.shape) != len(shape) or any(
        expected != -1 and observed != expected
        for observed, expected in zip(raw.shape, shape, strict=True)
    ):
        if field_name == "values":
            raise ModelingDataError(
                "values must have shape "
                "(num_windows, window_size, total_joint_count, coordinate_dimensions)."
            )
        raise ModelingDataError(f"{field_name} must have shape {shape}; got {raw.shape}.")
    return np.asarray(raw, dtype=np.float32).copy()


def _bool_array(value: np.ndarray, field_name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind != "b":
        raise ModelingDataError(f"{field_name} must be boolean with shape {shape}.")
    return np.asarray(raw, dtype=np.bool_).copy()


def _int_array(value: np.ndarray, field_name: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in {"i", "u"}:
        raise ModelingDataError(f"{field_name} must be an integer array with shape {shape}.")
    return np.asarray(raw, dtype=np.int64).copy()


def _validate_real_indices(
    source_frame_indices: np.ndarray,
    real_frame_mask: np.ndarray,
    source_frame_count: int,
) -> None:
    real_indices = source_frame_indices[real_frame_mask]
    if real_indices.size and np.any((real_indices < 0) | (real_indices >= source_frame_count)):
        raise ModelingDataError("real positions must have valid source_frame_indices.")
    if np.any(source_frame_indices[~real_frame_mask] != -1):
        raise ModelingDataError(
            "padded positions must have source_frame_indices=-1, zero confidence, "
            "and false masks."
        )


def _validate_padding_invariants(
    *,
    values: np.ndarray,
    validity_mask: np.ndarray,
    confidence_values: np.ndarray,
    frame_validity_mask: np.ndarray,
    real_frame_mask: np.ndarray,
    source_frame_indices: np.ndarray,
) -> None:
    padded = ~real_frame_mask
    if not np.any(padded):
        return
    if (
        np.any(source_frame_indices[padded] != -1)
        or np.any(values[padded] != 0.0)
        or np.any(validity_mask[padded])
        or np.any(confidence_values[padded] != 0.0)
        or np.any(frame_validity_mask[padded])
    ):
        raise ModelingDataError(
            "padded positions must have source_frame_indices=-1, zero confidence, "
            "and false masks."
        )


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


__all__ = [
    "TEMPORAL_WINDOW_SPEC_SCHEMA_VERSION",
    "BFH_POSE_WINDOW_SEQUENCE_SCHEMA_VERSION",
    "TEMPORAL_GRANULARITY_FRAME",
    "TEMPORAL_GRANULARITY_WINDOW",
    "TEMPORAL_COVERAGE_POLICY_EXACT",
    "TEMPORAL_COVERAGE_POLICY_COVER_ALL",
    "TEMPORAL_PADDING_POLICY_NONE",
    "TEMPORAL_PADDING_POLICY_RIGHT_ZERO",
    "WINDOW_MERGE_POLICY_MEAN_VALID_OBSERVATIONS",
    "TemporalWindowSpec",
    "BfhPoseWindowSequence",
    "BfhPoseWindowMergeResult",
    "temporal_window_starts",
    "extract_bfh_pose_windows",
    "flatten_bfh_pose_windows",
    "flatten_bfh_pose_window_validity",
    "merge_bfh_pose_windows",
]
