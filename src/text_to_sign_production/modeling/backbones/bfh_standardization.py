"""Deterministic observation-masked statistics for vectorized BFH coordinates."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhTensorLayout,
    BfhVectorizedPose,
    bfh_tensor_layout_from_dict,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError

BFH_STANDARDIZATION_SCHEMA_VERSION = "t2sp-bfh-standardization-v2"
BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES = frozenset(
    {"raise", "channel_fallback", "global_fallback", "identity_fallback"}
)


@dataclass(frozen=True, slots=True)
class BfhStandardizationStats:
    """Per-joint, per-coordinate valid-observation standardization statistics."""

    schema_version: str
    layout: BfhTensorLayout
    mean: np.ndarray
    std: np.ndarray
    epsilon: float
    sample_count: int
    frame_count: int
    missing_observation_policy: str
    observation_count: np.ndarray
    zero_observation_coordinate_count: int
    channel_fallback_coordinate_count: int
    global_fallback_coordinate_count: int
    identity_fallback_coordinate_count: int
    fallback_summary_by_channel: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if self.schema_version != BFH_STANDARDIZATION_SCHEMA_VERSION:
            raise ModelingDataError("BFH standardization schema_version is unsupported.")
        if not isinstance(self.layout, BfhTensorLayout):
            raise ModelingDataError("standardization layout must be a BfhTensorLayout.")
        if self.missing_observation_policy not in BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES:
            raise ModelingDataError(
                "standardization missing_observation_policy is unsupported."
            )
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ModelingDataError("standardization epsilon must be positive and finite.")
        for field_name in ("sample_count", "frame_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ModelingDataError(f"{field_name} must be a positive integer.")
        shape = (self.layout.total_joint_count, self.layout.coordinate_dimensions)
        mean = _finite_array(self.mean, "mean", shape)
        std = _finite_array(self.std, "std", shape)
        observation_count = _integer_array(self.observation_count, "observation_count", shape)
        if np.any(std < self.epsilon):
            raise ModelingDataError("standardization std must be at least epsilon.")
        zero_count = int(np.sum(observation_count == 0))
        diagnostic_counts = {
            "zero_observation_coordinate_count": self.zero_observation_coordinate_count,
            "channel_fallback_coordinate_count": self.channel_fallback_coordinate_count,
            "global_fallback_coordinate_count": self.global_fallback_coordinate_count,
            "identity_fallback_coordinate_count": self.identity_fallback_coordinate_count,
        }
        for name, value in diagnostic_counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelingDataError(f"{name} must be a non-negative integer.")
        if self.zero_observation_coordinate_count != zero_count:
            raise ModelingDataError(
                "zero_observation_coordinate_count must match observation_count."
            )
        fallback_total = (
            self.channel_fallback_coordinate_count
            + self.global_fallback_coordinate_count
            + self.identity_fallback_coordinate_count
        )
        if fallback_total != self.zero_observation_coordinate_count:
            raise ModelingDataError(
                "standardization fallback diagnostic totals are incoherent."
            )
        summary = _fallback_summary_by_channel(
            self.fallback_summary_by_channel,
            layout=self.layout,
        )
        for key in (
            "zero_observation",
            "channel_fallback",
            "global_fallback",
            "identity_fallback",
        ):
            total = sum(channel_summary[key] for channel_summary in summary.values())
            expected = {
                "zero_observation": self.zero_observation_coordinate_count,
                "channel_fallback": self.channel_fallback_coordinate_count,
                "global_fallback": self.global_fallback_coordinate_count,
                "identity_fallback": self.identity_fallback_coordinate_count,
            }[key]
            if total != expected:
                raise ModelingDataError(
                    f"fallback_summary_by_channel total for {key!r} is incoherent."
                )
        mean.setflags(write=False)
        std.setflags(write=False)
        observation_count.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "fallback_summary_by_channel", summary)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready statistics record."""

        return {
            "schema_version": self.schema_version,
            "layout": self.layout.to_dict(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon,
            "sample_count": self.sample_count,
            "frame_count": self.frame_count,
            "missing_observation_policy": self.missing_observation_policy,
            "observation_count": self.observation_count.tolist(),
            "zero_observation_coordinate_count": self.zero_observation_coordinate_count,
            "channel_fallback_coordinate_count": self.channel_fallback_coordinate_count,
            "global_fallback_coordinate_count": self.global_fallback_coordinate_count,
            "identity_fallback_coordinate_count": self.identity_fallback_coordinate_count,
            "fallback_summary_by_channel": {
                channel: dict(summary)
                for channel, summary in self.fallback_summary_by_channel.items()
            },
        }


class BfhStandardizationAccumulator:
    """Streaming accumulator for BFH standardization statistics."""

    def __init__(self) -> None:
        self._layout: BfhTensorLayout | None = None
        self._counts: np.ndarray | None = None
        self._sums: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None
        self._sample_count = 0
        self._frame_count = 0

    def update(self, pose: BfhVectorizedPose) -> None:
        if not isinstance(pose, BfhVectorizedPose):
            raise ModelingDataError("pose must be a BfhVectorizedPose.")
        if self._layout is None:
            self._layout = pose.layout
            shape = (pose.layout.total_joint_count, pose.layout.coordinate_dimensions)
            self._counts = np.zeros(shape, dtype=np.int64)
            self._sums = np.zeros(shape, dtype=np.float64)
            self._sumsq = np.zeros(shape, dtype=np.float64)
        elif pose.layout != self._layout:
            raise ModelingDataError("all BFH poses must use the same tensor layout.")
        observed = np.broadcast_to(pose.validity_mask[..., None], pose.values.shape)
        assert self._counts is not None
        assert self._sums is not None
        assert self._sumsq is not None
        self._counts += np.sum(observed, axis=0, dtype=np.int64)
        observed_values = np.where(observed, pose.values, 0.0)
        self._sums += np.sum(observed_values, axis=0, dtype=np.float64)
        self._sumsq += np.sum(observed_values * observed_values, axis=0, dtype=np.float64)
        self._sample_count += 1
        self._frame_count += pose.frame_count

    def finalize(
        self,
        *,
        epsilon: float,
        missing_observation_policy: str,
    ) -> BfhStandardizationStats:
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ModelingDataError("standardization epsilon must be positive and finite.")
        if missing_observation_policy not in BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES:
            raise ValueError(
                "missing_observation_policy must be one of "
                f"{sorted(BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES)!r}; "
                f"got {missing_observation_policy!r}."
            )
        if self._layout is None or self._counts is None or self._sums is None or self._sumsq is None:
            raise ModelingDataError("cannot compute BFH standardization statistics from empty input.")
        return _finalize_bfh_standardization_stats(
            layout=self._layout,
            counts=self._counts,
            sums=self._sums,
            sumsq=self._sumsq,
            sample_count=self._sample_count,
            frame_count=self._frame_count,
            epsilon=epsilon,
            missing_observation_policy=missing_observation_policy,
        )


def compute_bfh_standardization_stats(
    poses: Iterable[BfhVectorizedPose],
    *,
    epsilon: float = 1e-6,
    missing_observation_policy: str = "raise",
) -> BfhStandardizationStats:
    """Compute deterministic statistics over valid joint observations only."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ModelingDataError("standardization epsilon must be positive and finite.")
    if missing_observation_policy not in BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES:
        raise ValueError(
            "missing_observation_policy must be one of "
            f"{sorted(BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES)!r}; "
            f"got {missing_observation_policy!r}."
        )
    accumulator = BfhStandardizationAccumulator()
    for pose in poses:
        accumulator.update(pose)
    return accumulator.finalize(
        epsilon=epsilon,
        missing_observation_policy=missing_observation_policy,
    )


def _finalize_bfh_standardization_stats(
    *,
    layout: BfhTensorLayout,
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    sample_count: int,
    frame_count: int,
    epsilon: float,
    missing_observation_policy: str,
) -> BfhStandardizationStats:
    shape = (layout.total_joint_count, layout.coordinate_dimensions)
    missing = counts == 0
    if np.any(missing) and missing_observation_policy == "raise":
        raise ModelingDataError(
            "cannot compute BFH standardization statistics: at least one joint coordinate "
            "has zero valid observations."
        )
    mean = np.zeros(shape, dtype=np.float64)
    std = np.ones(shape, dtype=np.float64)
    observed_coordinates = counts > 0
    mean[observed_coordinates] = sums[observed_coordinates] / counts[observed_coordinates]
    variance = np.zeros(shape, dtype=np.float64)
    variance[observed_coordinates] = np.maximum(
        (sumsq[observed_coordinates] / counts[observed_coordinates])
        - (mean[observed_coordinates] * mean[observed_coordinates]),
        0.0,
    )
    std[observed_coordinates] = np.maximum(
        np.sqrt(variance[observed_coordinates]),
        float(epsilon),
    )
    fallback_summary = _empty_fallback_summary(layout)
    fallback_counts = {
        "channel_fallback": 0,
        "global_fallback": 0,
        "identity_fallback": 0,
    }
    for channel in layout.channels:
        channel_name = channel.value
        joint_slice = layout.channel_slices[channel]
        fallback_summary[channel_name]["zero_observation"] = int(np.sum(missing[joint_slice]))
    if np.any(missing):
        if missing_observation_policy == "identity_fallback":
            mean[missing] = 0.0
            std[missing] = 1.0
            fallback_counts["identity_fallback"] = int(np.sum(missing))
            for channel in layout.channels:
                channel_name = channel.value
                joint_slice = layout.channel_slices[channel]
                fallback_summary[channel_name]["identity_fallback"] = int(
                    np.sum(missing[joint_slice])
                )
        else:
            _apply_statistical_fallback(
                mean=mean,
                std=std,
                missing=missing,
                counts=counts,
                sums=sums,
                sumsq=sumsq,
                layout=layout,
                epsilon=float(epsilon),
                policy=missing_observation_policy,
                fallback_summary=fallback_summary,
                fallback_counts=fallback_counts,
            )
    return BfhStandardizationStats(
        schema_version=BFH_STANDARDIZATION_SCHEMA_VERSION,
        layout=layout,
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        epsilon=float(epsilon),
        sample_count=sample_count,
        frame_count=frame_count,
        missing_observation_policy=missing_observation_policy,
        observation_count=counts,
        zero_observation_coordinate_count=int(np.sum(missing)),
        channel_fallback_coordinate_count=fallback_counts["channel_fallback"],
        global_fallback_coordinate_count=fallback_counts["global_fallback"],
        identity_fallback_coordinate_count=fallback_counts["identity_fallback"],
        fallback_summary_by_channel=fallback_summary,
    )


def apply_bfh_standardization(
    pose: BfhVectorizedPose,
    stats: BfhStandardizationStats,
) -> BfhVectorizedPose:
    """Standardize valid coordinate values while retaining mask and confidence truth."""

    _require_compatible(pose, stats)
    observed = pose.validity_mask[..., None]
    normalized = np.where(observed, (pose.values - stats.mean) / stats.std, pose.values)
    return _with_values(pose, normalized)


def invert_bfh_standardization(
    pose: BfhVectorizedPose,
    stats: BfhStandardizationStats,
) -> BfhVectorizedPose:
    """Invert valid-coordinate standardization."""

    _require_compatible(pose, stats)
    observed = pose.validity_mask[..., None]
    values = np.where(observed, pose.values * stats.std + stats.mean, pose.values)
    return _with_values(pose, values)


def write_bfh_standardization_stats_json(path: Path, stats: BfhStandardizationStats) -> None:
    """Write strict deterministic UTF-8 standardization JSON."""

    if not isinstance(stats, BfhStandardizationStats):
        raise ModelingDataError("stats must be BfhStandardizationStats.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(stats.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_bfh_standardization_stats_json(path: Path) -> BfhStandardizationStats:
    """Read and validate strict standardization JSON."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelingDataError(f"malformed BFH standardization JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ModelingDataError("BFH standardization JSON document must be an object.")
    expected = {
        "schema_version",
        "layout",
        "mean",
        "std",
        "epsilon",
        "sample_count",
        "frame_count",
        "missing_observation_policy",
        "observation_count",
        "zero_observation_coordinate_count",
        "channel_fallback_coordinate_count",
        "global_fallback_coordinate_count",
        "identity_fallback_coordinate_count",
        "fallback_summary_by_channel",
    }
    if set(document) != expected:
        raise ModelingDataError("BFH standardization JSON keys do not match the schema.")
    try:
        return BfhStandardizationStats(
            schema_version=_text(document["schema_version"], "schema_version"),
            layout=bfh_tensor_layout_from_dict(_mapping(document["layout"], "layout")),
            mean=np.asarray(document["mean"], dtype=np.float32),
            std=np.asarray(document["std"], dtype=np.float32),
            epsilon=_number(document["epsilon"], "epsilon"),
            sample_count=_integer(document["sample_count"], "sample_count"),
            frame_count=_integer(document["frame_count"], "frame_count"),
            missing_observation_policy=_text(
                document["missing_observation_policy"],
                "missing_observation_policy",
            ),
            observation_count=np.asarray(document["observation_count"], dtype=np.int64),
            zero_observation_coordinate_count=_integer(
                document["zero_observation_coordinate_count"],
                "zero_observation_coordinate_count",
            ),
            channel_fallback_coordinate_count=_integer(
                document["channel_fallback_coordinate_count"],
                "channel_fallback_coordinate_count",
            ),
            global_fallback_coordinate_count=_integer(
                document["global_fallback_coordinate_count"],
                "global_fallback_coordinate_count",
            ),
            identity_fallback_coordinate_count=_integer(
                document["identity_fallback_coordinate_count"],
                "identity_fallback_coordinate_count",
            ),
            fallback_summary_by_channel=_mapping(
                document["fallback_summary_by_channel"],
                "fallback_summary_by_channel",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ModelingDataError(f"invalid BFH standardization JSON values: {exc}") from exc


def _require_compatible(pose: BfhVectorizedPose, stats: BfhStandardizationStats) -> None:
    if not isinstance(pose, BfhVectorizedPose):
        raise ModelingDataError("pose must be a BfhVectorizedPose.")
    if not isinstance(stats, BfhStandardizationStats):
        raise ModelingDataError("stats must be BfhStandardizationStats.")
    if pose.layout != stats.layout:
        raise ModelingDataError("pose and standardization stats tensor layouts do not match.")


def _with_values(pose: BfhVectorizedPose, values: np.ndarray) -> BfhVectorizedPose:
    return BfhVectorizedPose(
        layout=pose.layout,
        values=values,
        validity_mask=pose.validity_mask,
        frame_validity_mask=pose.frame_validity_mask,
        confidence_values=pose.confidence_values,
        frame_count=pose.frame_count,
        source_sample_id=pose.source_sample_id,
    )


def _finite_array(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).copy()
    if array.shape != shape:
        raise ModelingDataError(f"standardization {name} must have shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ModelingDataError(f"standardization {name} must contain finite values.")
    return array


def _integer_array(value: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).copy()
    if array.shape != shape:
        raise ModelingDataError(f"standardization {name} must have shape {shape}.")
    if np.any(array < 0):
        raise ModelingDataError(f"standardization {name} must be non-negative.")
    return array


def _empty_fallback_summary(layout: BfhTensorLayout) -> dict[str, dict[str, int]]:
    return {
        channel.value: {
            "zero_observation": 0,
            "channel_fallback": 0,
            "global_fallback": 0,
            "identity_fallback": 0,
        }
        for channel in layout.channels
    }


def _fallback_summary_by_channel(
    value: Mapping[str, Mapping[str, int]],
    *,
    layout: BfhTensorLayout,
) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping):
        raise ModelingDataError("fallback_summary_by_channel must be a mapping.")
    expected_channels = tuple(channel.value for channel in layout.channels)
    expected_counts = {
        "zero_observation",
        "channel_fallback",
        "global_fallback",
        "identity_fallback",
    }
    if set(value) != set(expected_channels):
        raise ModelingDataError("fallback_summary_by_channel channels do not match layout.")
    summary: dict[str, Mapping[str, int]] = {}
    for channel_name in expected_channels:
        raw = value[channel_name]
        if not isinstance(raw, Mapping) or set(raw) != expected_counts:
            raise ModelingDataError(
                f"fallback_summary_by_channel[{channel_name!r}] keys are invalid."
            )
        channel_summary: dict[str, int] = {}
        for key in expected_counts:
            count = raw[key]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ModelingDataError(
                    f"fallback_summary_by_channel[{channel_name!r}][{key!r}] "
                    "must be a non-negative integer."
                )
            channel_summary[key] = count
        summary[channel_name] = channel_summary
    return summary


def _apply_statistical_fallback(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    missing: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    layout: BfhTensorLayout,
    epsilon: float,
    policy: str,
    fallback_summary: dict[str, dict[str, int]],
    fallback_counts: dict[str, int],
) -> None:
    global_stats = tuple(
        _aggregate_dimension_stats(
            counts=counts[:, dimension],
            sums=sums[:, dimension],
            sumsq=sumsq[:, dimension],
            epsilon=epsilon,
        )
        for dimension in range(layout.coordinate_dimensions)
    )
    if any(item is None for item in global_stats):
        raise ModelingDataError(
            "cannot compute BFH standardization statistics: no valid coordinate "
            "observations are available for fallback."
        )
    for channel in layout.channels:
        channel_name = channel.value
        joint_slice = layout.channel_slices[channel]
        channel_missing = missing[joint_slice]
        if not np.any(channel_missing):
            continue
        for dimension in range(layout.coordinate_dimensions):
            dimension_missing = channel_missing[:, dimension]
            if not np.any(dimension_missing):
                continue
            target_indices = np.where(dimension_missing)[0] + joint_slice.start
            channel_stats = _aggregate_dimension_stats(
                counts=counts[joint_slice, dimension],
                sums=sums[joint_slice, dimension],
                sumsq=sumsq[joint_slice, dimension],
                epsilon=epsilon,
            )
            if policy == "channel_fallback" and channel_stats is not None:
                fallback_mean, fallback_std = channel_stats
                fallback_kind = "channel_fallback"
            elif policy in {"channel_fallback", "global_fallback"}:
                fallback_mean, fallback_std = global_stats[dimension]  # type: ignore[misc]
                fallback_kind = "global_fallback"
            else:
                raise ModelingDataError(f"unsupported missing observation policy: {policy!r}.")
            mean[target_indices, dimension] = fallback_mean
            std[target_indices, dimension] = fallback_std
            count = int(target_indices.shape[0])
            fallback_summary[channel_name][fallback_kind] += count
            fallback_counts[fallback_kind] += count


def _aggregate_dimension_stats(
    *,
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    epsilon: float,
) -> tuple[float, float] | None:
    total_count = int(np.sum(counts, dtype=np.int64))
    if total_count <= 0:
        return None
    total_sum = float(np.sum(sums, dtype=np.float64))
    total_sumsq = float(np.sum(sumsq, dtype=np.float64))
    mean = total_sum / total_count
    variance = max((total_sumsq / total_count) - (mean * mean), 0.0)
    return mean, max(float(np.sqrt(variance)), epsilon)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelingDataError(f"{name} must be a JSON object.")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingDataError(f"{name} must be non-empty text.")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelingDataError(f"{name} must be an integer.")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not np.isfinite(value):
        raise ModelingDataError(f"{name} must be a finite number.")
    return float(value)


__all__ = [
    "BFH_STANDARDIZATION_SCHEMA_VERSION",
    "BFH_STANDARDIZATION_MISSING_OBSERVATION_POLICIES",
    "BfhStandardizationAccumulator",
    "BfhStandardizationStats",
    "apply_bfh_standardization",
    "compute_bfh_standardization_stats",
    "invert_bfh_standardization",
    "read_bfh_standardization_stats_json",
    "write_bfh_standardization_stats_json",
]
