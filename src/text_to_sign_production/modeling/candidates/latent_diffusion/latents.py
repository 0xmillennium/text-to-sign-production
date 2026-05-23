"""Latent target, sequence, and manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_vectorization import BfhTensorLayout
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.data.temporal_windows import (
    TEMPORAL_GRANULARITY_FRAME,
    TEMPORAL_GRANULARITY_WINDOW,
    TemporalWindowSpec,
    temporal_window_starts,
)

LATENT_TARGET_SPEC_SCHEMA_VERSION = "t2sp-latent-target-spec-v1"
LATENT_SEQUENCE_SCHEMA_VERSION = "t2sp-latent-sequence-v2"
LATENT_SEQUENCE_SCHEMA_VERSION_V1 = "t2sp-latent-sequence-v1"
LATENT_MANIFEST_SCHEMA_VERSION = "t2sp-latent-manifest-v2"
LATENT_MANIFEST_SCHEMA_VERSION_V1 = "t2sp-latent-manifest-v1"
LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME = "standardized_bfh_frame"
LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT = "learned_bfh_window_latent"
LATENT_TARGET_TYPE = LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME


@dataclass(frozen=True, slots=True)
class LatentTargetSpec:
    schema_version: str
    target_type: str
    layout: BfhTensorLayout
    coordinate_mode: str
    confidence_policy: str
    temporal_granularity: str
    window_size: int
    stride: int
    latent_dim: int
    base_feature_dim: int

    def __post_init__(self) -> None:
        if self.schema_version != LATENT_TARGET_SPEC_SCHEMA_VERSION:
            raise LatentDiffusionError("latent target spec schema_version is unsupported.")
        if self.target_type not in {
            LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
            LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
        }:
            raise LatentDiffusionError(
                "latent_target.target_type must be one of "
                "{'standardized_bfh_frame', 'learned_bfh_window_latent'}."
            )
        if not isinstance(self.layout, BfhTensorLayout):
            raise LatentDiffusionError("latent target layout must be a BfhTensorLayout.")
        _require_equal(self.coordinate_mode, "xy", "coordinate_mode")
        _require_equal(self.confidence_policy, "mask_only", "confidence_policy")
        _require_positive_int(self.window_size, "window_size")
        _require_positive_int(self.stride, "stride")
        _require_positive_int(self.latent_dim, "latent_dim")
        _require_positive_int(self.base_feature_dim, "base_feature_dim")
        if self.base_feature_dim != self.layout.total_feature_dim:
            raise LatentDiffusionError("base_feature_dim must match layout.total_feature_dim.")
        if self.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
            if (
                self.temporal_granularity != TEMPORAL_GRANULARITY_FRAME
                or self.window_size != 1
                or self.stride != 1
            ):
                raise LatentDiffusionError(
                    "standardized_bfh_frame target requires temporal_granularity='frame', "
                    "window_size=1, and stride=1."
                )
            if self.latent_dim != self.base_feature_dim:
                raise LatentDiffusionError(
                    "standardized_bfh_frame latent_dim must equal layout.total_feature_dim."
                )
        if self.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
            if (
                self.temporal_granularity != TEMPORAL_GRANULARITY_WINDOW
                or self.window_size <= 1
                or self.stride < 1
            ):
                raise LatentDiffusionError(
                    "learned_bfh_window_latent target requires "
                    "temporal_granularity='window', window_size > 1, and stride >= 1."
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_type": self.target_type,
            "layout": self.layout.to_dict(),
            "coordinate_mode": self.coordinate_mode,
            "confidence_policy": self.confidence_policy,
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
            "latent_dim": self.latent_dim,
            "base_feature_dim": self.base_feature_dim,
        }

    def temporal_window_spec(self) -> TemporalWindowSpec:
        if self.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
            return TemporalWindowSpec.frame()
        return TemporalWindowSpec.window(window_size=self.window_size, stride=self.stride)


@dataclass(frozen=True, slots=True)
class LatentSequence:
    schema_version: str
    sample_id: str
    source_sentence_name: str
    split: SampleSplit
    values: np.ndarray
    validity_mask: np.ndarray
    frame_count: int
    latent_count: int
    latent_dim: int
    target_spec: LatentTargetSpec

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LATENT_SEQUENCE_SCHEMA_VERSION,
            LATENT_SEQUENCE_SCHEMA_VERSION_V1,
        }:
            raise LatentDiffusionError("latent sequence schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        _require_text(self.source_sentence_name, "source_sentence_name")
        object.__setattr__(self, "split", SampleSplit(self.split))
        _require_positive_int(self.frame_count, "frame_count")
        _require_positive_int(self.latent_count, "latent_count")
        _require_positive_int(self.latent_dim, "latent_dim")
        if not isinstance(self.target_spec, LatentTargetSpec):
            raise LatentDiffusionError("target_spec must be a LatentTargetSpec.")
        values = np.asarray(self.values, dtype=np.float32).copy()
        mask = np.asarray(self.validity_mask, dtype=np.bool_).copy()
        expected_latent_count = expected_latent_count_for_frame_count(
            frame_count=self.frame_count,
            target_spec=self.target_spec,
        )
        if self.latent_count != expected_latent_count:
            if self.target_spec.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
                raise LatentDiffusionError(
                    "latent_count must equal len(temporal_window_starts(frame_count, spec)) "
                    "for learned_bfh_window_latent."
                )
            raise LatentDiffusionError("latent_count must equal frame_count.")
        if self.latent_dim != self.target_spec.latent_dim:
            raise LatentDiffusionError("latent_dim must match target_spec.latent_dim.")
        expected = (self.latent_count, self.latent_dim)
        if values.shape != expected:
            raise LatentDiffusionError(
                f"latent values must have shape {expected}; got {values.shape}."
            )
        if mask.shape != expected:
            raise LatentDiffusionError(
                f"latent validity_mask must have shape {expected}; got {mask.shape}."
            )
        if not np.any(mask):
            raise LatentDiffusionError(
                f"latent validity_mask for sample_id={self.sample_id!r} must contain "
                "at least one valid latent feature."
            )
        if not np.all(np.isfinite(values[mask])):
            raise LatentDiffusionError("latent values must be finite where validity_mask is true.")
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "validity_mask", mask)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_sentence_name": self.source_sentence_name,
            "split": self.split.value,
            "frame_count": self.frame_count,
            "latent_count": self.latent_count,
            "latent_dim": self.latent_dim,
            "target_spec": self.target_spec.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LatentManifestEntry:
    schema_version: str
    sample_id: str
    source_sentence_name: str
    split: SampleSplit
    latent_path: Path
    frame_count: int
    latent_count: int
    latent_dim: int
    target_type: str
    temporal_granularity: str
    window_size: int
    stride: int
    issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LATENT_MANIFEST_SCHEMA_VERSION,
            LATENT_MANIFEST_SCHEMA_VERSION_V1,
        }:
            raise LatentDiffusionError("latent manifest schema_version is unsupported.")
        _require_text(self.sample_id, "sample_id")
        _require_text(self.source_sentence_name, "source_sentence_name")
        object.__setattr__(self, "split", SampleSplit(self.split))
        path = Path(self.latent_path)
        if not str(path).strip():
            raise LatentDiffusionError("latent_path must be non-empty.")
        object.__setattr__(self, "latent_path", path)
        _require_positive_int(self.frame_count, "frame_count")
        _require_positive_int(self.latent_count, "latent_count")
        _require_positive_int(self.latent_dim, "latent_dim")
        if self.target_type not in {
            LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
            LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
        }:
            raise LatentDiffusionError(
                "latent_target.target_type must be one of "
                "{'standardized_bfh_frame', 'learned_bfh_window_latent'}."
            )
        _require_text(self.temporal_granularity, "temporal_granularity")
        _require_positive_int(self.window_size, "window_size")
        _require_positive_int(self.stride, "stride")
        issues = tuple(self.issues)
        for issue in issues:
            _require_text(issue, "issue")
        object.__setattr__(self, "issues", issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_sentence_name": self.source_sentence_name,
            "split": self.split.value,
            "latent_path": str(self.latent_path),
            "frame_count": self.frame_count,
            "latent_count": self.latent_count,
            "latent_dim": self.latent_dim,
            "target_type": self.target_type,
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
            "issues": list(self.issues),
        }


def expected_latent_count_for_frame_count(
    *,
    frame_count: int,
    target_spec: LatentTargetSpec,
) -> int:
    _require_positive_int(frame_count, "frame_count")
    if target_spec.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
        return frame_count
    return len(
        temporal_window_starts(
            frame_count=frame_count,
            spec=target_spec.temporal_window_spec(),
        )
    )


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LatentDiffusionError(f"{name} must be non-empty.")


def _require_equal(value: object, expected: str, name: str) -> None:
    if value != expected:
        raise LatentDiffusionError(f"{name} must be {expected!r}.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LatentDiffusionError(f"{name} must be a positive integer.")


__all__ = [
    "LATENT_MANIFEST_SCHEMA_VERSION",
    "LATENT_MANIFEST_SCHEMA_VERSION_V1",
    "LATENT_SEQUENCE_SCHEMA_VERSION",
    "LATENT_SEQUENCE_SCHEMA_VERSION_V1",
    "LATENT_TARGET_SPEC_SCHEMA_VERSION",
    "LATENT_TARGET_TYPE",
    "LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT",
    "LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME",
    "LatentManifestEntry",
    "LatentSequence",
    "LatentTargetSpec",
    "expected_latent_count_for_frame_count",
]
