"""Immutable stage and execution result contracts for model providers."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.modeling.candidates.errors import ModelStageExecutionError
from text_to_sign_production.modeling.candidates.stages import PlannedModelStage
from text_to_sign_production.modeling.artifacts import GeneratedPoseSample
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.modeling.training.stages import ProviderStageArtifactRole

_PROVIDER_STAGE_ARTIFACT_ROLE_ALIASES = {
    "best_checkpoint": ProviderStageArtifactRole.CHECKPOINT_BEST,
    "last_checkpoint": ProviderStageArtifactRole.CHECKPOINT_LAST,
}


@dataclass(frozen=True, slots=True)
class ModelStageArtifactRef:
    """One artifact produced by a model provider stage."""

    role: ProviderStageArtifactRole | str
    path: Path
    kind: str
    description: str | None = None
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", provider_stage_artifact_role_from_value(self.role))
        _require_text(self.role, "role")
        try:
            object.__setattr__(self, "path", Path(self.path))
        except TypeError as exc:
            raise ModelStageExecutionError("artifact path must be coercible to Path.") from exc
        _require_text(self.kind, "kind")
        if self.description is not None:
            _require_text(self.description, "description")
        if not isinstance(self.metadata, Mapping):
            raise ModelStageExecutionError("artifact metadata must be a mapping.")
        metadata: dict[str, object] = {}
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise ModelStageExecutionError("artifact metadata keys must be strings.")
            metadata[key] = value
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


class ModelStageStatus(enum.StrEnum):
    """Provider execution outcome for one planned stage."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelStageResult:
    """Validated result produced for one planned provider stage."""

    stage: PlannedModelStage
    status: ModelStageStatus
    artifacts: tuple[ModelStageArtifactRef, ...] = ()
    generated_pose_surfaces: tuple[object, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.stage, PlannedModelStage):
            raise ModelStageExecutionError("stage must be a PlannedModelStage.")
        try:
            object.__setattr__(self, "status", ModelStageStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise ModelStageExecutionError(f"unknown stage result status: {self.status!r}") from exc
        if not isinstance(self.artifacts, tuple) or any(
            not isinstance(artifact, ModelStageArtifactRef) for artifact in self.artifacts
        ):
            raise ModelStageExecutionError(
                "artifacts must be a tuple of ModelStageArtifactRef values."
            )
        if not isinstance(self.generated_pose_surfaces, tuple):
            raise ModelStageExecutionError("generated_pose_surfaces must be a tuple.")
        for surface in self.generated_pose_surfaces:
            if surface.__class__.__name__ != "GeneratedPoseSurface":
                raise ModelStageExecutionError(
                    "generated_pose_surfaces must contain GeneratedPoseSurface values."
                )
        if not isinstance(self.metrics, Mapping):
            raise ModelStageExecutionError("metrics must be a mapping.")
        metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            _require_text(key, "metric key")
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
                raise ModelStageExecutionError("metric values must be finite numbers.")
            metrics[key] = float(value)
        if not isinstance(self.metadata, Mapping):
            raise ModelStageExecutionError("metadata must be a mapping.")
        metadata: dict[str, object] = {}
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise ModelStageExecutionError("metadata keys must be strings.")
            metadata[key] = value
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True, slots=True)
class ModelExecutionResult:
    """Aggregate sequential execution result for one model run."""

    model_key: ModelKey
    run_name: str
    stages: tuple[ModelStageResult, ...]

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "model_key", ModelKey(self.model_key))
        except (TypeError, ValueError) as exc:
            raise ModelStageExecutionError(f"unknown model key: {self.model_key!r}") from exc
        _require_text(self.run_name, "run_name")
        if not isinstance(self.stages, tuple) or not self.stages:
            raise ModelStageExecutionError("stages must be a non-empty tuple.")
        if any(not isinstance(stage, ModelStageResult) for stage in self.stages):
            raise ModelStageExecutionError("stages must contain ModelStageResult values.")
        indices = tuple(stage.stage.index for stage in self.stages)
        if len(set(indices)) != len(indices):
            raise ModelStageExecutionError("result stage indices must be unique.")
        if indices != tuple(sorted(indices)):
            raise ModelStageExecutionError("result stage indices must be sorted ascending.")

    @property
    def completed(self) -> bool:
        """Return whether all executed stages ended without a failure."""

        return all(
            stage.status in (ModelStageStatus.COMPLETED, ModelStageStatus.SKIPPED)
            for stage in self.stages
        )

    @property
    def artifact_refs(self) -> tuple[ModelStageArtifactRef, ...]:
        """Return stage artifacts in execution order."""

        return tuple(artifact for stage in self.stages for artifact in stage.artifacts)


@dataclass(frozen=True, slots=True)
class ModelSingleSampleInferenceResult:
    """Provider result for one strict single-sample generated-pose inference."""

    model_key: ModelKey
    run_name: str
    checkpoint_path: Path
    sample_id: str
    generated_sample: GeneratedPoseSample
    generated_payload_path: Path
    generated_manifest_path: Path
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "model_key", ModelKey(self.model_key))
        except (TypeError, ValueError) as exc:
            raise ModelStageExecutionError(f"unknown model key: {self.model_key!r}") from exc
        _require_text(self.run_name, "run_name")
        _require_text(self.sample_id, "sample_id")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "generated_payload_path", Path(self.generated_payload_path))
        object.__setattr__(self, "generated_manifest_path", Path(self.generated_manifest_path))
        if not isinstance(self.generated_sample, GeneratedPoseSample):
            raise ModelStageExecutionError("generated_sample must be a GeneratedPoseSample.")
        if self.generated_sample.run_name != self.run_name:
            raise ModelStageExecutionError("generated sample run_name must match result.")
        if self.generated_sample.sample_id != self.sample_id:
            raise ModelStageExecutionError("generated sample_id must match result.")
        if not isinstance(self.metadata, Mapping):
            raise ModelStageExecutionError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelStageExecutionError(f"{field_name} must be non-empty.")


def provider_stage_artifact_role_from_value(
    value: ProviderStageArtifactRole | str,
) -> ProviderStageArtifactRole | str:
    """Return the shared provider role enum for known roles, preserving legacy extensions."""

    if isinstance(value, ProviderStageArtifactRole):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ModelStageExecutionError("role must be non-empty.")
    stripped = value.strip()
    if stripped in _PROVIDER_STAGE_ARTIFACT_ROLE_ALIASES:
        return _PROVIDER_STAGE_ARTIFACT_ROLE_ALIASES[stripped]
    try:
        return ProviderStageArtifactRole(stripped)
    except ValueError:
        return stripped


def is_generated_pose_manifest_role(value: ProviderStageArtifactRole | str) -> bool:
    """Return whether an artifact role denotes a generated-pose manifest."""

    resolved = provider_stage_artifact_role_from_value(value)
    if resolved is ProviderStageArtifactRole.GENERATED_POSE_MANIFEST:
        return True
    return isinstance(resolved, str) and resolved.startswith("generated_pose_manifest:")


__all__ = [
    "ModelExecutionResult",
    "ModelSingleSampleInferenceResult",
    "ModelStageArtifactRef",
    "ModelStageResult",
    "ModelStageStatus",
    "is_generated_pose_manifest_role",
    "provider_stage_artifact_role_from_value",
]
