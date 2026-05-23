"""Generated-pose surface abstraction for candidate-agnostic validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.artifacts.store import ArtifactTopology
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import GeneratedPoseManifestEntry
from text_to_sign_production.modeling.artifacts.run_metadata import GeneratedPoseProducerType
from text_to_sign_production.modeling.data import (
    GeneratedPoseSurfaceSample,
    ModelingDataError,
    load_generated_pose_surface,
)
from text_to_sign_production.modeling.validation.config import ValidationSurfaceConfig
from text_to_sign_production.modeling.validation.errors import ModelValidationError


@dataclass(frozen=True, slots=True)
class ValidationGeneratedSurface:
    label: str
    producer_type: str
    producer_key: str
    canonical_id: str
    research_role: str
    run_name: str
    split: SampleSplit
    manifest_path: Path
    payload_root: Path
    entries: tuple[GeneratedPoseManifestEntry, ...]
    samples: tuple[GeneratedPoseSurfaceSample, ...]
    is_comparator: bool
    is_primary_model: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "payload_root", Path(self.payload_root))
        if self.producer_type not in {"model", "comparator"}:
            raise ModelValidationError("validation surface producer_type must be 'model' or 'comparator'.")
        if self.is_comparator != (self.producer_type == "comparator"):
            raise ModelValidationError("validation surface comparator flag is inconsistent.")
        if self.is_primary_model != (self.producer_type == "model"):
            raise ModelValidationError("validation surface primary-model flag is inconsistent.")


def resolve_generated_surface(
    *,
    topology: ArtifactTopology,
    surface_config: ValidationSurfaceConfig,
    split: SampleSplit | str = SampleSplit.VAL,
    run_name_override: str | None = None,
) -> ValidationGeneratedSurface | None:
    """Resolve one configured generated-pose surface, preserving failure manifest entries."""

    resolved_split = SampleSplit(split)
    if resolved_split is not SampleSplit.VAL:
        raise ModelValidationError("candidate-agnostic validation currently supports split='val' only.")
    run_name = run_name_override or surface_config.run_name
    if run_name is None:
        raise ModelValidationError(
            f"required validation surface {surface_config.label!r} is missing run_name."
        )
    manifest_path = topology.evaluations.generated_pose_manifest(
        surface_config.producer_key,
        run_name,
        resolved_split,
    ).path
    payload_root = topology.evaluations.generated_pose_samples_root(
        surface_config.producer_key,
        run_name,
        resolved_split,
    ).path
    try:
        samples = load_generated_pose_surface(
            topology,
            producer_key=surface_config.producer_key,
            run_name=run_name,
            split=resolved_split,
        )
    except FileNotFoundError as exc:
        if not surface_config.required:
            return None
        raise ModelValidationError(
            f"required validation surface {surface_config.label!r} is missing generated-pose "
            f"manifest for run_name={run_name!r}."
        ) from exc
    except ModelingDataError as exc:
        raise ModelValidationError(
            f"validation surface {surface_config.label!r} generated-pose inputs are invalid: {exc}"
        ) from exc
    if not samples:
        if not surface_config.required:
            return None
        raise ModelValidationError(
            f"required validation surface {surface_config.label!r} has no generated-pose entries."
        )
    entries = tuple(sample.entry for sample in samples)
    _validate_surface_metadata(surface_config, entries)
    first = entries[0]
    return ValidationGeneratedSurface(
        label=surface_config.label,
        producer_type=first.producer_type.value,
        producer_key=first.producer_key,
        canonical_id=first.canonical_id,
        research_role=first.research_role,
        run_name=first.run_name,
        split=resolved_split,
        manifest_path=manifest_path,
        payload_root=payload_root,
        entries=entries,
        samples=samples,
        is_comparator=first.producer_type is GeneratedPoseProducerType.COMPARATOR,
        is_primary_model=first.producer_type is GeneratedPoseProducerType.MODEL,
    )


def _validate_surface_metadata(
    surface_config: ValidationSurfaceConfig,
    entries: tuple[GeneratedPoseManifestEntry, ...],
) -> None:
    for entry in entries:
        if entry.producer_type.value != surface_config.producer_type:
            raise ModelValidationError(
                f"validation surface {surface_config.label!r} producer_type does not match config."
            )
        if entry.producer_key != surface_config.producer_key:
            raise ModelValidationError(
                f"validation surface {surface_config.label!r} producer_key does not match config."
            )
        if surface_config.canonical_id is not None and entry.canonical_id != surface_config.canonical_id:
            raise ModelValidationError(
                f"validation surface {surface_config.label!r} canonical_id does not match config."
            )
        if surface_config.research_role is not None and entry.research_role != surface_config.research_role:
            raise ModelValidationError(
                f"validation surface {surface_config.label!r} research_role does not match config."
            )


__all__ = ["ValidationGeneratedSurface", "resolve_generated_surface"]
