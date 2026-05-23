"""Typed stage vocabulary and stage-sequence validation for model candidates."""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from text_to_sign_production.modeling.candidates.errors import ModelStagePlanError
from text_to_sign_production.modeling.research import ModelSpec, ObjectiveKey


class ModelStageKind(enum.StrEnum):
    """Stage identities available to future model providers."""

    TRAIN = "train"
    GENERATE = "generate"
    EXPORT_GENERATED_POSE = "export_generated_pose"

    FIT_REPRESENTATION = "fit_representation"
    EVALUATE_RECONSTRUCTION = "evaluate_reconstruction"
    TRAIN_TEXT_TO_TOKEN = "train_text_to_token"
    DECODE_TO_POSE = "decode_to_pose"

    DEFINE_LATENT_TARGET = "define_latent_target"
    CACHE_LATENTS = "cache_latents"
    TRAIN_DENOISER = "train_denoiser"

    DEFINE_CHANNEL_PARTITIONS = "define_channel_partitions"
    DEFINE_MASK_STRATEGY = "define_mask_strategy"
    DEFINE_LOSS_WEIGHTING = "define_loss_weighting"
    TRAIN_STRUCTURE_AWARE = "train_structure_aware"

    ATTACH_AUXILIARY_OBJECTIVE = "attach_auxiliary_objective"


def model_stage_kind_from_value(value: ModelStageKind | str) -> ModelStageKind:
    """Coerce one provider stage identity."""

    try:
        return ModelStageKind(value)
    except (TypeError, ValueError) as exc:
        raise ModelStagePlanError(f"unknown model stage kind: {value!r}") from exc


class ModelStageCategory(enum.StrEnum):
    """Broad operational category of a model stage."""

    CONFIGURATION = "configuration"
    REPRESENTATION = "representation"
    TRAINING = "training"
    GENERATION = "generation"
    EXPORT = "export"
    REPORTING = "reporting"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True, slots=True)
class ModelStageSpec:
    """Shared semantics for one provider stage identity."""

    kind: ModelStageKind
    label: str
    category: ModelStageCategory
    description: str
    required: bool = True
    produces_generated_pose: bool = False
    expected_artifact_roles: tuple[str, ...] = ()
    expected_counters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ModelStageKind):
            raise ModelStagePlanError("stage spec kind must be a ModelStageKind.")
        if not isinstance(self.category, ModelStageCategory):
            raise ModelStagePlanError("stage spec category must be a ModelStageCategory.")
        _require_text(self.label, "label")
        _require_text(self.description, "description")
        _validate_unique_strings(self.expected_artifact_roles, "expected_artifact_roles")
        _validate_unique_strings(self.expected_counters, "expected_counters")


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelStagePlanError(f"{field_name} must be non-empty.")


def _validate_unique_strings(values: tuple[str, ...], field_name: str) -> None:
    if not isinstance(values, tuple):
        raise ModelStagePlanError(f"{field_name} must be a tuple.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ModelStagePlanError(f"{field_name} must contain non-empty strings.")
    if len(set(values)) != len(values):
        raise ModelStagePlanError(f"{field_name} must not contain duplicates.")


MODEL_STAGE_SPECS: Mapping[ModelStageKind, ModelStageSpec] = MappingProxyType(
    {
        ModelStageKind.TRAIN: ModelStageSpec(
            kind=ModelStageKind.TRAIN,
            label="Train",
            category=ModelStageCategory.TRAINING,
            description="Train a direct model candidate under its provider contract.",
        ),
        ModelStageKind.GENERATE: ModelStageSpec(
            kind=ModelStageKind.GENERATE,
            label="Generate",
            category=ModelStageCategory.GENERATION,
            description="Generate provider-native model outputs for requested splits.",
        ),
        ModelStageKind.EXPORT_GENERATED_POSE: ModelStageSpec(
            kind=ModelStageKind.EXPORT_GENERATED_POSE,
            label="Export generated pose",
            category=ModelStageCategory.EXPORT,
            description="Export generated outputs through the generated-pose artifact contract.",
            produces_generated_pose=True,
            expected_artifact_roles=("generated_pose_manifest", "generated_pose_samples"),
        ),
        ModelStageKind.FIT_REPRESENTATION: ModelStageSpec(
            kind=ModelStageKind.FIT_REPRESENTATION,
            label="Fit representation",
            category=ModelStageCategory.REPRESENTATION,
            description="Fit a provider-defined learned pose representation.",
        ),
        ModelStageKind.EVALUATE_RECONSTRUCTION: ModelStageSpec(
            kind=ModelStageKind.EVALUATE_RECONSTRUCTION,
            label="Evaluate reconstruction",
            category=ModelStageCategory.REPRESENTATION,
            description="Evaluate reconstruction behavior of a learned representation.",
        ),
        ModelStageKind.TRAIN_TEXT_TO_TOKEN: ModelStageSpec(
            kind=ModelStageKind.TRAIN_TEXT_TO_TOKEN,
            label="Train text to token",
            category=ModelStageCategory.TRAINING,
            description="Train a text-conditioned token predictor.",
        ),
        ModelStageKind.DECODE_TO_POSE: ModelStageSpec(
            kind=ModelStageKind.DECODE_TO_POSE,
            label="Decode to pose",
            category=ModelStageCategory.GENERATION,
            description="Decode generated representations into provider-native pose outputs.",
        ),
        ModelStageKind.DEFINE_LATENT_TARGET: ModelStageSpec(
            kind=ModelStageKind.DEFINE_LATENT_TARGET,
            label="Define latent target",
            category=ModelStageCategory.CONFIGURATION,
            description="Define the provider latent target contract.",
        ),
        ModelStageKind.CACHE_LATENTS: ModelStageSpec(
            kind=ModelStageKind.CACHE_LATENTS,
            label="Cache latents",
            category=ModelStageCategory.REPRESENTATION,
            description="Prepare provider-managed latent representations.",
        ),
        ModelStageKind.TRAIN_DENOISER: ModelStageSpec(
            kind=ModelStageKind.TRAIN_DENOISER,
            label="Train denoiser",
            category=ModelStageCategory.TRAINING,
            description="Train the provider denoising component.",
        ),
        ModelStageKind.DEFINE_CHANNEL_PARTITIONS: ModelStageSpec(
            kind=ModelStageKind.DEFINE_CHANNEL_PARTITIONS,
            label="Define channel partitions",
            category=ModelStageCategory.CONFIGURATION,
            description="Define provider-managed articulator channel partitions.",
        ),
        ModelStageKind.DEFINE_MASK_STRATEGY: ModelStageSpec(
            kind=ModelStageKind.DEFINE_MASK_STRATEGY,
            label="Define mask strategy",
            category=ModelStageCategory.CONFIGURATION,
            description="Define provider-managed masking behavior.",
        ),
        ModelStageKind.DEFINE_LOSS_WEIGHTING: ModelStageSpec(
            kind=ModelStageKind.DEFINE_LOSS_WEIGHTING,
            label="Define loss weighting",
            category=ModelStageCategory.CONFIGURATION,
            description="Define provider-managed loss weighting behavior.",
        ),
        ModelStageKind.TRAIN_STRUCTURE_AWARE: ModelStageSpec(
            kind=ModelStageKind.TRAIN_STRUCTURE_AWARE,
            label="Train structure aware",
            category=ModelStageCategory.TRAINING,
            description="Train a structure-aware model candidate.",
        ),
        ModelStageKind.ATTACH_AUXILIARY_OBJECTIVE: ModelStageSpec(
            kind=ModelStageKind.ATTACH_AUXILIARY_OBJECTIVE,
            label="Attach auxiliary objective",
            category=ModelStageCategory.AUXILIARY,
            description="Plan a compatible auxiliary objective attachment.",
            required=False,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class PlannedModelStage:
    """One ordered provider stage selected for a model run."""

    index: int
    spec: ModelStageSpec
    provider_stage_id: str
    objective_key: ObjectiveKey | None = None
    depends_on: tuple[ModelStageKind, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ModelStagePlanError("planned stage index must be non-negative.")
        if not isinstance(self.spec, ModelStageSpec):
            raise ModelStagePlanError("planned stage spec must be a ModelStageSpec.")
        _require_text(self.provider_stage_id, "provider_stage_id")
        if self.objective_key is not None and not isinstance(self.objective_key, ObjectiveKey):
            raise ModelStagePlanError("objective_key must be an ObjectiveKey or None.")
        if not isinstance(self.depends_on, tuple):
            raise ModelStagePlanError("depends_on must be a tuple.")
        if any(not isinstance(stage, ModelStageKind) for stage in self.depends_on):
            raise ModelStagePlanError("depends_on must contain ModelStageKind values.")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ModelStagePlanError("depends_on must not contain duplicate stage kinds.")


def stage_sequence_from_values(
    values: Iterable[ModelStageKind | str],
) -> tuple[ModelStageKind, ...]:
    """Coerce a non-empty stage sequence and reject duplicates."""

    if isinstance(values, (ModelStageKind, str)):
        raise ModelStagePlanError("stage sequence must be an iterable of stage values.")
    stages = tuple(model_stage_kind_from_value(value) for value in values)
    if not stages:
        raise ModelStagePlanError("stage sequence must be non-empty.")
    if len(set(stages)) != len(stages):
        raise ModelStagePlanError("stage sequence must not contain duplicate stage kinds.")
    return stages


def validate_stage_sequence_for_spec(
    model_spec: ModelSpec,
    stages: Iterable[ModelStageKind | str],
) -> tuple[ModelStageKind, ...]:
    """Validate that a provider sequence contains its research-backed defaults."""

    if not isinstance(model_spec, ModelSpec):
        raise ModelStagePlanError("model_spec must be a ModelSpec.")
    resolved = stage_sequence_from_values(stages)
    defaults = stage_sequence_from_values(model_spec.default_stage_sequence)
    observed_defaults = tuple(stage for stage in resolved if stage in defaults)
    if observed_defaults != defaults:
        raise ModelStagePlanError(
            "stage sequence must contain every default model stage in default order."
        )
    if (
        model_spec.generated_pose_required
        and ModelStageKind.EXPORT_GENERATED_POSE not in resolved
    ):
        raise ModelStagePlanError(
            "generated-pose model stage sequence must include export_generated_pose."
        )
    return resolved


__all__ = [
    "MODEL_STAGE_SPECS",
    "ModelStageCategory",
    "ModelStageKind",
    "ModelStageSpec",
    "PlannedModelStage",
    "model_stage_kind_from_value",
    "stage_sequence_from_values",
    "validate_stage_sequence_for_spec",
]
