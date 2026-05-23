"""Strict configuration contracts for the semantic-consistency objective foundation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.spec import (
    SEMANTIC_OBJECTIVE_CANONICAL_ID,
    SEMANTIC_OBJECTIVE_KEY,
    SEMANTIC_OBJECTIVE_PHASE_NUMBER,
    require_semantic_consistency_spec,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey

SEMANTIC_OBJECTIVE_CONFIG_SCHEMA_VERSION = "t2sp-semantic-objective-config-v2"

_EXPECTED_TOP_LEVEL_KEYS = {
    "identity",
    "attachment",
    "text_embedding",
    "pose_embedding",
    "alignment",
    "ablation",
    "reports",
    "training_objective",
}
_V1_TOP_LEVEL_KEYS = _EXPECTED_TOP_LEVEL_KEYS - {"training_objective"}
_FORBIDDEN_FIELD_KEYS = frozenset(
    {
        "manifest_family",
        "retrieval",
        "retrieval_bank",
        "retrieval_comparator",
        "gloss",
        "gloss_label",
        "gloss_supervision",
        "dictionary",
        "dictionary_stitching",
        "avatar",
        "smpl",
        "smplx",
        "smpl-x",
        "3d",
        "rendering",
        "provider",
        "standalone_provider",
        "standalone_model",
        "model_provider",
    }
)
_CANONICAL_POSE_CHANNELS = (
    PoseChannel.BODY,
    PoseChannel.LEFT_HAND,
    PoseChannel.RIGHT_HAND,
    PoseChannel.FACE,
)


@dataclass(frozen=True, slots=True)
class SemanticObjectiveIdentityConfig:
    objective_key: ObjectiveKey
    canonical_id: str
    phase_number: int
    research_role: str
    standalone_model_allowed: bool
    requires_ablation: bool

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "objective_key", ObjectiveKey(self.objective_key))
        except (TypeError, ValueError) as exc:
            raise SemanticConsistencyError(
                f"identity.objective_key is invalid: {self.objective_key!r}."
            ) from exc
        if self.objective_key is not SEMANTIC_OBJECTIVE_KEY:
            raise SemanticConsistencyError("identity.objective_key must be 'semantic_consistency'.")
        _require_equal(self.canonical_id, SEMANTIC_OBJECTIVE_CANONICAL_ID, "identity.canonical_id")
        if self.phase_number != SEMANTIC_OBJECTIVE_PHASE_NUMBER:
            raise SemanticConsistencyError("identity.phase_number must be 9.")
        _require_equal(
            self.research_role,
            "auxiliary_additive_objective",
            "identity.research_role",
        )
        if self.standalone_model_allowed is not False:
            raise SemanticConsistencyError(
                "identity.standalone_model_allowed must be false; semantic consistency is auxiliary."
            )
        if self.requires_ablation is not True:
            raise SemanticConsistencyError(
                "identity.requires_ablation must be true; with/without ablation is required."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_key": self.objective_key.value,
            "canonical_id": self.canonical_id,
            "phase_number": self.phase_number,
            "research_role": self.research_role,
            "standalone_model_allowed": self.standalone_model_allowed,
            "requires_ablation": self.requires_ablation,
        }


@dataclass(frozen=True, slots=True)
class SemanticAttachmentConfig:
    allowed_models: tuple[ModelKey, ...]
    forbidden_models: tuple[ModelKey, ...]
    stage_id: str
    require_same_manifest_family: bool
    require_same_splits: bool
    require_same_evaluation_protocol: bool

    def __post_init__(self) -> None:
        allowed = _model_keys(self.allowed_models, "attachment.allowed_models")
        forbidden = _model_keys(self.forbidden_models, "attachment.forbidden_models")
        registry_allowed = set(require_semantic_consistency_spec().allowed_attachment_models)
        if not allowed or not set(allowed).issubset(registry_allowed):
            raise SemanticConsistencyError(
                "attachment.allowed_models must be a non-empty subset of the objective "
                "registry attachments: learned_pose_token, latent_diffusion, articulator_aware."
            )
        if ModelKey.BASE_DIRECT in allowed:
            raise SemanticConsistencyError(
                "attachment.allowed_models must not include base_direct; it cannot receive "
                "semantic_consistency."
            )
        if ModelKey.BASE_DIRECT not in forbidden:
            raise SemanticConsistencyError(
                "attachment.forbidden_models must include base_direct."
            )
        if set(allowed) & set(forbidden):
            raise SemanticConsistencyError(
                "attachment allowed_models and forbidden_models must not overlap."
            )
        _require_equal(self.stage_id, "semantic_consistency.attach", "attachment.stage_id")
        for name in (
            "require_same_manifest_family",
            "require_same_splits",
            "require_same_evaluation_protocol",
        ):
            _require_true(getattr(self, name), f"attachment.{name}")
        object.__setattr__(self, "allowed_models", allowed)
        object.__setattr__(self, "forbidden_models", forbidden)

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_models": [model.value for model in self.allowed_models],
            "forbidden_models": [model.value for model in self.forbidden_models],
            "stage_id": self.stage_id,
            "require_same_manifest_family": self.require_same_manifest_family,
            "require_same_splits": self.require_same_splits,
            "require_same_evaluation_protocol": self.require_same_evaluation_protocol,
        }


@dataclass(frozen=True, slots=True)
class SemanticTextEmbeddingConfig:
    backend: str
    embedding_dim: int
    pooling: str
    trainable: bool
    proxy_only: bool

    def __post_init__(self) -> None:
        _require_equal(self.backend, "deterministic_hash", "text_embedding.backend")
        _positive_int(self.embedding_dim, "text_embedding.embedding_dim")
        _require_equal(self.pooling, "mean", "text_embedding.pooling")
        if self.trainable is not False:
            raise SemanticConsistencyError("text_embedding.trainable must be false.")
        _require_true(self.proxy_only, "text_embedding.proxy_only")

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "trainable": self.trainable,
            "proxy_only": self.proxy_only,
        }


@dataclass(frozen=True, slots=True)
class SemanticPoseEmbeddingConfig:
    backend: str
    embedding_dim: int
    coordinate_mode: str
    channel_groups: tuple[PoseChannel, ...]
    include_velocity_statistics: bool
    include_validity_statistics: bool
    projection_seed: int
    proxy_only: bool

    def __post_init__(self) -> None:
        _require_equal(self.backend, "bfh_statistics_projection", "pose_embedding.backend")
        _positive_int(self.embedding_dim, "pose_embedding.embedding_dim")
        _require_equal(self.coordinate_mode, "xy", "pose_embedding.coordinate_mode")
        channels = _pose_channels(self.channel_groups, "pose_embedding.channel_groups")
        if channels != _CANONICAL_POSE_CHANNELS:
            raise SemanticConsistencyError(
                "pose_embedding.channel_groups must be body,left_hand,right_hand,face "
                "in canonical order."
            )
        for name in ("include_velocity_statistics", "include_validity_statistics"):
            if not isinstance(getattr(self, name), bool):
                raise SemanticConsistencyError(f"pose_embedding.{name} must be a boolean.")
        if (
            not isinstance(self.projection_seed, int)
            or isinstance(self.projection_seed, bool)
            or self.projection_seed < 0
        ):
            raise SemanticConsistencyError(
                "pose_embedding.projection_seed must be a non-negative integer."
            )
        _require_true(self.proxy_only, "pose_embedding.proxy_only")
        object.__setattr__(self, "channel_groups", channels)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "embedding_dim": self.embedding_dim,
            "coordinate_mode": self.coordinate_mode,
            "channel_groups": [channel.value for channel in self.channel_groups],
            "include_velocity_statistics": self.include_velocity_statistics,
            "include_validity_statistics": self.include_validity_statistics,
            "projection_seed": self.projection_seed,
            "proxy_only": self.proxy_only,
        }


@dataclass(frozen=True, slots=True)
class SemanticAlignmentConfig:
    metric: str
    loss_weight: float
    zero_norm_policy: str
    non_finite_policy: str
    normalize_embeddings: bool

    def __post_init__(self) -> None:
        _require_equal(self.metric, "cosine_distance", "alignment.metric")
        _non_negative_float(self.loss_weight, "alignment.loss_weight")
        _require_equal(self.zero_norm_policy, "fail", "alignment.zero_norm_policy")
        _require_equal(self.non_finite_policy, "fail", "alignment.non_finite_policy")
        _require_true(self.normalize_embeddings, "alignment.normalize_embeddings")

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "loss_weight": float(self.loss_weight),
            "zero_norm_policy": self.zero_norm_policy,
            "non_finite_policy": self.non_finite_policy,
            "normalize_embeddings": self.normalize_embeddings,
        }


@dataclass(frozen=True, slots=True)
class SemanticAblationConfig:
    required: bool
    comparison: str
    same_model_config_required: bool
    same_manifest_family_required: bool
    same_seed_required: bool
    same_validation_protocol_required: bool

    def __post_init__(self) -> None:
        _require_true(self.required, "ablation.required")
        _require_equal(self.comparison, "with_without_objective", "ablation.comparison")
        for name in (
            "same_model_config_required",
            "same_manifest_family_required",
            "same_seed_required",
            "same_validation_protocol_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SemanticConsistencyError(f"ablation.{name} must be a boolean.")
        for name in (
            "same_model_config_required",
            "same_manifest_family_required",
            "same_validation_protocol_required",
        ):
            _require_true(getattr(self, name), f"ablation.{name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "required": self.required,
            "comparison": self.comparison,
            "same_model_config_required": self.same_model_config_required,
            "same_manifest_family_required": self.same_manifest_family_required,
            "same_seed_required": self.same_seed_required,
            "same_validation_protocol_required": self.same_validation_protocol_required,
        }


@dataclass(frozen=True, slots=True)
class SemanticObjectiveReportConfig:
    write_attachment_decision_report: bool
    write_objective_config_report: bool
    write_ablation_plan_report: bool
    write_metric_limitations_report: bool
    write_risk_controls_report: bool

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_true(getattr(self, name), f"reports.{name}")

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SemanticTrainingPoseProjectionConfig:
    backend: str
    embedding_dim_policy: str
    coordinate_mode: str
    channel_groups: tuple[PoseChannel, ...]
    include_velocity_statistics: bool
    include_validity_statistics: bool
    projection_seed: int
    trainable_projection: bool

    def __post_init__(self) -> None:
        _require_equal(
            self.backend,
            "differentiable_bfh_statistics_projection",
            "training_objective.pose_projection.backend",
        )
        _require_equal(
            self.embedding_dim_policy,
            "match_text_embedding",
            "training_objective.pose_projection.embedding_dim_policy",
        )
        _require_equal(self.coordinate_mode, "xy", "training_objective.pose_projection.coordinate_mode")
        channels = _pose_channels(
            self.channel_groups,
            "training_objective.pose_projection.channel_groups",
        )
        if channels != _CANONICAL_POSE_CHANNELS:
            raise SemanticConsistencyError(
                "training_objective.pose_projection.channel_groups must be "
                "body,left_hand,right_hand,face in canonical order."
            )
        if not isinstance(self.include_velocity_statistics, bool):
            raise SemanticConsistencyError(
                "training_objective.pose_projection.include_velocity_statistics must be a boolean."
            )
        if not isinstance(self.include_validity_statistics, bool):
            raise SemanticConsistencyError(
                "training_objective.pose_projection.include_validity_statistics must be a boolean."
            )
        if (
            not isinstance(self.projection_seed, int)
            or isinstance(self.projection_seed, bool)
            or self.projection_seed < 0
        ):
            raise SemanticConsistencyError(
                "training_objective.pose_projection.projection_seed must be a non-negative integer."
            )
        if self.trainable_projection is not False:
            raise SemanticConsistencyError(
                "training_objective.pose_projection.trainable_projection must be false."
            )
        object.__setattr__(self, "channel_groups", channels)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "embedding_dim_policy": self.embedding_dim_policy,
            "coordinate_mode": self.coordinate_mode,
            "channel_groups": [channel.value for channel in self.channel_groups],
            "include_velocity_statistics": self.include_velocity_statistics,
            "include_validity_statistics": self.include_validity_statistics,
            "projection_seed": self.projection_seed,
            "trainable_projection": self.trainable_projection,
        }


@dataclass(frozen=True, slots=True)
class SemanticTrainingAlignmentConfig:
    metric: str
    loss_weight: float
    normalize_embeddings: bool
    zero_norm_policy: str
    non_finite_policy: str

    def __post_init__(self) -> None:
        _require_equal(self.metric, "cosine_distance", "training_objective.alignment.metric")
        if _float(self.loss_weight, "training_objective.alignment.loss_weight") <= 0.0:
            raise SemanticConsistencyError("training_objective.alignment.loss_weight must be > 0.")
        _require_true(
            self.normalize_embeddings,
            "training_objective.alignment.normalize_embeddings",
        )
        _require_equal(
            self.zero_norm_policy,
            "fail",
            "training_objective.alignment.zero_norm_policy",
        )
        _require_equal(
            self.non_finite_policy,
            "fail",
            "training_objective.alignment.non_finite_policy",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "loss_weight": float(self.loss_weight),
            "normalize_embeddings": self.normalize_embeddings,
            "zero_norm_policy": self.zero_norm_policy,
            "non_finite_policy": self.non_finite_policy,
        }


@dataclass(frozen=True, slots=True)
class SemanticTrainingObjectiveConfig:
    enabled: bool
    role: str
    text_source: str
    detach_text_embedding: bool
    pose_projection: SemanticTrainingPoseProjectionConfig
    alignment: SemanticTrainingAlignmentConfig
    supported_models: tuple[ModelKey, ...]
    supported_stages: Mapping[ModelKey, str]
    requires_ablation: bool
    proxy_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise SemanticConsistencyError("training_objective.enabled must be a boolean.")
        models = _model_keys(self.supported_models, "training_objective.supported_models")
        stages = {
            _model_key(key, "training_objective.supported_stages"): _text(
                value,
                f"training_objective.supported_stages.{key}",
            )
            for key, value in dict(self.supported_stages).items()
        }
        object.__setattr__(self, "supported_models", models)
        object.__setattr__(self, "supported_stages", MappingProxyType(stages))
        if not self.enabled:
            return
        _require_equal(
            self.role,
            "differentiable_auxiliary_training_loss",
            "training_objective.role",
        )
        _require_equal(
            self.text_source,
            "candidate_text_embedding",
            "training_objective.text_source",
        )
        _require_true(self.detach_text_embedding, "training_objective.detach_text_embedding")
        if ModelKey.BASE_DIRECT in models:
            raise SemanticConsistencyError(
                "training_objective.supported_models must not include base_direct."
            )
        expected = {
            ModelKey.LEARNED_POSE_TOKEN: "train_text_to_token",
            ModelKey.LATENT_DIFFUSION: "train_denoiser",
            ModelKey.ARTICULATOR_AWARE: "train_structure_aware",
        }
        if set(models) != set(expected) or stages != expected:
            raise SemanticConsistencyError(
                "training_objective must support learned_pose_token/train_text_to_token, "
                "latent_diffusion/train_denoiser, and articulator_aware/train_structure_aware."
            )
        _require_true(self.requires_ablation, "training_objective.requires_ablation")
        if self.proxy_only is not False:
            raise SemanticConsistencyError("training_objective.proxy_only must be false.")

    @classmethod
    def disabled(cls) -> "SemanticTrainingObjectiveConfig":
        return cls(
            enabled=False,
            role="proxy_only_diagnostic",
            text_source="none",
            detach_text_embedding=True,
            pose_projection=SemanticTrainingPoseProjectionConfig(
                backend="differentiable_bfh_statistics_projection",
                embedding_dim_policy="match_text_embedding",
                coordinate_mode="xy",
                channel_groups=_CANONICAL_POSE_CHANNELS,
                include_velocity_statistics=True,
                include_validity_statistics=True,
                projection_seed=9001,
                trainable_projection=False,
            ),
            alignment=SemanticTrainingAlignmentConfig(
                metric="cosine_distance",
                loss_weight=0.02,
                normalize_embeddings=True,
                zero_norm_policy="fail",
                non_finite_policy="fail",
            ),
            supported_models=(
                ModelKey.LEARNED_POSE_TOKEN,
                ModelKey.LATENT_DIFFUSION,
                ModelKey.ARTICULATOR_AWARE,
            ),
            supported_stages={
                ModelKey.LEARNED_POSE_TOKEN: "train_text_to_token",
                ModelKey.LATENT_DIFFUSION: "train_denoiser",
                ModelKey.ARTICULATOR_AWARE: "train_structure_aware",
            },
            requires_ablation=True,
            proxy_only=True,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "role": self.role,
            "text_source": self.text_source,
            "detach_text_embedding": self.detach_text_embedding,
            "pose_projection": self.pose_projection.to_dict(),
            "alignment": self.alignment.to_dict(),
            "supported_models": [model.value for model in self.supported_models],
            "supported_stages": {
                model.value: stage for model, stage in self.supported_stages.items()
            },
            "requires_ablation": self.requires_ablation,
            "proxy_only": self.proxy_only,
        }


@dataclass(frozen=True, slots=True)
class SemanticConsistencyObjectiveConfig:
    identity: SemanticObjectiveIdentityConfig
    attachment: SemanticAttachmentConfig
    text_embedding: SemanticTextEmbeddingConfig
    pose_embedding: SemanticPoseEmbeddingConfig
    alignment: SemanticAlignmentConfig
    ablation: SemanticAblationConfig
    reports: SemanticObjectiveReportConfig
    training_objective: SemanticTrainingObjectiveConfig

    def __post_init__(self) -> None:
        if self.identity.requires_ablation is not self.ablation.required:
            raise SemanticConsistencyError(
                "identity.requires_ablation must agree with ablation.required."
            )
        if self.text_embedding.embedding_dim != self.pose_embedding.embedding_dim:
            raise SemanticConsistencyError(
                "text_embedding.embedding_dim must match pose_embedding.embedding_dim "
                "for cosine alignment."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "attachment": self.attachment.to_dict(),
            "text_embedding": self.text_embedding.to_dict(),
            "pose_embedding": self.pose_embedding.to_dict(),
            "alignment": self.alignment.to_dict(),
            "ablation": self.ablation.to_dict(),
            "reports": self.reports.to_dict(),
            "training_objective": self.training_objective.to_dict(),
        }


def load_semantic_consistency_config(path: Path) -> SemanticConsistencyObjectiveConfig:
    """Load and strictly validate the objective YAML configuration."""

    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SemanticConsistencyError(
            f"semantic_consistency config YAML could not be loaded: {exc}"
        ) from exc
    return semantic_consistency_config_from_mapping(loaded)


def semantic_consistency_config_from_mapping(
    raw_config: object,
) -> SemanticConsistencyObjectiveConfig:
    """Build a validated semantic config from a strict mapping."""

    if not isinstance(raw_config, Mapping):
        raise SemanticConsistencyError("semantic_consistency config must be a mapping.")
    raw = copy.deepcopy(dict(cast(Mapping[str, Any], raw_config)))
    _reject_forbidden_fields(raw, path=())
    if set(raw) == _V1_TOP_LEVEL_KEYS:
        raw["training_objective"] = None
    elif set(raw) != _EXPECTED_TOP_LEVEL_KEYS:
        raise SemanticConsistencyError(
            "semantic_consistency config keys mismatch: "
            f"expected={sorted(_EXPECTED_TOP_LEVEL_KEYS)}, observed={sorted(raw)}."
        )
    identity = _section(raw["identity"], "identity", {
        "objective_key", "canonical_id", "phase_number", "research_role",
        "standalone_model_allowed", "requires_ablation",
    })
    attachment = _section(raw["attachment"], "attachment", {
        "allowed_models", "forbidden_models", "stage_id",
        "require_same_manifest_family", "require_same_splits",
        "require_same_evaluation_protocol",
    })
    text_embedding = _section(raw["text_embedding"], "text_embedding", {
        "backend", "embedding_dim", "pooling", "trainable", "proxy_only",
    })
    pose_embedding = _section(raw["pose_embedding"], "pose_embedding", {
        "backend", "embedding_dim", "coordinate_mode", "channel_groups",
        "include_velocity_statistics", "include_validity_statistics",
        "projection_seed", "proxy_only",
    })
    alignment = _section(raw["alignment"], "alignment", {
        "metric", "loss_weight", "zero_norm_policy", "non_finite_policy",
        "normalize_embeddings",
    })
    ablation = _section(raw["ablation"], "ablation", {
        "required", "comparison", "same_model_config_required",
        "same_manifest_family_required", "same_seed_required",
        "same_validation_protocol_required",
    })
    reports = _section(raw["reports"], "reports", {
        "write_attachment_decision_report", "write_objective_config_report",
        "write_ablation_plan_report", "write_metric_limitations_report",
        "write_risk_controls_report",
    })
    training_objective = _training_objective(raw["training_objective"])
    return SemanticConsistencyObjectiveConfig(
        identity=SemanticObjectiveIdentityConfig(
            objective_key=_text(identity["objective_key"], "identity.objective_key"),
            canonical_id=_text(identity["canonical_id"], "identity.canonical_id"),
            phase_number=_int(identity["phase_number"], "identity.phase_number"),
            research_role=_text(identity["research_role"], "identity.research_role"),
            standalone_model_allowed=_bool(
                identity["standalone_model_allowed"], "identity.standalone_model_allowed"
            ),
            requires_ablation=_bool(identity["requires_ablation"], "identity.requires_ablation"),
        ),
        attachment=SemanticAttachmentConfig(
            allowed_models=_raw_model_keys(attachment["allowed_models"], "attachment.allowed_models"),
            forbidden_models=_raw_model_keys(
                attachment["forbidden_models"], "attachment.forbidden_models"
            ),
            stage_id=_text(attachment["stage_id"], "attachment.stage_id"),
            require_same_manifest_family=_bool(
                attachment["require_same_manifest_family"],
                "attachment.require_same_manifest_family",
            ),
            require_same_splits=_bool(attachment["require_same_splits"], "attachment.require_same_splits"),
            require_same_evaluation_protocol=_bool(
                attachment["require_same_evaluation_protocol"],
                "attachment.require_same_evaluation_protocol",
            ),
        ),
        text_embedding=SemanticTextEmbeddingConfig(
            backend=_text(text_embedding["backend"], "text_embedding.backend"),
            embedding_dim=_int(text_embedding["embedding_dim"], "text_embedding.embedding_dim"),
            pooling=_text(text_embedding["pooling"], "text_embedding.pooling"),
            trainable=_bool(text_embedding["trainable"], "text_embedding.trainable"),
            proxy_only=_bool(text_embedding["proxy_only"], "text_embedding.proxy_only"),
        ),
        pose_embedding=SemanticPoseEmbeddingConfig(
            backend=_text(pose_embedding["backend"], "pose_embedding.backend"),
            embedding_dim=_int(pose_embedding["embedding_dim"], "pose_embedding.embedding_dim"),
            coordinate_mode=_text(pose_embedding["coordinate_mode"], "pose_embedding.coordinate_mode"),
            channel_groups=_raw_pose_channels(pose_embedding["channel_groups"]),
            include_velocity_statistics=_bool(
                pose_embedding["include_velocity_statistics"],
                "pose_embedding.include_velocity_statistics",
            ),
            include_validity_statistics=_bool(
                pose_embedding["include_validity_statistics"],
                "pose_embedding.include_validity_statistics",
            ),
            projection_seed=_int(pose_embedding["projection_seed"], "pose_embedding.projection_seed"),
            proxy_only=_bool(pose_embedding["proxy_only"], "pose_embedding.proxy_only"),
        ),
        alignment=SemanticAlignmentConfig(
            metric=_text(alignment["metric"], "alignment.metric"),
            loss_weight=_float(alignment["loss_weight"], "alignment.loss_weight"),
            zero_norm_policy=_text(alignment["zero_norm_policy"], "alignment.zero_norm_policy"),
            non_finite_policy=_text(alignment["non_finite_policy"], "alignment.non_finite_policy"),
            normalize_embeddings=_bool(
                alignment["normalize_embeddings"], "alignment.normalize_embeddings"
            ),
        ),
        ablation=SemanticAblationConfig(
            required=_bool(ablation["required"], "ablation.required"),
            comparison=_text(ablation["comparison"], "ablation.comparison"),
            same_model_config_required=_bool(
                ablation["same_model_config_required"], "ablation.same_model_config_required"
            ),
            same_manifest_family_required=_bool(
                ablation["same_manifest_family_required"], "ablation.same_manifest_family_required"
            ),
            same_seed_required=_bool(ablation["same_seed_required"], "ablation.same_seed_required"),
            same_validation_protocol_required=_bool(
                ablation["same_validation_protocol_required"],
                "ablation.same_validation_protocol_required",
            ),
        ),
        reports=SemanticObjectiveReportConfig(**reports),
        training_objective=training_objective,
    )


def _training_objective(value: object) -> SemanticTrainingObjectiveConfig:
    if value is None:
        return SemanticTrainingObjectiveConfig.disabled()
    raw = _section(value, "training_objective", {
        "enabled", "role", "text_source", "detach_text_embedding", "pose_projection",
        "alignment", "supported_models", "supported_stages", "requires_ablation", "proxy_only",
    })
    pose_projection = _section(raw["pose_projection"], "training_objective.pose_projection", {
        "backend", "embedding_dim_policy", "coordinate_mode", "channel_groups",
        "include_velocity_statistics", "include_validity_statistics", "projection_seed",
        "trainable_projection",
    })
    alignment = _section(raw["alignment"], "training_objective.alignment", {
        "metric", "loss_weight", "normalize_embeddings", "zero_norm_policy",
        "non_finite_policy",
    })
    stages = _section(raw["supported_stages"], "training_objective.supported_stages", {
        "learned_pose_token", "latent_diffusion", "articulator_aware",
    })
    return SemanticTrainingObjectiveConfig(
        enabled=_bool(raw["enabled"], "training_objective.enabled"),
        role=_text(raw["role"], "training_objective.role"),
        text_source=_text(raw["text_source"], "training_objective.text_source"),
        detach_text_embedding=_bool(
            raw["detach_text_embedding"],
            "training_objective.detach_text_embedding",
        ),
        pose_projection=SemanticTrainingPoseProjectionConfig(
            backend=_text(pose_projection["backend"], "training_objective.pose_projection.backend"),
            embedding_dim_policy=_text(
                pose_projection["embedding_dim_policy"],
                "training_objective.pose_projection.embedding_dim_policy",
            ),
            coordinate_mode=_text(
                pose_projection["coordinate_mode"],
                "training_objective.pose_projection.coordinate_mode",
            ),
            channel_groups=_raw_training_pose_channels(pose_projection["channel_groups"]),
            include_velocity_statistics=_bool(
                pose_projection["include_velocity_statistics"],
                "training_objective.pose_projection.include_velocity_statistics",
            ),
            include_validity_statistics=_bool(
                pose_projection["include_validity_statistics"],
                "training_objective.pose_projection.include_validity_statistics",
            ),
            projection_seed=_int(
                pose_projection["projection_seed"],
                "training_objective.pose_projection.projection_seed",
            ),
            trainable_projection=_bool(
                pose_projection["trainable_projection"],
                "training_objective.pose_projection.trainable_projection",
            ),
        ),
        alignment=SemanticTrainingAlignmentConfig(
            metric=_text(alignment["metric"], "training_objective.alignment.metric"),
            loss_weight=_float(
                alignment["loss_weight"],
                "training_objective.alignment.loss_weight",
            ),
            normalize_embeddings=_bool(
                alignment["normalize_embeddings"],
                "training_objective.alignment.normalize_embeddings",
            ),
            zero_norm_policy=_text(
                alignment["zero_norm_policy"],
                "training_objective.alignment.zero_norm_policy",
            ),
            non_finite_policy=_text(
                alignment["non_finite_policy"],
                "training_objective.alignment.non_finite_policy",
            ),
        ),
        supported_models=_raw_model_keys(
            raw["supported_models"],
            "training_objective.supported_models",
        ),
        supported_stages={
            ModelKey.LEARNED_POSE_TOKEN: _text(
                stages["learned_pose_token"],
                "training_objective.supported_stages.learned_pose_token",
            ),
            ModelKey.LATENT_DIFFUSION: _text(
                stages["latent_diffusion"],
                "training_objective.supported_stages.latent_diffusion",
            ),
            ModelKey.ARTICULATOR_AWARE: _text(
                stages["articulator_aware"],
                "training_objective.supported_stages.articulator_aware",
            ),
        },
        requires_ablation=_bool(raw["requires_ablation"], "training_objective.requires_ablation"),
        proxy_only=_bool(raw["proxy_only"], "training_objective.proxy_only"),
    )


def _reject_forbidden_fields(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SemanticConsistencyError("semantic_consistency config keys must be strings.")
            if key.lower() in _FORBIDDEN_FIELD_KEYS:
                location = ".".join((*path, key))
                raise SemanticConsistencyError(
                    f"{location} is forbidden in the semantic objective foundation; "
                    "retrieval, gloss, dictionary, avatar, rendering, provider, and "
                    "manifest-family mechanisms are out of scope."
                )
            _reject_forbidden_fields(child, path=(*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=(*path, str(index)))


def _section(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SemanticConsistencyError(f"{name} must be a mapping with string keys.")
    if set(value) != keys:
        raise SemanticConsistencyError(
            f"{name} keys mismatch: expected={sorted(keys)}, observed={sorted(value)}."
        )
    return dict(cast(Mapping[str, Any], value))


def _raw_model_keys(value: object, name: str) -> tuple[ModelKey, ...]:
    if not isinstance(value, list):
        raise SemanticConsistencyError(f"{name} must be a list.")
    return tuple(_model_key(item, name) for item in value)


def _raw_pose_channels(value: object) -> tuple[PoseChannel, ...]:
    if not isinstance(value, list):
        raise SemanticConsistencyError("pose_embedding.channel_groups must be a list.")
    try:
        return tuple(PoseChannel(item) for item in value)
    except ValueError as exc:
        raise SemanticConsistencyError(
            "pose_embedding.channel_groups contains an unknown BFH channel."
        ) from exc


def _raw_training_pose_channels(value: object) -> tuple[PoseChannel, ...]:
    if not isinstance(value, list):
        raise SemanticConsistencyError(
            "training_objective.pose_projection.channel_groups must be a list."
        )
    try:
        return tuple(PoseChannel(item) for item in value)
    except ValueError as exc:
        raise SemanticConsistencyError(
            "training_objective.pose_projection.channel_groups contains an unknown BFH channel."
        ) from exc


def _model_keys(values: tuple[ModelKey, ...], name: str) -> tuple[ModelKey, ...]:
    resolved = tuple(_model_key(value, name) for value in values)
    if len(set(resolved)) != len(resolved):
        raise SemanticConsistencyError(f"{name} must not contain duplicates.")
    return resolved


def _model_key(value: object, name: str) -> ModelKey:
    try:
        return ModelKey(value)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError(f"{name} contains an unknown model key: {value!r}.") from exc


def _pose_channels(values: tuple[PoseChannel, ...], name: str) -> tuple[PoseChannel, ...]:
    try:
        resolved = tuple(PoseChannel(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError(f"{name} contains an unknown BFH channel.") from exc
    if len(set(resolved)) != len(resolved):
        raise SemanticConsistencyError(f"{name} must not contain duplicates.")
    return resolved


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{name} must be non-empty.")
    return value


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be a boolean.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be an integer.")
    return value


def _float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SemanticConsistencyError(f"{name} must be a number.")
    amount = float(value)
    if not math.isfinite(amount):
        raise SemanticConsistencyError(f"{name} must be finite.")
    return amount


def _positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SemanticConsistencyError(f"{name} must be a positive integer.")


def _non_negative_float(value: object, name: str) -> None:
    if _float(value, name) < 0.0:
        raise SemanticConsistencyError(f"{name} must be non-negative.")


def _require_equal(value: object, expected: str, name: str) -> None:
    if value != expected:
        raise SemanticConsistencyError(f"{name} must be {expected!r}.")


def _require_true(value: object, name: str) -> None:
    if value is not True:
        raise SemanticConsistencyError(f"{name} must be true.")


__all__ = [
    "SEMANTIC_OBJECTIVE_CONFIG_SCHEMA_VERSION",
    "SemanticAblationConfig",
    "SemanticAlignmentConfig",
    "SemanticAttachmentConfig",
    "SemanticConsistencyObjectiveConfig",
    "SemanticObjectiveIdentityConfig",
    "SemanticObjectiveReportConfig",
    "SemanticPoseEmbeddingConfig",
    "SemanticTextEmbeddingConfig",
    "SemanticTrainingAlignmentConfig",
    "SemanticTrainingObjectiveConfig",
    "SemanticTrainingPoseProjectionConfig",
    "load_semantic_consistency_config",
    "semantic_consistency_config_from_mapping",
]
