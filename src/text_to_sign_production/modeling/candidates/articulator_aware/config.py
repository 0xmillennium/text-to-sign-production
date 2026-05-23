"""Strict config contract for the articulator-aware foundation."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.text_encoder import (
    TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH,
    TextEncoderConfig,
)
from text_to_sign_production.modeling.candidates.articulator_aware.errors import (
    ArticulatorAwareError,
)

_PRIMARY_CHANNELS = (
    PoseChannel.BODY,
    PoseChannel.LEFT_HAND,
    PoseChannel.RIGHT_HAND,
    PoseChannel.FACE,
)

_EXPECTED_TOP_LEVEL_KEYS = {
    "identity",
    "data",
    "partition_policy",
    "mask_strategy",
    "loss_weighting",
    "structure_variant",
    "text_encoder",
    "training",
    "length",
    "generation",
    "checkpoints",
    "reports",
}

_FORBIDDEN_FIELD_KEYS = frozenset(
    {
        "manifest_family",
        "learned_token",
        "learned_pose_token",
        "pose_token",
        "latent_diffusion",
        "diffusion",
        "retrieval",
        "retrieval_comparator",
        "semantic",
        "semantic_objective",
        "gloss",
        "gloss_supervision",
        "non_manual_annotation",
        "non_manual_linguistic_annotation",
        "manual_non_manual_linguistic_annotation",
        "avatar",
        "smpl",
        "smplx",
        "SMPL-X",
        "rendering",
    }
)

_LEGACY_ARTICULATOR_TEXT_EMBEDDING_DIM = 256
ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP = "channel_fusion_mlp"
ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL = "articulator_factorized_temporal"
ARTICULATOR_ARCHITECTURES = frozenset(
    {
        ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP,
        ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL,
    }
)


@dataclass(frozen=True, slots=True)
class ArticulatorAwareIdentityConfig:
    model_key: str
    research_role: str
    phase_number: int

    def __post_init__(self) -> None:
        _require_equal(self.model_key, "articulator_aware", "identity.model_key")
        _require_text(self.research_role, "identity.research_role")
        if self.phase_number != 8:
            raise ArticulatorAwareError("identity.phase_number must be 8.")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "research_role": self.research_role,
            "phase_number": self.phase_number,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareDataConfig:
    min_frame_count: int
    max_frame_count: int | None

    def __post_init__(self) -> None:
        _require_positive_int(self.min_frame_count, "data.min_frame_count")
        if self.max_frame_count is not None:
            _require_positive_int(self.max_frame_count, "data.max_frame_count")
            if self.max_frame_count < self.min_frame_count:
                raise ArticulatorAwareError(
                    "data.max_frame_count must be null or at least data.min_frame_count."
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "min_frame_count": self.min_frame_count,
            "max_frame_count": self.max_frame_count,
        }


@dataclass(frozen=True, slots=True)
class ChannelPartitionPolicyConfig:
    source: str
    primary_channels: tuple[PoseChannel, ...]
    composite_groups: Mapping[str, tuple[PoseChannel, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _require_equal(self.source, "canonical_bfh_channels", "partition_policy.source")
        primary = _channel_tuple(self.primary_channels, "partition_policy.primary_channels")
        if primary != _PRIMARY_CHANNELS:
            raise ArticulatorAwareError(
                "partition_policy.primary_channels must be exactly "
                "body,left_hand,right_hand,face in canonical order."
            )
        if len(set(primary)) != len(primary):
            raise ArticulatorAwareError("partition_policy.primary_channels must not contain duplicates.")
        groups = _coerce_composite_groups(self.composite_groups, primary)
        object.__setattr__(self, "primary_channels", primary)
        object.__setattr__(self, "composite_groups", MappingProxyType(groups))

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "primary_channels": [channel.value for channel in self.primary_channels],
            "composite_groups": {
                name: [channel.value for channel in self.composite_groups[name]]
                for name in sorted(self.composite_groups)
            },
        }


@dataclass(frozen=True, slots=True)
class ChannelMaskStrategyConfig:
    validity_source: str
    confidence_policy: str
    missing_channel_policy: str
    all_invalid_sample_policy: str
    channel_min_valid_fraction: Mapping[PoseChannel, float]

    def __post_init__(self) -> None:
        _require_equal(
            self.validity_source,
            "bfh_confidence_and_frame_validity",
            "mask_strategy.validity_source",
        )
        _require_equal(self.confidence_policy, "mask_only", "mask_strategy.confidence_policy")
        if self.missing_channel_policy not in {"skip_channel_loss", "fail"}:
            raise ArticulatorAwareError(
                "mask_strategy.missing_channel_policy must be 'skip_channel_loss' or 'fail'."
            )
        _require_equal(
            self.all_invalid_sample_policy,
            "fail",
            "mask_strategy.all_invalid_sample_policy",
        )
        fractions = _channel_number_mapping(
            self.channel_min_valid_fraction,
            "mask_strategy.channel_min_valid_fraction",
            allow_zero=True,
            require_positive=False,
        )
        for channel, value in fractions.items():
            if value > 1.0:
                raise ArticulatorAwareError(
                    f"mask_strategy.channel_min_valid_fraction.{channel.value} must be in [0, 1]."
                )
        object.__setattr__(self, "channel_min_valid_fraction", MappingProxyType(fractions))

    def to_dict(self) -> dict[str, object]:
        return {
            "validity_source": self.validity_source,
            "confidence_policy": self.confidence_policy,
            "missing_channel_policy": self.missing_channel_policy,
            "all_invalid_sample_policy": self.all_invalid_sample_policy,
            "channel_min_valid_fraction": _channel_mapping_to_dict(
                self.channel_min_valid_fraction
            ),
        }


@dataclass(frozen=True, slots=True)
class ChannelLossWeightingConfig:
    policy: str
    normalize_weights: bool
    channel_weights: Mapping[PoseChannel, float]
    velocity_weight: float
    symmetry_weight: float
    cross_channel_consistency_weight: float

    def __post_init__(self) -> None:
        _require_equal(self.policy, "static_channel_weights", "loss_weighting.policy")
        if not isinstance(self.normalize_weights, bool):
            raise ArticulatorAwareError("loss_weighting.normalize_weights must be a boolean.")
        weights = _channel_number_mapping(
            self.channel_weights,
            "loss_weighting.channel_weights",
            allow_zero=False,
            require_positive=True,
        )
        object.__setattr__(self, "channel_weights", MappingProxyType(weights))
        for value, name in (
            (self.velocity_weight, "loss_weighting.velocity_weight"),
            (self.symmetry_weight, "loss_weighting.symmetry_weight"),
            (
                self.cross_channel_consistency_weight,
                "loss_weighting.cross_channel_consistency_weight",
            ),
        ):
            _require_non_negative_number(value, name)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "normalize_weights": self.normalize_weights,
            "channel_weights": _channel_mapping_to_dict(self.channel_weights),
            "velocity_weight": float(self.velocity_weight),
            "symmetry_weight": float(self.symmetry_weight),
            "cross_channel_consistency_weight": float(self.cross_channel_consistency_weight),
        }


@dataclass(frozen=True, slots=True)
class StructureVariantConfig:
    architecture: str
    shared_text_encoder: str
    channel_specific_heads: bool
    fusion: str
    hidden_dim: int
    dropout: float
    temporal_layers: int = 1
    fusion_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        if self.architecture not in ARTICULATOR_ARCHITECTURES:
            raise ArticulatorAwareError(
                "structure_variant.architecture must be one of "
                "{'channel_fusion_mlp', 'articulator_factorized_temporal'}."
            )
        if self.shared_text_encoder not in {
            TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH,
            "huggingface",
        }:
            raise ArticulatorAwareError(
                "structure_variant.shared_text_encoder must be "
                "'deterministic_hash' or 'huggingface'."
            )
        if not isinstance(self.channel_specific_heads, bool):
            raise ArticulatorAwareError(
                "structure_variant.channel_specific_heads must be a boolean."
            )
        if not self.channel_specific_heads:
            raise ArticulatorAwareError(
                "structure_variant.channel_specific_heads must be true for articulator_aware."
            )
        _require_positive_int(self.hidden_dim, "structure_variant.hidden_dim")
        _require_dropout(self.dropout, "structure_variant.dropout")
        _require_positive_int(self.temporal_layers, "structure_variant.temporal_layers")
        fusion_hidden_dim = (
            self.hidden_dim if self.fusion_hidden_dim is None else self.fusion_hidden_dim
        )
        _require_positive_int(fusion_hidden_dim, "structure_variant.fusion_hidden_dim")
        object.__setattr__(self, "fusion_hidden_dim", fusion_hidden_dim)
        if self.architecture == ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP:
            _require_equal(self.fusion, "concat", "structure_variant.fusion")
        if self.architecture == ARTICULATOR_ARCHITECTURE_FACTORIZED_TEMPORAL:
            if self.fusion != "gated_cross_channel":
                raise ArticulatorAwareError(
                    "articulator_factorized_temporal requires "
                    "structure_variant.fusion='gated_cross_channel'."
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "shared_text_encoder": self.shared_text_encoder,
            "channel_specific_heads": self.channel_specific_heads,
            "fusion": self.fusion,
            "hidden_dim": self.hidden_dim,
            "dropout": float(self.dropout),
            "temporal_layers": self.temporal_layers,
            "fusion_hidden_dim": self.fusion_hidden_dim,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareTrainingConfig:
    max_epochs: int
    batch_size: int
    frame_batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int | None
    num_workers: int
    device: str

    def __post_init__(self) -> None:
        _require_positive_int(self.max_epochs, "training.max_epochs")
        _require_positive_int(self.batch_size, "training.batch_size")
        _require_positive_int(self.frame_batch_size, "training.frame_batch_size")
        _require_positive_number(self.learning_rate, "training.learning_rate")
        _require_non_negative_number(self.weight_decay, "training.weight_decay")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ArticulatorAwareError("training.seed must be an integer or null.")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise ArticulatorAwareError("training.num_workers must be a non-negative integer.")
        _require_text(self.device, "training.device")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "frame_batch_size": self.frame_batch_size,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "seed": self.seed,
            "num_workers": self.num_workers,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareGenerationConfig:
    generation_mode: str
    confidence_policy: str
    length_policy: str

    def __post_init__(self) -> None:
        _require_equal(self.generation_mode, "deterministic", "generation.generation_mode")
        _require_equal(
            self.confidence_policy,
            "synthetic_validity",
            "generation.confidence_policy",
        )
        _require_equal(self.length_policy, "predicted_length", "generation.length_policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "generation_mode": self.generation_mode,
            "confidence_policy": self.confidence_policy,
            "length_policy": self.length_policy,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareLengthConfig:
    policy: str
    predictor_type: str
    min_generated_frames: int
    max_generated_frames: int | None
    loss_weight: float
    max_positions: int

    def __post_init__(self) -> None:
        _require_equal(self.policy, "predicted_length", "length.policy")
        _require_equal(self.predictor_type, "mlp", "length.predictor_type")
        _require_positive_int(self.min_generated_frames, "length.min_generated_frames")
        if self.max_generated_frames is not None:
            _require_positive_int(self.max_generated_frames, "length.max_generated_frames")
            if self.max_generated_frames < self.min_generated_frames:
                raise ArticulatorAwareError(
                    "length.max_generated_frames must be null or at least min_generated_frames."
                )
        _require_non_negative_number(self.loss_weight, "length.loss_weight")
        _require_positive_int(self.max_positions, "length.max_positions")
        if self.max_positions < self.min_generated_frames:
            raise ArticulatorAwareError(
                "length.max_positions must be at least length.min_generated_frames."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "predictor_type": self.predictor_type,
            "min_generated_frames": self.min_generated_frames,
            "max_generated_frames": self.max_generated_frames,
            "loss_weight": float(self.loss_weight),
            "max_positions": self.max_positions,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareCheckpointConfig:
    selection_metric: str
    lower_is_better: bool

    def __post_init__(self) -> None:
        _require_equal(
            self.selection_metric,
            "validation_channel_weighted_loss",
            "checkpoints.selection_metric",
        )
        if not isinstance(self.lower_is_better, bool):
            raise ArticulatorAwareError("checkpoints.lower_is_better must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_metric": self.selection_metric,
            "lower_is_better": self.lower_is_better,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareReportConfig:
    write_partition_report: bool
    write_mask_report: bool
    write_loss_weighting_report: bool
    write_channel_diagnostics_report: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if not isinstance(getattr(self, field_name), bool):
                raise ArticulatorAwareError(f"reports.{field_name} must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "write_partition_report": self.write_partition_report,
            "write_mask_report": self.write_mask_report,
            "write_loss_weighting_report": self.write_loss_weighting_report,
            "write_channel_diagnostics_report": self.write_channel_diagnostics_report,
        }


@dataclass(frozen=True, slots=True)
class ArticulatorAwareConfig:
    identity: ArticulatorAwareIdentityConfig
    data: ArticulatorAwareDataConfig
    partition_policy: ChannelPartitionPolicyConfig
    mask_strategy: ChannelMaskStrategyConfig
    loss_weighting: ChannelLossWeightingConfig
    structure_variant: StructureVariantConfig
    text_encoder: TextEncoderConfig
    training: ArticulatorAwareTrainingConfig
    length: ArticulatorAwareLengthConfig
    generation: ArticulatorAwareGenerationConfig
    checkpoints: ArticulatorAwareCheckpointConfig
    reports: ArticulatorAwareReportConfig

    def __post_init__(self) -> None:
        if (
            self.structure_variant.architecture == ARTICULATOR_ARCHITECTURE_CHANNEL_FUSION_MLP
            and (
                self.loss_weighting.velocity_weight != 0.0
                or self.loss_weighting.symmetry_weight != 0.0
                or self.loss_weighting.cross_channel_consistency_weight != 0.0
            )
        ):
            raise ArticulatorAwareError(
                "channel_fusion_mlp does not support non-zero temporal/channel auxiliary "
                "losses; use articulator_factorized_temporal."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "data": self.data.to_dict(),
            "partition_policy": self.partition_policy.to_dict(),
            "mask_strategy": self.mask_strategy.to_dict(),
            "loss_weighting": self.loss_weighting.to_dict(),
            "structure_variant": self.structure_variant.to_dict(),
            "text_encoder": self.text_encoder.to_dict(),
            "training": self.training.to_dict(),
            "length": self.length.to_dict(),
            "generation": self.generation.to_dict(),
            "checkpoints": self.checkpoints.to_dict(),
            "reports": self.reports.to_dict(),
        }


def load_articulator_aware_config(path: Path) -> ArticulatorAwareConfig:
    """Load and validate the articulator-aware foundation YAML config."""

    try:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ArticulatorAwareError(f"articulator_aware config YAML is invalid: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ArticulatorAwareError("articulator_aware config must be a YAML mapping.")
    return articulator_aware_config_from_mapping(loaded)


def articulator_aware_config_from_mapping(
    raw_config: Mapping[str, object],
) -> ArticulatorAwareConfig:
    """Build a validated articulator-aware config from a strict mapping."""

    raw = copy.deepcopy(dict(cast(Mapping[str, Any], raw_config)))
    _reject_forbidden_fields(raw, path=())
    has_text_encoder = "text_encoder" in raw
    expected_keys = _EXPECTED_TOP_LEVEL_KEYS if has_text_encoder else _EXPECTED_TOP_LEVEL_KEYS - {"text_encoder"}
    if set(raw) != expected_keys:
        raise ArticulatorAwareError(
            f"articulator_aware config keys mismatch: expected={sorted(expected_keys)}, "
            f"observed={sorted(raw)}."
        )
    structure_variant = StructureVariantConfig(
        **_mapping(raw["structure_variant"], "structure_variant")
    )
    text_encoder = (
        _text_encoder_config(_mapping(raw["text_encoder"], "text_encoder"), structure_variant=structure_variant)
        if has_text_encoder
        else _legacy_text_encoder_config(structure_variant)
    )
    return ArticulatorAwareConfig(
        identity=ArticulatorAwareIdentityConfig(**_mapping(raw["identity"], "identity")),
        data=ArticulatorAwareDataConfig(**_mapping(raw["data"], "data")),
        partition_policy=ChannelPartitionPolicyConfig(
            source=_text(
                _mapping(raw["partition_policy"], "partition_policy")["source"],
                "partition_policy.source",
            ),
            primary_channels=_channel_tuple(
                _mapping(raw["partition_policy"], "partition_policy")["primary_channels"],
                "partition_policy.primary_channels",
            ),
            composite_groups=_composite_groups_from_raw(
                _mapping(raw["partition_policy"], "partition_policy").get("composite_groups", {})
            ),
        ),
        mask_strategy=ChannelMaskStrategyConfig(
            validity_source=_text(
                _mapping(raw["mask_strategy"], "mask_strategy")["validity_source"],
                "mask_strategy.validity_source",
            ),
            confidence_policy=_text(
                _mapping(raw["mask_strategy"], "mask_strategy")["confidence_policy"],
                "mask_strategy.confidence_policy",
            ),
            missing_channel_policy=_text(
                _mapping(raw["mask_strategy"], "mask_strategy")["missing_channel_policy"],
                "mask_strategy.missing_channel_policy",
            ),
            all_invalid_sample_policy=_text(
                _mapping(raw["mask_strategy"], "mask_strategy")["all_invalid_sample_policy"],
                "mask_strategy.all_invalid_sample_policy",
            ),
            channel_min_valid_fraction=_channel_value_mapping_from_raw(
                _mapping(raw["mask_strategy"], "mask_strategy")["channel_min_valid_fraction"],
                "mask_strategy.channel_min_valid_fraction",
            ),
        ),
        loss_weighting=ChannelLossWeightingConfig(
            policy=_text(
                _mapping(raw["loss_weighting"], "loss_weighting")["policy"],
                "loss_weighting.policy",
            ),
            normalize_weights=_bool(
                _mapping(raw["loss_weighting"], "loss_weighting")["normalize_weights"],
                "loss_weighting.normalize_weights",
            ),
            channel_weights=_channel_value_mapping_from_raw(
                _mapping(raw["loss_weighting"], "loss_weighting")["channel_weights"],
                "loss_weighting.channel_weights",
            ),
            velocity_weight=_number(
                _mapping(raw["loss_weighting"], "loss_weighting")["velocity_weight"],
                "loss_weighting.velocity_weight",
            ),
            symmetry_weight=_number(
                _mapping(raw["loss_weighting"], "loss_weighting")["symmetry_weight"],
                "loss_weighting.symmetry_weight",
            ),
            cross_channel_consistency_weight=_number(
                _mapping(raw["loss_weighting"], "loss_weighting")[
                    "cross_channel_consistency_weight"
                ],
                "loss_weighting.cross_channel_consistency_weight",
            ),
        ),
        structure_variant=structure_variant,
        text_encoder=text_encoder,
        training=ArticulatorAwareTrainingConfig(**_mapping(raw["training"], "training")),
        length=ArticulatorAwareLengthConfig(**_mapping(raw["length"], "length")),
        generation=ArticulatorAwareGenerationConfig(**_mapping(raw["generation"], "generation")),
        checkpoints=ArticulatorAwareCheckpointConfig(
            **_mapping(raw["checkpoints"], "checkpoints")
        ),
        reports=ArticulatorAwareReportConfig(**_mapping(raw["reports"], "reports")),
    )


def _legacy_text_encoder_config(structure_variant: StructureVariantConfig) -> TextEncoderConfig:
    if structure_variant.shared_text_encoder != TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH:
        raise ArticulatorAwareError(
            "articulator_aware config without text_encoder must use legacy "
            "structure_variant.shared_text_encoder='deterministic_hash'."
        )
    return TextEncoderConfig(
        encoder_key="deterministic_hash",
        model_name_or_path=None,
        pooling="mean",
        trainable=False,
        max_length=256,
        embedding_dim=_LEGACY_ARTICULATOR_TEXT_EMBEDDING_DIM,
        backend=TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH,
    )


def _text_encoder_config(
    raw: Mapping[str, object],
    *,
    structure_variant: StructureVariantConfig,
) -> TextEncoderConfig:
    canonical_keys = {
        "encoder_key",
        "model_name_or_path",
        "pooling",
        "trainable",
        "max_length",
        "embedding_dim",
        "backend",
    }
    optional_keys = {"revision", "local_files_only", "output_dim"}
    if not (canonical_keys <= set(raw) <= canonical_keys | optional_keys):
        raise ArticulatorAwareError("text_encoder config keys do not match foundation schema.")
    config = TextEncoderConfig(
        encoder_key=_text(raw["encoder_key"], "text_encoder.encoder_key"),
        model_name_or_path=(
            None
            if raw["model_name_or_path"] is None
            else _text(raw["model_name_or_path"], "text_encoder.model_name_or_path")
        ),
        pooling=_text(raw["pooling"], "text_encoder.pooling"),
        trainable=_bool(raw["trainable"], "text_encoder.trainable"),
        max_length=_int(raw["max_length"], "text_encoder.max_length"),
        embedding_dim=_optional_int(raw, "embedding_dim"),
        backend=_text(raw["backend"], "text_encoder.backend"),
        revision=_text(raw.get("revision", "main"), "text_encoder.revision"),
        local_files_only=_bool(
            raw.get("local_files_only", False),
            "text_encoder.local_files_only",
        ),
    )
    if structure_variant.shared_text_encoder != config.backend:
        raise ArticulatorAwareError(
            "text_encoder.backend conflicts with structure_variant.shared_text_encoder."
        )
    return config


def _reject_forbidden_fields(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArticulatorAwareError("articulator_aware config keys must be strings.")
            if key in _FORBIDDEN_FIELD_KEYS:
                dotted = ".".join((*path, key))
                raise ArticulatorAwareError(
                    f"{dotted} is out of scope for this articulator_aware foundation stage."
                )
            _reject_forbidden_fields(child, path=(*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=(*path, str(index)))


def _coerce_composite_groups(
    value: Mapping[str, tuple[PoseChannel, ...]],
    primary_channels: tuple[PoseChannel, ...],
) -> dict[str, tuple[PoseChannel, ...]]:
    if not isinstance(value, Mapping):
        raise ArticulatorAwareError("partition_policy.composite_groups must be a mapping.")
    groups: dict[str, tuple[PoseChannel, ...]] = {}
    for name, channels in value.items():
        _require_text(name, "partition_policy.composite_groups key")
        resolved = _channel_tuple(channels, f"partition_policy.composite_groups.{name}")
        if not resolved:
            raise ArticulatorAwareError(
                f"partition_policy.composite_groups.{name} must include at least one channel."
            )
        unknown = tuple(channel.value for channel in resolved if channel not in primary_channels)
        if unknown:
            raise ArticulatorAwareError(
                f"partition_policy.composite_groups.{name} references channels outside "
                f"primary_channels: {', '.join(unknown)}."
            )
        groups[str(name)] = resolved
    return groups


def _composite_groups_from_raw(value: object) -> Mapping[str, tuple[PoseChannel, ...]]:
    raw = _mapping(value, "partition_policy.composite_groups")
    return {
        name: _channel_tuple(channels, f"partition_policy.composite_groups.{name}")
        for name, channels in raw.items()
    }


def _channel_value_mapping_from_raw(value: object, name: str) -> Mapping[PoseChannel, float]:
    raw = _mapping(value, name)
    resolved: dict[PoseChannel, float] = {}
    for channel, amount in raw.items():
        try:
            resolved_channel = PoseChannel(_text(channel, f"{name} channel"))
        except ValueError as exc:
            raise ArticulatorAwareError(f"{name} contains an unknown pose channel.") from exc
        resolved[resolved_channel] = _number(amount, f"{name}.{resolved_channel.value}")
    return resolved


def _channel_number_mapping(
    value: Mapping[PoseChannel, float],
    name: str,
    *,
    allow_zero: bool,
    require_positive: bool,
) -> dict[PoseChannel, float]:
    if not isinstance(value, Mapping):
        raise ArticulatorAwareError(f"{name} must be a channel mapping.")
    resolved: dict[PoseChannel, float] = {}
    for channel, raw_amount in value.items():
        resolved_channel = PoseChannel(channel)
        amount = _number(raw_amount, f"{name}.{resolved_channel.value}")
        if require_positive and amount <= 0.0:
            raise ArticulatorAwareError(f"{name}.{resolved_channel.value} must be > 0.")
        if not allow_zero and amount == 0.0:
            raise ArticulatorAwareError(f"{name}.{resolved_channel.value} must be non-zero.")
        if amount < 0.0:
            raise ArticulatorAwareError(f"{name}.{resolved_channel.value} must be non-negative.")
        resolved[resolved_channel] = amount
    if set(resolved) != set(_PRIMARY_CHANNELS):
        raise ArticulatorAwareError(f"{name} must define exactly body,left_hand,right_hand,face.")
    return {channel: resolved[channel] for channel in _PRIMARY_CHANNELS}


def _channel_mapping_to_dict(value: Mapping[PoseChannel, float]) -> dict[str, float]:
    return {channel.value: float(value[channel]) for channel in _PRIMARY_CHANNELS}


def _channel_tuple(value: object, name: str) -> tuple[PoseChannel, ...]:
    if not isinstance(value, (list, tuple)):
        raise ArticulatorAwareError(f"{name} must be a list or tuple of channels.")
    try:
        channels = tuple(PoseChannel(channel) for channel in value)
    except ValueError as exc:
        raise ArticulatorAwareError(f"{name} contains an unknown pose channel.") from exc
    if len(set(channels)) != len(channels):
        raise ArticulatorAwareError(f"{name} must not contain duplicate channels.")
    return channels


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ArticulatorAwareError(f"{name} must be a mapping with string keys.")
    return cast(Mapping[str, Any], value)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArticulatorAwareError(f"{name} must be non-empty.")


def _text(value: object, name: str) -> str:
    _require_text(value, name)
    return str(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be a boolean.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be an integer.")
    return value


def _optional_int(mapping: Mapping[str, object], name: str) -> int | None:
    value = mapping.get(name)
    if value is None:
        return None
    return _int(value, name)


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ArticulatorAwareError(f"{name} must be a number.")
    amount = float(value)
    if not math.isfinite(amount):
        raise ArticulatorAwareError(f"{name} must be finite.")
    return amount


def _require_equal(value: object, expected: str, name: str) -> None:
    if value != expected:
        raise ArticulatorAwareError(f"{name} must be {expected!r}.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArticulatorAwareError(f"{name} must be a positive integer.")


def _require_positive_number(value: object, name: str) -> None:
    if _number(value, name) <= 0.0:
        raise ArticulatorAwareError(f"{name} must be a positive number.")


def _require_non_negative_number(value: object, name: str) -> None:
    if _number(value, name) < 0.0:
        raise ArticulatorAwareError(f"{name} must be a non-negative number.")


def _require_dropout(value: object, name: str) -> None:
    amount = _number(value, name)
    if not 0.0 <= amount < 1.0:
        raise ArticulatorAwareError(f"{name} must be in [0, 1).")


__all__ = [
    "ArticulatorAwareCheckpointConfig",
    "ArticulatorAwareConfig",
    "ArticulatorAwareDataConfig",
    "ArticulatorAwareGenerationConfig",
    "ArticulatorAwareIdentityConfig",
    "ArticulatorAwareLengthConfig",
    "ArticulatorAwareReportConfig",
    "ArticulatorAwareTrainingConfig",
    "ChannelLossWeightingConfig",
    "ChannelMaskStrategyConfig",
    "ChannelPartitionPolicyConfig",
    "StructureVariantConfig",
    "articulator_aware_config_from_mapping",
    "load_articulator_aware_config",
]
