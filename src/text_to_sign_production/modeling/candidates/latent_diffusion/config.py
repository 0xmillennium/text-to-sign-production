"""Strict configuration contract for the latent_diffusion foundation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from text_to_sign_production.modeling.backbones.text_encoder import TextEncoderConfig
from text_to_sign_production.modeling.candidates.latent_diffusion.errors import (
    LatentDiffusionError,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.latents import (
    LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
    LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
)

_FORBIDDEN_SECTION_KEYS = frozenset(
    {
        "semantic",
        "semantic_objective",
        "retrieval",
        "retrieval_comparator",
        "audio",
        "audio_conditioning",
        "sparse_keyframe",
        "sparse-keyframe",
        "cfm",
        "CFM",
        "avatar",
        "smpl",
        "SMPL",
        "smplx",
        "SMPL-X",
        "articulator",
        "articulator_aware",
    }
)


@dataclass(frozen=True, slots=True)
class LatentDiffusionIdentityConfig:
    model_key: str
    research_role: str
    phase_number: int

    def __post_init__(self) -> None:
        if self.model_key != "latent_diffusion":
            raise LatentDiffusionError("identity.model_key must be 'latent_diffusion'.")
        if self.phase_number != 7:
            raise LatentDiffusionError("identity.phase_number must be 7.")
        _require_text(self.research_role, "identity.research_role")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "research_role": self.research_role,
            "phase_number": self.phase_number,
        }


@dataclass(frozen=True, slots=True)
class LatentDiffusionDataConfig:
    min_frame_count: int
    max_frame_count: int | None

    def __post_init__(self) -> None:
        _require_positive_int(self.min_frame_count, "data.min_frame_count")
        if self.max_frame_count is not None and (
            not isinstance(self.max_frame_count, int)
            or isinstance(self.max_frame_count, bool)
            or self.max_frame_count < self.min_frame_count
        ):
            raise LatentDiffusionError(
                "data.max_frame_count must be null or at least data.min_frame_count."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "min_frame_count": self.min_frame_count,
            "max_frame_count": self.max_frame_count,
        }


@dataclass(frozen=True, slots=True)
class LatentTargetConfig:
    target_type: str
    temporal_granularity: str
    window_size: int
    stride: int
    coordinate_mode: str
    confidence_policy: str
    standardization: str
    standardization_missing_observation_policy: str = "raise"

    def __post_init__(self) -> None:
        if self.target_type not in {
            LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME,
            LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT,
        }:
            raise LatentDiffusionError(
                "latent_target.target_type must be one of "
                "{'standardized_bfh_frame', 'learned_bfh_window_latent'}."
            )
        _require_positive_int(self.window_size, "latent_target.window_size")
        _require_positive_int(self.stride, "latent_target.stride")
        if self.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
            if (
                self.temporal_granularity != "frame"
                or self.window_size != 1
                or self.stride != 1
            ):
                raise LatentDiffusionError(
                    "standardized_bfh_frame target requires temporal_granularity='frame', "
                    "window_size=1, and stride=1."
                )
        if self.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
            if (
                self.temporal_granularity != "window"
                or self.window_size <= 1
                or self.stride < 1
            ):
                raise LatentDiffusionError(
                    "learned_bfh_window_latent target requires "
                    "temporal_granularity='window', window_size > 1, and stride >= 1."
                )
        _require_equal(self.coordinate_mode, "xy", "latent_target.coordinate_mode")
        _require_equal(self.confidence_policy, "mask_only", "latent_target.confidence_policy")
        _require_equal(self.standardization, "train_split", "latent_target.standardization")
        if self.standardization_missing_observation_policy not in {
            "raise",
            "channel_fallback",
            "global_fallback",
            "identity_fallback",
        }:
            raise LatentDiffusionError(
                "latent_target.standardization_missing_observation_policy must be one of "
                "{'raise', 'channel_fallback', 'global_fallback', 'identity_fallback'}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_type": self.target_type,
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
            "coordinate_mode": self.coordinate_mode,
            "confidence_policy": self.confidence_policy,
            "standardization": self.standardization,
            "standardization_missing_observation_policy": (
                self.standardization_missing_observation_policy
            ),
        }


@dataclass(frozen=True, slots=True)
class LatentAutoencoderConfig:
    architecture: str
    latent_dim: int
    hidden_dim: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    reconstruction_loss: str
    velocity_loss_weight: float

    def __post_init__(self) -> None:
        _require_equal(
            self.architecture,
            "window_mlp_autoencoder",
            "latent_autoencoder.architecture",
        )
        _require_positive_int(self.latent_dim, "latent_autoencoder.latent_dim")
        _require_positive_int(self.hidden_dim, "latent_autoencoder.hidden_dim")
        _require_positive_int(self.max_epochs, "latent_autoencoder.max_epochs")
        _require_positive_int(self.batch_size, "latent_autoencoder.batch_size")
        _require_positive_number(
            self.learning_rate,
            "latent_autoencoder.learning_rate",
        )
        _require_non_negative_number(
            self.weight_decay,
            "latent_autoencoder.weight_decay",
        )
        _require_equal(
            self.reconstruction_loss,
            "masked_mse",
            "latent_autoencoder.reconstruction_loss",
        )
        _require_non_negative_number(
            self.velocity_loss_weight,
            "latent_autoencoder.velocity_loss_weight",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "reconstruction_loss": self.reconstruction_loss,
            "velocity_loss_weight": float(self.velocity_loss_weight),
        }


@dataclass(frozen=True, slots=True)
class LatentDenoiserConfig:
    architecture: str
    latent_dim: int | None
    hidden_dim: int
    timestep_embedding_dim: int
    position_embedding_dim: int
    max_positions: int
    dropout: float
    prediction_type: str

    def __post_init__(self) -> None:
        _require_equal(self.architecture, "temporal_mlp", "denoiser.architecture")
        if self.latent_dim is not None:
            _require_positive_int(self.latent_dim, "denoiser.latent_dim")
        _require_positive_int(self.hidden_dim, "denoiser.hidden_dim")
        _require_positive_int(self.timestep_embedding_dim, "denoiser.timestep_embedding_dim")
        _require_positive_int(self.position_embedding_dim, "denoiser.position_embedding_dim")
        _require_positive_int(self.max_positions, "denoiser.max_positions")
        if (
            not isinstance(self.dropout, int | float)
            or isinstance(self.dropout, bool)
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise LatentDiffusionError("denoiser.dropout must be in [0, 1).")
        _require_equal(self.prediction_type, "epsilon", "denoiser.prediction_type")

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "timestep_embedding_dim": self.timestep_embedding_dim,
            "position_embedding_dim": self.position_embedding_dim,
            "max_positions": self.max_positions,
            "dropout": float(self.dropout),
            "prediction_type": self.prediction_type,
        }


@dataclass(frozen=True, slots=True)
class DiffusionScheduleConfig:
    schedule: str
    timesteps: int
    beta_start: float
    beta_end: float
    loss: str
    sampler: str
    sampling_steps: int

    def __post_init__(self) -> None:
        _require_equal(self.schedule, "linear_beta", "diffusion.schedule")
        if not isinstance(self.timesteps, int) or isinstance(self.timesteps, bool) or self.timesteps <= 1:
            raise LatentDiffusionError(
                "diffusion.timesteps must be an integer greater than 1."
            )
        if not (
            isinstance(self.beta_start, int | float)
            and isinstance(self.beta_end, int | float)
            and not isinstance(self.beta_start, bool)
            and not isinstance(self.beta_end, bool)
            and 0.0 < float(self.beta_start) < float(self.beta_end) < 1.0
        ):
            raise LatentDiffusionError(
                "diffusion beta range must satisfy 0 < beta_start < beta_end < 1."
            )
        _require_equal(self.loss, "masked_mse", "diffusion.loss")
        _require_equal(self.sampler, "ddpm", "diffusion.sampler")
        _require_positive_int(self.sampling_steps, "diffusion.sampling_steps")
        if self.sampling_steps > self.timesteps:
            raise LatentDiffusionError(
                "diffusion.sampling_steps must be less than or equal to diffusion.timesteps."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule": self.schedule,
            "timesteps": self.timesteps,
            "beta_start": float(self.beta_start),
            "beta_end": float(self.beta_end),
            "loss": self.loss,
            "sampler": self.sampler,
            "sampling_steps": self.sampling_steps,
        }


@dataclass(frozen=True, slots=True)
class LatentLengthConfig:
    policy: str
    predictor_type: str
    min_generated_frames: int
    max_generated_frames: int | None
    loss_weight: float

    def __post_init__(self) -> None:
        _require_equal(self.policy, "predicted_length", "length.policy")
        _require_equal(self.predictor_type, "mlp", "length.predictor_type")
        _require_positive_int(self.min_generated_frames, "length.min_generated_frames")
        if self.max_generated_frames is not None and (
            not isinstance(self.max_generated_frames, int)
            or isinstance(self.max_generated_frames, bool)
            or self.max_generated_frames < self.min_generated_frames
        ):
            raise LatentDiffusionError(
                "length.max_generated_frames must be null or at least min_generated_frames."
            )
        _require_non_negative_number(self.loss_weight, "length.loss_weight")

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "predictor_type": self.predictor_type,
            "min_generated_frames": self.min_generated_frames,
            "max_generated_frames": self.max_generated_frames,
            "loss_weight": float(self.loss_weight),
        }


@dataclass(frozen=True, slots=True)
class LatentGenerationConfig:
    generation_mode: str
    candidate_count: int
    seed_policy: str
    confidence_policy: str

    def __post_init__(self) -> None:
        _require_equal(self.generation_mode, "stochastic", "generation.generation_mode")
        _require_positive_int(self.candidate_count, "generation.candidate_count")
        if self.seed_policy not in {"fixed_per_run", "fixed_per_sample"}:
            raise LatentDiffusionError(
                "generation.seed_policy must be 'fixed_per_run' or 'fixed_per_sample'."
            )
        _require_equal(self.confidence_policy, "synthetic_validity", "generation.confidence_policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "generation_mode": self.generation_mode,
            "candidate_count": self.candidate_count,
            "seed_policy": self.seed_policy,
            "confidence_policy": self.confidence_policy,
        }


@dataclass(frozen=True, slots=True)
class LatentTrainingConfig:
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int | None
    num_workers: int
    device: str

    def __post_init__(self) -> None:
        _require_positive_int(self.max_epochs, "training.max_epochs")
        _require_positive_int(self.batch_size, "training.batch_size")
        _require_positive_number(self.learning_rate, "training.learning_rate")
        _require_non_negative_number(self.weight_decay, "training.weight_decay")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise LatentDiffusionError("training.seed must be an integer or null.")
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool) or self.num_workers < 0:
            raise LatentDiffusionError("training.num_workers must be a non-negative integer.")
        _require_text(self.device, "training.device")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "seed": self.seed,
            "num_workers": self.num_workers,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class LatentCheckpointConfig:
    selection_metric: str
    lower_is_better: bool

    def __post_init__(self) -> None:
        _require_equal(
            self.selection_metric,
            "validation_denoising_loss",
            "checkpoints.selection_metric",
        )
        if not isinstance(self.lower_is_better, bool):
            raise LatentDiffusionError("checkpoints.lower_is_better must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_metric": self.selection_metric,
            "lower_is_better": self.lower_is_better,
        }


@dataclass(frozen=True, slots=True)
class LatentDiffusionReportConfig:
    write_latent_target_report: bool
    write_seed_policy_report: bool
    write_compute_failure_cost_report: bool
    write_generation_report: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if not isinstance(getattr(self, field_name), bool):
                raise LatentDiffusionError(f"reports.{field_name} must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "write_latent_target_report": self.write_latent_target_report,
            "write_seed_policy_report": self.write_seed_policy_report,
            "write_compute_failure_cost_report": self.write_compute_failure_cost_report,
            "write_generation_report": self.write_generation_report,
        }


@dataclass(frozen=True, slots=True)
class LatentDiffusionConfig:
    identity: LatentDiffusionIdentityConfig
    data: LatentDiffusionDataConfig
    latent_target: LatentTargetConfig
    latent_autoencoder: LatentAutoencoderConfig | None
    text_encoder: TextEncoderConfig
    denoiser: LatentDenoiserConfig
    diffusion: DiffusionScheduleConfig
    length: LatentLengthConfig
    generation: LatentGenerationConfig
    training: LatentTrainingConfig
    checkpoints: LatentCheckpointConfig
    reports: LatentDiffusionReportConfig

    def __post_init__(self) -> None:
        if not isinstance(self.text_encoder, TextEncoderConfig):
            raise LatentDiffusionError("text_encoder must be a TextEncoderConfig.")
        if self.latent_target.target_type == LATENT_TARGET_TYPE_STANDARDIZED_BFH_FRAME:
            if self.latent_autoencoder is not None:
                raise LatentDiffusionError(
                    "standardized_bfh_frame target requires latent_autoencoder to be null."
                )
        if self.latent_target.target_type == LATENT_TARGET_TYPE_LEARNED_BFH_WINDOW_LATENT:
            if self.latent_autoencoder is None:
                raise LatentDiffusionError(
                    "latent_autoencoder config is required when "
                    "latent_target.target_type='learned_bfh_window_latent'."
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "data": self.data.to_dict(),
            "latent_target": self.latent_target.to_dict(),
            "latent_autoencoder": (
                None
                if self.latent_autoencoder is None
                else self.latent_autoencoder.to_dict()
            ),
            "text_encoder": self.text_encoder.to_dict(),
            "denoiser": self.denoiser.to_dict(),
            "diffusion": self.diffusion.to_dict(),
            "length": self.length.to_dict(),
            "generation": self.generation.to_dict(),
            "training": self.training.to_dict(),
            "checkpoints": self.checkpoints.to_dict(),
            "reports": self.reports.to_dict(),
        }


def load_latent_diffusion_config(path: Path) -> LatentDiffusionConfig:
    """Load and validate the latent_diffusion foundation YAML config."""

    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LatentDiffusionError(f"latent_diffusion config YAML is invalid: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise LatentDiffusionError("latent_diffusion config must be a YAML mapping.")
    raw = copy.deepcopy(dict(cast(Mapping[str, Any], loaded)))
    required = {
        "identity",
        "data",
        "latent_target",
        "text_encoder",
        "denoiser",
        "diffusion",
        "length",
        "generation",
        "training",
        "checkpoints",
        "reports",
    }
    optional = {"latent_autoencoder"}
    if not required <= set(raw) <= required | optional:
        raise LatentDiffusionError(
            f"latent_diffusion config keys mismatch: expected={sorted(required | optional)}, "
            f"observed={sorted(raw)}."
        )
    _reject_forbidden_fields(raw, path=())
    data = _mapping(raw["data"], "data")
    if "manifest_family" in data:
        raise LatentDiffusionError(
            "data.manifest_family is workflow-owned and must not appear in latent_diffusion YAML."
        )
    return latent_diffusion_config_from_mapping(raw)


def latent_diffusion_config_from_mapping(
    raw_config: Mapping[str, object],
) -> LatentDiffusionConfig:
    """Build a validated latent_diffusion config from a strict mapping."""

    raw = copy.deepcopy(dict(cast(Mapping[str, Any], raw_config)))
    required = {
        "identity",
        "data",
        "latent_target",
        "text_encoder",
        "denoiser",
        "diffusion",
        "length",
        "generation",
        "training",
        "checkpoints",
        "reports",
    }
    optional = {"latent_autoencoder"}
    if not required <= set(raw) <= required | optional:
        raise LatentDiffusionError(
            f"latent_diffusion config keys mismatch: expected={sorted(required | optional)}, "
            f"observed={sorted(raw)}."
        )
    _reject_forbidden_fields(raw, path=())
    data = _mapping(raw["data"], "data")
    if "manifest_family" in data:
        raise LatentDiffusionError(
            "data.manifest_family is workflow-owned and must not appear in latent_diffusion YAML."
        )
    latent_autoencoder_raw = raw.get("latent_autoencoder")
    if latent_autoencoder_raw is not None:
        latent_autoencoder = LatentAutoencoderConfig(
            **_mapping(latent_autoencoder_raw, "latent_autoencoder")
        )
    else:
        latent_autoencoder = None
    return LatentDiffusionConfig(
        identity=LatentDiffusionIdentityConfig(**_mapping(raw["identity"], "identity")),
        data=LatentDiffusionDataConfig(**data),
        latent_target=LatentTargetConfig(**_mapping(raw["latent_target"], "latent_target")),
        latent_autoencoder=latent_autoencoder,
        text_encoder=_text_encoder_config(_mapping(raw["text_encoder"], "text_encoder")),
        denoiser=LatentDenoiserConfig(**_mapping(raw["denoiser"], "denoiser")),
        diffusion=DiffusionScheduleConfig(**_mapping(raw["diffusion"], "diffusion")),
        length=LatentLengthConfig(**_mapping(raw["length"], "length")),
        generation=LatentGenerationConfig(**_mapping(raw["generation"], "generation")),
        training=LatentTrainingConfig(**_mapping(raw["training"], "training")),
        checkpoints=LatentCheckpointConfig(**_mapping(raw["checkpoints"], "checkpoints")),
        reports=LatentDiffusionReportConfig(**_mapping(raw["reports"], "reports")),
    )


def _text_encoder_config(raw: Mapping[str, object]) -> TextEncoderConfig:
    yaml_keys = {"backend", "embedding_dim", "pooling", "trainable", "max_length"}
    effective_keys = yaml_keys | {"encoder_key", "model_name_or_path"}
    optional_keys = {"revision", "local_files_only", "output_dim"}
    if set(raw) != yaml_keys and not (
        effective_keys <= set(raw) <= effective_keys | optional_keys
    ):
        raise LatentDiffusionError("text_encoder config keys do not match foundation schema.")
    return TextEncoderConfig(
        encoder_key=str(raw.get("encoder_key", "latent_diffusion_text_encoder")),
        model_name_or_path=(
            None
            if raw.get("model_name_or_path") is None
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


def _optional_int(mapping: Mapping[str, object], name: str) -> int | None:
    value = mapping.get(name)
    if value is None:
        return None
    return _int(value, name)


def _reject_forbidden_fields(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise LatentDiffusionError("latent_diffusion config keys must be strings.")
            if key in _FORBIDDEN_SECTION_KEYS:
                dotted = ".".join((*path, key))
                raise LatentDiffusionError(
                    f"{dotted} is out of scope for this latent_diffusion foundation stage."
                )
            _reject_forbidden_fields(child, path=(*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, path=(*path, str(index)))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LatentDiffusionError(f"{name} must be a mapping with string keys.")
    return cast(Mapping[str, Any], value)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LatentDiffusionError(f"{name} must be non-empty.")


def _text(value: object, name: str) -> str:
    _require_text(value, name)
    return str(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise LatentDiffusionError(f"{name} must be a boolean.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LatentDiffusionError(f"{name} must be an integer.")
    return value


def _require_equal(value: object, expected: str, name: str) -> None:
    if value != expected:
        raise LatentDiffusionError(f"{name} must be {expected!r}.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LatentDiffusionError(f"{name} must be a positive integer.")


def _require_positive_number(value: object, name: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or float(value) <= 0.0
    ):
        raise LatentDiffusionError(f"{name} must be a positive number.")


def _require_non_negative_number(value: object, name: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or float(value) < 0.0
    ):
        raise LatentDiffusionError(f"{name} must be a non-negative number.")


__all__ = [
    "DiffusionScheduleConfig",
    "LatentAutoencoderConfig",
    "LatentCheckpointConfig",
    "LatentDenoiserConfig",
    "LatentDiffusionConfig",
    "LatentDiffusionDataConfig",
    "LatentDiffusionIdentityConfig",
    "LatentDiffusionReportConfig",
    "LatentGenerationConfig",
    "LatentLengthConfig",
    "LatentTargetConfig",
    "LatentTrainingConfig",
    "latent_diffusion_config_from_mapping",
    "load_latent_diffusion_config",
]
