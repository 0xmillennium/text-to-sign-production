"""Configuration contracts for the learned pose-token foundation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.text_encoder import TextEncoderConfig
from text_to_sign_production.modeling.candidates import (
    ModelRunMode,
    ModelRunModePolicy,
    ModelRunRequest,
    resolve_model_run_mode_policy,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    TemporalWindowSpec,
    parse_modeling_manifest_family,
)
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.modeling.candidates.learned_pose_token.spec import (
    LEARNED_POSE_TOKEN_MODEL_KEY,
    LEARNED_POSE_TOKEN_PHASE_NUMBER,
)
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    provider_active_overrides,
)
from text_to_sign_production.workflows.model.contracts.compute_profile_application import (
    compute_profile_dataloader_section,
    provider_candidate_overrides_for_profile,
    provider_compute_profile_application,
)


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenIdentityConfig:
    model_key: str
    research_role: str
    phase_number: int

    def __post_init__(self) -> None:
        if self.model_key != LEARNED_POSE_TOKEN_MODEL_KEY:
            raise LearnedPoseTokenError("identity.model_key must be 'learned_pose_token'.")
        if self.phase_number != LEARNED_POSE_TOKEN_PHASE_NUMBER:
            raise LearnedPoseTokenError("identity.phase_number must be 6.")
        _require_text(self.research_role, "identity.research_role")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "research_role": self.research_role,
            "phase_number": self.phase_number,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenDataConfig:
    max_frame_count: int | None
    min_frame_count: int
    manifest_family: ModelingManifestFamily | None = None
    train_split: SampleSplit = SampleSplit.TRAIN
    validation_split: SampleSplit = SampleSplit.VAL
    prediction_splits: tuple[SampleSplit, ...] = (SampleSplit.VAL,)
    limit_train_samples: int | None = None
    limit_validation_samples: int | None = None
    limit_prediction_samples: int | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.min_frame_count, "data.min_frame_count")
        if self.max_frame_count is not None:
            _require_positive_int(self.max_frame_count, "data.max_frame_count")
            if self.max_frame_count < self.min_frame_count:
                raise LearnedPoseTokenError(
                    "data.max_frame_count must be greater than or equal to min_frame_count."
                )
        if self.manifest_family is not None:
            object.__setattr__(
                self,
                "manifest_family",
                self.manifest_family
                if isinstance(self.manifest_family, ModelingManifestFamily)
                else parse_modeling_manifest_family(self.manifest_family),
            )
        object.__setattr__(self, "train_split", SampleSplit(self.train_split))
        object.__setattr__(self, "validation_split", SampleSplit(self.validation_split))
        object.__setattr__(
            self,
            "prediction_splits",
            tuple(SampleSplit(split) for split in self.prediction_splits),
        )
        if not self.prediction_splits or len(set(self.prediction_splits)) != len(self.prediction_splits):
            raise LearnedPoseTokenError("data.prediction_splits must be non-empty and unique.")
        for field_name in (
            "limit_train_samples",
            "limit_validation_samples",
            "limit_prediction_samples",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_int(value, f"data.{field_name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_frame_count": self.max_frame_count,
            "min_frame_count": self.min_frame_count,
            "manifest_family": (
                None if self.manifest_family is None else self.manifest_family.family_id
            ),
            "train_split": self.train_split.value,
            "validation_split": self.validation_split.value,
            "prediction_splits": [split.value for split in self.prediction_splits],
            "limit_train_samples": self.limit_train_samples,
            "limit_validation_samples": self.limit_validation_samples,
            "limit_prediction_samples": self.limit_prediction_samples,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenRepresentationConfig:
    coordinate_mode: str
    confidence_policy: str
    standardization: str
    standardization_missing_observation_policy: str = "raise"

    def __post_init__(self) -> None:
        if self.coordinate_mode != "xy":
            raise LearnedPoseTokenError("pose_representation.coordinate_mode must be 'xy'.")
        if self.confidence_policy != "mask_only":
            raise LearnedPoseTokenError(
                "pose_representation.confidence_policy must be 'mask_only'."
            )
        if self.standardization != "train_split":
            raise LearnedPoseTokenError(
                "pose_representation.standardization must be 'train_split'."
            )
        if self.standardization_missing_observation_policy not in {
            "raise",
            "channel_fallback",
            "global_fallback",
            "identity_fallback",
        }:
            raise LearnedPoseTokenError(
                "pose_representation.standardization_missing_observation_policy "
                "must be one of {'raise', 'channel_fallback', 'global_fallback', "
                "'identity_fallback'}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_mode": self.coordinate_mode,
            "confidence_policy": self.confidence_policy,
            "standardization": self.standardization,
            "standardization_missing_observation_policy": (
                self.standardization_missing_observation_policy
            ),
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenizerConfig:
    temporal_granularity: str
    window_size: int
    stride: int
    latent_dim: int
    hidden_dim: int
    commitment_weight: float
    reconstruction_loss: str
    velocity_loss_weight: float

    def __post_init__(self) -> None:
        if self.temporal_granularity not in {"frame", "window"}:
            raise LearnedPoseTokenError(
                "tokenizer.temporal_granularity must be one of {'frame', 'window'}."
            )
        _require_positive_int(self.window_size, "tokenizer.window_size")
        _require_positive_int(self.stride, "tokenizer.stride")
        if self.temporal_granularity == "frame" and (
            self.window_size != 1 or self.stride != 1
        ):
            raise LearnedPoseTokenError(
                "frame tokenizer requires window_size=1 and stride=1."
            )
        if self.temporal_granularity == "window" and self.window_size <= 1:
            raise LearnedPoseTokenError(
                "window tokenizer requires window_size > 1 and stride >= 1."
            )
        if self.temporal_granularity == "window" and self.stride < 1:
            raise LearnedPoseTokenError(
                "window tokenizer requires window_size > 1 and stride >= 1."
            )
        try:
            self.temporal_window_spec()
        except Exception as exc:
            raise LearnedPoseTokenError(str(exc)) from exc
        _require_positive_int(self.latent_dim, "tokenizer.latent_dim")
        _require_positive_int(self.hidden_dim, "tokenizer.hidden_dim")
        _require_non_negative_float(self.commitment_weight, "tokenizer.commitment_weight")
        if self.reconstruction_loss != "masked_l1_l2":
            raise LearnedPoseTokenError(
                "tokenizer.reconstruction_loss must be 'masked_l1_l2'."
            )
        _require_non_negative_float(self.velocity_loss_weight, "tokenizer.velocity_loss_weight")

    def to_dict(self) -> dict[str, object]:
        return {
            "temporal_granularity": self.temporal_granularity,
            "window_size": self.window_size,
            "stride": self.stride,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "commitment_weight": self.commitment_weight,
            "reconstruction_loss": self.reconstruction_loss,
            "velocity_loss_weight": self.velocity_loss_weight,
        }

    def temporal_window_spec(self) -> TemporalWindowSpec:
        if self.temporal_granularity == "frame":
            return TemporalWindowSpec.frame()
        if self.temporal_granularity == "window":
            return TemporalWindowSpec.window(
                window_size=self.window_size,
                stride=self.stride,
            )
        raise LearnedPoseTokenError(
            "tokenizer.temporal_granularity must be one of {'frame', 'window'}."
        )


@dataclass(frozen=True, slots=True)
class LearnedPoseCodebookConfig:
    size: int
    embedding_dim: int
    initialization: str
    dead_code_threshold: int
    collapse_perplexity_threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size <= 1:
            raise LearnedPoseTokenError("codebook.size must be greater than 1.")
        _require_positive_int(self.embedding_dim, "codebook.embedding_dim")
        if self.initialization != "random_normal":
            raise LearnedPoseTokenError("codebook.initialization must be 'random_normal'.")
        _require_non_negative_int(self.dead_code_threshold, "codebook.dead_code_threshold")
        _require_positive_float(
            self.collapse_perplexity_threshold,
            "codebook.collapse_perplexity_threshold",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "size": self.size,
            "embedding_dim": self.embedding_dim,
            "initialization": self.initialization,
            "dead_code_threshold": self.dead_code_threshold,
            "collapse_perplexity_threshold": self.collapse_perplexity_threshold,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenizerTrainingConfig:
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int | None
    num_workers: int
    device: str
    tokenizer_batch_size: int | None = None
    text_to_token_batch_size: int | None = None
    reconstruction_batch_size: int | None = None
    decode_batch_size: int | None = None
    materialization_workers: int = 0
    cache_materialized_sources: bool = False

    def __post_init__(self) -> None:
        _require_positive_int(self.max_epochs, "training.max_epochs")
        _require_positive_int(self.batch_size, "training.batch_size")
        for field_name in (
            "tokenizer_batch_size",
            "text_to_token_batch_size",
            "reconstruction_batch_size",
            "decode_batch_size",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_positive_int(value, f"training.{field_name}")
        _require_non_negative_int(
            self.materialization_workers,
            "training.materialization_workers",
        )
        if not isinstance(self.cache_materialized_sources, bool):
            raise LearnedPoseTokenError(
                "training.cache_materialized_sources must be a boolean."
            )
        _require_positive_float(self.learning_rate, "training.learning_rate")
        _require_non_negative_float(self.weight_decay, "training.weight_decay")
        if self.seed is not None:
            _require_non_negative_int(self.seed, "training.seed")
        _require_non_negative_int(self.num_workers, "training.num_workers")
        _require_text(self.device, "training.device")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "tokenizer_batch_size": self.tokenizer_batch_size,
            "text_to_token_batch_size": self.text_to_token_batch_size,
            "reconstruction_batch_size": self.reconstruction_batch_size,
            "decode_batch_size": self.decode_batch_size,
            "materialization_workers": self.materialization_workers,
            "cache_materialized_sources": self.cache_materialized_sources,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "device": self.device,
        }

    @property
    def effective_tokenizer_batch_size(self) -> int:
        return self.batch_size if self.tokenizer_batch_size is None else self.tokenizer_batch_size

    @property
    def effective_text_to_token_batch_size(self) -> int:
        return self.batch_size if self.text_to_token_batch_size is None else self.text_to_token_batch_size

    @property
    def effective_reconstruction_batch_size(self) -> int:
        return self.batch_size if self.reconstruction_batch_size is None else self.reconstruction_batch_size

    @property
    def effective_decode_batch_size(self) -> int:
        return self.batch_size if self.decode_batch_size is None else self.decode_batch_size


@dataclass(frozen=True, slots=True)
class LearnedPoseTextToTokenConfig:
    predictor_type: str
    hidden_dim: int
    position_dim: int
    dropout: float
    length_predictor: str
    min_generated_tokens: int
    max_generated_tokens: int | None
    token_loss_weight: float
    length_loss_weight: float

    def __post_init__(self) -> None:
        if self.predictor_type != "temporal_mlp":
            raise LearnedPoseTokenError("text_to_token.predictor_type must be 'temporal_mlp'.")
        _require_positive_int(self.hidden_dim, "text_to_token.hidden_dim")
        _require_positive_int(self.position_dim, "text_to_token.position_dim")
        _require_probability(self.dropout, "text_to_token.dropout")
        if self.length_predictor != "mlp":
            raise LearnedPoseTokenError("text_to_token.length_predictor must be 'mlp'.")
        _require_positive_int(self.min_generated_tokens, "text_to_token.min_generated_tokens")
        if self.max_generated_tokens is not None:
            _require_positive_int(self.max_generated_tokens, "text_to_token.max_generated_tokens")
            if self.max_generated_tokens < self.min_generated_tokens:
                raise LearnedPoseTokenError(
                    "text_to_token.max_generated_tokens must be at least min_generated_tokens."
                )
        _require_positive_float(self.token_loss_weight, "text_to_token.token_loss_weight")
        _require_non_negative_float(self.length_loss_weight, "text_to_token.length_loss_weight")

    def to_dict(self) -> dict[str, object]:
        return {
            "predictor_type": self.predictor_type,
            "hidden_dim": self.hidden_dim,
            "position_dim": self.position_dim,
            "dropout": self.dropout,
            "length_predictor": self.length_predictor,
            "min_generated_tokens": self.min_generated_tokens,
            "max_generated_tokens": self.max_generated_tokens,
            "token_loss_weight": self.token_loss_weight,
            "length_loss_weight": self.length_loss_weight,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseGenerationConfig:
    token_selection: str
    generation_mode: str
    max_generation_tokens: int | None
    confidence_policy: str
    length_policy: str

    def __post_init__(self) -> None:
        if self.token_selection != "argmax":
            raise LearnedPoseTokenError("generation.token_selection must be 'argmax'.")
        if self.generation_mode != "deterministic":
            raise LearnedPoseTokenError("generation.generation_mode must be 'deterministic'.")
        if self.max_generation_tokens is not None:
            _require_positive_int(self.max_generation_tokens, "generation.max_generation_tokens")
        if self.confidence_policy != "synthetic_validity":
            raise LearnedPoseTokenError(
                "generation.confidence_policy must be 'synthetic_validity'."
            )
        if self.length_policy != "predicted_length":
            raise LearnedPoseTokenError("generation.length_policy must be 'predicted_length'.")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_selection": self.token_selection,
            "generation_mode": self.generation_mode,
            "max_generation_tokens": self.max_generation_tokens,
            "confidence_policy": self.confidence_policy,
            "length_policy": self.length_policy,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseCheckpointConfig:
    selection_metric: str
    lower_is_better: bool

    def __post_init__(self) -> None:
        if self.selection_metric != "validation_token_loss":
            raise LearnedPoseTokenError(
                "checkpoints.selection_metric must be 'validation_token_loss'."
            )
        if self.lower_is_better is not True:
            raise LearnedPoseTokenError("checkpoints.lower_is_better must be true.")

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_metric": self.selection_metric,
            "lower_is_better": self.lower_is_better,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenExportConfig:
    reconstruction_split: SampleSplit
    max_reconstruction_samples: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_split", SampleSplit(self.reconstruction_split))
        if self.reconstruction_split is not SampleSplit.VAL:
            raise LearnedPoseTokenError("export.reconstruction_split must be 'val'.")
        if self.max_reconstruction_samples is not None:
            _require_positive_int(
                self.max_reconstruction_samples,
                "export.max_reconstruction_samples",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "reconstruction_split": self.reconstruction_split.value,
            "max_reconstruction_samples": self.max_reconstruction_samples,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenReportConfig:
    write_tokenizer_report: bool
    write_reconstruction_report: bool
    write_codebook_report: bool

    def __post_init__(self) -> None:
        for field_name in (
            "write_tokenizer_report",
            "write_reconstruction_report",
            "write_codebook_report",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise LearnedPoseTokenError(f"reports.{field_name} must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "write_tokenizer_report": self.write_tokenizer_report,
            "write_reconstruction_report": self.write_reconstruction_report,
            "write_codebook_report": self.write_codebook_report,
        }


@dataclass(frozen=True, slots=True)
class LearnedPoseTokenConfig:
    source_path: Path | None
    raw_config: Mapping[str, object]
    identity: LearnedPoseTokenIdentityConfig
    data: LearnedPoseTokenDataConfig
    representation: LearnedPoseTokenRepresentationConfig
    tokenizer: LearnedPoseTokenizerConfig
    codebook: LearnedPoseCodebookConfig
    training: LearnedPoseTokenizerTrainingConfig
    text_encoder: TextEncoderConfig
    text_to_token: LearnedPoseTextToTokenConfig
    generation: LearnedPoseGenerationConfig
    checkpoints: LearnedPoseCheckpointConfig
    export: LearnedPoseTokenExportConfig
    reports: LearnedPoseTokenReportConfig
    run_mode_policy: ModelRunModePolicy
    run_mode_overrides: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    compute_profile_active_overrides: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    compute_profile_dataloader_overrides: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    compute_profile_application: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.source_path is not None and not isinstance(self.source_path, Path):
            raise LearnedPoseTokenError("source_path must be a Path or None.")
        if not isinstance(self.raw_config, Mapping):
            raise LearnedPoseTokenError("raw_config must be a mapping.")
        object.__setattr__(self, "raw_config", MappingProxyType(dict(self.raw_config)))
        if self.codebook.embedding_dim != self.tokenizer.latent_dim:
            raise LearnedPoseTokenError(
                "codebook.embedding_dim must equal tokenizer.latent_dim in this foundation."
            )
        if self.generation.max_generation_tokens is not None:
            max_text_tokens = self.text_to_token.max_generated_tokens
            if max_text_tokens is not None and self.generation.max_generation_tokens > max_text_tokens:
                raise LearnedPoseTokenError(
                    "generation.max_generation_tokens must not exceed text_to_token.max_generated_tokens."
                )
        if not isinstance(self.run_mode_policy, ModelRunModePolicy):
            raise LearnedPoseTokenError("run_mode_policy must be a ModelRunModePolicy.")
        if not isinstance(self.run_mode_overrides, Mapping):
            raise LearnedPoseTokenError("run_mode_overrides must be a mapping.")
        object.__setattr__(
            self,
            "run_mode_overrides",
            MappingProxyType(dict(self.run_mode_overrides)),
        )
        for field_name in (
            "compute_profile_active_overrides",
            "compute_profile_dataloader_overrides",
            "compute_profile_application",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping) or any(
                not isinstance(key, str) for key in value
            ):
                raise LearnedPoseTokenError(f"{field_name} must be a mapping.")
            object.__setattr__(self, field_name, MappingProxyType(dict(value)))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": None if self.source_path is None else str(self.source_path),
            "identity": self.identity.to_dict(),
            "data": self.data.to_dict(),
            "pose_representation": self.representation.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "codebook": self.codebook.to_dict(),
            "training": self.training.to_dict(),
            "text_encoder": self.text_encoder.to_dict(),
            "text_to_token": self.text_to_token.to_dict(),
            "generation": self.generation.to_dict(),
            "checkpoints": self.checkpoints.to_dict(),
            "export": self.export.to_dict(),
            "reports": self.reports.to_dict(),
            "run_mode_policy": self.run_mode_policy.to_dict(),
            "run_mode_overrides": dict(self.run_mode_overrides),
            "compute_profile_active_overrides": dict(self.compute_profile_active_overrides),
            "compute_profile_dataloader_overrides": dict(
                self.compute_profile_dataloader_overrides
            ),
            "compute_profile_application": dict(self.compute_profile_application),
            "runtime_truth_contract": _learned_pose_token_runtime_truth_contract(),
        }


def _learned_pose_token_runtime_truth_contract() -> dict[str, object]:
    return {
        "schema_version": "model.runtime_truth_contract.v1",
        "provider_key": LEARNED_POSE_TOKEN_MODEL_KEY,
        "required_runtime_evidence": {
            "num_workers": [
                "surface_reader_num_workers_used",
                "surface_reader_worker_mode",
            ],
        },
    }


def load_learned_pose_token_config(
    path: Path,
    *,
    request: ModelRunRequest | None = None,
) -> LearnedPoseTokenConfig:
    """Load and validate a learned pose-token foundation YAML config."""

    if request is not None and request.model_key is not ModelKey.LEARNED_POSE_TOKEN:
        raise LearnedPoseTokenError("learned_pose_token config requires learned_pose_token request.")
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LearnedPoseTokenError(f"learned_pose_token config YAML is invalid: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise LearnedPoseTokenError("learned_pose_token config must be a YAML mapping.")
    raw = copy.deepcopy(dict(cast(Mapping[str, Any], loaded)))
    expected = {
        "identity",
        "data",
        "text_encoder",
        "pose_representation",
        "tokenizer",
        "codebook",
        "training",
        "text_to_token",
        "generation",
        "checkpoints",
        "export",
        "reports",
    }
    if set(raw) != expected:
        raise LearnedPoseTokenError(
            f"learned_pose_token config keys mismatch: expected={sorted(expected)}, "
            f"observed={sorted(raw)}."
        )
    data = _mapping(raw["data"], "data")
    if "manifest_family" in data:
        raise LearnedPoseTokenError(
            "data.manifest_family is workflow-owned and must not appear in learned_pose_token YAML."
        )
    tokenizer = LearnedPoseTokenizerConfig(**_mapping(raw["tokenizer"], "tokenizer"))
    codebook = LearnedPoseCodebookConfig(**_mapping(raw["codebook"], "codebook"))
    policy = resolve_model_run_mode_policy(ModelRunMode.SMOKE if request is None else request.run_mode)
    data_config = LearnedPoseTokenDataConfig(
        **data,
        manifest_family=None if request is None else request.manifest_family,
        train_split=SampleSplit.TRAIN if request is None else request.train_split,
        validation_split=SampleSplit.VAL if request is None else request.validation_split,
        prediction_splits=(SampleSplit.VAL,) if request is None else request.prediction_splits,
    )
    training_config = LearnedPoseTokenizerTrainingConfig(
        **_mapping(raw["training"], "training")
    )
    if request is not None and request.seed is not None:
        training_config = replace(training_config, seed=request.seed)
    data_config, training_config, overrides = _apply_run_mode(
        policy,
        data_config,
        training_config,
    )
    if request is not None:
        (
            training_config,
            dataloader_applied,
            dataloader_not_applicable,
            dataloader_unsupported,
        ) = _apply_compute_profile_dataloader_overrides(request, training_config)
        training_config, compute_overrides = _apply_compute_profile_active_overrides(
            request,
            training_config,
        )
        if compute_overrides:
            overrides = MappingProxyType(
                {
                    **dict(overrides),
                    "compute_profile_active": dict(compute_overrides),
                }
            )
        candidate_overrides = provider_candidate_overrides_for_profile(
            request.compute_profile,
            provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        )
        candidate_allowed = {
            "tokenizer_batch_size",
            "text_to_token_batch_size",
            "reconstruction_batch_size",
            "decode_batch_size",
        }
        candidate_unknown = set(candidate_overrides) - candidate_allowed
        compute_profile_application = provider_compute_profile_application(
            compute_profile=request.compute_profile,
            provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
            run_mode=request.run_mode.value,
            active_applied=compute_overrides,
            dataloader_applied=dataloader_applied,
            dataloader_not_applicable=dataloader_not_applicable,
            dataloader_unsupported=dataloader_unsupported,
            candidates_applicable={
                key: candidate_overrides[key]
                for key in candidate_overrides
                if key in candidate_allowed
            },
            candidates_unsupported={
                key: "learned_pose_token supports calibration candidates only for provider batch fields."
                for key in candidate_unknown
            },
            telemetry_required_fields=(
                "tokenizer_batch_size",
                "text_to_token_batch_size",
                "reconstruction_batch_size",
                "decode_batch_size",
                "cache_materialized_sources",
                "num_workers",
            ),
            calibration_candidate_keys=(
                "tokenizer_batch_size",
                "text_to_token_batch_size",
                "reconstruction_batch_size",
                "decode_batch_size",
            ),
            calibration_override_targets={
                "tokenizer_batch_size": "training.tokenizer_batch_size",
                "text_to_token_batch_size": "training.text_to_token_batch_size",
                "reconstruction_batch_size": "training.reconstruction_batch_size",
                "decode_batch_size": "training.decode_batch_size",
            },
        ).to_dict()
    else:
        compute_overrides = MappingProxyType({})
        dataloader_applied = MappingProxyType({})
        compute_profile_application = MappingProxyType({})
    return LearnedPoseTokenConfig(
        source_path=config_path.expanduser().resolve(),
        raw_config=raw,
        identity=LearnedPoseTokenIdentityConfig(**_mapping(raw["identity"], "identity")),
        data=data_config,
        representation=LearnedPoseTokenRepresentationConfig(
            **_mapping(raw["pose_representation"], "pose_representation")
        ),
        tokenizer=tokenizer,
        codebook=codebook,
        training=training_config,
        text_encoder=_text_encoder_config(_mapping(raw["text_encoder"], "text_encoder")),
        text_to_token=LearnedPoseTextToTokenConfig(
            **_mapping(raw["text_to_token"], "text_to_token")
        ),
        generation=LearnedPoseGenerationConfig(
            **_mapping(raw["generation"], "generation")
        ),
        checkpoints=LearnedPoseCheckpointConfig(
            **_mapping(raw["checkpoints"], "checkpoints")
        ),
        export=LearnedPoseTokenExportConfig(**_mapping(raw["export"], "export")),
        reports=LearnedPoseTokenReportConfig(**_mapping(raw["reports"], "reports")),
        run_mode_policy=policy,
        run_mode_overrides=overrides,
        compute_profile_active_overrides=compute_overrides,
        compute_profile_dataloader_overrides=dataloader_applied,
        compute_profile_application=compute_profile_application,
    )


def learned_pose_token_config_from_effective_dict(
    effective: Mapping[str, object],
    *,
    source_path: Path | None,
) -> LearnedPoseTokenConfig:
    """Rehydrate a validated learned_pose_token config from an effective snapshot."""

    data = _mapping(effective["data"], "data")
    policy = _run_mode_policy_from_effective(effective)
    return LearnedPoseTokenConfig(
        source_path=source_path,
        raw_config=effective,
        identity=LearnedPoseTokenIdentityConfig(
            **_mapping(effective["identity"], "identity")
        ),
        data=LearnedPoseTokenDataConfig(
            max_frame_count=_optional_int(data, "max_frame_count"),
            min_frame_count=_int(data["min_frame_count"], "data.min_frame_count"),
            manifest_family=(
                None
                if data.get("manifest_family") is None
                else parse_modeling_manifest_family(_text(data["manifest_family"], "data.manifest_family"))
            ),
            train_split=SampleSplit(_text(data["train_split"], "data.train_split")),
            validation_split=SampleSplit(_text(data["validation_split"], "data.validation_split")),
            prediction_splits=tuple(
                SampleSplit(_text(value, "data.prediction_splits"))
                for value in _sequence(data["prediction_splits"], "data.prediction_splits")
            ),
            limit_train_samples=_optional_int(data, "limit_train_samples"),
            limit_validation_samples=_optional_int(data, "limit_validation_samples"),
            limit_prediction_samples=_optional_int(data, "limit_prediction_samples"),
        ),
        representation=LearnedPoseTokenRepresentationConfig(
            **_mapping(effective["pose_representation"], "pose_representation")
        ),
        tokenizer=LearnedPoseTokenizerConfig(**_mapping(effective["tokenizer"], "tokenizer")),
        codebook=LearnedPoseCodebookConfig(**_mapping(effective["codebook"], "codebook")),
        training=LearnedPoseTokenizerTrainingConfig(
            **_mapping(effective["training"], "training")
        ),
        text_encoder=_text_encoder_config(_mapping(effective["text_encoder"], "text_encoder")),
        text_to_token=LearnedPoseTextToTokenConfig(
            **_mapping(effective["text_to_token"], "text_to_token")
        ),
        generation=LearnedPoseGenerationConfig(
            **_mapping(effective["generation"], "generation")
        ),
        checkpoints=LearnedPoseCheckpointConfig(
            **_mapping(effective["checkpoints"], "checkpoints")
        ),
        export=LearnedPoseTokenExportConfig(**_mapping(effective["export"], "export")),
        reports=LearnedPoseTokenReportConfig(**_mapping(effective["reports"], "reports")),
        run_mode_policy=policy,
        run_mode_overrides=(
            _mapping(effective["run_mode_overrides"], "run_mode_overrides")
            if isinstance(effective.get("run_mode_overrides"), Mapping)
            else {}
        ),
        compute_profile_active_overrides=(
            _mapping(
                effective["compute_profile_active_overrides"],
                "compute_profile_active_overrides",
            )
            if isinstance(effective.get("compute_profile_active_overrides"), Mapping)
            else {}
        ),
        compute_profile_dataloader_overrides=(
            _mapping(
                effective["compute_profile_dataloader_overrides"],
                "compute_profile_dataloader_overrides",
            )
            if isinstance(effective.get("compute_profile_dataloader_overrides"), Mapping)
            else {}
        ),
        compute_profile_application=(
            _mapping(effective["compute_profile_application"], "compute_profile_application")
            if isinstance(effective.get("compute_profile_application"), Mapping)
            else {}
        ),
    )


def _text_encoder_config(raw: Mapping[str, object]) -> TextEncoderConfig:
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
    if canonical_keys <= set(raw) <= canonical_keys | optional_keys:
        return TextEncoderConfig(
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
    if set(raw) != {"backend", "embedding_dim", "pooling", "trainable", "max_length"}:
        raise LearnedPoseTokenError("text_encoder config keys do not match foundation schema.")
    return TextEncoderConfig(
        encoder_key="learned_pose_token_text_encoder",
        model_name_or_path=None,
        pooling=_text(raw["pooling"], "text_encoder.pooling"),
        trainable=_bool(raw["trainable"], "text_encoder.trainable"),
        max_length=_int(raw["max_length"], "text_encoder.max_length"),
        embedding_dim=_int(raw["embedding_dim"], "text_encoder.embedding_dim"),
        backend=_text(raw["backend"], "text_encoder.backend"),
    )


def _apply_run_mode(
    policy: ModelRunModePolicy,
    data: LearnedPoseTokenDataConfig,
    training: LearnedPoseTokenizerTrainingConfig,
) -> tuple[LearnedPoseTokenDataConfig, LearnedPoseTokenizerTrainingConfig, Mapping[str, object]]:
    if policy.mode is ModelRunMode.FULL:
        return data, training, MappingProxyType({})
    data_overrides = {
        "limit_train_samples": policy.limit_train_samples,
        "limit_validation_samples": policy.limit_validation_samples,
        "limit_prediction_samples": policy.limit_prediction_samples,
    }
    training_overrides = {
        "max_epochs": (
            training.max_epochs
            if policy.max_epochs is None
            else min(training.max_epochs, policy.max_epochs)
        ),
        "batch_size": policy.batch_size or training.batch_size,
        "num_workers": policy.num_workers if policy.num_workers is not None else training.num_workers,
    }
    return (
        replace(data, **data_overrides),
        replace(training, **training_overrides),
        MappingProxyType({"data": data_overrides, "training": training_overrides}),
    )


def _apply_compute_profile_active_overrides(
    request: ModelRunRequest,
    training: LearnedPoseTokenizerTrainingConfig,
) -> tuple[LearnedPoseTokenizerTrainingConfig, Mapping[str, object]]:
    active = provider_active_overrides(
        request.compute_profile,
        provider_key=LEARNED_POSE_TOKEN_MODEL_KEY,
        run_mode=request.run_mode.value,
    )
    if not active:
        return training, MappingProxyType({})
    allowed = {
        "tokenizer_batch_size",
        "text_to_token_batch_size",
        "reconstruction_batch_size",
        "decode_batch_size",
        "cache_materialized_sources",
    }
    unknown = set(active) - allowed
    if unknown:
        raise LearnedPoseTokenError(
            "learned_pose_token compute profile active overrides contain "
            f"unsupported keys: {sorted(unknown)}."
        )
    return replace(training, **dict(active)), active


def _apply_compute_profile_dataloader_overrides(
    request: ModelRunRequest,
    training: LearnedPoseTokenizerTrainingConfig,
) -> tuple[
    LearnedPoseTokenizerTrainingConfig,
    Mapping[str, object],
    Mapping[str, str],
    Mapping[str, str],
]:
    requested = compute_profile_dataloader_section(request.compute_profile)
    if not requested:
        return training, MappingProxyType({}), MappingProxyType({}), MappingProxyType({})

    applied: dict[str, object] = {}
    not_applicable: dict[str, str] = {}
    unsupported: dict[str, str] = {}
    updated = training
    allowed = {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "materialization_workers",
    }
    for key, value in requested.items():
        if key not in allowed:
            unsupported[key] = (
                "learned_pose_token does not recognize this dataloader profile key."
            )
            continue
        if key == "num_workers":
            if value is None:
                not_applicable[key] = "compute profile does not request a num_workers override."
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LearnedPoseTokenError(
                    "learned_pose_token dataloader.num_workers must be non-negative."
                )
            updated = replace(updated, num_workers=value)
            applied[key] = value
            continue
        not_applicable[key] = (
            "learned_pose_token currently wires only num_workers into runtime dataloaders."
        )
    return (
        updated,
        MappingProxyType(applied),
        MappingProxyType(not_applicable),
        MappingProxyType(unsupported),
    )


def _run_mode_policy_from_effective(effective: Mapping[str, object]) -> ModelRunModePolicy:
    policy_section = _mapping(effective["run_mode_policy"], "run_mode_policy")
    policy = resolve_model_run_mode_policy(_text(policy_section["mode"], "run_mode_policy.mode"))
    if dict(policy_section) != policy.to_dict():
        raise LearnedPoseTokenError(
            "learned_pose_token effective run_mode_policy does not match shared policy."
        )
    return policy


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise LearnedPoseTokenError(f"{name} must be a mapping with string keys.")
    return cast(Mapping[str, Any], value)


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LearnedPoseTokenError(f"{name} must be non-empty.")


def _text(value: object, name: str) -> str:
    _require_text(value, name)
    return str(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise LearnedPoseTokenError(f"{name} must be a boolean.")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LearnedPoseTokenError(f"{name} must be an integer.")
    return value


def _optional_int(mapping: Mapping[str, object], name: str) -> int | None:
    value = mapping.get(name)
    if value is None:
        return None
    return _int(value, name)


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, str):
        raise LearnedPoseTokenError(f"{name} must be a sequence, not a string.")
    try:
        sequence = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise LearnedPoseTokenError(f"{name} must be a sequence.") from exc
    return sequence


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LearnedPoseTokenError(f"{name} must be a positive integer.")


def _require_non_negative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LearnedPoseTokenError(f"{name} must be a non-negative integer.")


def _require_positive_float(value: object, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0.0:
        raise LearnedPoseTokenError(f"{name} must be a positive number.")


def _require_non_negative_float(value: object, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) < 0.0:
        raise LearnedPoseTokenError(f"{name} must be a non-negative number.")


def _require_probability(value: object, name: str) -> None:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or float(value) < 0.0
        or float(value) >= 1.0
    ):
        raise LearnedPoseTokenError(f"{name} must be in [0, 1).")


__all__ = [
    "LearnedPoseCodebookConfig",
    "LearnedPoseTokenConfig",
    "LearnedPoseTokenDataConfig",
    "LearnedPoseTokenExportConfig",
    "LearnedPoseTokenIdentityConfig",
    "LearnedPoseTokenReportConfig",
    "LearnedPoseTokenRepresentationConfig",
    "LearnedPoseTextToTokenConfig",
    "LearnedPoseGenerationConfig",
    "LearnedPoseCheckpointConfig",
    "LearnedPoseTokenizerConfig",
    "LearnedPoseTokenizerTrainingConfig",
    "learned_pose_token_config_from_effective_dict",
    "load_learned_pose_token_config",
]
