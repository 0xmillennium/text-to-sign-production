"""Provider configuration contract for the M0 direct baseline."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
)
from text_to_sign_production.modeling.candidates import (
    ModelCandidateError,
    ModelRunMode,
    ModelRunModePolicy,
    ModelRunRequest,
    resolve_model_run_mode_policy,
)
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
)
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    provider_active_overrides,
)
from text_to_sign_production.workflows.model.contracts.compute_profile_application import (
    compute_profile_dataloader_section,
    provider_candidate_overrides_for_profile,
    provider_compute_profile_application,
)

_EXPECTED_IDENTITY = {
    "model_key": "base_direct",
    "canonical_id": "m0_direct_text_to_pose",
    "research_role": "baseline_or_ablation_floor",
    "channel_policy": GENERATED_POSE_CHANNEL_POLICY,
    "length_policy": "reference_length",
    "confidence_policy": "synthetic_validity",
    "output_contract": GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
    "manifest_contract": GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
}
_CHANNELS = ("body", "left_hand", "right_hand", "face")


@dataclass(frozen=True, slots=True)
class BaseDirectIdentityConfig:
    model_key: str
    canonical_id: str
    display_name: str
    research_role: str
    channel_policy: str
    length_policy: str
    confidence_policy: str
    output_contract: str
    manifest_contract: str

    def __post_init__(self) -> None:
        for field_name, expected in _EXPECTED_IDENTITY.items():
            if getattr(self, field_name) != expected:
                raise ModelCandidateError(
                    f"base_direct identity.{field_name} must be {expected!r}."
                )
        _require_text(self.display_name, "identity.display_name")


@dataclass(frozen=True, slots=True)
class BaseDirectDataConfig:
    manifest_family: ModelingManifestFamily
    train_split: SampleSplit
    validation_split: SampleSplit
    prediction_splits: tuple[SampleSplit, ...]
    limit_train_samples: int | None
    limit_validation_samples: int | None
    limit_prediction_samples: int | None

    def __post_init__(self) -> None:
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
        if not self.prediction_splits or len(set(self.prediction_splits)) != len(
            self.prediction_splits
        ):
            raise ModelCandidateError("base_direct prediction_splits must be non-empty and unique.")
        for field_name in (
            "limit_train_samples",
            "limit_validation_samples",
            "limit_prediction_samples",
        ):
            _require_optional_positive_int(getattr(self, field_name), f"data.{field_name}")


@dataclass(frozen=True, slots=True)
class BaseDirectTextEncoderConfig:
    model_name: str
    revision: str
    max_length: int
    local_files_only: bool
    trainable: bool
    freeze_strategy: str
    encoder_learning_rate: float

    def __post_init__(self) -> None:
        _require_text(self.model_name, "text_encoder.model_name")
        _require_text(self.revision, "text_encoder.revision")
        _require_positive_int(self.max_length, "text_encoder.max_length")
        _require_text(self.freeze_strategy, "text_encoder.freeze_strategy")
        _require_positive_float(
            self.encoder_learning_rate,
            "text_encoder.encoder_learning_rate",
        )


@dataclass(frozen=True, slots=True)
class BaseDirectModelConfig:
    decoder_hidden_dim: int
    decoder_layers: int
    decoder_dropout: float
    frame_position_encoding_dim: int

    def __post_init__(self) -> None:
        _require_positive_int(self.decoder_hidden_dim, "model.decoder_hidden_dim")
        _require_positive_int(self.decoder_layers, "model.decoder_layers")
        _require_probability(self.decoder_dropout, "model.decoder_dropout")
        _require_non_negative_int(
            self.frame_position_encoding_dim,
            "model.frame_position_encoding_dim",
        )


@dataclass(frozen=True, slots=True)
class BaseDirectLossConfig:
    channel_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.channel_weights, Mapping):
            raise ModelCandidateError("loss.channel_weights must be a mapping.")
        if set(self.channel_weights) != set(_CHANNELS):
            raise ModelCandidateError(
                "loss.channel_weights must include body, left_hand, right_hand, and face."
            )
        resolved = {
            channel: _non_negative_float(
                self.channel_weights[channel],
                f"loss.channel_weights.{channel}",
            )
            for channel in _CHANNELS
        }
        if not any(value > 0.0 for value in resolved.values()):
            raise ModelCandidateError("loss.channel_weights must contain a positive weight.")
        object.__setattr__(self, "channel_weights", MappingProxyType(resolved))


@dataclass(frozen=True, slots=True)
class BaseDirectTrainingConfig:
    epochs: int
    min_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    early_stopping_mode: str
    validate_every_epochs: int
    batch_size: int
    shuffle_train: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    non_blocking_transfers: bool
    seed: int | None
    device: str
    gradient_accumulation_steps: int
    max_grad_norm: float
    mixed_precision: str
    length_bucketed_batching: bool

    def __post_init__(self) -> None:
        for field_name in (
            "epochs",
            "min_epochs",
            "early_stopping_patience",
            "validate_every_epochs",
            "batch_size",
            "gradient_accumulation_steps",
        ):
            _require_positive_int(getattr(self, field_name), f"training.{field_name}")
        if self.min_epochs > self.epochs:
            raise ModelCandidateError("training.min_epochs must not exceed training.epochs.")
        _require_non_negative_int(self.num_workers, "training.num_workers")
        _require_optional_positive_int(self.prefetch_factor, "training.prefetch_factor")
        if self.num_workers == 0 and (self.persistent_workers or self.prefetch_factor is not None):
            raise ModelCandidateError(
                "training worker persistence/prefetch requires positive num_workers."
            )
        if self.seed is not None:
            _require_non_negative_int(self.seed, "training.seed")
        _require_text(self.early_stopping_metric, "training.early_stopping_metric")
        if self.early_stopping_mode != "min":
            raise ModelCandidateError("training.early_stopping_mode must be 'min'.")
        _require_text(self.device, "training.device")
        _require_positive_float(self.max_grad_norm, "training.max_grad_norm")
        _require_text(self.mixed_precision, "training.mixed_precision")


@dataclass(frozen=True, slots=True)
class BaseDirectOptimizerConfig:
    name: str
    decoder_learning_rate: float
    weight_decay: float

    def __post_init__(self) -> None:
        if self.name.lower() != "adamw":
            raise ModelCandidateError("optimizer.name must be 'adamw'.")
        _require_positive_float(self.decoder_learning_rate, "optimizer.decoder_learning_rate")
        _non_negative_float(self.weight_decay, "optimizer.weight_decay")


@dataclass(frozen=True, slots=True)
class BaseDirectSchedulerConfig:
    name: str
    warmup_ratio: float

    def __post_init__(self) -> None:
        if self.name != "cosine_with_warmup":
            raise ModelCandidateError("scheduler.name must be 'cosine_with_warmup'.")
        _require_probability(self.warmup_ratio, "scheduler.warmup_ratio")


@dataclass(frozen=True, slots=True)
class BaseDirectTargetStandardizationConfig:
    enabled: bool
    epsilon: float

    def __post_init__(self) -> None:
        _require_positive_float(self.epsilon, "target_standardization.epsilon")


@dataclass(frozen=True, slots=True)
class BaseDirectRunConfig:
    source_path: Path | None
    raw_config: Mapping[str, object]
    identity: BaseDirectIdentityConfig
    data: BaseDirectDataConfig
    text_encoder: BaseDirectTextEncoderConfig
    model: BaseDirectModelConfig
    loss: BaseDirectLossConfig
    training: BaseDirectTrainingConfig
    optimizer: BaseDirectOptimizerConfig
    scheduler: BaseDirectSchedulerConfig
    target_standardization: BaseDirectTargetStandardizationConfig
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
            raise ModelCandidateError("base_direct source_path must be a Path or None.")
        object.__setattr__(self, "raw_config", _mapping_proxy(self.raw_config, "raw_config"))
        if not isinstance(self.run_mode_policy, ModelRunModePolicy):
            raise ModelCandidateError(
                "base_direct run_mode_policy must be a ModelRunModePolicy."
            )
        object.__setattr__(
            self,
            "run_mode_overrides",
            _mapping_proxy(self.run_mode_overrides, "run_mode_overrides"),
        )
        object.__setattr__(
            self,
            "compute_profile_active_overrides",
            _mapping_proxy(
                self.compute_profile_active_overrides,
                "compute_profile_active_overrides",
            ),
        )
        object.__setattr__(
            self,
            "compute_profile_dataloader_overrides",
            _mapping_proxy(
                self.compute_profile_dataloader_overrides,
                "compute_profile_dataloader_overrides",
            ),
        )
        object.__setattr__(
            self,
            "compute_profile_application",
            _mapping_proxy(
                self.compute_profile_application,
                "compute_profile_application",
            ),
        )


def load_base_direct_config(
    path: Path | str | None,
    *,
    request: ModelRunRequest,
) -> BaseDirectRunConfig:
    """Load provider configuration with request-owned manifest-family and run overrides."""

    if request.model_key is not ModelKey.BASE_DIRECT:
        raise ModelCandidateError("base_direct config requires a base_direct request.")
    if request.auxiliary_objectives:
        raise ModelCandidateError("base_direct does not accept auxiliary objectives.")
    source_path: Path | None
    if path is None:
        source_path = None
        raw = _default_config_dict()
    else:
        source_path = Path(path).expanduser().resolve()
        try:
            loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ModelCandidateError(f"could not read base_direct config: {source_path}") from exc
        except yaml.YAMLError as exc:
            raise ModelCandidateError(f"could not parse base_direct config: {source_path}") from exc
        if not isinstance(loaded, Mapping):
            raise ModelCandidateError("base_direct config root must be a mapping.")
        raw = copy.deepcopy(dict(loaded))

    identity_section = _required_mapping(raw, "identity")
    data_section = _required_mapping(raw, "data")
    encoder_section = _required_mapping(raw, "text_encoder")
    model_section = _required_mapping(raw, "model")
    loss_section = _required_mapping(raw, "loss")
    training_section = _required_mapping(raw, "training")
    optimizer_section = _required_mapping(raw, "optimizer")
    scheduler_section = _required_mapping(raw, "scheduler")
    standardization_section = _required_mapping(raw, "target_standardization")
    data = BaseDirectDataConfig(
        manifest_family=request.manifest_family,
        train_split=request.train_split,
        validation_split=request.validation_split,
        prediction_splits=request.prediction_splits,
        limit_train_samples=_optional_int(data_section, "limit_train_samples"),
        limit_validation_samples=_optional_int(data_section, "limit_validation_samples"),
        limit_prediction_samples=_optional_int(data_section, "limit_prediction_samples"),
    )
    training = BaseDirectTrainingConfig(
        epochs=_int(training_section, "epochs"),
        min_epochs=_int(training_section, "min_epochs"),
        early_stopping_patience=_int(training_section, "early_stopping_patience"),
        early_stopping_metric=_str(training_section, "early_stopping_metric"),
        early_stopping_mode=_str(training_section, "early_stopping_mode"),
        validate_every_epochs=_int(training_section, "validate_every_epochs"),
        batch_size=_int(training_section, "batch_size"),
        shuffle_train=_bool(training_section, "shuffle_train"),
        num_workers=_int(training_section, "num_workers"),
        pin_memory=_bool(training_section, "pin_memory"),
        persistent_workers=_bool(training_section, "persistent_workers"),
        prefetch_factor=_optional_int(training_section, "prefetch_factor"),
        non_blocking_transfers=_bool(training_section, "non_blocking_transfers"),
        seed=(
            request.seed
            if request.seed is not None
            else _optional_int(training_section, "seed")
        ),
        device=_str(training_section, "device"),
        gradient_accumulation_steps=_int(training_section, "gradient_accumulation_steps"),
        max_grad_norm=_float(training_section, "max_grad_norm"),
        mixed_precision=_str(training_section, "mixed_precision"),
        length_bucketed_batching=_bool(training_section, "length_bucketed_batching"),
    )
    policy = resolve_model_run_mode_policy(request.run_mode)
    data, training, overrides = _apply_run_mode(policy, data, training)
    (
        training,
        dataloader_applied,
        dataloader_not_applicable,
        dataloader_unsupported,
    ) = _apply_compute_profile_dataloader_overrides(request, training)
    training, active_overrides = _apply_compute_profile_active_overrides(
        request,
        training,
    )
    if active_overrides:
        overrides = MappingProxyType(
            {
                **dict(overrides),
                "compute_profile_active": dict(active_overrides),
            }
        )
    candidate_overrides = provider_candidate_overrides_for_profile(
        request.compute_profile,
        provider_key="base_direct",
    )
    candidate_allowed = {"batch_size"}
    candidate_unknown = set(candidate_overrides) - candidate_allowed
    compute_profile_application = provider_compute_profile_application(
        compute_profile=request.compute_profile,
        provider_key="base_direct",
        run_mode=request.run_mode.value,
        active_applied=active_overrides,
        dataloader_applied=dataloader_applied,
        dataloader_not_applicable=dataloader_not_applicable,
        dataloader_unsupported=dataloader_unsupported,
        candidates_applicable={
            key: candidate_overrides[key]
            for key in candidate_overrides
            if key in candidate_allowed
        },
        candidates_unsupported={
            key: "base_direct supports calibration candidates only for batch_size."
            for key in candidate_unknown
        },
        telemetry_required_fields=("batch_size", "num_workers"),
        calibration_candidate_keys=("batch_size",),
        calibration_override_targets={"batch_size": "training.batch_size"},
    ).to_dict()
    return BaseDirectRunConfig(
        source_path=source_path,
        raw_config=raw,
        identity=BaseDirectIdentityConfig(
            model_key=_str(identity_section, "model_key"),
            canonical_id=_str(identity_section, "canonical_id"),
            display_name=_str(identity_section, "display_name"),
            research_role=_str(identity_section, "research_role"),
            channel_policy=_str(identity_section, "channel_policy"),
            length_policy=_str(identity_section, "length_policy"),
            confidence_policy=_str(identity_section, "confidence_policy"),
            output_contract=_str(identity_section, "output_contract"),
            manifest_contract=_str(identity_section, "manifest_contract"),
        ),
        data=data,
        text_encoder=BaseDirectTextEncoderConfig(
            model_name=_str(encoder_section, "model_name"),
            revision=_str(encoder_section, "revision"),
            max_length=_int(encoder_section, "max_length"),
            local_files_only=_bool(encoder_section, "local_files_only"),
            trainable=_bool(encoder_section, "trainable"),
            freeze_strategy=_str(encoder_section, "freeze_strategy"),
            encoder_learning_rate=_float(encoder_section, "encoder_learning_rate"),
        ),
        model=BaseDirectModelConfig(
            decoder_hidden_dim=_int(model_section, "decoder_hidden_dim"),
            decoder_layers=_int(model_section, "decoder_layers"),
            decoder_dropout=_float(model_section, "decoder_dropout"),
            frame_position_encoding_dim=_int(model_section, "frame_position_encoding_dim"),
        ),
        loss=BaseDirectLossConfig(
            channel_weights=_required_mapping(loss_section, "channel_weights"),
        ),
        training=training,
        optimizer=BaseDirectOptimizerConfig(
            name=_str(optimizer_section, "name"),
            decoder_learning_rate=_float(optimizer_section, "decoder_learning_rate"),
            weight_decay=_float(optimizer_section, "weight_decay"),
        ),
        scheduler=BaseDirectSchedulerConfig(
            name=_str(scheduler_section, "name"),
            warmup_ratio=_float(scheduler_section, "warmup_ratio"),
        ),
        target_standardization=BaseDirectTargetStandardizationConfig(
            enabled=_bool(standardization_section, "enabled"),
            epsilon=_float(standardization_section, "epsilon"),
        ),
        run_mode_policy=policy,
        run_mode_overrides=overrides,
        compute_profile_active_overrides=active_overrides,
        compute_profile_dataloader_overrides=dataloader_applied,
        compute_profile_application=compute_profile_application,
    )


def base_direct_config_to_dict(config: BaseDirectRunConfig) -> dict[str, object]:
    """Return JSON-serializable effective base-direct provider configuration."""

    return {
        "source_path": None if config.source_path is None else str(config.source_path),
        "identity": {
            field_name: getattr(config.identity, field_name)
            for field_name in _EXPECTED_IDENTITY | {"display_name": ""}
        },
        "data": {
            "manifest_family": config.data.manifest_family.family_id,
            "train_split": config.data.train_split.value,
            "validation_split": config.data.validation_split.value,
            "prediction_splits": [split.value for split in config.data.prediction_splits],
            "limit_train_samples": config.data.limit_train_samples,
            "limit_validation_samples": config.data.limit_validation_samples,
            "limit_prediction_samples": config.data.limit_prediction_samples,
        },
        "text_encoder": _dataclass_values(
            config.text_encoder,
            (
                "model_name",
                "revision",
                "max_length",
                "local_files_only",
                "trainable",
                "freeze_strategy",
                "encoder_learning_rate",
            ),
        ),
        "model": _dataclass_values(
            config.model,
            (
                "decoder_hidden_dim",
                "decoder_layers",
                "decoder_dropout",
                "frame_position_encoding_dim",
            ),
        ),
        "loss": {"channel_weights": dict(config.loss.channel_weights)},
        "training": _dataclass_values(
            config.training,
            (
                "epochs",
                "min_epochs",
                "early_stopping_patience",
                "early_stopping_metric",
                "early_stopping_mode",
                "validate_every_epochs",
                "batch_size",
                "shuffle_train",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
                "non_blocking_transfers",
                "seed",
                "device",
                "gradient_accumulation_steps",
                "max_grad_norm",
                "mixed_precision",
                "length_bucketed_batching",
            ),
        ),
        "optimizer": _dataclass_values(
            config.optimizer,
            ("name", "decoder_learning_rate", "weight_decay"),
        ),
        "scheduler": _dataclass_values(config.scheduler, ("name", "warmup_ratio")),
        "target_standardization": _dataclass_values(
            config.target_standardization,
            ("enabled", "epsilon"),
        ),
        "run_mode_policy": config.run_mode_policy.to_dict(),
        "run_mode_overrides": dict(config.run_mode_overrides),
        "compute_profile_active_overrides": dict(
            config.compute_profile_active_overrides
        ),
        "compute_profile_dataloader_overrides": dict(
            config.compute_profile_dataloader_overrides
        ),
        "compute_profile_application": dict(config.compute_profile_application),
        "runtime_truth_contract": _base_direct_runtime_truth_contract(),
    }


def _base_direct_runtime_truth_contract() -> dict[str, object]:
    return {
        "schema_version": "model.runtime_truth_contract.v1",
        "provider_key": ModelKey.BASE_DIRECT.value,
        "required_runtime_evidence": {
            "batch_size": ["batch_size"],
            "num_workers": ["num_workers"],
            "pin_memory": ["pin_memory"],
            "persistent_workers": ["persistent_workers"],
            "prefetch_factor": ["prefetch_factor"],
        },
    }


def base_direct_config_from_effective_dict(
    effective: Mapping[str, object],
    *,
    source_path: Path | None,
) -> BaseDirectRunConfig:
    """Rehydrate a validated base_direct config from an effective-config snapshot."""

    identity_section = _required_mapping(effective, "identity")
    data_section = _required_mapping(effective, "data")
    encoder_section = _required_mapping(effective, "text_encoder")
    model_section = _required_mapping(effective, "model")
    loss_section = _required_mapping(effective, "loss")
    training_section = _required_mapping(effective, "training")
    optimizer_section = _required_mapping(effective, "optimizer")
    scheduler_section = _required_mapping(effective, "scheduler")
    standardization_section = _required_mapping(effective, "target_standardization")
    return BaseDirectRunConfig(
        source_path=source_path,
        raw_config=effective,
        identity=BaseDirectIdentityConfig(
            model_key=_str(identity_section, "model_key"),
            canonical_id=_str(identity_section, "canonical_id"),
            display_name=_str(identity_section, "display_name"),
            research_role=_str(identity_section, "research_role"),
            channel_policy=_str(identity_section, "channel_policy"),
            length_policy=_str(identity_section, "length_policy"),
            confidence_policy=_str(identity_section, "confidence_policy"),
            output_contract=_str(identity_section, "output_contract"),
            manifest_contract=_str(identity_section, "manifest_contract"),
        ),
        data=BaseDirectDataConfig(
            manifest_family=parse_modeling_manifest_family(
                _str(data_section, "manifest_family")
            ),
            train_split=SampleSplit(_str(data_section, "train_split")),
            validation_split=SampleSplit(_str(data_section, "validation_split")),
            prediction_splits=tuple(
                SampleSplit(value) for value in _string_sequence(data_section, "prediction_splits")
            ),
            limit_train_samples=_optional_int(data_section, "limit_train_samples"),
            limit_validation_samples=_optional_int(data_section, "limit_validation_samples"),
            limit_prediction_samples=_optional_int(data_section, "limit_prediction_samples"),
        ),
        text_encoder=BaseDirectTextEncoderConfig(
            model_name=_str(encoder_section, "model_name"),
            revision=_str(encoder_section, "revision"),
            max_length=_int(encoder_section, "max_length"),
            local_files_only=_bool(encoder_section, "local_files_only"),
            trainable=_bool(encoder_section, "trainable"),
            freeze_strategy=_str(encoder_section, "freeze_strategy"),
            encoder_learning_rate=_float(encoder_section, "encoder_learning_rate"),
        ),
        model=BaseDirectModelConfig(
            decoder_hidden_dim=_int(model_section, "decoder_hidden_dim"),
            decoder_layers=_int(model_section, "decoder_layers"),
            decoder_dropout=_float(model_section, "decoder_dropout"),
            frame_position_encoding_dim=_int(model_section, "frame_position_encoding_dim"),
        ),
        loss=BaseDirectLossConfig(
            channel_weights=_required_mapping(loss_section, "channel_weights"),
        ),
        training=BaseDirectTrainingConfig(
            epochs=_int(training_section, "epochs"),
            min_epochs=_int(training_section, "min_epochs"),
            early_stopping_patience=_int(training_section, "early_stopping_patience"),
            early_stopping_metric=_str(training_section, "early_stopping_metric"),
            early_stopping_mode=_str(training_section, "early_stopping_mode"),
            validate_every_epochs=_int(training_section, "validate_every_epochs"),
            batch_size=_int(training_section, "batch_size"),
            shuffle_train=_bool(training_section, "shuffle_train"),
            num_workers=_int(training_section, "num_workers"),
            pin_memory=_bool(training_section, "pin_memory"),
            persistent_workers=_bool(training_section, "persistent_workers"),
            prefetch_factor=_optional_int(training_section, "prefetch_factor"),
            non_blocking_transfers=_bool(training_section, "non_blocking_transfers"),
            seed=_optional_int(training_section, "seed"),
            device=_str(training_section, "device"),
            gradient_accumulation_steps=_int(training_section, "gradient_accumulation_steps"),
            max_grad_norm=_float(training_section, "max_grad_norm"),
            mixed_precision=_str(training_section, "mixed_precision"),
            length_bucketed_batching=_bool(training_section, "length_bucketed_batching"),
        ),
        optimizer=BaseDirectOptimizerConfig(
            name=_str(optimizer_section, "name"),
            decoder_learning_rate=_float(optimizer_section, "decoder_learning_rate"),
            weight_decay=_float(optimizer_section, "weight_decay"),
        ),
        scheduler=BaseDirectSchedulerConfig(
            name=_str(scheduler_section, "name"),
            warmup_ratio=_float(scheduler_section, "warmup_ratio"),
        ),
        target_standardization=BaseDirectTargetStandardizationConfig(
            enabled=_bool(standardization_section, "enabled"),
            epsilon=_float(standardization_section, "epsilon"),
        ),
        run_mode_policy=_run_mode_policy_from_effective(effective),
        run_mode_overrides=_required_mapping(effective, "run_mode_overrides")
        if isinstance(effective.get("run_mode_overrides"), Mapping)
        else {},
        compute_profile_active_overrides=(
            _required_mapping(effective, "compute_profile_active_overrides")
            if isinstance(effective.get("compute_profile_active_overrides"), Mapping)
            else {}
        ),
        compute_profile_dataloader_overrides=(
            _required_mapping(effective, "compute_profile_dataloader_overrides")
            if isinstance(effective.get("compute_profile_dataloader_overrides"), Mapping)
            else {}
        ),
        compute_profile_application=(
            _required_mapping(effective, "compute_profile_application")
            if isinstance(effective.get("compute_profile_application"), Mapping)
            else {}
        ),
    )


def _apply_run_mode(
    policy: ModelRunModePolicy,
    data: BaseDirectDataConfig,
    training: BaseDirectTrainingConfig,
) -> tuple[BaseDirectDataConfig, BaseDirectTrainingConfig, Mapping[str, object]]:
    if policy.mode is ModelRunMode.FULL:
        return data, training, MappingProxyType({})
    if policy.mode is ModelRunMode.SMOKE:
        overrides: dict[str, object] = {
            "data": {
                "limit_train_samples": policy.limit_train_samples,
                "limit_validation_samples": policy.limit_validation_samples,
                "limit_prediction_samples": policy.limit_prediction_samples,
            },
            "training": {
                "epochs": policy.max_epochs,
                "min_epochs": 1,
                "early_stopping_patience": 1,
                "batch_size": policy.batch_size,
                "num_workers": policy.num_workers,
                "persistent_workers": policy.persistent_workers,
                "prefetch_factor": policy.prefetch_factor,
            },
        }
        return (
            replace(
                data,
                limit_train_samples=policy.limit_train_samples,
                limit_validation_samples=policy.limit_validation_samples,
                limit_prediction_samples=policy.limit_prediction_samples,
            ),
            replace(
                training,
                epochs=policy.max_epochs or training.epochs,
                min_epochs=1,
                early_stopping_patience=1,
                batch_size=policy.batch_size or training.batch_size,
                num_workers=(
                    policy.num_workers
                    if policy.num_workers is not None
                    else training.num_workers
                ),
                persistent_workers=(
                    policy.persistent_workers
                    if policy.persistent_workers is not None
                    else training.persistent_workers
                ),
                prefetch_factor=policy.prefetch_factor,
            ),
            MappingProxyType(overrides),
        )
    overrides = {
        "data": {
            "limit_train_samples": policy.limit_train_samples,
            "limit_validation_samples": policy.limit_validation_samples,
            "limit_prediction_samples": policy.limit_prediction_samples,
        },
        "training": {
            "epochs": min(training.epochs, policy.max_epochs or training.epochs),
            "min_epochs": 1,
            "early_stopping_patience": min(training.early_stopping_patience, 2),
            "num_workers": policy.num_workers,
            "persistent_workers": policy.persistent_workers,
            "prefetch_factor": policy.prefetch_factor,
        },
    }
    return (
        replace(
            data,
            limit_train_samples=policy.limit_train_samples,
            limit_validation_samples=policy.limit_validation_samples,
            limit_prediction_samples=policy.limit_prediction_samples,
        ),
        replace(
            training,
            epochs=min(training.epochs, policy.max_epochs or training.epochs),
            min_epochs=1,
            early_stopping_patience=min(training.early_stopping_patience, 2),
            num_workers=(
                policy.num_workers
                if policy.num_workers is not None
                else training.num_workers
            ),
            persistent_workers=(
                policy.persistent_workers
                if policy.persistent_workers is not None
                else training.persistent_workers
            ),
            prefetch_factor=policy.prefetch_factor,
        ),
        MappingProxyType(overrides),
    )


def _apply_compute_profile_active_overrides(
    request: ModelRunRequest,
    training: BaseDirectTrainingConfig,
) -> tuple[BaseDirectTrainingConfig, Mapping[str, object]]:
    active = provider_active_overrides(
        request.compute_profile,
        provider_key="base_direct",
        run_mode=request.run_mode.value,
    )
    if not active:
        return training, MappingProxyType({})

    allowed = {"batch_size"}
    unknown = set(active) - allowed
    if unknown:
        raise ModelCandidateError(
            "base_direct compute profile active overrides contain unsupported keys: "
            f"{sorted(unknown)}."
        )

    batch_size = active.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ModelCandidateError(
            "base_direct compute profile active batch_size must be a positive integer."
        )

    return replace(training, batch_size=batch_size), active


def _apply_compute_profile_dataloader_overrides(
    request: ModelRunRequest,
    training: BaseDirectTrainingConfig,
) -> tuple[
    BaseDirectTrainingConfig,
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
    updates: dict[str, object] = {}
    allowed = {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "materialization_workers",
    }
    for key, value in requested.items():
        if key not in allowed:
            unsupported[key] = "base_direct does not recognize this dataloader profile key."
            continue
        if key == "materialization_workers":
            not_applicable[key] = "base_direct uses lazy dataloaders, not materialized surfaces."
            continue
        if key == "num_workers":
            if value is None:
                not_applicable[key] = "compute profile does not request a num_workers override."
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ModelCandidateError(
                    "base_direct dataloader.num_workers override must be a non-negative integer."
                )
            updates[key] = value
            applied[key] = value
            continue
        if key == "pin_memory":
            if not isinstance(value, bool):
                raise ModelCandidateError("base_direct dataloader.pin_memory must be boolean.")
            updates[key] = value
            applied[key] = value
            continue
        if key == "persistent_workers":
            if value is None:
                not_applicable[key] = (
                    "compute profile does not request a persistent_workers override."
                )
                continue
            if not isinstance(value, bool):
                raise ModelCandidateError(
                    "base_direct dataloader.persistent_workers must be boolean."
                )
            effective_num_workers = cast(int, updates.get("num_workers", training.num_workers))
            if effective_num_workers <= 0 and value:
                not_applicable[key] = (
                    "persistent dataloader workers require num_workers greater than zero."
                )
            else:
                updates[key] = value
                applied[key] = value
            continue
        if key == "prefetch_factor":
            if value is None:
                not_applicable[key] = "compute profile does not request a prefetch_factor override."
                continue
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ModelCandidateError(
                    "base_direct dataloader.prefetch_factor must be a positive integer or null."
                )
            effective_num_workers = cast(int, updates.get("num_workers", training.num_workers))
            if effective_num_workers <= 0:
                not_applicable[key] = (
                    "dataloader prefetch_factor requires num_workers greater than zero."
                )
            else:
                updates[key] = value
                applied[key] = value

    return (
        replace(training, **updates) if updates else training,
        MappingProxyType(applied),
        MappingProxyType(not_applicable),
        MappingProxyType(unsupported),
    )


def _default_config_dict() -> dict[str, object]:
    return {
        "identity": {
            **_EXPECTED_IDENTITY,
            "display_name": "M0 Direct Text-to-Pose Baseline",
        },
        "data": {
            "limit_train_samples": None,
            "limit_validation_samples": None,
            "limit_prediction_samples": None,
        },
        "text_encoder": {
            "model_name": "google/flan-t5-base",
            "revision": "main",
            "max_length": 128,
            "local_files_only": False,
            "trainable": True,
            "freeze_strategy": "none",
            "encoder_learning_rate": 0.00002,
        },
        "model": {
            "decoder_hidden_dim": 512,
            "decoder_layers": 4,
            "decoder_dropout": 0.10,
            "frame_position_encoding_dim": 32,
        },
        "loss": {
            "channel_weights": {
                "body": 1.0,
                "left_hand": 1.5,
                "right_hand": 1.5,
                "face": 1.25,
            }
        },
        "training": {
            "epochs": 50,
            "min_epochs": 10,
            "early_stopping_patience": 8,
            "early_stopping_metric": "validation_masked_l2_mean",
            "early_stopping_mode": "min",
            "validate_every_epochs": 1,
            "batch_size": 2,
            "shuffle_train": True,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "non_blocking_transfers": True,
            "seed": 13,
            "device": "auto",
            "gradient_accumulation_steps": 8,
            "max_grad_norm": 1.0,
            "mixed_precision": "auto",
            "length_bucketed_batching": True,
        },
        "optimizer": {
            "name": "adamw",
            "decoder_learning_rate": 0.0001,
            "weight_decay": 0.01,
        },
        "scheduler": {"name": "cosine_with_warmup", "warmup_ratio": 0.05},
        "target_standardization": {"enabled": True, "epsilon": 0.000001},
    }


def _run_mode_policy_from_effective(
    effective: Mapping[str, object],
) -> ModelRunModePolicy:
    policy_section = _required_mapping(effective, "run_mode_policy")
    policy = resolve_model_run_mode_policy(_str(policy_section, "mode"))
    if dict(policy_section) != policy.to_dict():
        raise ModelCandidateError(
            "base_direct effective config run_mode_policy does not match the shared policy."
        )
    return policy


def _required_mapping(mapping: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = mapping.get(name)
    if not isinstance(value, Mapping):
        raise ModelCandidateError(f"base_direct config {name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ModelCandidateError(f"base_direct config {name} keys must be strings.")
    return cast(Mapping[str, object], value)


def _mapping_proxy(value: Mapping[str, object], field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ModelCandidateError(f"{field_name} must be a string-keyed mapping.")
    return MappingProxyType(dict(value))


def _str(section: Mapping[str, object], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ModelCandidateError(f"base_direct config {name} must be non-empty text.")
    return value


def _bool(section: Mapping[str, object], name: str) -> bool:
    value = section.get(name)
    if not isinstance(value, bool):
        raise ModelCandidateError(f"base_direct config {name} must be boolean.")
    return value


def _int(section: Mapping[str, object], name: str) -> int:
    value = section.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelCandidateError(f"base_direct config {name} must be an integer.")
    return value


def _optional_int(section: Mapping[str, object], name: str) -> int | None:
    value = section.get(name)
    return None if value is None else _int(section, name)


def _float(section: Mapping[str, object], name: str) -> float:
    value = section.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ModelCandidateError(f"base_direct config {name} must be numeric.")
    return float(value)


def _string_sequence(section: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = section.get(name)
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
        raise ModelCandidateError(f"base_direct config {name} must be a string list.")
    return tuple(value)


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelCandidateError(f"{field_name} must be non-empty.")


def _require_positive_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelCandidateError(f"{field_name} must be positive.")


def _require_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ModelCandidateError(f"{field_name} must be non-negative.")


def _require_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None:
        _require_positive_int(value, field_name)


def _require_positive_float(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) <= 0.0:
        raise ModelCandidateError(f"{field_name} must be positive.")


def _non_negative_float(value: object, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or float(value) < 0.0:
        raise ModelCandidateError(f"{field_name} must be non-negative.")
    return float(value)


def _require_probability(value: float, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise ModelCandidateError(f"{field_name} must be within [0, 1].")


def _dataclass_values(instance: object, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: getattr(instance, field_name) for field_name in fields}


__all__ = [
    "BaseDirectDataConfig",
    "BaseDirectIdentityConfig",
    "BaseDirectLossConfig",
    "BaseDirectModelConfig",
    "BaseDirectOptimizerConfig",
    "BaseDirectRunConfig",
    "BaseDirectSchedulerConfig",
    "BaseDirectTargetStandardizationConfig",
    "BaseDirectTextEncoderConfig",
    "BaseDirectTrainingConfig",
    "base_direct_config_from_effective_dict",
    "base_direct_config_to_dict",
    "load_base_direct_config",
]
