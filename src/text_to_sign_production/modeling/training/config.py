"""Config loading for M0 full-BFH baseline training runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import yaml

from text_to_sign_production.data.constants import SPLITS
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH
from text_to_sign_production.modeling.contracts import (
    BASELINE_ID,
    BASELINE_NAME,
    BASELINE_ROLE,
    CONFIDENCE_POLICY,
    LENGTH_POLICY,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    validate_channel_weights,
)
from text_to_sign_production.modeling.data import M0_CHANNEL_POLICY, M0_TARGET_CHANNELS

SUPPORTED_OPTIMIZERS = frozenset({"adamw"})


class BaselineTrainingConfigError(ValueError):
    """Raised when a baseline training config is missing or malformed."""


@dataclass(frozen=True, slots=True)
class BaselineIdentityConfig:
    """Identity and policy constants for the M0 baseline contract."""

    baseline_id: str
    name: str
    role: str
    channels: tuple[str, ...]
    channel_policy: str
    length_policy: str
    confidence_policy: str
    prediction_schema_version: str
    prediction_manifest_schema_version: str


@dataclass(frozen=True, slots=True)
class BaselineDataConfig:
    """Processed Dataset Build references for baseline train/val loading."""

    train_manifest: Path
    val_manifest: Path
    train_split: str
    val_split: str
    prediction_splits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BaselineBackboneConfig:
    """Text backbone configuration for the baseline model."""

    name: str
    revision: str
    max_length: int
    local_files_only: bool
    trainable: bool
    freeze_strategy: str
    encoder_learning_rate: float


@dataclass(frozen=True, slots=True)
class BaselineModelConfig:
    """Baseline decoder dimensions."""

    decoder_hidden_dim: int
    decoder_layers: int
    decoder_dropout: float
    frame_position_encoding_dim: int


@dataclass(frozen=True, slots=True)
class BaselineLossConfig:
    """Channel-balanced masked regression loss settings."""

    channel_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class BaselineLoopConfig:
    """Epoch loop and reproducibility settings."""

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
    progress_interval_batches: int
    length_bucketed_batching: bool


@dataclass(frozen=True, slots=True)
class BaselineOptimizerConfig:
    """Optimizer settings for the baseline training loop."""

    name: str
    decoder_learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class BaselineSchedulerConfig:
    """Learning-rate scheduler settings."""

    name: str
    warmup_ratio: float


@dataclass(frozen=True, slots=True)
class BaselineTargetStandardizationConfig:
    """Model-side target standardization policy."""

    enabled: bool
    epsilon: float


@dataclass(frozen=True, slots=True)
class BaselineCheckpointConfig:
    """Checkpoint output settings."""

    output_dir: Path


@dataclass(frozen=True, slots=True)
class BaselineTrainingConfig:
    """Validated effective baseline training configuration."""

    source_path: Path
    raw_config: dict[str, Any]
    baseline: BaselineIdentityConfig
    data: BaselineDataConfig
    backbone: BaselineBackboneConfig
    model: BaselineModelConfig
    loss: BaselineLossConfig
    training: BaselineLoopConfig
    optimizer: BaselineOptimizerConfig
    scheduler: BaselineSchedulerConfig
    target_standardization: BaselineTargetStandardizationConfig
    checkpoint: BaselineCheckpointConfig


def load_baseline_training_config(
    path: Path | str = DEFAULT_BASELINE_CONFIG_PATH,
    *,
    validate_paths: bool = True,
    checkpoint_output_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> BaselineTrainingConfig:
    """Load, validate, and resolve an M0 baseline training config."""

    config_path = _resolve_path(path, repo_root=repo_root)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BaselineTrainingConfigError(f"Could not read config file: {config_path}") from exc
    except yaml.YAMLError as exc:
        message = f"Could not parse YAML config: {config_path}: {exc}"
        raise BaselineTrainingConfigError(message) from exc

    if not isinstance(loaded, dict):
        raise BaselineTrainingConfigError("Baseline training config root must be a mapping.")
    raw_config = cast(dict[str, Any], loaded)

    baseline_section = _required_mapping(raw_config, "baseline")
    data_section = _required_mapping(raw_config, "data")
    text_encoder_section = _required_mapping(raw_config, "text_encoder")
    model_section = _required_mapping(raw_config, "model")
    loss_section = _required_mapping(raw_config, "loss")
    training_section = _required_mapping(raw_config, "training")
    optimizer_section = _required_mapping(raw_config, "optimizer")
    scheduler_section = _required_mapping(raw_config, "scheduler")
    target_standardization_section = _required_mapping(raw_config, "target_standardization")
    checkpoint_section = _optional_mapping(raw_config, "checkpoint")

    train_manifest = _required_path(data_section, "data.train_manifest", repo_root=repo_root)
    val_manifest = _required_path(data_section, "data.val_manifest", repo_root=repo_root)
    if validate_paths:
        _require_existing_file(train_manifest, "data.train_manifest")
        _require_existing_file(val_manifest, "data.val_manifest")

    checkpoint_dir = _required_injected_checkpoint_output_dir(
        checkpoint_output_dir,
        checkpoint_section=checkpoint_section,
        repo_root=repo_root,
    )

    optimizer_name = _required_str(optimizer_section, "optimizer.name").lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        expected = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise BaselineTrainingConfigError(
            f"optimizer.name must be one of: {expected}; got {optimizer_name!r}."
        )

    num_workers, persistent_workers, prefetch_factor = _validated_worker_config(training_section)

    train_split = _required_split(data_section, "data.train_split")
    val_split = _required_split(data_section, "data.val_split")
    prediction_splits = _required_split_sequence(data_section, "data.prediction_splits")

    return BaselineTrainingConfig(
        source_path=config_path.resolve(),
        raw_config=raw_config,
        baseline=BaselineIdentityConfig(
            baseline_id=_required_literal(baseline_section, "baseline.id", BASELINE_ID),
            name=_required_literal(baseline_section, "baseline.name", BASELINE_NAME),
            role=_required_literal(baseline_section, "baseline.role", BASELINE_ROLE),
            channels=_required_channel_list(baseline_section, "baseline.channels"),
            channel_policy=_required_literal(
                baseline_section,
                "baseline.channel_policy",
                M0_CHANNEL_POLICY,
            ),
            length_policy=_required_literal(
                baseline_section,
                "baseline.length_policy",
                LENGTH_POLICY,
            ),
            confidence_policy=_required_literal(
                baseline_section,
                "baseline.confidence_policy",
                CONFIDENCE_POLICY,
            ),
            prediction_schema_version=_required_literal(
                baseline_section,
                "baseline.prediction_schema_version",
                PREDICTION_SCHEMA_VERSION,
            ),
            prediction_manifest_schema_version=_required_literal(
                baseline_section,
                "baseline.prediction_manifest_schema_version",
                PREDICTION_MANIFEST_SCHEMA_VERSION,
            ),
        ),
        data=BaselineDataConfig(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            train_split=train_split,
            val_split=val_split,
            prediction_splits=prediction_splits,
        ),
        backbone=BaselineBackboneConfig(
            name=_required_str(text_encoder_section, "text_encoder.model_name"),
            revision=_required_str(text_encoder_section, "text_encoder.revision"),
            max_length=_required_positive_int(text_encoder_section, "text_encoder.max_length"),
            local_files_only=_required_bool(
                text_encoder_section,
                "text_encoder.local_files_only",
            ),
            trainable=_required_bool(text_encoder_section, "text_encoder.trainable"),
            freeze_strategy=_required_freeze_strategy(
                text_encoder_section,
                "text_encoder.freeze_strategy",
            ),
            encoder_learning_rate=_required_positive_float(
                text_encoder_section,
                "text_encoder.encoder_learning_rate",
            ),
        ),
        model=BaselineModelConfig(
            decoder_hidden_dim=_required_positive_int(
                model_section,
                "model.decoder_hidden_dim",
            ),
            decoder_layers=_required_positive_int(model_section, "model.decoder_layers"),
            decoder_dropout=_required_probability(model_section, "model.decoder_dropout"),
            frame_position_encoding_dim=_required_non_negative_int(
                model_section,
                "model.frame_position_encoding_dim",
            ),
        ),
        loss=BaselineLossConfig(
            channel_weights=_required_channel_weights(loss_section, "loss.channel_weights"),
        ),
        training=BaselineLoopConfig(
            epochs=_required_positive_int(training_section, "training.epochs"),
            min_epochs=_required_positive_int(training_section, "training.min_epochs"),
            early_stopping_patience=_required_positive_int(
                training_section,
                "training.early_stopping_patience",
            ),
            early_stopping_metric=_required_literal(
                training_section,
                "training.early_stopping_metric",
                "validation_masked_l2_mean",
            ),
            early_stopping_mode=_required_literal(
                training_section,
                "training.early_stopping_mode",
                "min",
            ),
            validate_every_epochs=_required_positive_int(
                training_section,
                "training.validate_every_epochs",
            ),
            batch_size=_required_positive_int(training_section, "training.batch_size"),
            shuffle_train=_required_bool(training_section, "training.shuffle_train"),
            num_workers=num_workers,
            pin_memory=_required_bool(training_section, "training.pin_memory"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            non_blocking_transfers=_required_bool(
                training_section,
                "training.non_blocking_transfers",
            ),
            seed=_required_nullable_non_negative_int(training_section, "training.seed"),
            device=_required_str(training_section, "training.device"),
            gradient_accumulation_steps=_required_positive_int(
                training_section,
                "training.gradient_accumulation_steps",
            ),
            max_grad_norm=_required_positive_float(training_section, "training.max_grad_norm"),
            mixed_precision=_required_mixed_precision(
                training_section,
                "training.mixed_precision",
            ),
            progress_interval_batches=_required_positive_int(
                training_section,
                "training.progress_interval_batches",
            ),
            length_bucketed_batching=_required_bool(
                training_section,
                "training.length_bucketed_batching",
            ),
        ),
        optimizer=BaselineOptimizerConfig(
            name=optimizer_name,
            decoder_learning_rate=_required_positive_float(
                optimizer_section,
                "optimizer.decoder_learning_rate",
            ),
            weight_decay=_required_non_negative_float(optimizer_section, "optimizer.weight_decay"),
        ),
        scheduler=BaselineSchedulerConfig(
            name=_required_literal(scheduler_section, "scheduler.name", "cosine_with_warmup"),
            warmup_ratio=_required_probability(scheduler_section, "scheduler.warmup_ratio"),
        ),
        target_standardization=BaselineTargetStandardizationConfig(
            enabled=_required_bool(
                target_standardization_section, "target_standardization.enabled"
            ),
            epsilon=_required_positive_float(
                target_standardization_section, "target_standardization.epsilon"
            ),
        ),
        checkpoint=BaselineCheckpointConfig(output_dir=checkpoint_dir),
    )


def baseline_config_to_dict(
    config: BaselineTrainingConfig,
    *,
    path_formatter: Callable[[Path], str],
) -> dict[str, Any]:
    """Return JSON-serializable effective config content for provenance."""

    return {
        "baseline": {
            "id": config.baseline.baseline_id,
            "name": config.baseline.name,
            "role": config.baseline.role,
            "channels": list(config.baseline.channels),
            "channel_policy": config.baseline.channel_policy,
            "length_policy": config.baseline.length_policy,
            "confidence_policy": config.baseline.confidence_policy,
            "prediction_schema_version": config.baseline.prediction_schema_version,
            "prediction_manifest_schema_version": (
                config.baseline.prediction_manifest_schema_version
            ),
        },
        "data": {
            "train_manifest": path_formatter(config.data.train_manifest),
            "val_manifest": path_formatter(config.data.val_manifest),
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
            "prediction_splits": list(config.data.prediction_splits),
        },
        "text_encoder": {
            "model_name": config.backbone.name,
            "revision": config.backbone.revision,
            "max_length": config.backbone.max_length,
            "local_files_only": config.backbone.local_files_only,
            "trainable": config.backbone.trainable,
            "freeze_strategy": config.backbone.freeze_strategy,
            "encoder_learning_rate": config.backbone.encoder_learning_rate,
        },
        "model": {
            "decoder_hidden_dim": config.model.decoder_hidden_dim,
            "decoder_layers": config.model.decoder_layers,
            "decoder_dropout": config.model.decoder_dropout,
            "frame_position_encoding_dim": config.model.frame_position_encoding_dim,
        },
        "loss": {
            "channel_weights": dict(config.loss.channel_weights),
        },
        "training": {
            "epochs": config.training.epochs,
            "min_epochs": config.training.min_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_metric": config.training.early_stopping_metric,
            "early_stopping_mode": config.training.early_stopping_mode,
            "validate_every_epochs": config.training.validate_every_epochs,
            "batch_size": config.training.batch_size,
            "shuffle_train": config.training.shuffle_train,
            "num_workers": config.training.num_workers,
            "pin_memory": config.training.pin_memory,
            "persistent_workers": config.training.persistent_workers,
            "prefetch_factor": config.training.prefetch_factor,
            "non_blocking_transfers": config.training.non_blocking_transfers,
            "seed": config.training.seed,
            "device": config.training.device,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "max_grad_norm": config.training.max_grad_norm,
            "mixed_precision": config.training.mixed_precision,
            "progress_interval_batches": config.training.progress_interval_batches,
            "length_bucketed_batching": config.training.length_bucketed_batching,
        },
        "optimizer": {
            "name": config.optimizer.name,
            "decoder_learning_rate": config.optimizer.decoder_learning_rate,
            "weight_decay": config.optimizer.weight_decay,
        },
        "scheduler": {
            "name": config.scheduler.name,
            "warmup_ratio": config.scheduler.warmup_ratio,
        },
        "target_standardization": {
            "enabled": config.target_standardization.enabled,
            "epsilon": config.target_standardization.epsilon,
        },
        "checkpoint": {
            "output_dir": path_formatter(config.checkpoint.output_dir),
        },
    }


def baseline_config_with_training_overrides(
    config: BaselineTrainingConfig,
    *,
    epochs: int | None = None,
    min_epochs: int | None = None,
    early_stopping_patience: int | None = None,
    shuffle_train: bool | None = None,
) -> BaselineTrainingConfig:
    """Return a runtime-effective config with workflow-owned training overrides applied."""

    resolved_epochs = config.training.epochs
    if epochs is not None:
        resolved_epochs = _validated_positive_int_override(epochs, label="training.epochs")
    resolved_min_epochs = config.training.min_epochs
    if min_epochs is not None:
        resolved_min_epochs = _validated_positive_int_override(
            min_epochs,
            label="training.min_epochs",
        )
    if resolved_min_epochs > resolved_epochs:
        raise BaselineTrainingConfigError("training.min_epochs must not exceed training.epochs.")
    resolved_patience = config.training.early_stopping_patience
    if early_stopping_patience is not None:
        resolved_patience = _validated_positive_int_override(
            early_stopping_patience,
            label="training.early_stopping_patience",
        )
    resolved_shuffle_train = config.training.shuffle_train
    if shuffle_train is not None:
        resolved_shuffle_train = _validated_bool_override(
            shuffle_train,
            label="training.shuffle_train",
        )
    if (
        resolved_epochs == config.training.epochs
        and resolved_min_epochs == config.training.min_epochs
        and resolved_patience == config.training.early_stopping_patience
        and resolved_shuffle_train == config.training.shuffle_train
    ):
        return config
    return replace(
        config,
        training=replace(
            config.training,
            epochs=resolved_epochs,
            min_epochs=resolved_min_epochs,
            early_stopping_patience=resolved_patience,
            shuffle_train=resolved_shuffle_train,
        ),
    )


def baseline_config_with_data_overrides(
    config: BaselineTrainingConfig,
    *,
    train_manifest: Path | None = None,
    val_manifest: Path | None = None,
) -> BaselineTrainingConfig:
    """Return a config with workflow-owned dataset-surface overrides applied."""

    resolved_train_manifest = (
        config.data.train_manifest if train_manifest is None else train_manifest
    )
    resolved_val_manifest = config.data.val_manifest if val_manifest is None else val_manifest
    if (
        resolved_train_manifest == config.data.train_manifest
        and resolved_val_manifest == config.data.val_manifest
    ):
        return config
    return replace(
        config,
        data=replace(
            config.data,
            train_manifest=resolved_train_manifest,
            val_manifest=resolved_val_manifest,
        ),
    )


def _required_mapping(config: dict[str, Any], field_path: str) -> dict[str, Any]:
    value = _required_value(config, field_path)
    if not isinstance(value, dict):
        raise BaselineTrainingConfigError(f"{field_path} must be a mapping.")
    return cast(dict[str, Any], value)


def _optional_mapping(config: dict[str, Any], field_path: str) -> dict[str, Any]:
    key = field_path.rsplit(".", maxsplit=1)[-1]
    if key not in config or config[key] is None:
        return {}
    value = config[key]
    if not isinstance(value, dict):
        raise BaselineTrainingConfigError(f"{field_path} must be a mapping when provided.")
    return cast(dict[str, Any], value)


def _required_injected_checkpoint_output_dir(
    checkpoint_output_dir: Path | str | None,
    *,
    checkpoint_section: dict[str, Any],
    repo_root: Path | str | None,
) -> Path:
    if "output_dir" in checkpoint_section:
        raise BaselineTrainingConfigError(
            "checkpoint.output_dir must be injected by the workflow at runtime, "
            "not stored in the modeling source config."
        )
    if checkpoint_output_dir is None:
        raise BaselineTrainingConfigError(
            "checkpoint_output_dir is required and must be injected by the workflow."
        )
    return _resolve_path(checkpoint_output_dir, repo_root=repo_root)


def _required_value(config: dict[str, Any], field_path: str) -> Any:
    key = field_path.rsplit(".", maxsplit=1)[-1]
    if key not in config:
        raise BaselineTrainingConfigError(f"Config field {field_path} is required.")
    return config[key]


def _validated_worker_config(config: dict[str, Any]) -> tuple[int, bool, int | None]:
    num_workers = _required_non_negative_int(config, "training.num_workers")
    persistent_workers = _required_bool(config, "training.persistent_workers")
    prefetch_factor = _required_nullable_positive_int(
        config,
        "training.prefetch_factor",
    )
    if persistent_workers and num_workers == 0:
        raise BaselineTrainingConfigError(
            "training.persistent_workers requires training.num_workers to be positive."
        )
    if prefetch_factor is not None and num_workers == 0:
        raise BaselineTrainingConfigError(
            "training.prefetch_factor requires training.num_workers to be positive."
        )
    return num_workers, persistent_workers, prefetch_factor


def _required_str(config: dict[str, Any], field_path: str) -> str:
    value = _required_value(config, field_path)
    if not isinstance(value, str):
        raise BaselineTrainingConfigError(f"{field_path} must be a string.")
    if not value.strip():
        raise BaselineTrainingConfigError(f"{field_path} must not be blank.")
    return value


def _required_literal(config: dict[str, Any], field_path: str, expected: str) -> str:
    value = _required_str(config, field_path)
    if value != expected:
        raise BaselineTrainingConfigError(f"{field_path} must be {expected!r}; got {value!r}.")
    return value


def _required_split(config: dict[str, Any], field_path: str) -> str:
    value = _required_str(config, field_path)
    if value not in SPLITS:
        expected = ", ".join(SPLITS)
        raise BaselineTrainingConfigError(
            f"{field_path} must be one of: {expected}; got {value!r}."
        )
    return value


def _required_split_sequence(config: dict[str, Any], field_path: str) -> tuple[str, ...]:
    values = _required_str_sequence(config, field_path)
    for value in values:
        if value not in SPLITS:
            expected = ", ".join(SPLITS)
            raise BaselineTrainingConfigError(
                f"{field_path} values must be one of: {expected}; got {value!r}."
            )
    if len(set(values)) != len(values):
        raise BaselineTrainingConfigError(f"{field_path} must not contain duplicates.")
    return values


def _required_channel_list(config: dict[str, Any], field_path: str) -> tuple[str, ...]:
    values = _required_str_sequence(config, field_path)
    if values != M0_TARGET_CHANNELS:
        raise BaselineTrainingConfigError(
            f"{field_path} must be {M0_TARGET_CHANNELS!r}; got {values!r}."
        )
    return values


def _required_str_sequence(config: dict[str, Any], field_path: str) -> tuple[str, ...]:
    value = _required_value(config, field_path)
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise BaselineTrainingConfigError(f"{field_path} must be a list of strings.")
    if not value:
        raise BaselineTrainingConfigError(f"{field_path} must not be empty.")

    resolved: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise BaselineTrainingConfigError(
                f"{field_path}[{index}] must be a string; got {type(item).__name__}."
            )
        if not item.strip():
            raise BaselineTrainingConfigError(f"{field_path}[{index}] must not be blank.")
        if item != item.strip():
            raise BaselineTrainingConfigError(
                f"{field_path}[{index}] must not contain leading or trailing whitespace."
            )
        resolved.append(item)
    return tuple(resolved)


def _required_channel_weights(config: dict[str, Any], field_path: str) -> dict[str, float]:
    weights_section = _required_mapping(config, field_path)
    raw_weights: dict[str, float] = {}
    for key, value in weights_section.items():
        if not isinstance(key, str):
            raise BaselineTrainingConfigError(f"{field_path} keys must be strings.")
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise BaselineTrainingConfigError(f"{field_path}.{key} must be numeric.")
        raw_weights[key] = float(value)
    try:
        return validate_channel_weights(raw_weights)
    except ValueError as exc:
        raise BaselineTrainingConfigError(str(exc)) from exc


def _required_bool(config: dict[str, Any], field_path: str) -> bool:
    value = _required_value(config, field_path)
    if not isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be a boolean.")
    return value


def _required_freeze_strategy(config: dict[str, Any], field_path: str) -> str:
    value = _required_str(config, field_path)
    if value not in {"none", "partial", "frozen"}:
        raise BaselineTrainingConfigError(
            f"{field_path} must be one of: none, partial, frozen; got {value!r}."
        )
    return value


def _required_mixed_precision(config: dict[str, Any], field_path: str) -> str:
    value = _required_str(config, field_path)
    if value not in {"auto", "bf16", "fp16", "off"}:
        raise BaselineTrainingConfigError(
            f"{field_path} must be one of: auto, bf16, fp16, off; got {value!r}."
        )
    return value


def _required_positive_int(config: dict[str, Any], field_path: str) -> int:
    value = _required_value(config, field_path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be an integer.")
    if value <= 0:
        raise BaselineTrainingConfigError(f"{field_path} must be positive.")
    return value


def _validated_positive_int_override(value: int, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{label} must be an integer.")
    if value <= 0:
        raise BaselineTrainingConfigError(f"{label} must be positive.")
    return value


def _validated_bool_override(value: bool, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{label} must be a boolean.")
    return value


def _required_non_negative_int(config: dict[str, Any], field_path: str) -> int:
    value = _required_value(config, field_path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be an integer.")
    if value < 0:
        raise BaselineTrainingConfigError(f"{field_path} must not be negative.")
    return value


def _required_nullable_non_negative_int(config: dict[str, Any], field_path: str) -> int | None:
    value = _required_value(config, field_path)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be an integer or null.")
    if value < 0:
        raise BaselineTrainingConfigError(f"{field_path} must not be negative.")
    return value


def _required_nullable_positive_int(config: dict[str, Any], field_path: str) -> int | None:
    value = _required_value(config, field_path)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be an integer or null.")
    if value <= 0:
        raise BaselineTrainingConfigError(f"{field_path} must be positive when set.")
    return value


def _required_positive_float(config: dict[str, Any], field_path: str) -> float:
    value = _required_value(config, field_path)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be numeric.")
    resolved = float(value)
    if resolved <= 0.0:
        raise BaselineTrainingConfigError(f"{field_path} must be positive.")
    return resolved


def _required_probability(config: dict[str, Any], field_path: str) -> float:
    value = _required_value(config, field_path)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be numeric.")
    resolved = float(value)
    if resolved < 0.0 or resolved > 1.0:
        raise BaselineTrainingConfigError(f"{field_path} must be in [0, 1].")
    return resolved


def _required_non_negative_float(config: dict[str, Any], field_path: str) -> float:
    value = _required_value(config, field_path)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be numeric.")
    resolved = float(value)
    if resolved < 0.0:
        raise BaselineTrainingConfigError(f"{field_path} must not be negative.")
    return resolved


def _required_path(
    config: dict[str, Any],
    field_path: str,
    *,
    repo_root: Path | str | None = None,
) -> Path:
    return _resolve_path(_required_str(config, field_path), repo_root=repo_root)


def _require_existing_file(path: Path, field_path: str) -> None:
    if not path.is_file():
        raise BaselineTrainingConfigError(f"{field_path} does not exist or is not a file: {path}")


def _resolve_path(path: Path | str, *, repo_root: Path | str | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    if repo_root is not None:
        root = Path(repo_root).expanduser().resolve()
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root):
            raise BaselineTrainingConfigError(f"Path must stay under {root}: {path}")
        return resolved

    raise BaselineTrainingConfigError(
        f"Relative path requires explicit repo_root in baseline config loading: {path}"
    )
