"""Config loading for Sprint 3 baseline training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from text_to_sign_production.data.utils import resolve_repo_path
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH

SUPPORTED_OPTIMIZERS = frozenset({"adamw"})


class BaselineTrainingConfigError(ValueError):
    """Raised when a baseline training config is missing or malformed."""


@dataclass(frozen=True, slots=True)
class BaselineDataConfig:
    """Processed Dataset Build references for baseline train/val loading."""

    train_manifest: Path
    val_manifest: Path
    train_split: str
    val_split: str


@dataclass(frozen=True, slots=True)
class BaselineBackboneConfig:
    """Text backbone configuration for the baseline model."""

    name: str
    max_length: int
    local_files_only: bool
    freeze: bool


@dataclass(frozen=True, slots=True)
class BaselineModelConfig:
    """Small baseline decoder dimensions."""

    decoder_hidden_dim: int
    latent_dim: int


@dataclass(frozen=True, slots=True)
class BaselineLoopConfig:
    """Epoch loop and reproducibility settings."""

    epochs: int
    batch_size: int
    shuffle_train: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    non_blocking_transfers: bool
    seed: int | None
    device: str


@dataclass(frozen=True, slots=True)
class BaselineOptimizerConfig:
    """Optimizer settings for the baseline training loop."""

    name: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class BaselineCheckpointConfig:
    """Checkpoint output settings."""

    output_dir: Path


@dataclass(frozen=True, slots=True)
class BaselineTrainingConfig:
    """Validated effective baseline training configuration."""

    source_path: Path
    raw_config: dict[str, Any]
    data: BaselineDataConfig
    backbone: BaselineBackboneConfig
    model: BaselineModelConfig
    training: BaselineLoopConfig
    optimizer: BaselineOptimizerConfig
    checkpoint: BaselineCheckpointConfig


def load_baseline_training_config(
    path: Path | str = DEFAULT_BASELINE_CONFIG_PATH,
    *,
    validate_paths: bool = True,
    checkpoint_output_dir: Path | str | None = None,
) -> BaselineTrainingConfig:
    """Load, validate, and resolve a Sprint 3 baseline training config."""

    config_path = Path(path)
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

    data_section = _required_mapping(raw_config, "data")
    backbone_section = _required_mapping(raw_config, "backbone")
    model_section = _required_mapping(raw_config, "model")
    training_section = _required_mapping(raw_config, "training")
    optimizer_section = _required_mapping(raw_config, "optimizer")
    checkpoint_section = _required_mapping(raw_config, "checkpoint")

    train_manifest = _required_path(data_section, "data.train_manifest")
    val_manifest = _required_path(data_section, "data.val_manifest")
    if validate_paths:
        _require_existing_file(train_manifest, "data.train_manifest")
        _require_existing_file(val_manifest, "data.val_manifest")

    checkpoint_dir = (
        resolve_repo_path(checkpoint_output_dir)
        if checkpoint_output_dir is not None
        else _required_path(checkpoint_section, "checkpoint.output_dir")
    )

    optimizer_name = _required_str(optimizer_section, "optimizer.name").lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        expected = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise BaselineTrainingConfigError(
            f"optimizer.name must be one of: {expected}; got {optimizer_name!r}."
        )

    num_workers, persistent_workers, prefetch_factor = _validated_worker_config(training_section)

    return BaselineTrainingConfig(
        source_path=config_path.resolve(),
        raw_config=raw_config,
        data=BaselineDataConfig(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            train_split=_required_str(data_section, "data.train_split"),
            val_split=_required_str(data_section, "data.val_split"),
        ),
        backbone=BaselineBackboneConfig(
            name=_required_str(backbone_section, "backbone.name"),
            max_length=_required_positive_int(backbone_section, "backbone.max_length"),
            local_files_only=_required_bool(backbone_section, "backbone.local_files_only"),
            freeze=_required_bool(backbone_section, "backbone.freeze"),
        ),
        model=BaselineModelConfig(
            decoder_hidden_dim=_required_positive_int(
                model_section,
                "model.decoder_hidden_dim",
            ),
            latent_dim=_required_positive_int(model_section, "model.latent_dim"),
        ),
        training=BaselineLoopConfig(
            epochs=_required_positive_int(training_section, "training.epochs"),
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
        ),
        optimizer=BaselineOptimizerConfig(
            name=optimizer_name,
            learning_rate=_required_positive_float(optimizer_section, "optimizer.learning_rate"),
            weight_decay=_required_non_negative_float(optimizer_section, "optimizer.weight_decay"),
        ),
        checkpoint=BaselineCheckpointConfig(output_dir=checkpoint_dir),
    )


def baseline_config_to_dict(config: BaselineTrainingConfig) -> dict[str, Any]:
    """Return JSON-serializable effective config content for provenance."""

    return {
        "data": {
            "train_manifest": config.data.train_manifest.as_posix(),
            "val_manifest": config.data.val_manifest.as_posix(),
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
        },
        "backbone": {
            "name": config.backbone.name,
            "max_length": config.backbone.max_length,
            "local_files_only": config.backbone.local_files_only,
            "freeze": config.backbone.freeze,
        },
        "model": {
            "decoder_hidden_dim": config.model.decoder_hidden_dim,
            "latent_dim": config.model.latent_dim,
        },
        "training": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "shuffle_train": config.training.shuffle_train,
            "num_workers": config.training.num_workers,
            "pin_memory": config.training.pin_memory,
            "persistent_workers": config.training.persistent_workers,
            "prefetch_factor": config.training.prefetch_factor,
            "non_blocking_transfers": config.training.non_blocking_transfers,
            "seed": config.training.seed,
            "device": config.training.device,
        },
        "optimizer": {
            "name": config.optimizer.name,
            "learning_rate": config.optimizer.learning_rate,
            "weight_decay": config.optimizer.weight_decay,
        },
        "checkpoint": {
            "output_dir": config.checkpoint.output_dir.as_posix(),
        },
    }


def _required_mapping(config: dict[str, Any], field_path: str) -> dict[str, Any]:
    value = _required_value(config, field_path)
    if not isinstance(value, dict):
        raise BaselineTrainingConfigError(f"{field_path} must be a mapping.")
    return cast(dict[str, Any], value)


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


def _required_bool(config: dict[str, Any], field_path: str) -> bool:
    value = _required_value(config, field_path)
    if not isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be a boolean.")
    return value


def _required_positive_int(config: dict[str, Any], field_path: str) -> int:
    value = _required_value(config, field_path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be an integer.")
    if value <= 0:
        raise BaselineTrainingConfigError(f"{field_path} must be positive.")
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


def _required_non_negative_float(config: dict[str, Any], field_path: str) -> float:
    value = _required_value(config, field_path)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BaselineTrainingConfigError(f"{field_path} must be numeric.")
    resolved = float(value)
    if resolved < 0.0:
        raise BaselineTrainingConfigError(f"{field_path} must not be negative.")
    return resolved


def _required_path(config: dict[str, Any], field_path: str) -> Path:
    return resolve_repo_path(_required_str(config, field_path))


def _require_existing_file(path: Path, field_path: str) -> None:
    if not path.is_file():
        raise BaselineTrainingConfigError(f"{field_path} does not exist or is not a file: {path}")
