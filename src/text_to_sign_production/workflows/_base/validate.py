"""Base workflow input validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from text_to_sign_production.artifacts.layout import DatasetArtifactLayout
from text_to_sign_production.core.files import (
    OutputExistsPolicy,
    require_dir_contains,
    require_file,
)
from text_to_sign_production.core.paths import ProjectLayout
from text_to_sign_production.modeling.data import (
    ProcessedModelingDataError,
    read_processed_modeling_manifest,
)
from text_to_sign_production.modeling.training.config import (
    BaselineTrainingConfig,
    BaselineTrainingConfigError,
    load_baseline_training_config,
)
from text_to_sign_production.workflows._base.constants import (
    BASE_ALLOWED_SPLITS,
    BASE_RUN_MODE_POLICIES,
    BASE_RUN_MODES,
    BaseRunMode,
    BaseRunModePolicy,
)
from text_to_sign_production.workflows._base.layout import (
    _config_path,
    _layout_from_config,
    build_base_run_layout,
)
from text_to_sign_production.workflows._base.types import BaseWorkflowConfig, BaseWorkflowInputError


def validate_base_inputs(config: BaseWorkflowConfig) -> None:
    """Validate baseline workflow inputs without constructing a model."""

    config = _workflow_config_with_run_mode_policy(config)
    layout = _layout_from_config(config)
    config_path = _config_path(config, layout)
    _output_policy(config.output_policy)
    if config.panel_size <= 0:
        raise BaseWorkflowInputError("panel_size must be positive.")
    _positive_optional_limit(
        config.limit_prediction_samples,
        label="limit_prediction_samples",
    )

    try:
        require_file(config_path, label="Baseline config")
        run_layout = build_base_run_layout(project_root=layout.root, run_name=config.run_name)
        training_config = load_baseline_training_config(
            config_path,
            validate_paths=config.validate_processed_inputs,
            checkpoint_output_dir=run_layout.checkpoints_dir,
            repo_root=layout.root,
        )
        _prediction_splits(config.prediction_splits, training_config)
        if config.validate_processed_inputs:
            _validate_processed_training_inputs(training_config, layout)
            _validate_processed_prediction_inputs(
                training_config,
                layout,
                prediction_splits=config.prediction_splits,
            )
    except (
        BaselineTrainingConfigError,
        FileNotFoundError,
        IsADirectoryError,
        ProcessedModelingDataError,
        ValueError,
    ) as exc:
        raise BaseWorkflowInputError(str(exc)) from exc


def _workflow_config_with_run_mode_policy(config: BaseWorkflowConfig) -> BaseWorkflowConfig:
    policy = _run_mode_policy(config.run_mode)
    _validate_run_name_for_mode(run_mode=policy.run_mode, run_name=config.run_name)
    prediction_splits = tuple(_validate_split(split) for split in policy.prediction_splits)
    return replace(
        config,
        run_mode=policy.run_mode,
        prediction_splits=prediction_splits,
        run_qualitative_export=policy.run_qualitative_export,
        panel_size=policy.panel_size,
        limit_train_samples=policy.limit_train_samples,
        limit_validation_samples=policy.limit_validation_samples,
        limit_prediction_samples=policy.limit_prediction_samples,
        epoch_count=policy.epoch_count,
        min_epochs=policy.min_epochs,
        early_stopping_patience=policy.early_stopping_patience,
        shuffle_train=policy.shuffle_train,
    )


def _run_mode_policy(value: BaseRunMode | str) -> BaseRunModePolicy:
    return BASE_RUN_MODE_POLICIES[_run_mode(value)]


def _run_mode(value: BaseRunMode | str) -> BaseRunMode:
    if not isinstance(value, str):
        raise BaseWorkflowInputError("run_mode must be a string.")
    resolved = value.strip()
    if resolved != value or resolved not in BASE_RUN_MODE_POLICIES:
        expected = ", ".join(BASE_RUN_MODES)
        raise BaseWorkflowInputError(
            f"Unsupported run_mode {value!r}; expected one of: {expected}."
        )
    return cast(BaseRunMode, resolved)


def _validate_run_name_for_mode(*, run_mode: BaseRunMode, run_name: str) -> None:
    if not isinstance(run_name, str):
        raise BaseWorkflowInputError("run_name must be a string.")
    lowered = run_name.casefold()
    if run_mode == "complete" and ("smoke" in lowered or "check" in lowered):
        raise BaseWorkflowInputError(
            "complete run_mode run_name must not contain 'smoke' or 'check'."
        )
    if run_mode == "check" and "smoke" in lowered:
        raise BaseWorkflowInputError("check run_mode run_name must not contain 'smoke'.")


def _validate_processed_training_inputs(
    config: BaselineTrainingConfig,
    layout: ProjectLayout,
) -> None:
    artifact_layout = DatasetArtifactLayout(layout)
    require_file(config.data.train_manifest, label="Processed train manifest")
    require_file(config.data.val_manifest, label="Processed validation manifest")
    require_dir_contains(
        artifact_layout.processed_samples_split_root(config.data.train_split),
        "*.npz",
        label="Processed train samples",
    )
    require_dir_contains(
        artifact_layout.processed_samples_split_root(config.data.val_split),
        "*.npz",
        label="Processed validation samples",
    )
    read_processed_modeling_manifest(
        config.data.train_manifest,
        split=config.data.train_split,
        data_root=layout.data_root,
    )
    read_processed_modeling_manifest(
        config.data.val_manifest,
        split=config.data.val_split,
        data_root=layout.data_root,
    )


def _validate_processed_prediction_inputs(
    config: BaselineTrainingConfig,
    layout: ProjectLayout,
    *,
    prediction_splits: tuple[str, ...] | None,
) -> None:
    artifact_layout = DatasetArtifactLayout(layout)
    for split in _prediction_splits(prediction_splits, config):
        manifest_path = artifact_layout.processed_manifest_path(split)
        require_file(manifest_path, label=f"Processed {split} prediction manifest")
        require_dir_contains(
            artifact_layout.processed_samples_split_root(split),
            "*.npz",
            label=f"Processed {split} prediction samples",
        )
        read_processed_modeling_manifest(
            manifest_path,
            split=split,
            data_root=layout.data_root,
        )


def _prediction_splits(
    configured_splits: tuple[str, ...] | None,
    config: BaselineTrainingConfig,
) -> tuple[str, ...]:
    values = config.data.prediction_splits if configured_splits is None else configured_splits
    if not values:
        raise BaseWorkflowInputError("At least one prediction split is required.")
    return tuple(_validate_split(split) for split in values)


def _ordered_unique_splits(splits: Sequence[str]) -> tuple[str, ...]:
    requested = {_validate_split(split) for split in splits}
    return tuple(split for split in BASE_ALLOWED_SPLITS if split in requested)


def _validate_split(split: str) -> str:
    if split not in BASE_ALLOWED_SPLITS:
        expected = ", ".join(BASE_ALLOWED_SPLITS)
        raise BaseWorkflowInputError(
            f"Unsupported split {split!r}; expected values from: {expected}."
        )
    return split


def _output_policy(value: OutputExistsPolicy | str) -> OutputExistsPolicy:
    try:
        return value if isinstance(value, OutputExistsPolicy) else OutputExistsPolicy(value)
    except ValueError as exc:
        expected = ", ".join(policy.value for policy in OutputExistsPolicy)
        raise BaseWorkflowInputError(
            f"Unsupported output_policy {value!r}; expected one of: {expected}."
        ) from exc


def _positive_optional_limit(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{label} must be positive when set.")
    return value


__all__ = ["validate_base_inputs"]
