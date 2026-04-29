"""Notebook-facing baseline training workflow over canonical processed files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text_to_sign_production.core.files import (
    OutputExistsPolicy,
    prepare_output_dir,
    require_dir_contains,
    require_file,
)
from text_to_sign_production.core.paths import DEFAULT_REPO_ROOT, ProjectLayout, resolve_repo_path
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH
from text_to_sign_production.modeling.data import (
    ProcessedModelingDataError,
    read_processed_modeling_manifest,
)
from text_to_sign_production.modeling.inference.qualitative import (
    DEFAULT_QUALITATIVE_PANEL_SIZE,
    QualitativeExportResult,
    export_qualitative_panel,
)
from text_to_sign_production.modeling.training.config import (
    BaselineTrainingConfig,
    BaselineTrainingConfigError,
    load_baseline_training_config,
)
from text_to_sign_production.modeling.training.train import (
    BaselineTrainingRunResult,
    run_baseline_training,
)


@dataclass(frozen=True, slots=True)
class BaseWorkflowConfig:
    project_root: Path | str | None = None
    run_name: str = "base_run"
    config_path: Path | str | None = None
    run_qualitative_export: bool = False
    panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE
    validate_processed_inputs: bool = True
    output_policy: OutputExistsPolicy = OutputExistsPolicy.FAIL


@dataclass(frozen=True, slots=True)
class BaseWorkflowResult:
    config: BaselineTrainingConfig
    training: BaselineTrainingRunResult
    qualitative: QualitativeExportResult | None
    run_root: Path


class BaseWorkflowError(RuntimeError):
    """Raised when baseline workflow orchestration fails."""


class BaseWorkflowInputError(ValueError):
    """Raised when baseline workflow inputs are missing or invalid."""


def validate_base_inputs(config: BaseWorkflowConfig) -> None:
    """Validate baseline workflow inputs without constructing a model."""

    layout = _layout_from_config(config)
    config_path = _config_path(config, layout)
    if config.panel_size <= 0:
        raise BaseWorkflowInputError("panel_size must be positive.")

    try:
        require_file(config_path, label="Baseline config")
        training_config = load_baseline_training_config(
            config_path,
            validate_paths=config.validate_processed_inputs,
            checkpoint_output_dir=_checkpoint_output_dir(layout, config.run_name),
            repo_root=layout.root,
        )
        if config.validate_processed_inputs:
            _validate_processed_training_inputs(training_config, layout)
    except (
        BaselineTrainingConfigError,
        FileNotFoundError,
        IsADirectoryError,
        ProcessedModelingDataError,
        ValueError,
    ) as exc:
        raise BaseWorkflowInputError(str(exc)) from exc


def run_base_workflow(config: BaseWorkflowConfig) -> BaseWorkflowResult:
    """Run baseline training and optionally export a qualitative validation panel."""

    validate_base_inputs(config)
    layout = _layout_from_config(config)
    config_path = _config_path(config, layout)
    run_root = prepare_output_dir(
        layout.base_run_root(config.run_name),
        policy=config.output_policy,
        label="Base run root",
    )
    checkpoint_output_dir = run_root / "checkpoints"
    qualitative_output_dir = run_root / "qualitative"

    try:
        training_config = load_baseline_training_config(
            config_path,
            validate_paths=config.validate_processed_inputs,
            checkpoint_output_dir=checkpoint_output_dir,
            repo_root=layout.root,
        )
        training_result = run_baseline_training(
            config_path,
            checkpoint_output_dir=checkpoint_output_dir,
            repo_root=layout.root,
        )

        qualitative_result: QualitativeExportResult | None = None
        if config.run_qualitative_export:
            qualitative_result = export_qualitative_panel(
                config_path,
                output_dir=qualitative_output_dir,
                checkpoint_path=(
                    training_result.best_checkpoint_path or training_result.last_checkpoint_path
                ),
                run_summary_path=training_result.summary_path,
                panel_size=config.panel_size,
                repo_root=layout.root,
            )
    except BaseWorkflowInputError:
        raise
    except Exception as exc:
        raise BaseWorkflowError(f"Baseline workflow failed: {exc}") from exc

    return BaseWorkflowResult(
        config=training_config,
        training=training_result,
        qualitative=qualitative_result,
        run_root=run_root,
    )


def _layout_from_config(config: BaseWorkflowConfig) -> ProjectLayout:
    root = DEFAULT_REPO_ROOT if config.project_root is None else Path(config.project_root)
    return ProjectLayout(root)


def _config_path(config: BaseWorkflowConfig, layout: ProjectLayout) -> Path:
    configured = config.config_path or DEFAULT_BASELINE_CONFIG_PATH
    return resolve_repo_path(configured, repo_root=layout.root)


def _checkpoint_output_dir(layout: ProjectLayout, run_name: str) -> Path:
    return layout.base_run_root(run_name) / "checkpoints"


def _validate_processed_training_inputs(
    config: BaselineTrainingConfig,
    layout: ProjectLayout,
) -> None:
    require_file(config.data.train_manifest, label="Processed train manifest")
    require_file(config.data.val_manifest, label="Processed validation manifest")
    require_dir_contains(
        layout.processed_samples_split_root(config.data.train_split),
        "*.npz",
        label="Processed train samples",
    )
    require_dir_contains(
        layout.processed_samples_split_root(config.data.val_split),
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


__all__ = [
    "BaseWorkflowConfig",
    "BaseWorkflowError",
    "BaseWorkflowInputError",
    "BaseWorkflowResult",
    "run_base_workflow",
    "validate_base_inputs",
]
