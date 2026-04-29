"""Base workflow run layout and path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from text_to_sign_production.core.paths import ProjectLayout, resolve_repo_path
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH
from text_to_sign_production.workflows._base.constants import (
    BASE_RUN_SUMMARY_FILENAME,
    BASELINE_CONFIG_COPY_FILENAME,
    BASELINE_REPORT_JSON_FILENAME,
    BASELINE_REPORT_MARKDOWN_FILENAME,
    FAILURE_MODES_JSON_FILENAME,
    FAILURE_MODES_MARKDOWN_FILENAME,
)
from text_to_sign_production.workflows._base.types import (
    BaseRunLayout,
    BaseWorkflowConfig,
    BaseWorkflowError,
    BaseWorkflowInputError,
)


def build_base_run_layout(
    *,
    project_root: Path | str,
    run_name: str = "smoke_run",
) -> BaseRunLayout:
    """Build the canonical M0 run-root layout without touching the filesystem."""

    layout = ProjectLayout(Path(project_root))
    run_root = layout.base_m0_run_root(run_name)
    config_dir = run_root / "config"
    training_dir = run_root / "training"
    checkpoints_dir = run_root / "checkpoints"
    predictions_dir = run_root / "predictions"
    reports_dir = run_root / "reports"
    qualitative_dir = run_root / "qualitative"
    return BaseRunLayout(
        project_root=layout.root,
        run_name=run_name,
        run_root=run_root,
        config_dir=config_dir,
        config_snapshot_path=config_dir / "config.json",
        baseline_config_copy_path=config_dir / BASELINE_CONFIG_COPY_FILENAME,
        training_dir=training_dir,
        training_metrics_path=training_dir / "metrics.jsonl",
        training_live_log_path=training_dir / "live.log",
        training_summary_path=training_dir / "summary.json",
        checkpoints_dir=checkpoints_dir,
        best_checkpoint_path=checkpoints_dir / "best.pt",
        last_checkpoint_path=checkpoints_dir / "last.pt",
        predictions_dir=predictions_dir,
        reports_dir=reports_dir,
        baseline_report_json_path=reports_dir / BASELINE_REPORT_JSON_FILENAME,
        baseline_report_markdown_path=reports_dir / BASELINE_REPORT_MARKDOWN_FILENAME,
        failure_modes_json_path=reports_dir / FAILURE_MODES_JSON_FILENAME,
        failure_modes_markdown_path=reports_dir / FAILURE_MODES_MARKDOWN_FILENAME,
        qualitative_dir=qualitative_dir,
        run_summary_path=run_root / BASE_RUN_SUMMARY_FILENAME,
    )


def _layout_from_config(config: BaseWorkflowConfig) -> ProjectLayout:
    return ProjectLayout(_required_project_root(config.project_root))


def _config_path(config: BaseWorkflowConfig, layout: ProjectLayout) -> Path:
    configured = config.config_path or DEFAULT_BASELINE_CONFIG_PATH
    return resolve_repo_path(configured, repo_root=layout.root)


def _required_project_root(value: Path | str) -> Path:
    if value is None:
        raise BaseWorkflowInputError("project_root is required.")
    return Path(value)


def _resolve_manifest_or_repo_path(path: str, *, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise BaseWorkflowError(
            f"Prediction manifest path must be project-root-relative, got absolute path: {path}"
        )
    return resolve_repo_path(candidate, repo_root=project_root)


__all__ = ["build_base_run_layout"]
