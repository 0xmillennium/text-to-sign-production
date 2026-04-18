"""Stage-oriented orchestration for Sprint 3 baseline modeling."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml

from text_to_sign_production.data.constants import COLAB_DRIVE_PROJECT_ROOT, REPO_ROOT
from text_to_sign_production.data.utils import ensure_directory, utc_timestamp, write_json
from text_to_sign_production.modeling.config import DEFAULT_BASELINE_CONFIG_PATH
from text_to_sign_production.modeling.data import read_processed_modeling_manifest
from text_to_sign_production.modeling.inference.qualitative import DEFAULT_QUALITATIVE_PANEL_SIZE
from text_to_sign_production.modeling.training.config import load_baseline_training_config
from text_to_sign_production.ops.archive_ops import (
    create_tar_zst_archive,
    extract_tar_zst_with_progress,
)

BaselineModelingStep = Literal["prepare", "train", "export-panel", "package", "all"]
BaselineStepName = Literal["prepare", "train", "export-panel", "package"]
BaselineStepAction = Literal["prepared", "reuse_extracted", "extract_archive", "run_step"]

BASELINE_MODELING_RECORD_SCHEMA_VERSION = "t2sp-baseline-modeling-package-v1"
DEFAULT_BASELINE_RUN_NAME = "baseline-default"
LOCAL_BASELINE_ARTIFACT_RUNS_ROOT = (
    REPO_ROOT / "outputs" / "modeling" / "baseline-modeling" / "runs"
)
COLAB_BASELINE_ARTIFACT_RUNS_ROOT = (
    COLAB_DRIVE_PROJECT_ROOT / "artifacts" / "baseline-modeling" / "runs"
)

TRAINING_ARCHIVE_NAME = "baseline_training_outputs.tar.zst"
QUALITATIVE_ARCHIVE_NAME = "baseline_qualitative_outputs.tar.zst"
RECORD_ARCHIVE_NAME = "baseline_record_package.tar.zst"
SOURCE_BASELINE_CONFIG_NAME = "source_baseline.yaml"

_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class BaselineTrainingRunner(Protocol):
    """Callable surface used by the workflow to run baseline training."""

    def __call__(
        self,
        config_path: Path,
        /,
        *,
        checkpoint_output_dir: Path | None = None,
    ) -> Any:
        """Run training and return the lower-level training result."""


class QualitativeExportRunner(Protocol):
    """Callable surface used by the workflow to export qualitative artifacts."""

    def __call__(
        self,
        config_path: Path,
        /,
        *,
        output_dir: Path,
        checkpoint_path: Path | None = None,
        panel_definition_path: Path | None = None,
        run_summary_path: Path | None = None,
        panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    ) -> Any:
        """Run qualitative export and return the lower-level export result."""


@dataclass(frozen=True, slots=True)
class BaselineRunLayout:
    """Stable Sprint 3 run layout under a baseline-modeling artifact root."""

    run_name: str
    artifact_runs_root: Path

    @property
    def run_root(self) -> Path:
        return self.artifact_runs_root / self.run_name

    @property
    def config_dir(self) -> Path:
        return self.run_root / "config"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.run_root / "metrics"

    @property
    def qualitative_dir(self) -> Path:
        return self.run_root / "qualitative"

    @property
    def record_dir(self) -> Path:
        return self.run_root / "record"

    @property
    def archives_dir(self) -> Path:
        return self.run_root / "archives"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "baseline.yaml"

    @property
    def source_config_path(self) -> Path:
        return self.config_dir / SOURCE_BASELINE_CONFIG_NAME

    @property
    def checkpoint_run_summary_path(self) -> Path:
        return self.checkpoints_dir / "run_summary.json"

    @property
    def metrics_run_summary_path(self) -> Path:
        return self.metrics_dir / "run_summary.json"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "best.pt"

    @property
    def panel_definition_path(self) -> Path:
        return self.qualitative_dir / "panel_definition.json"

    @property
    def qualitative_records_path(self) -> Path:
        return self.qualitative_dir / "records.jsonl"

    @property
    def panel_summary_path(self) -> Path:
        return self.qualitative_dir / "panel_summary.json"

    @property
    def evidence_bundle_path(self) -> Path:
        return self.qualitative_dir / "baseline_evidence_bundle.json"

    @property
    def package_manifest_path(self) -> Path:
        return self.record_dir / "baseline_modeling_package.json"

    @property
    def record_evidence_bundle_path(self) -> Path:
        return self.record_dir / "baseline_evidence_bundle.json"

    @property
    def record_run_summary_path(self) -> Path:
        return self.record_dir / "run_summary.json"

    @property
    def training_archive_path(self) -> Path:
        return self.archives_dir / TRAINING_ARCHIVE_NAME

    @property
    def qualitative_archive_path(self) -> Path:
        return self.archives_dir / QUALITATIVE_ARCHIVE_NAME

    @property
    def record_archive_path(self) -> Path:
        return self.archives_dir / RECORD_ARCHIVE_NAME


@dataclass(frozen=True, slots=True)
class BaselineModelingStepResult:
    """Result for one resolved baseline-modeling workflow step."""

    step: BaselineStepName
    action: BaselineStepAction
    output_path: Path
    archive_path: Path | None
    paths: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class BaselineModelingWorkflowResult:
    """Summary of one public baseline-modeling workflow invocation."""

    run_name: str
    layout: BaselineRunLayout
    steps: tuple[BaselineModelingStepResult, ...]


@dataclass(frozen=True, slots=True)
class _ResumeDecision:
    step: BaselineStepName
    action: Literal["reuse_extracted", "extract_archive", "run_step"]
    output_path: Path
    archive_path: Path
    required_paths: tuple[Path, ...]


def resolve_baseline_run_layout(
    *,
    run_name: str = DEFAULT_BASELINE_RUN_NAME,
    artifact_runs_root: Path | str | None = None,
) -> BaselineRunLayout:
    """Resolve and validate the stable artifact layout for a Sprint 3 baseline run."""

    validated_run_name = validate_baseline_run_name(run_name)
    root = (
        LOCAL_BASELINE_ARTIFACT_RUNS_ROOT
        if artifact_runs_root is None
        else Path(artifact_runs_root)
    )
    return BaselineRunLayout(
        run_name=validated_run_name,
        artifact_runs_root=root.expanduser().resolve(),
    )


def validate_baseline_run_name(run_name: str) -> str:
    """Validate a run name that is safe to use as one directory segment."""

    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("Baseline run_name must be a non-empty string.")
    resolved = run_name.strip()
    if resolved != run_name:
        raise ValueError("Baseline run_name must not have leading or trailing whitespace.")
    if not _RUN_NAME_PATTERN.fullmatch(resolved):
        raise ValueError(
            "Baseline run_name may contain only letters, numbers, dots, underscores, "
            "and hyphens, and must start with a letter or number."
        )
    return resolved


def run_baseline_modeling(
    *,
    step: BaselineModelingStep = "all",
    config_path: Path | str = DEFAULT_BASELINE_CONFIG_PATH,
    run_name: str = DEFAULT_BASELINE_RUN_NAME,
    artifact_runs_root: Path | str | None = None,
    panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    training_runner: BaselineTrainingRunner | None = None,
    qualitative_runner: QualitativeExportRunner | None = None,
) -> BaselineModelingWorkflowResult:
    """Run the public Sprint 3 baseline-modeling workflow with archive-aware resume."""

    if step not in {"prepare", "train", "export-panel", "package", "all"}:
        raise ValueError(f"Unsupported baseline-modeling step: {step!r}")
    if panel_size <= 0:
        raise ValueError("panel_size must be positive.")

    layout = resolve_baseline_run_layout(
        run_name=run_name,
        artifact_runs_root=artifact_runs_root,
    )
    resolved_config_path = Path(config_path).expanduser().resolve()
    steps: list[BaselineModelingStepResult] = []
    steps.append(
        prepare_baseline_modeling_run(
            resolved_config_path,
            layout=layout,
            validate_processed_inputs=(
                step == "prepare" or not _run_config_matches(resolved_config_path, layout)
            ),
        )
    )

    if step == "prepare":
        return BaselineModelingWorkflowResult(
            run_name=layout.run_name,
            layout=layout,
            steps=tuple(steps),
        )

    steps.append(
        ensure_baseline_training_outputs(
            layout=layout,
            training_runner=training_runner,
        )
    )

    if step == "train":
        return BaselineModelingWorkflowResult(
            run_name=layout.run_name,
            layout=layout,
            steps=tuple(steps),
        )

    steps.append(
        ensure_baseline_qualitative_outputs(
            layout=layout,
            panel_size=panel_size,
            qualitative_runner=qualitative_runner,
        )
    )

    if step == "export-panel":
        return BaselineModelingWorkflowResult(
            run_name=layout.run_name,
            layout=layout,
            steps=tuple(steps),
        )

    steps.append(ensure_baseline_record_package(layout=layout))
    return BaselineModelingWorkflowResult(
        run_name=layout.run_name,
        layout=layout,
        steps=tuple(steps),
    )


def prepare_baseline_modeling_run(
    config_path: Path | str,
    *,
    layout: BaselineRunLayout,
    validate_processed_inputs: bool = True,
) -> BaselineModelingStepResult:
    """Create the run layout, validate processed inputs, and copy the baseline config."""

    source_config_path = Path(config_path).expanduser().resolve()
    if validate_processed_inputs:
        validate_baseline_processed_inputs(source_config_path)
    for directory in (
        layout.config_dir,
        layout.checkpoints_dir,
        layout.metrics_dir,
        layout.qualitative_dir,
        layout.record_dir,
        layout.archives_dir,
    ):
        ensure_directory(directory)
    _write_run_configs_once(source_config_path, layout)
    return BaselineModelingStepResult(
        step="prepare",
        action="prepared",
        output_path=layout.run_root,
        archive_path=None,
        paths={
            "config_path": layout.config_path,
            "source_config_path": layout.source_config_path,
        },
    )


def validate_baseline_processed_inputs(config_path: Path | str) -> None:
    """Validate that the baseline config points at processed Dataset Build outputs only."""

    config = load_baseline_training_config(config_path, validate_paths=True)
    train_records = read_processed_modeling_manifest(
        config.data.train_manifest,
        split=config.data.train_split,
    )
    val_records = read_processed_modeling_manifest(
        config.data.val_manifest,
        split=config.data.val_split,
    )
    if not train_records:
        raise ValueError(f"Baseline train manifest has no records: {config.data.train_manifest}")
    if not val_records:
        raise ValueError(f"Baseline val manifest has no records: {config.data.val_manifest}")


def resolve_training_resume_decision(layout: BaselineRunLayout) -> _ResumeDecision:
    """Resolve whether training outputs should be reused, extracted, or produced."""

    return _resolve_resume_decision(
        step="train",
        output_path=layout.checkpoints_dir,
        archive_path=layout.training_archive_path,
        required_paths=(
            layout.config_path,
            layout.source_config_path,
            layout.checkpoint_run_summary_path,
            layout.last_checkpoint_path,
        ),
    )


def resolve_qualitative_resume_decision(layout: BaselineRunLayout) -> _ResumeDecision:
    """Resolve whether qualitative outputs should be reused, extracted, or produced."""

    return _resolve_resume_decision(
        step="export-panel",
        output_path=layout.qualitative_dir,
        archive_path=layout.qualitative_archive_path,
        required_paths=(
            layout.panel_definition_path,
            layout.qualitative_records_path,
            layout.panel_summary_path,
            layout.evidence_bundle_path,
        ),
    )


def resolve_record_resume_decision(layout: BaselineRunLayout) -> _ResumeDecision:
    """Resolve whether record/package outputs should be reused, extracted, or produced."""

    return _resolve_resume_decision(
        step="package",
        output_path=layout.record_dir,
        archive_path=layout.record_archive_path,
        required_paths=(
            layout.package_manifest_path,
            layout.record_evidence_bundle_path,
            layout.record_run_summary_path,
        ),
    )


def ensure_baseline_training_outputs(
    *,
    layout: BaselineRunLayout,
    training_runner: BaselineTrainingRunner | None = None,
) -> BaselineModelingStepResult:
    """Ensure baseline training outputs exist under the run layout."""

    decision = resolve_training_resume_decision(layout)
    if decision.action == "reuse_extracted":
        _sync_training_metrics(layout)
        return _step_result_from_decision(decision)
    if decision.action == "extract_archive":
        _extract_archive_into_run_root(
            decision.archive_path,
            run_root=layout.run_root,
            label="[baseline train] Extract archived outputs",
        )
        _require_decision_outputs(decision)
        _sync_training_metrics(layout)
        return _step_result_from_decision(decision)

    runner = training_runner if training_runner is not None else _load_training_runner()
    runner(layout.config_path, checkpoint_output_dir=layout.checkpoints_dir)
    _sync_training_metrics(layout)
    _require_decision_outputs(decision)
    _archive_training_outputs(layout)
    return _step_result_from_decision(decision)


def ensure_baseline_qualitative_outputs(
    *,
    layout: BaselineRunLayout,
    panel_size: int = DEFAULT_QUALITATIVE_PANEL_SIZE,
    qualitative_runner: QualitativeExportRunner | None = None,
) -> BaselineModelingStepResult:
    """Ensure baseline qualitative panel outputs exist under the run layout."""

    decision = resolve_qualitative_resume_decision(layout)
    if decision.action == "reuse_extracted":
        return _step_result_from_decision(decision)
    if decision.action == "extract_archive":
        _extract_archive_into_run_root(
            decision.archive_path,
            run_root=layout.run_root,
            label="[baseline qualitative] Extract archived outputs",
        )
        _require_decision_outputs(decision)
        return _step_result_from_decision(decision)

    runner = qualitative_runner if qualitative_runner is not None else _load_qualitative_runner()
    runner(
        layout.config_path,
        output_dir=layout.qualitative_dir,
        checkpoint_path=_select_export_checkpoint(layout),
        run_summary_path=layout.checkpoint_run_summary_path,
        panel_size=panel_size,
    )
    _require_decision_outputs(decision)
    _archive_qualitative_outputs(layout)
    return _step_result_from_decision(decision)


def ensure_baseline_record_package(
    *,
    layout: BaselineRunLayout,
) -> BaselineModelingStepResult:
    """Ensure runtime-side package/evidence outputs exist under the run layout."""

    decision = resolve_record_resume_decision(layout)
    if decision.action == "reuse_extracted":
        return _step_result_from_decision(decision)
    if decision.action == "extract_archive":
        _extract_archive_into_run_root(
            decision.archive_path,
            run_root=layout.run_root,
            label="[baseline record] Extract archived package",
        )
        _require_decision_outputs(decision)
        return _step_result_from_decision(decision)

    _write_record_package(layout)
    _require_decision_outputs(decision)
    _archive_record_package(layout)
    return _step_result_from_decision(decision)


def _load_training_runner() -> BaselineTrainingRunner:
    from text_to_sign_production.modeling.training.train import run_baseline_training

    return run_baseline_training


def _load_qualitative_runner() -> QualitativeExportRunner:
    from text_to_sign_production.modeling.inference.qualitative import export_qualitative_panel

    return export_qualitative_panel


def _resolve_resume_decision(
    *,
    step: BaselineStepName,
    output_path: Path,
    archive_path: Path,
    required_paths: tuple[Path, ...],
) -> _ResumeDecision:
    if all(path.is_file() for path in required_paths):
        action: Literal["reuse_extracted", "extract_archive", "run_step"] = "reuse_extracted"
    elif archive_path.is_file():
        action = "extract_archive"
    else:
        action = "run_step"
    return _ResumeDecision(
        step=step,
        action=action,
        output_path=output_path,
        archive_path=archive_path,
        required_paths=required_paths,
    )


def _require_decision_outputs(decision: _ResumeDecision) -> None:
    missing_paths = [path for path in decision.required_paths if not path.is_file()]
    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Baseline workflow step {decision.step!r} did not produce required file outputs:\n"
            f"{formatted}"
        )


def _step_result_from_decision(decision: _ResumeDecision) -> BaselineModelingStepResult:
    return BaselineModelingStepResult(
        step=decision.step,
        action=decision.action,
        output_path=decision.output_path,
        archive_path=decision.archive_path if decision.archive_path.is_file() else None,
        paths={},
    )


def _run_config_matches(source: Path, layout: BaselineRunLayout) -> bool:
    if (
        not source.is_file()
        or not layout.config_path.is_file()
        or not layout.source_config_path.is_file()
    ):
        return False
    return (
        layout.source_config_path.read_bytes() == source.read_bytes()
        and layout.config_path.read_bytes() == _effective_config_bytes(source, layout)
    )


def _write_run_configs_once(source: Path, layout: BaselineRunLayout) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Baseline config does not exist: {source}")
    source_config_bytes = source.read_bytes()
    effective_config_bytes = _effective_config_bytes(source, layout)
    ensure_directory(layout.config_dir)

    if layout.source_config_path.exists():
        if not layout.source_config_path.is_file():
            raise ValueError(
                f"Baseline source config path exists and is not a file: {layout.source_config_path}"
            )
        if layout.source_config_path.read_bytes() != source_config_bytes:
            raise ValueError(
                "Baseline run source config already exists with different content. "
                f"Use a new run_name or remove the existing run directory: "
                f"{layout.source_config_path}"
            )
    else:
        layout.source_config_path.write_bytes(source_config_bytes)

    if layout.config_path.exists():
        if not layout.config_path.is_file():
            raise ValueError(
                f"Baseline run config path exists and is not a file: {layout.config_path}"
            )
        current_config_bytes = layout.config_path.read_bytes()
        if current_config_bytes == effective_config_bytes:
            return
        if current_config_bytes == source_config_bytes:
            layout.config_path.write_bytes(effective_config_bytes)
            return
        raise ValueError(
            "Baseline run config already exists with different content. "
            f"Use a new run_name or remove the existing run directory: {layout.config_path}"
        )
    layout.config_path.write_bytes(effective_config_bytes)


def _effective_config_bytes(source: Path, layout: BaselineRunLayout) -> bytes:
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse baseline config YAML: {source}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Baseline config root must be a mapping: {source}")
    effective_config = deepcopy(cast(dict[str, Any], loaded))
    checkpoint_config = effective_config.get("checkpoint")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Baseline config checkpoint section must be a mapping: {source}")
    checkpoint_config["output_dir"] = layout.checkpoints_dir.as_posix()
    return yaml.safe_dump(effective_config, sort_keys=False).encode("utf-8")


def _sync_training_metrics(layout: BaselineRunLayout) -> None:
    if not layout.checkpoint_run_summary_path.is_file():
        raise FileNotFoundError(
            "Baseline training did not write run_summary.json: "
            f"{layout.checkpoint_run_summary_path}"
        )
    ensure_directory(layout.metrics_dir)
    shutil.copy2(layout.checkpoint_run_summary_path, layout.metrics_run_summary_path)


def _select_export_checkpoint(layout: BaselineRunLayout) -> Path:
    if layout.best_checkpoint_path.is_file():
        return layout.best_checkpoint_path
    if layout.last_checkpoint_path.is_file():
        return layout.last_checkpoint_path
    raise FileNotFoundError(
        "No baseline checkpoint is available for qualitative export. "
        f"Expected {layout.best_checkpoint_path} or {layout.last_checkpoint_path}."
    )


def _archive_training_outputs(layout: BaselineRunLayout) -> Path:
    return _create_baseline_archive(
        archive_path=layout.training_archive_path,
        members=(layout.config_dir, layout.checkpoints_dir, layout.metrics_dir),
        cwd=layout.run_root,
        label="[baseline train] Archive outputs",
        artifact_description="baseline training outputs",
    )


def _archive_qualitative_outputs(layout: BaselineRunLayout) -> Path:
    return _create_baseline_archive(
        archive_path=layout.qualitative_archive_path,
        members=(layout.qualitative_dir,),
        cwd=layout.run_root,
        label="[baseline qualitative] Archive outputs",
        artifact_description="baseline qualitative outputs",
    )


def _archive_record_package(layout: BaselineRunLayout) -> Path:
    return _create_baseline_archive(
        archive_path=layout.record_archive_path,
        members=(layout.record_dir,),
        cwd=layout.run_root,
        label="[baseline record] Archive package",
        artifact_description="baseline record package",
    )


def _create_baseline_archive(
    *,
    archive_path: Path,
    members: tuple[Path, ...],
    cwd: Path,
    label: str,
    artifact_description: str,
) -> Path:
    _require_baseline_archive_members(
        members=members,
        cwd=cwd,
        artifact_description=artifact_description,
    )
    try:
        return create_tar_zst_archive(
            archive_path=archive_path,
            members=members,
            cwd=cwd,
            label=label,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            _baseline_archive_missing_members_message(
                members=members,
                cwd=cwd,
                artifact_description=artifact_description,
            )
        ) from exc


def _require_baseline_archive_members(
    *,
    members: tuple[Path, ...],
    cwd: Path,
    artifact_description: str,
) -> None:
    missing_members = _missing_baseline_archive_members(members=members, cwd=cwd)
    if missing_members:
        formatted = "\n".join(f"- {member}" for member in missing_members)
        raise FileNotFoundError(
            f"Cannot archive {artifact_description}; missing required baseline-modeling "
            f"run artifacts:\n{formatted}"
        )


def _baseline_archive_missing_members_message(
    *,
    members: tuple[Path, ...],
    cwd: Path,
    artifact_description: str,
) -> str:
    missing_members = _missing_baseline_archive_members(members=members, cwd=cwd)
    if missing_members:
        formatted = "\n".join(f"- {member}" for member in missing_members)
        return (
            f"Cannot archive {artifact_description}; missing required baseline-modeling "
            f"run artifacts:\n{formatted}"
        )
    formatted = "\n".join(f"- {member}" for member in members)
    return (
        f"Cannot archive {artifact_description}; baseline-modeling archive member "
        f"validation failed while reading run artifacts under {cwd}. Expected members:\n"
        f"{formatted}"
    )


def _missing_baseline_archive_members(
    *,
    members: tuple[Path, ...],
    cwd: Path,
) -> tuple[Path, ...]:
    return tuple(
        member for member in members if not _baseline_archive_member_path(member, cwd).exists()
    )


def _baseline_archive_member_path(member: Path, cwd: Path) -> Path:
    return member if member.is_absolute() else cwd / member


def _extract_archive_into_run_root(archive_path: Path, *, run_root: Path, label: str) -> None:
    ensure_directory(run_root)
    with tempfile.TemporaryDirectory(
        prefix=f".{archive_path.name}.extract-",
        dir=run_root.parent,
    ) as temp_root:
        extraction_destination = Path(temp_root) / "extracted"
        extract_tar_zst_with_progress(archive_path, extraction_destination, label=label)
        if not extraction_destination.is_dir():
            raise RuntimeError(f"Archive extraction did not create outputs: {archive_path}")
        extracted_members = sorted(extraction_destination.iterdir(), key=lambda path: path.name)
        if not extracted_members:
            raise RuntimeError(f"Archive did not contain any outputs: {archive_path}")
        _validate_extracted_config_compatible(
            extracted_config_dir=extraction_destination / "config",
            current_config_dir=run_root / "config",
        )
        for source in extracted_members:
            _move_extracted_member_into_run_root(source, run_root=run_root)


def _move_extracted_member_into_run_root(source: Path, *, run_root: Path) -> None:
    destination = run_root / source.name
    if source.name == "config" and destination.exists():
        return
    if source.name == "archives" and destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"Baseline archives path exists and is not a directory: {destination}"
            )
        return
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.move(str(source), str(destination))


def _validate_extracted_config_compatible(
    *,
    extracted_config_dir: Path,
    current_config_dir: Path,
) -> None:
    if not extracted_config_dir.exists():
        return
    if not extracted_config_dir.is_dir():
        raise RuntimeError(
            f"Archived baseline config member is not a directory: {extracted_config_dir}"
        )
    if not current_config_dir.exists():
        return
    if not current_config_dir.is_dir():
        raise FileExistsError(
            f"Baseline config path exists and is not a directory: {current_config_dir}"
        )

    current_source_config = current_config_dir / SOURCE_BASELINE_CONFIG_NAME
    archived_source_config = extracted_config_dir / SOURCE_BASELINE_CONFIG_NAME
    if not current_source_config.is_file() or not archived_source_config.is_file():
        raise ValueError(
            "Cannot extract archived baseline config over an existing config directory "
            f"because {SOURCE_BASELINE_CONFIG_NAME} is missing. Use a new run_name or "
            f"repair the run directory: {current_config_dir}"
        )
    if current_source_config.read_bytes() != archived_source_config.read_bytes():
        raise ValueError(
            "Archived baseline source config differs from the current run config. "
            "Use a new run_name for a different config or remove the existing run "
            f"directory: {current_config_dir}"
        )


def _write_record_package(layout: BaselineRunLayout) -> None:
    _require_package_inputs(layout)
    ensure_directory(layout.record_dir)
    shutil.copy2(layout.evidence_bundle_path, layout.record_evidence_bundle_path)
    shutil.copy2(layout.metrics_run_summary_path, layout.record_run_summary_path)
    write_json(layout.package_manifest_path, _record_package_payload(layout))


def _require_package_inputs(layout: BaselineRunLayout) -> None:
    required_paths = (
        layout.config_path,
        layout.source_config_path,
        layout.metrics_run_summary_path,
        layout.last_checkpoint_path,
        layout.panel_definition_path,
        layout.qualitative_records_path,
        layout.panel_summary_path,
        layout.evidence_bundle_path,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            f"Cannot package baseline run; missing required file outputs:\n{formatted}"
        )


def _record_package_payload(layout: BaselineRunLayout) -> dict[str, Any]:
    run_summary = _read_json_mapping(layout.metrics_run_summary_path)
    best_checkpoint_path = (
        layout.best_checkpoint_path.as_posix() if layout.best_checkpoint_path.is_file() else None
    )
    return {
        "schema_version": BASELINE_MODELING_RECORD_SCHEMA_VERSION,
        "artifact_role": "runtime-side baseline evidence, not a formal experiment record",
        "created_at": utc_timestamp(),
        "run_name": layout.run_name,
        "run_root": layout.run_root.as_posix(),
        "config": {
            "path": layout.config_path.as_posix(),
            "source_path": layout.source_config_path.as_posix(),
            "backbone_name": run_summary.get("backbone_name"),
            "target_channels": run_summary.get("target_channels"),
        },
        "training": {
            "run_summary_path": layout.metrics_run_summary_path.as_posix(),
            "checkpoint_run_summary_path": layout.checkpoint_run_summary_path.as_posix(),
            "last_checkpoint_path": layout.last_checkpoint_path.as_posix(),
            "best_checkpoint_path": best_checkpoint_path,
        },
        "qualitative": {
            "panel_definition_path": layout.panel_definition_path.as_posix(),
            "records_path": layout.qualitative_records_path.as_posix(),
            "panel_summary_path": layout.panel_summary_path.as_posix(),
            "evidence_bundle_path": layout.evidence_bundle_path.as_posix(),
        },
        "record": {
            "package_manifest_path": layout.package_manifest_path.as_posix(),
            "copied_evidence_bundle_path": layout.record_evidence_bundle_path.as_posix(),
            "copied_run_summary_path": layout.record_run_summary_path.as_posix(),
        },
        "archives": {
            "training": layout.training_archive_path.as_posix(),
            "qualitative": layout.qualitative_archive_path.as_posix(),
            "record": layout.record_archive_path.as_posix(),
        },
    }


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON artifact is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON artifact root must be a mapping: {path}")
    return cast(Mapping[str, Any], loaded)
