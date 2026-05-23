"""Configuration contracts for the model production workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from typing import cast

import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates import (
    ModelCandidateError,
    ModelRunMode,
    ModelRunRequest,
)
from text_to_sign_production.modeling.data import (
    ModelingDataError,
    ModelingManifestFamily,
    parse_modeling_manifest_family,
)
from text_to_sign_production.modeling.registry import require_model_spec
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    DEFAULT_MODEL_COMPUTE_PROFILE,
    ModelComputeProfile,
    load_model_compute_profile,
)


MODEL_TRAIN_SPLIT = SampleSplit.TRAIN
MODEL_VALIDATION_SPLIT = SampleSplit.VAL
MODEL_TEST_SPLIT = SampleSplit.TEST
MODEL_PREDICTION_SPLITS = (SampleSplit.VAL,)
MODEL_AUXILIARY_OBJECTIVES: tuple[ObjectiveKey, ...] = ()


class ModelWorkflowError(Exception):
    """Base error for model workflow failures."""


class ModelWorkflowInputError(ModelWorkflowError):
    """Raised when model workflow input configuration is invalid."""


class ModelWorkflowInvariantError(ModelWorkflowError):
    """Raised when model workflow invariants are violated."""


@dataclass(frozen=True, slots=True)
class ModelWorkflowConfig:
    """Operator configuration for one provider-backed model run."""

    project_root: Path
    drive_project_root: Path
    model_key: ModelKey
    manifest_family: ModelingManifestFamily
    model_config_relpath: Path | None = None
    run_mode: ModelRunMode = ModelRunMode.SMOKE
    compute_profile: str = DEFAULT_MODEL_COMPUTE_PROFILE
    run_name: str | None = None
    auxiliary_objectives: tuple[ObjectiveKey, ...] = MODEL_AUXILIARY_OBJECTIVES

    train_split: SampleSplit = field(init=False, default=MODEL_TRAIN_SPLIT)
    validation_split: SampleSplit = field(init=False, default=MODEL_VALIDATION_SPLIT)
    test_split: SampleSplit = field(init=False, default=MODEL_TEST_SPLIT)
    prediction_splits: tuple[SampleSplit, ...] = field(
        init=False,
        default=MODEL_PREDICTION_SPLITS,
    )
    seed: int | None = field(init=False, default=None)
    runtime_root: Path | None = field(init=False, default=None)
    resolved_compute_profile: ModelComputeProfile = field(init=False)

    def __post_init__(self) -> None:
        project_root = _coerce_path("project_root", self.project_root)
        object.__setattr__(self, "project_root", project_root)
        drive_project_root = _coerce_path("drive_project_root", self.drive_project_root)
        object.__setattr__(self, "drive_project_root", drive_project_root)
        try:
            model_key = ModelKey(self.model_key)
            manifest_family = (
                self.manifest_family
                if isinstance(self.manifest_family, ModelingManifestFamily)
                else parse_modeling_manifest_family(self.manifest_family)
            )
            run_mode = ModelRunMode(self.run_mode)
            object.__setattr__(self, "model_key", model_key)
            object.__setattr__(
                self,
                "manifest_family",
                manifest_family,
            )
            object.__setattr__(self, "run_mode", run_mode)
            auxiliary_objectives = _coerce_auxiliary_objectives(self.auxiliary_objectives)
            require_model_spec(model_key)
            object.__setattr__(self, "auxiliary_objectives", auxiliary_objectives)
        except (TypeError, ValueError, ModelingDataError) as exc:
            raise ModelWorkflowInputError(str(exc)) from exc
        run_name = (
            build_model_run_name(
                model_key=model_key,
                manifest_family=manifest_family,
                run_mode=run_mode,
                auxiliary_objectives=auxiliary_objectives,
            )
            if self.run_name is None
            else self.run_name
        )
        _validate_run_name(run_name)
        object.__setattr__(self, "run_name", run_name)
        if self.model_config_relpath is not None:
            model_config_relpath = _coerce_relative_path(
                "model_config_relpath",
                self.model_config_relpath,
            )
            resolved_project_root = project_root.expanduser().resolve(strict=False)
            resolved_config_path = (resolved_project_root / model_config_relpath).resolve(
                strict=False
            )
            if not resolved_config_path.is_relative_to(resolved_project_root):
                raise ModelWorkflowInputError(
                    "model_config_relpath must remain under project_root when resolved"
                )
            if not model_config_relpath.is_relative_to(Path("configs") / "modeling"):
                raise ModelWorkflowInputError("MODEL_CONFIG_RELATIVE_PATH must be under configs/modeling")
            if not resolved_config_path.is_file():
                raise ModelWorkflowInputError(f"MODEL_CONFIG_RELATIVE_PATH does not exist: {model_config_relpath}")
            _validate_raw_model_config_identity(
                path=resolved_config_path,
                model_key=model_key,
            )
            object.__setattr__(
                self,
                "model_config_relpath",
                model_config_relpath,
            )
        runtime_root = (
            project_root / "runtime" / "model"
        )
        object.__setattr__(self, "runtime_root", runtime_root)
        try:
            profile = load_model_compute_profile(project_root, self.compute_profile)
        except ValueError as exc:
            raise ModelWorkflowInputError(str(exc)) from exc
        object.__setattr__(self, "compute_profile", profile.name)
        object.__setattr__(self, "resolved_compute_profile", profile)
        _validate_full_compute_profile_calibration_policy(
            model_key=model_key,
            run_mode=run_mode.value,
            compute_profile=profile.to_dict(),
        )
        try:
            self.to_model_run_request()
        except ModelCandidateError as exc:
            raise ModelWorkflowInputError(str(exc)) from exc

    def to_model_run_request(
        self,
        *,
        config_path: Path | None = None,
        objective_config_paths: object | None = None,
    ) -> ModelRunRequest:
        """Build the shared provider-engine request for this workflow run."""

        return ModelRunRequest(
            model_key=self.model_key,
            run_name=self.run_name,
            manifest_family=self.manifest_family,
            train_split=self.train_split,
            validation_split=self.validation_split,
            prediction_splits=self.prediction_splits,
            auxiliary_objectives=self.auxiliary_objectives,
            run_mode=self.run_mode,
            seed=self.seed,
            config_path=config_path,
            objective_config_paths={} if objective_config_paths is None else objective_config_paths,
            compute_profile=self.resolved_compute_profile.to_dict(),
        )


def _coerce_path(field_name: str, value: object) -> Path:
    if isinstance(value, str):
        raw_path = value
    elif isinstance(value, os.PathLike):
        raw_path = os.fspath(cast(os.PathLike[str], value))
    else:
        raise ModelWorkflowInputError(f"{field_name} must be a path-like value")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ModelWorkflowInputError(f"{field_name} must be a non-empty path")
    return Path(raw_path)


def _coerce_relative_path(field_name: str, value: object) -> Path:
    path = _coerce_path(field_name, value)
    if path.is_absolute():
        raise ModelWorkflowInputError(f"{field_name} must be relative")
    if not path.parts or path in {Path("."), Path("..")} or ".." in path.parts:
        raise ModelWorkflowInputError(
            f"{field_name} must be a concrete relative path without parent references"
        )
    return path


def _validate_run_name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelWorkflowInputError("run_name must be non-empty")
    if value in {".", ".."} or "/" in value or "\\" in value or Path(value).name != value:
        raise ModelWorkflowInputError("run_name must be a safe path token")


def _validate_raw_model_config_identity(*, path: Path, model_key: ModelKey) -> None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelWorkflowInputError(f"model config YAML could not be read: {path}") from exc
    if not isinstance(raw, dict):
        raise ModelWorkflowInputError("model config YAML must be a mapping")
    identity = raw.get("identity")
    if isinstance(identity, dict):
        for key in ("model_key", "provider_key"):
            value = identity.get(key)
            if value is not None and value != model_key.value:
                raise ModelWorkflowInputError(
                    f"model config identity.{key}={value!r} does not match MODEL_KEY={model_key.value!r}"
                )


def _validate_full_compute_profile_calibration_policy(
    *,
    model_key: ModelKey,
    run_mode: str,
    compute_profile: dict[str, object],
) -> None:
    if run_mode != "full":
        return
    if compute_profile.get("name") != "colab_a100_80gb":
        return
    from text_to_sign_production.modeling.candidates.bootstrap import ensure_model_provider_registered
    from text_to_sign_production.modeling.candidates.registry import require_model_provider

    ensure_model_provider_registered(model_key)
    provider = require_model_provider(model_key)
    policy_fn = getattr(provider, "calibration_policy", None)
    if not callable(policy_fn):
        raise ModelWorkflowInputError("full A100 run requires provider calibration policy")


def build_model_run_name(
    *,
    model_key: ModelKey | str,
    manifest_family: ModelingManifestFamily | str,
    run_mode: ModelRunMode | str,
    auxiliary_objectives: tuple[ObjectiveKey, ...] | list[ObjectiveKey | str] = (),
) -> str:
    """Build a unique safe run token for one model workflow transaction."""

    try:
        model = ModelKey(model_key)
        family = (
            manifest_family
            if isinstance(manifest_family, ModelingManifestFamily)
            else parse_modeling_manifest_family(manifest_family)
        )
        mode = ModelRunMode(run_mode)
        objectives = _coerce_auxiliary_objectives(auxiliary_objectives)
        require_model_spec(model)
    except (TypeError, ValueError, ModelingDataError) as exc:
        raise ModelWorkflowInputError(str(exc)) from exc
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    objective_slug = "".join(f"__{objective.value}" for objective in objectives)
    run_name = (
        f"{model.value}__{family.path_slug}__{mode.value}{objective_slug}__"
        f"{timestamp}_{uuid4().hex[:8]}"
    )
    _validate_run_name(run_name)
    return run_name


def _coerce_auxiliary_objectives(
    values: tuple[ObjectiveKey, ...] | list[ObjectiveKey | str],
) -> tuple[ObjectiveKey, ...]:
    if isinstance(values, (str, ObjectiveKey)):
        raise ModelWorkflowInputError(
            "auxiliary_objectives must be a tuple or list of objective keys, not a bare string."
        )
    if not isinstance(values, (tuple, list)):
        raise ModelWorkflowInputError(
            "auxiliary_objectives must be a tuple or list of objective keys."
        )
    try:
        objectives = tuple(ObjectiveKey(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ModelWorkflowInputError(
            f"unknown auxiliary objective in {values!r}."
        ) from exc
    if len(set(objectives)) != len(objectives):
        raise ModelWorkflowInputError("auxiliary_objectives must not contain duplicates.")
    return objectives


__all__ = [
    "MODEL_AUXILIARY_OBJECTIVES",
    "MODEL_PREDICTION_SPLITS",
    "MODEL_TEST_SPLIT",
    "MODEL_TRAIN_SPLIT",
    "MODEL_VALIDATION_SPLIT",
    "ModelWorkflowConfig",
    "ModelWorkflowError",
    "ModelWorkflowInputError",
    "ModelWorkflowInvariantError",
    "build_model_run_name",
]
