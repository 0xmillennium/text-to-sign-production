"""Run requests and objective compatibility validation for model candidates."""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.candidates.errors import (
    ModelCandidateError,
    ObjectiveAttachmentError,
)
from text_to_sign_production.modeling.data.errors import ModelingDataError
from text_to_sign_production.modeling.data.manifest_families import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
)
from text_to_sign_production.modeling.registry import (
    ModelingRegistryError,
    require_model_spec,
    require_objective_spec,
)
from text_to_sign_production.modeling.research import ModelKey, ModelSpec, ObjectiveKey, ObjectiveSpec


class ModelRunMode(enum.StrEnum):
    """Intended execution scale for a model run."""

    SMOKE = "smoke"
    DEBUG = "debug"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ModelRunRequest:
    """Provider-independent request for one candidate model run."""

    model_key: ModelKey
    run_name: str
    manifest_family: ModelingManifestFamily
    train_split: SampleSplit = SampleSplit.TRAIN
    validation_split: SampleSplit = SampleSplit.VAL
    prediction_splits: tuple[SampleSplit, ...] = (SampleSplit.VAL,)
    auxiliary_objectives: tuple[ObjectiveKey, ...] = ()
    run_mode: ModelRunMode = ModelRunMode.SMOKE
    seed: int | None = None
    config_path: Path | None = None
    objective_config_paths: Mapping[ObjectiveKey, Path] = MappingProxyType({})
    compute_profile: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_key", _coerce_model_key(self.model_key))
        object.__setattr__(
            self,
            "manifest_family",
            _coerce_manifest_family(self.manifest_family),
        )
        object.__setattr__(self, "train_split", _coerce_split(self.train_split, "train_split"))
        object.__setattr__(
            self,
            "validation_split",
            _coerce_split(self.validation_split, "validation_split"),
        )
        object.__setattr__(
            self,
            "prediction_splits",
            _coerce_prediction_splits(self.prediction_splits),
        )
        object.__setattr__(
            self,
            "auxiliary_objectives",
            _coerce_objective_keys(self.auxiliary_objectives),
        )
        try:
            object.__setattr__(self, "run_mode", ModelRunMode(self.run_mode))
        except (TypeError, ValueError) as exc:
            raise ModelCandidateError(f"unknown model run mode: {self.run_mode!r}") from exc
        _validate_run_name(self.run_name)
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0
        ):
            raise ModelCandidateError("seed must be a non-negative integer or None.")
        if self.config_path is not None and not isinstance(self.config_path, Path):
            raise ModelCandidateError("config_path must be a Path or None.")
        object.__setattr__(
            self,
            "objective_config_paths",
            MappingProxyType(_coerce_objective_config_paths(self.objective_config_paths)),
        )
        if not isinstance(self.compute_profile, Mapping) or any(
            not isinstance(key, str) for key in self.compute_profile
        ):
            raise ModelCandidateError("compute_profile must be a mapping with string keys.")
        object.__setattr__(self, "compute_profile", MappingProxyType(dict(self.compute_profile)))


def validate_objective_attachments(
    model_spec: ModelSpec,
    objective_keys: Iterable[ObjectiveKey | str],
) -> tuple[ObjectiveSpec, ...]:
    """Resolve and validate auxiliary objectives compatible with one model."""

    if not isinstance(model_spec, ModelSpec):
        raise ObjectiveAttachmentError("model_spec must be a ModelSpec.")
    keys = _coerce_objective_keys(objective_keys)
    objectives: list[ObjectiveSpec] = []
    for key in keys:
        try:
            objective = require_objective_spec(key)
        except ModelingRegistryError as exc:
            raise ObjectiveAttachmentError(f"unknown auxiliary objective: {key.value!r}.") from exc
        if objective.standalone_model_allowed is not False:
            raise ObjectiveAttachmentError(
                f"objective {objective.key.value!r} must not be a standalone model."
            )
        if model_spec.key not in objective.allowed_attachment_models:
            raise ObjectiveAttachmentError(
                f"objective {objective.key.value!r} cannot attach to model "
                f"{model_spec.key.value!r}."
            )
        if objective.key not in model_spec.compatible_objectives:
            raise ObjectiveAttachmentError(
                f"model {model_spec.key.value!r} does not declare objective "
                f"{objective.key.value!r} as compatible."
            )
        objectives.append(objective)
    return tuple(objectives)


def model_run_request_to_dict(request: ModelRunRequest) -> dict[str, object]:
    """Return a JSON-serializable model run request record."""

    if not isinstance(request, ModelRunRequest):
        raise ModelCandidateError("request must be a ModelRunRequest.")
    return {
        "model_key": request.model_key.value,
        "run_name": request.run_name,
        **request.manifest_family.to_dict(),
        "train_split": request.train_split.value,
        "validation_split": request.validation_split.value,
        "prediction_splits": [split.value for split in request.prediction_splits],
        "auxiliary_objectives": [
            objective.value for objective in request.auxiliary_objectives
        ],
        "run_mode": request.run_mode.value,
        "seed": request.seed,
        "config_path": None if request.config_path is None else str(request.config_path),
        "objective_config_paths": {
            objective.value: str(path)
            for objective, path in request.objective_config_paths.items()
        },
        "compute_profile": dict(request.compute_profile),
    }


def _coerce_objective_config_paths(
    value: Mapping[ObjectiveKey | str, Path | str] | None,
) -> dict[ObjectiveKey, Path]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelCandidateError("objective_config_paths must be a mapping.")
    result: dict[ObjectiveKey, Path] = {}
    for raw_key, raw_path in value.items():
        try:
            key = ObjectiveKey(raw_key)
        except (TypeError, ValueError) as exc:
            raise ObjectiveAttachmentError(
                f"objective_config_paths contains unknown objective: {raw_key!r}."
            ) from exc
        if not isinstance(raw_path, Path):
            raw_path = Path(raw_path)
        result[key] = raw_path
    return result


def _coerce_model_key(value: ModelKey | str) -> ModelKey:
    try:
        return ModelKey(value)
    except (TypeError, ValueError) as exc:
        raise ModelCandidateError(f"unknown model key: {value!r}") from exc


def _coerce_manifest_family(
    value: ModelingManifestFamily | str,
) -> ModelingManifestFamily:
    if isinstance(value, ModelingManifestFamily):
        return value
    try:
        return parse_modeling_manifest_family(value)
    except ModelingDataError as exc:
        raise ModelCandidateError(str(exc)) from exc


def _coerce_split(value: SampleSplit | str, field_name: str) -> SampleSplit:
    try:
        return SampleSplit(value)
    except (TypeError, ValueError) as exc:
        raise ModelCandidateError(f"unknown {field_name}: {value!r}") from exc


def _coerce_prediction_splits(
    values: Iterable[SampleSplit | str],
) -> tuple[SampleSplit, ...]:
    if isinstance(values, (SampleSplit, str)):
        raise ModelCandidateError("prediction_splits must be a tuple of split values.")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ModelCandidateError("prediction_splits must be iterable.") from exc
    splits = tuple(_coerce_split(value, "prediction_splits") for value in values)
    if not splits:
        raise ModelCandidateError("prediction_splits must be non-empty.")
    if len(set(splits)) != len(splits):
        raise ModelCandidateError("prediction_splits must not contain duplicates.")
    return splits


def _coerce_objective_keys(
    values: Iterable[ObjectiveKey | str],
) -> tuple[ObjectiveKey, ...]:
    if isinstance(values, (ObjectiveKey, str)):
        raise ObjectiveAttachmentError(
            "auxiliary objectives must be an iterable of objective values."
        )
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ObjectiveAttachmentError("auxiliary objectives must be iterable.") from exc
    keys: list[ObjectiveKey] = []
    for value in raw_values:
        try:
            keys.append(ObjectiveKey(value))
        except (TypeError, ValueError) as exc:
            raise ObjectiveAttachmentError(f"unknown auxiliary objective: {value!r}.") from exc
    if len(set(keys)) != len(keys):
        raise ObjectiveAttachmentError("auxiliary objectives must not contain duplicates.")
    return tuple(keys)


def _validate_run_name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelCandidateError("run_name must be non-empty.")
    if value in {".", ".."} or "/" in value or "\\" in value or Path(value).name != value:
        raise ModelCandidateError("run_name must be a safe path token.")


__all__ = [
    "ModelRunMode",
    "ModelRunRequest",
    "model_run_request_to_dict",
    "validate_objective_attachments",
]
