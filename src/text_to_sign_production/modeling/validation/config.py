"""Candidate-agnostic validation engine configuration."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
)
from text_to_sign_production.modeling.validation.errors import ModelValidationError
from text_to_sign_production.modeling.validation.identifiers import (
    ValidationChannelMetricKey,
    ValidationMetricKey,
)

VALIDATION_ENGINE_SCHEMA_VERSION = "t2sp-validation-engine-v1"
_PRODUCER_TYPES = frozenset({"model", "comparator"})
_ALIGNMENT_POLICY = "prefix_min_length"


@dataclass(frozen=True, slots=True)
class ValidationEngineIdentityConfig:
    schema_version: str
    name: str = "candidate_agnostic_validation"
    manifest_family: ModelingManifestFamily | None = None

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_ENGINE_SCHEMA_VERSION:
            raise ModelValidationError("validation engine schema_version is unsupported.")
        _require_text(self.name, "identity.name")
        if self.manifest_family is not None:
            object.__setattr__(
                self,
                "manifest_family",
                self.manifest_family
                if isinstance(self.manifest_family, ModelingManifestFamily)
                else parse_modeling_manifest_family(self.manifest_family),
            )


@dataclass(frozen=True, slots=True)
class ValidationEngineDataConfig:
    split: SampleSplit = SampleSplit.VAL
    minimum_paired_count: int = 1
    alignment_policy: str = _ALIGNMENT_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        if self.split is not SampleSplit.VAL:
            raise ModelValidationError(
                "candidate-agnostic validation currently supports split='val' only."
            )
        _require_positive_int(self.minimum_paired_count, "data.minimum_paired_count")
        if self.alignment_policy != _ALIGNMENT_POLICY:
            raise ModelValidationError("data.alignment_policy must be 'prefix_min_length'.")


@dataclass(frozen=True, slots=True)
class ValidationSurfaceConfig:
    label: str
    producer_type: str
    producer_key: str
    canonical_id: str | None = None
    research_role: str | None = None
    run_name: str | None = None
    required: bool = False

    def __post_init__(self) -> None:
        _require_text(self.label, "surface.label")
        _require_text(self.producer_key, "surface.producer_key")
        if self.producer_type not in _PRODUCER_TYPES:
            raise ModelValidationError("validation surface producer_type must be 'model' or 'comparator'.")
        for value, name in (
            (self.canonical_id, "surface.canonical_id"),
            (self.research_role, "surface.research_role"),
            (self.run_name, "surface.run_name"),
        ):
            if value is not None:
                _require_text(value, name)
        if not isinstance(self.required, bool):
            raise ModelValidationError("surface.required must be boolean.")


@dataclass(frozen=True, slots=True)
class ValidationMetricSelectionConfig:
    full_bfh_metrics: bool = True
    channel_metrics: bool = True

    def __post_init__(self) -> None:
        for field_name in ("full_bfh_metrics", "channel_metrics"):
            if not isinstance(getattr(self, field_name), bool):
                raise ModelValidationError(f"metrics.{field_name} must be boolean.")


@dataclass(frozen=True, slots=True)
class ValidationComparisonConfig:
    enabled: bool = True
    require_shared_pairing_subset: bool = True
    lower_is_better: tuple[str, ...] = ()
    higher_is_better: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("enabled", "require_shared_pairing_subset"):
            if not isinstance(getattr(self, field_name), bool):
                raise ModelValidationError(f"comparison.{field_name} must be boolean.")
        lower = _metric_key_tuple(self.lower_is_better, "comparison.lower_is_better")
        higher = _metric_key_tuple(self.higher_is_better, "comparison.higher_is_better")
        if set(lower) & set(higher):
            raise ModelValidationError("comparison lower_is_better and higher_is_better must not overlap.")
        object.__setattr__(self, "lower_is_better", lower)
        object.__setattr__(self, "higher_is_better", higher)


@dataclass(frozen=True, slots=True)
class ValidationEngineReportsConfig:
    enabled: bool = True
    output_dir_name: str = "validation_engine"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ModelValidationError("reports.enabled must be boolean.")
        _require_text(self.output_dir_name, "reports.output_dir_name")


@dataclass(frozen=True, slots=True)
class ValidationEngineConfig:
    source_path: Path | None
    raw_config: Mapping[str, object]
    identity: ValidationEngineIdentityConfig
    data: ValidationEngineDataConfig
    surfaces: tuple[ValidationSurfaceConfig, ...]
    metrics: ValidationMetricSelectionConfig
    comparison: ValidationComparisonConfig
    reports: ValidationEngineReportsConfig

    def __post_init__(self) -> None:
        if self.source_path is not None and not isinstance(self.source_path, Path):
            raise ModelValidationError("validation engine source_path must be a Path or None.")
        if not self.surfaces:
            raise ModelValidationError("validation engine surfaces must be non-empty.")
        labels = tuple(surface.label for surface in self.surfaces)
        if len(set(labels)) != len(labels):
            raise ModelValidationError("validation surface labels must be unique.")
        object.__setattr__(self, "raw_config", MappingProxyType(dict(self.raw_config)))


def load_validation_engine_config(path: Path | str) -> ValidationEngineConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelValidationError(f"could not read validation engine config: {source_path}") from exc
    except yaml.YAMLError as exc:
        raise ModelValidationError(f"could not parse validation engine config: {source_path}") from exc
    if not isinstance(loaded, Mapping):
        raise ModelValidationError("validation engine config root must be a mapping.")
    raw = copy.deepcopy(dict(loaded))
    identity = _mapping(raw, "identity")
    data = _mapping(raw, "data")
    metrics = _mapping(raw, "metrics")
    comparison = _mapping(raw, "comparison")
    reports = _mapping(raw, "reports")
    surfaces = _sequence(raw, "surfaces")
    return ValidationEngineConfig(
        source_path=source_path,
        raw_config=raw,
        identity=ValidationEngineIdentityConfig(
            schema_version=_str(identity, "schema_version"),
            name=_str(identity, "name"),
            manifest_family=(
                None
                if identity.get("manifest_family") is None
                else parse_modeling_manifest_family(_str(identity, "manifest_family"))
            ),
        ),
        data=ValidationEngineDataConfig(
            split=SampleSplit(_str(data, "split")),
            minimum_paired_count=_int(data, "minimum_paired_count"),
            alignment_policy=_str(data, "alignment_policy"),
        ),
        surfaces=tuple(_surface_config(value) for value in surfaces),
        metrics=ValidationMetricSelectionConfig(
            full_bfh_metrics=_bool(metrics, "full_bfh_metrics"),
            channel_metrics=_bool(metrics, "channel_metrics"),
        ),
        comparison=ValidationComparisonConfig(
            enabled=_bool(comparison, "enabled"),
            require_shared_pairing_subset=_bool(comparison, "require_shared_pairing_subset"),
            lower_is_better=_str_sequence(comparison, "lower_is_better"),
            higher_is_better=_str_sequence(comparison, "higher_is_better"),
        ),
        reports=ValidationEngineReportsConfig(
            enabled=_bool(reports, "enabled"),
            output_dir_name=_str(reports, "output_dir_name"),
        ),
    )


def default_lower_is_better_metric_keys() -> tuple[str, ...]:
    return (
        ValidationMetricKey.MASKED_L1_MEAN.value,
        ValidationMetricKey.MASKED_L2_MEAN.value,
        ValidationMetricKey.VELOCITY_L1_MEAN.value,
        ValidationMetricKey.VELOCITY_L2_MEAN.value,
        ValidationMetricKey.SEQUENCE_LENGTH_ABSOLUTE_ERROR.value,
        *(
            f"{metric.value}:*"
            for metric in (
                ValidationChannelMetricKey.CHANNEL_MASKED_L1_MEAN,
                ValidationChannelMetricKey.CHANNEL_MASKED_L2_MEAN,
                ValidationChannelMetricKey.CHANNEL_VELOCITY_L1_MEAN,
                ValidationChannelMetricKey.CHANNEL_VELOCITY_L2_MEAN,
            )
        ),
    )


def default_higher_is_better_metric_keys() -> tuple[str, ...]:
    return (
        ValidationMetricKey.VALID_JOINT_COVERAGE.value,
        f"{ValidationChannelMetricKey.CHANNEL_VALID_JOINT_COVERAGE.value}:*",
    )


def _surface_config(value: object) -> ValidationSurfaceConfig:
    mapping = _as_mapping(value, "surface")
    return ValidationSurfaceConfig(
        label=_str(mapping, "label"),
        producer_type=_str(mapping, "producer_type"),
        producer_key=_str(mapping, "producer_key"),
        canonical_id=_optional_str(mapping, "canonical_id"),
        research_role=_optional_str(mapping, "research_role"),
        run_name=_optional_str(mapping, "run_name"),
        required=_bool(mapping, "required"),
    )


def _metric_key_tuple(values: Sequence[str], label: str) -> tuple[str, ...]:
    resolved = tuple(values)
    for value in resolved:
        _require_text(value, label)
    return resolved


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    try:
        value = mapping[key]
    except KeyError as exc:
        raise ModelValidationError(f"validation engine config missing section: {key}.") from exc
    return _as_mapping(value, key)


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelValidationError(f"validation engine config {label} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise ModelValidationError(f"validation engine config {label} keys must be strings.")
    return value


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ModelValidationError(f"validation engine config {key} must be a sequence.")
    return value


def _str_sequence(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = _sequence(mapping, key)
    return tuple(_text(value, key) for value in values)


def _str(mapping: Mapping[str, object], key: str) -> str:
    if key not in mapping:
        raise ModelValidationError(f"validation engine config missing key: {key}.")
    return _text(mapping[key], key)


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _text(value, key)


def _text(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{key} must be non-empty.")
    return value


def _int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelValidationError(f"{key} must be an integer.")
    return value


def _bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ModelValidationError(f"{key} must be boolean.")
    return value


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ModelValidationError(f"{label} must be non-empty.")


def _require_positive_int(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ModelValidationError(f"{label} must be at least 1.")


__all__ = [
    "VALIDATION_ENGINE_SCHEMA_VERSION",
    "ValidationComparisonConfig",
    "ValidationEngineConfig",
    "ValidationEngineDataConfig",
    "ValidationEngineIdentityConfig",
    "ValidationEngineReportsConfig",
    "ValidationMetricSelectionConfig",
    "ValidationSurfaceConfig",
    "default_higher_is_better_metric_keys",
    "default_lower_is_better_metric_keys",
    "load_validation_engine_config",
]
