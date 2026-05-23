"""Configuration for evaluation protocol and ablation orchestration."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
)
from text_to_sign_production.modeling.evaluation.errors import EvaluationProtocolError

EVALUATION_PROTOCOL_SCHEMA_VERSION = "t2sp-evaluation-protocol-v1"
_PRODUCER_TYPES = frozenset({"model", "comparator"})
_CONTRAST_TYPES = frozenset({"ablation", "comparator_baseline", "readiness_only"})
_RUN_NAME_REQUIRED_MESSAGE = (
    "evaluation variant run_name is required because Aşama 9 validates existing generated "
    "surfaces and does not execute missing runs."
)


@dataclass(frozen=True, slots=True)
class EvaluationProtocolIdentityConfig:
    schema_version: str
    name: str = "evaluation_protocol"
    manifest_family: ModelingManifestFamily | None = None

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_PROTOCOL_SCHEMA_VERSION:
            raise EvaluationProtocolError("evaluation protocol schema_version is unsupported.")
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
class EvaluationProtocolExecutionConfig:
    mode: str = "validate_existing"
    run_validation_engine: bool = True

    def __post_init__(self) -> None:
        if self.mode != "validate_existing":
            raise EvaluationProtocolError("evaluation protocol execution.mode must be 'validate_existing'.")
        if not isinstance(self.run_validation_engine, bool):
            raise EvaluationProtocolError("execution.run_validation_engine must be boolean.")


@dataclass(frozen=True, slots=True)
class EvaluationVariantConfig:
    variant_id: str
    label: str
    producer_type: str
    producer_key: str
    config_path: str
    run_name: str
    auxiliary_objectives: tuple[str, ...] = ()
    objective_config_paths: Mapping[str, str] = MappingProxyType({})
    tags: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        for value, name in (
            (self.variant_id, "variant.variant_id"),
            (self.label, "variant.label"),
            (self.producer_key, "variant.producer_key"),
            (self.config_path, "variant.config_path"),
        ):
            _require_text(value, name)
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise EvaluationProtocolError(_RUN_NAME_REQUIRED_MESSAGE)
        if self.producer_type not in _PRODUCER_TYPES:
            raise EvaluationProtocolError("evaluation variant producer_type must be 'model' or 'comparator'.")
        if self.variant_id == "retrieval_pose_baseline" and self.producer_type != "comparator":
            raise EvaluationProtocolError("retrieval_pose_baseline must use producer_type='comparator'.")
        objectives = _text_tuple(self.auxiliary_objectives, "variant.auxiliary_objectives")
        tags = _text_tuple(self.tags, "variant.tags")
        paths = dict(self.objective_config_paths)
        for key, value in paths.items():
            _require_text(key, "variant.objective_config_paths key")
            _require_text(value, "variant.objective_config_paths value")
        if "semantic_consistency" in objectives:
            if "semantic_consistency" not in paths:
                raise EvaluationProtocolError(
                    "semantic_consistency auxiliary objective requires objective_config_paths.semantic_consistency."
                )
        elif "semantic_consistency" in paths:
            raise EvaluationProtocolError(
                "semantic objective config path is only allowed when auxiliary_objectives includes semantic_consistency."
            )
        if self.producer_key == "base_direct" and "semantic_consistency" in objectives:
            raise EvaluationProtocolError("base_direct cannot use semantic_consistency auxiliary objective.")
        if not isinstance(self.required, bool):
            raise EvaluationProtocolError("variant.required must be boolean.")
        object.__setattr__(self, "auxiliary_objectives", objectives)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "objective_config_paths", MappingProxyType(paths))


@dataclass(frozen=True, slots=True)
class EvaluationContrastConfig:
    contrast_id: str
    contrast_type: str
    control_variant_id: str
    treatment_variant_id: str
    intended_difference: str
    required: bool = True
    allow_config_path_mismatch: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.contrast_id, "contrast.contrast_id"),
            (self.control_variant_id, "contrast.control_variant_id"),
            (self.treatment_variant_id, "contrast.treatment_variant_id"),
            (self.intended_difference, "contrast.intended_difference"),
        ):
            _require_text(value, name)
        if self.contrast_type not in _CONTRAST_TYPES:
            raise EvaluationProtocolError(
                "evaluation contrast_type must be 'ablation', 'comparator_baseline', or 'readiness_only'."
            )
        if self.control_variant_id == self.treatment_variant_id:
            raise EvaluationProtocolError("evaluation contrast control and treatment variants must differ.")
        for field_name in ("required", "allow_config_path_mismatch"):
            if not isinstance(getattr(self, field_name), bool):
                raise EvaluationProtocolError(f"contrast.{field_name} must be boolean.")


@dataclass(frozen=True, slots=True)
class EvaluationValidationEngineConfig:
    enabled: bool = True
    minimum_shared_paired_count: int = 1
    require_shared_pairing_subset: bool = True
    output_dir_name: str = "validation_engine"

    def __post_init__(self) -> None:
        for field_name in ("enabled", "require_shared_pairing_subset"):
            if not isinstance(getattr(self, field_name), bool):
                raise EvaluationProtocolError(f"validation_engine.{field_name} must be boolean.")
        if not isinstance(self.minimum_shared_paired_count, int) or isinstance(
            self.minimum_shared_paired_count, bool
        ) or self.minimum_shared_paired_count < 1:
            raise EvaluationProtocolError("validation_engine.minimum_shared_paired_count must be >= 1.")
        _require_text(self.output_dir_name, "validation_engine.output_dir_name")


@dataclass(frozen=True, slots=True)
class EvaluationProtocolReportsConfig:
    enabled: bool = True
    output_dir_name: str = "evaluation_protocol"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise EvaluationProtocolError("reports.enabled must be boolean.")
        _require_text(self.output_dir_name, "reports.output_dir_name")


@dataclass(frozen=True, slots=True)
class EvaluationProtocolConfig:
    source_path: Path | None
    raw_config: Mapping[str, object]
    identity: EvaluationProtocolIdentityConfig
    split: SampleSplit
    execution: EvaluationProtocolExecutionConfig
    variants: tuple[EvaluationVariantConfig, ...]
    contrasts: tuple[EvaluationContrastConfig, ...]
    validation_engine: EvaluationValidationEngineConfig
    reports: EvaluationProtocolReportsConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "split", SampleSplit(self.split))
        if self.split is not SampleSplit.VAL:
            raise EvaluationProtocolError("evaluation protocol currently supports split='val' only.")
        if not self.variants:
            raise EvaluationProtocolError("evaluation protocol variants must be non-empty.")
        variant_ids = tuple(variant.variant_id for variant in self.variants)
        if len(set(variant_ids)) != len(variant_ids):
            raise EvaluationProtocolError("evaluation variant_id values must be unique.")
        contrast_ids = tuple(contrast.contrast_id for contrast in self.contrasts)
        if len(set(contrast_ids)) != len(contrast_ids):
            raise EvaluationProtocolError("evaluation contrast_id values must be unique.")
        known = set(variant_ids)
        for contrast in self.contrasts:
            if contrast.control_variant_id not in known:
                raise EvaluationProtocolError(
                    f"evaluation contrast {contrast.contrast_id!r} references unknown "
                    f"control_variant_id={contrast.control_variant_id!r}."
                )
            if contrast.treatment_variant_id not in known:
                raise EvaluationProtocolError(
                    f"evaluation contrast {contrast.contrast_id!r} references unknown "
                    f"treatment_variant_id={contrast.treatment_variant_id!r}."
                )
        object.__setattr__(self, "raw_config", MappingProxyType(dict(self.raw_config)))


def load_evaluation_protocol_config(path: Path | str) -> EvaluationProtocolConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationProtocolError(f"could not read evaluation protocol config: {source_path}") from exc
    except yaml.YAMLError as exc:
        raise EvaluationProtocolError(f"could not parse evaluation protocol config: {source_path}") from exc
    if not isinstance(loaded, Mapping):
        raise EvaluationProtocolError("evaluation protocol config root must be a mapping.")
    raw = copy.deepcopy(dict(loaded))
    identity = _mapping(raw, "identity")
    execution = _mapping(raw, "execution")
    validation_engine = _mapping(raw, "validation_engine")
    reports = _mapping(raw, "reports")
    return EvaluationProtocolConfig(
        source_path=source_path,
        raw_config=raw,
        identity=EvaluationProtocolIdentityConfig(
            schema_version=_str(identity, "schema_version"),
            name=_str(identity, "name"),
            manifest_family=(
                None
                if identity.get("manifest_family") is None
                else parse_modeling_manifest_family(_str(identity, "manifest_family"))
            ),
        ),
        split=SampleSplit(_str(raw, "split")),
        execution=EvaluationProtocolExecutionConfig(
            mode=_str(execution, "mode"),
            run_validation_engine=_bool(execution, "run_validation_engine"),
        ),
        variants=tuple(_variant_config(value) for value in _sequence(raw, "variants")),
        contrasts=tuple(_contrast_config(value) for value in _sequence(raw, "contrasts")),
        validation_engine=EvaluationValidationEngineConfig(
            enabled=_bool(validation_engine, "enabled"),
            minimum_shared_paired_count=_int(validation_engine, "minimum_shared_paired_count"),
            require_shared_pairing_subset=_bool(validation_engine, "require_shared_pairing_subset"),
            output_dir_name=_str(validation_engine, "output_dir_name"),
        ),
        reports=EvaluationProtocolReportsConfig(
            enabled=_bool(reports, "enabled"),
            output_dir_name=_str(reports, "output_dir_name"),
        ),
    )


def _variant_config(value: object) -> EvaluationVariantConfig:
    mapping = _as_mapping(value, "variant")
    return EvaluationVariantConfig(
        variant_id=_str(mapping, "variant_id"),
        label=_str(mapping, "label"),
        producer_type=_str(mapping, "producer_type"),
        producer_key=_str(mapping, "producer_key"),
        config_path=_str(mapping, "config_path"),
        run_name=_run_name(mapping),
        auxiliary_objectives=_str_sequence(mapping, "auxiliary_objectives", default=()),
        objective_config_paths=_optional_mapping(mapping, "objective_config_paths"),
        tags=_str_sequence(mapping, "tags", default=()),
        required=_bool(mapping, "required"),
    )


def _contrast_config(value: object) -> EvaluationContrastConfig:
    mapping = _as_mapping(value, "contrast")
    return EvaluationContrastConfig(
        contrast_id=_str(mapping, "contrast_id"),
        contrast_type=_str(mapping, "contrast_type"),
        control_variant_id=_str(mapping, "control_variant_id"),
        treatment_variant_id=_str(mapping, "treatment_variant_id"),
        intended_difference=_str(mapping, "intended_difference"),
        required=_bool(mapping, "required"),
        allow_config_path_mismatch=_bool(mapping, "allow_config_path_mismatch", default=False),
    )


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    return _as_mapping(value, key)


def _optional_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, str]:
    value = mapping.get(key, {})
    if value is None:
        return {}
    result = _as_mapping(value, key)
    return {str(k): _text(v, f"{key}.{k}") for k, v in result.items()}


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvaluationProtocolError(f"{name} must be a mapping.")
    return value


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationProtocolError(f"{key} must be a sequence.")
    return value


def _str_sequence(
    mapping: Mapping[str, object],
    key: str,
    *,
    default: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = mapping.get(key, default)
    return _text_tuple(value, key)


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvaluationProtocolError(f"{name} must be a sequence of strings.")
    return tuple(_text(item, name) for item in value)


def _str(mapping: Mapping[str, object], key: str) -> str:
    return _text(mapping.get(key), key)


def _run_name(mapping: Mapping[str, object]) -> str:
    value = mapping.get("run_name")
    if not isinstance(value, str) or not value.strip():
        raise EvaluationProtocolError(_RUN_NAME_REQUIRED_MESSAGE)
    return value


def _bool(mapping: Mapping[str, object], key: str, *, default: bool | None = None) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise EvaluationProtocolError(f"{key} must be boolean.")
    return value


def _int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvaluationProtocolError(f"{key} must be an integer.")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationProtocolError(f"{name} must be non-empty text.")
    return value


def _require_text(value: object, name: str) -> None:
    _text(value, name)


__all__ = [
    "EVALUATION_PROTOCOL_SCHEMA_VERSION",
    "EvaluationContrastConfig",
    "EvaluationProtocolConfig",
    "EvaluationProtocolExecutionConfig",
    "EvaluationProtocolIdentityConfig",
    "EvaluationProtocolReportsConfig",
    "EvaluationValidationEngineConfig",
    "EvaluationVariantConfig",
    "load_evaluation_protocol_config",
]
