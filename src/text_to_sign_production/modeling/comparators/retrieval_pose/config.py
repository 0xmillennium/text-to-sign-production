"""Configuration contract for the whole-pose retrieval comparator."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.artifacts import (
    GENERATED_POSE_CHANNEL_POLICY,
    GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
    GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
)
from text_to_sign_production.modeling.backbones.text_encoder import TextEncoderConfig
from text_to_sign_production.modeling.comparators.retrieval_pose.errors import (
    RetrievalPoseComparatorError,
)
from text_to_sign_production.modeling.data import (
    ModelingManifestFamily,
    parse_modeling_manifest_family,
)

_FORBIDDEN_KEYS = frozenset(
    {
        "gloss",
        "dictionary",
        "stitching",
        "avatar",
        "rendering",
        "learned_generator",
        "diffusion",
        "articulator",
        "pose_token",
    }
)


@dataclass(frozen=True, slots=True)
class RetrievalComparatorIdentityConfig:
    comparator_key: str
    canonical_id: str
    display_name: str
    phase_number: int
    research_role: str
    producer_type: str
    channel_policy: str
    output_contract: str
    manifest_contract: str

    def __post_init__(self) -> None:
        expected = {
            "comparator_key": "retrieval_pose",
            "canonical_id": "retrieval_augmented_pose_comparator",
            "phase_number": 9,
            "research_role": "counter_alternative_comparator",
            "producer_type": "comparator",
            "channel_policy": GENERATED_POSE_CHANNEL_POLICY,
            "output_contract": GENERATED_POSE_PAYLOAD_SCHEMA_VERSION,
            "manifest_contract": GENERATED_POSE_MANIFEST_SCHEMA_VERSION,
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise RetrievalPoseComparatorError(
                    f"retrieval identity.{field_name} must be {value!r}."
                )
        _require_text(self.display_name, "identity.display_name")


@dataclass(frozen=True, slots=True)
class RetrievalComparatorDataConfig:
    manifest_family: ModelingManifestFamily | None
    bank_split: SampleSplit = SampleSplit.TRAIN
    query_splits: tuple[SampleSplit, ...] = (SampleSplit.VAL,)
    max_bank_size: int | None = None
    max_query_samples: int | None = None

    def __post_init__(self) -> None:
        if self.manifest_family is not None:
            object.__setattr__(
                self,
                "manifest_family",
                self.manifest_family
                if isinstance(self.manifest_family, ModelingManifestFamily)
                else parse_modeling_manifest_family(self.manifest_family),
            )
        object.__setattr__(self, "bank_split", SampleSplit(self.bank_split))
        object.__setattr__(
            self,
            "query_splits",
            tuple(SampleSplit(split) for split in self.query_splits),
        )
        if not self.query_splits or len(set(self.query_splits)) != len(self.query_splits):
            raise RetrievalPoseComparatorError("retrieval data.query_splits must be non-empty and unique.")
        _require_optional_positive_int(self.max_bank_size, "data.max_bank_size")
        _require_optional_positive_int(self.max_query_samples, "data.max_query_samples")


@dataclass(frozen=True, slots=True)
class RetrievalScoringConfig:
    score_metric: str = "cosine_similarity"
    selection_policy: str = "top1"
    top_k: int = 5
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.score_metric != "cosine_similarity":
            raise RetrievalPoseComparatorError("retrieval.score_metric must be 'cosine_similarity'.")
        if self.selection_policy != "top1":
            raise RetrievalPoseComparatorError("retrieval.selection_policy must be 'top1'.")
        _require_positive_int(self.top_k, "retrieval.top_k")
        if not isinstance(self.normalize_embeddings, bool):
            raise RetrievalPoseComparatorError("retrieval.normalize_embeddings must be boolean.")


@dataclass(frozen=True, slots=True)
class RetrievalLeakagePolicyConfig:
    allow_same_split: bool = False
    exclude_same_sample_id: bool = True
    exclude_same_source_sentence_id: bool = True
    exclude_same_source_video_id: bool = False
    exclude_identical_text: bool = False
    fail_on_leakage: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "allow_same_split",
            "exclude_same_sample_id",
            "exclude_same_source_sentence_id",
            "exclude_same_source_video_id",
            "exclude_identical_text",
            "fail_on_leakage",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise RetrievalPoseComparatorError(f"leakage.{field_name} must be boolean.")


@dataclass(frozen=True, slots=True)
class RetrievalGenerationConfig:
    candidates_per_query: int = 1
    generation_mode: str = "retrieval"
    length_policy: str = "retrieved_length"
    confidence_policy: str = "retrieved_confidence"

    def __post_init__(self) -> None:
        if self.candidates_per_query != 1:
            raise RetrievalPoseComparatorError("generation.candidates_per_query must be 1.")
        expected = {
            "generation_mode": "retrieval",
            "length_policy": "retrieved_length",
            "confidence_policy": "retrieved_confidence",
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise RetrievalPoseComparatorError(f"generation.{field_name} must be {value!r}.")


@dataclass(frozen=True, slots=True)
class RetrievalReportsConfig:
    enabled: bool = True
    output_dir_name: str = "retrieval_pose"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise RetrievalPoseComparatorError("reports.enabled must be boolean.")
        _require_text(self.output_dir_name, "reports.output_dir_name")


@dataclass(frozen=True, slots=True)
class RetrievalPoseComparatorConfig:
    source_path: Path | None
    raw_config: Mapping[str, object]
    identity: RetrievalComparatorIdentityConfig
    data: RetrievalComparatorDataConfig
    text_encoder: TextEncoderConfig
    retrieval: RetrievalScoringConfig
    leakage: RetrievalLeakagePolicyConfig
    generation: RetrievalGenerationConfig
    reports: RetrievalReportsConfig

    def __post_init__(self) -> None:
        if self.source_path is not None and not isinstance(self.source_path, Path):
            raise RetrievalPoseComparatorError("retrieval source_path must be a Path or None.")
        if self.text_encoder.trainable:
            raise RetrievalPoseComparatorError("text_encoder.trainable must be false.")
        object.__setattr__(self, "raw_config", MappingProxyType(dict(self.raw_config)))


def load_retrieval_pose_comparator_config(path: Path | str) -> RetrievalPoseComparatorConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RetrievalPoseComparatorError(f"could not read retrieval config: {source_path}") from exc
    except yaml.YAMLError as exc:
        raise RetrievalPoseComparatorError(f"could not parse retrieval config: {source_path}") from exc
    if not isinstance(loaded, Mapping):
        raise RetrievalPoseComparatorError("retrieval config root must be a mapping.")
    raw = copy.deepcopy(dict(loaded))
    _reject_forbidden_keys(raw)

    identity = _required_mapping(raw, "identity")
    data = _required_mapping(raw, "data")
    text_encoder = _required_mapping(raw, "text_encoder")
    retrieval = _required_mapping(raw, "retrieval")
    leakage = _required_mapping(raw, "leakage")
    generation = _required_mapping(raw, "generation")
    reports = _required_mapping(raw, "reports")
    text_encoder_trainable = _bool(text_encoder, "trainable")
    if text_encoder_trainable:
        raise RetrievalPoseComparatorError("text_encoder.trainable must be false.")
    return RetrievalPoseComparatorConfig(
        source_path=source_path,
        raw_config=raw,
        identity=RetrievalComparatorIdentityConfig(
            comparator_key=_str(identity, "comparator_key"),
            canonical_id=_str(identity, "canonical_id"),
            display_name=_str(identity, "display_name"),
            phase_number=_int(identity, "phase_number"),
            research_role=_str(identity, "research_role"),
            producer_type=_str(identity, "producer_type"),
            channel_policy=_str(identity, "channel_policy"),
            output_contract=_str(identity, "output_contract"),
            manifest_contract=_str(identity, "manifest_contract"),
        ),
        data=RetrievalComparatorDataConfig(
            manifest_family=(
                None
                if data.get("manifest_family") is None
                else parse_modeling_manifest_family(_str(data, "manifest_family"))
            ),
            bank_split=SampleSplit(_str(data, "bank_split")),
            query_splits=tuple(SampleSplit(value) for value in _str_sequence(data, "query_splits")),
            max_bank_size=_optional_int(data, "max_bank_size"),
            max_query_samples=_optional_int(data, "max_query_samples"),
        ),
        text_encoder=TextEncoderConfig(
            encoder_key=_str(text_encoder, "encoder_key"),
            backend=_str(text_encoder, "backend"),
            model_name_or_path=_optional_str(text_encoder, "model_name_or_path"),
            revision=_str(text_encoder, "revision"),
            local_files_only=_bool(text_encoder, "local_files_only"),
            pooling=_str(text_encoder, "pooling"),
            trainable=text_encoder_trainable,
            max_length=_int(text_encoder, "max_length"),
            embedding_dim=_optional_int(text_encoder, "embedding_dim"),
        ),
        retrieval=RetrievalScoringConfig(
            score_metric=_str(retrieval, "score_metric"),
            selection_policy=_str(retrieval, "selection_policy"),
            top_k=_int(retrieval, "top_k"),
            normalize_embeddings=_bool(retrieval, "normalize_embeddings"),
        ),
        leakage=RetrievalLeakagePolicyConfig(
            allow_same_split=_bool(leakage, "allow_same_split"),
            exclude_same_sample_id=_bool(leakage, "exclude_same_sample_id"),
            exclude_same_source_sentence_id=_bool(leakage, "exclude_same_source_sentence_id"),
            exclude_same_source_video_id=_bool(leakage, "exclude_same_source_video_id"),
            exclude_identical_text=_bool(leakage, "exclude_identical_text"),
            fail_on_leakage=_bool(leakage, "fail_on_leakage"),
        ),
        generation=RetrievalGenerationConfig(
            candidates_per_query=_int(generation, "candidates_per_query"),
            generation_mode=_str(generation, "generation_mode"),
            length_policy=_str(generation, "length_policy"),
            confidence_policy=_str(generation, "confidence_policy"),
        ),
        reports=RetrievalReportsConfig(
            enabled=_bool(reports, "enabled"),
            output_dir_name=_str(reports, "output_dir_name"),
        ),
    )


def _reject_forbidden_keys(value: object, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RetrievalPoseComparatorError("retrieval config keys must be strings.")
            child_path = key if not path else f"{path}.{key}"
            if key.lower() in _FORBIDDEN_KEYS:
                raise RetrievalPoseComparatorError(
                    f"retrieval config contains forbidden experimental key {child_path!r}."
                )
            _reject_forbidden_keys(child, path=child_path)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, path=f"{path}[{index}]")


def _required_mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise RetrievalPoseComparatorError(f"retrieval config section {key!r} must be a mapping.")
    return value


def _str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RetrievalPoseComparatorError(f"{key} must be a non-empty string.")
    return value


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RetrievalPoseComparatorError(f"{key} must be null or a non-empty string.")
    return value


def _int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrievalPoseComparatorError(f"{key} must be an integer.")
    return value


def _optional_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrievalPoseComparatorError(f"{key} must be an integer or null.")
    return value


def _bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise RetrievalPoseComparatorError(f"{key} must be boolean.")
    return value


def _str_sequence(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RetrievalPoseComparatorError(f"{key} must be a string sequence.")
    return tuple(_sequence_text(item, key) for item in value)


def _sequence_text(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalPoseComparatorError(f"{key} must contain non-empty strings.")
    return value


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalPoseComparatorError(f"{name} must be non-empty.")


def _require_positive_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RetrievalPoseComparatorError(f"{name} must be a positive integer.")


def _require_optional_positive_int(value: object, name: str) -> None:
    if value is None:
        return
    _require_positive_int(value, name)


__all__ = [
    "RetrievalComparatorDataConfig",
    "RetrievalComparatorIdentityConfig",
    "RetrievalGenerationConfig",
    "RetrievalLeakagePolicyConfig",
    "RetrievalPoseComparatorConfig",
    "RetrievalReportsConfig",
    "RetrievalScoringConfig",
    "load_retrieval_pose_comparator_config",
]
