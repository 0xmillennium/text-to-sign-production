"""Run record and aggregate contracts for proxy semantic alignment."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey

if TYPE_CHECKING:
    from text_to_sign_production.modeling.objectives.semantic_consistency.losses import (
        SemanticAlignmentResult,
    )

SEMANTIC_OBJECTIVE_RUN_RECORD_SCHEMA_VERSION = "t2sp-semantic-objective-run-record-v1"
SEMANTIC_ALIGNMENT_AGGREGATE_SCHEMA_VERSION = "t2sp-semantic-alignment-aggregate-v2"
SEMANTIC_GENERATED_CANDIDATE_POLICY_SCHEMA_VERSION = "t2sp-semantic-generated-candidate-policy-v1"
SEMANTIC_SINGLE_CANDIDATE_POLICY = "single_candidate_only"

_ELIGIBLE_MODELS = frozenset(
    {ModelKey.LEARNED_POSE_TOKEN, ModelKey.LATENT_DIFFUSION, ModelKey.ARTICULATOR_AWARE}
)
_AGGREGATE_CAVEATS = (
    "Embedding similarity is not proof of sign intelligibility.",
    "Embedding similarity is not proof of linguistic semantic correctness.",
    "Deterministic hash text embeddings are proxy-only.",
    "Pose statistics projection is proxy-only.",
)


@dataclass(frozen=True, slots=True)
class SemanticGeneratedCandidatePolicy:
    schema_version: str
    policy: str
    required_generation_index: int
    max_candidates_per_sample: int

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_GENERATED_CANDIDATE_POLICY_SCHEMA_VERSION, "candidate policy")
        if self.policy != SEMANTIC_SINGLE_CANDIDATE_POLICY:
            raise SemanticConsistencyError("semantic generated candidate policy must be single_candidate_only.")
        if self.required_generation_index != 0:
            raise SemanticConsistencyError("semantic generated candidate policy requires generation_index=0.")
        if self.max_candidates_per_sample != 1:
            raise SemanticConsistencyError("semantic generated candidate policy supports one candidate per sample.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "required_generation_index": self.required_generation_index,
            "max_candidates_per_sample": self.max_candidates_per_sample,
        }


def single_candidate_policy() -> SemanticGeneratedCandidatePolicy:
    return SemanticGeneratedCandidatePolicy(
        schema_version=SEMANTIC_GENERATED_CANDIDATE_POLICY_SCHEMA_VERSION,
        policy=SEMANTIC_SINGLE_CANDIDATE_POLICY,
        required_generation_index=0,
        max_candidates_per_sample=1,
    )


@dataclass(frozen=True, slots=True)
class SemanticObjectiveRunRecord:
    schema_version: str
    objective_key: ObjectiveKey
    model_key: ModelKey
    run_name: str
    split: SampleSplit
    objective_attached: bool
    ablation_required: bool
    proxy_only: bool
    records_count: int
    skipped_count: int

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_OBJECTIVE_RUN_RECORD_SCHEMA_VERSION, "run record")
        objective, model, split = _identity(self.objective_key, self.model_key, self.split)
        object.__setattr__(self, "objective_key", objective)
        object.__setattr__(self, "model_key", model)
        object.__setattr__(self, "split", split)
        _text(self.run_name, "run_name")
        if not isinstance(self.objective_attached, bool):
            raise SemanticConsistencyError("objective_attached must be a boolean.")
        if self.ablation_required is not True:
            raise SemanticConsistencyError("semantic run record ablation_required must be true.")
        if self.proxy_only is not True:
            raise SemanticConsistencyError("semantic run record proxy_only must be true.")
        _counts(self.records_count, self.skipped_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "objective_key": ObjectiveKey(self.objective_key).value,
            "model_key": ModelKey(self.model_key).value,
            "run_name": self.run_name,
            "split": SampleSplit(self.split).value,
            "objective_attached": self.objective_attached,
            "ablation_required": self.ablation_required,
            "proxy_only": self.proxy_only,
            "records_count": self.records_count,
            "skipped_count": self.skipped_count,
        }


@dataclass(frozen=True, slots=True)
class SemanticAlignmentAggregate:
    schema_version: str
    objective_key: ObjectiveKey
    model_key: ModelKey
    split: SampleSplit
    candidate_policy: str
    records_count: int
    skipped_count: int
    mean_cosine_similarity: float | None
    mean_cosine_distance: float | None
    mean_loss_value: float | None
    caveats: tuple[str, ...]

    def __post_init__(self) -> None:
        _schema(self.schema_version, SEMANTIC_ALIGNMENT_AGGREGATE_SCHEMA_VERSION, "aggregate")
        objective, model, split = _identity(self.objective_key, self.model_key, self.split)
        object.__setattr__(self, "objective_key", objective)
        object.__setattr__(self, "model_key", model)
        object.__setattr__(self, "split", split)
        if self.candidate_policy != SEMANTIC_SINGLE_CANDIDATE_POLICY:
            raise SemanticConsistencyError("semantic aggregate candidate_policy must be single_candidate_only.")
        _counts(self.records_count, self.skipped_count)
        values = (self.mean_cosine_similarity, self.mean_cosine_distance, self.mean_loss_value)
        observed_count = self.records_count - self.skipped_count
        if observed_count == 0 and any(value is not None for value in values):
            raise SemanticConsistencyError(
                "all-skipped semantic aggregate numeric means must be None, not zero."
            )
        if observed_count > 0 and any(value is None for value in values):
            raise SemanticConsistencyError(
                "semantic aggregate with computed results must include all numeric means."
            )
        for value, name in zip(
            values,
            ("mean_cosine_similarity", "mean_cosine_distance", "mean_loss_value"),
            strict=True,
        ):
            if value is not None:
                _finite(value, name)
        caveats = tuple(self.caveats)
        for caveat in _AGGREGATE_CAVEATS:
            if caveat not in caveats:
                raise SemanticConsistencyError(f"semantic aggregate caveats must include: {caveat}")
        object.__setattr__(self, "caveats", caveats)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "objective_key": ObjectiveKey(self.objective_key).value,
            "model_key": ModelKey(self.model_key).value,
            "split": SampleSplit(self.split).value,
            "candidate_policy": self.candidate_policy,
            "records_count": self.records_count,
            "skipped_count": self.skipped_count,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "mean_cosine_distance": self.mean_cosine_distance,
            "mean_loss_value": self.mean_loss_value,
            "caveats": list(self.caveats),
        }


def aggregate_semantic_alignment_results(
    records: Iterable[SemanticAlignmentResult],
    *,
    model_key: ModelKey,
    split: SampleSplit,
) -> SemanticAlignmentAggregate:
    """Aggregate non-skipped proxy alignments while keeping unavailable means absent."""

    try:
        model = ModelKey(model_key)
        resolved_split = SampleSplit(split)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError("semantic aggregate request has an invalid model or split.") from exc
    if model not in _ELIGIBLE_MODELS:
        raise SemanticConsistencyError(
            f"semantic alignment aggregate cannot be produced for model {model.value!r}."
        )
    materialized = tuple(records)
    from text_to_sign_production.modeling.objectives.semantic_consistency.losses import (
        SemanticAlignmentResult,
    )

    computed: list[SemanticAlignmentResult] = []
    for record in materialized:
        if not isinstance(record, SemanticAlignmentResult):
            raise SemanticConsistencyError("records must contain SemanticAlignmentResult values.")
        if record.model_key is not model:
            raise SemanticConsistencyError("semantic alignment record model_key does not match aggregate.")
        if record.split is not resolved_split:
            raise SemanticConsistencyError("semantic alignment record split does not match aggregate.")
        if record.candidate_policy != SEMANTIC_SINGLE_CANDIDATE_POLICY:
            raise SemanticConsistencyError("semantic alignment candidate_policy does not match aggregate policy.")
        if not record.skipped:
            computed.append(record)
    return SemanticAlignmentAggregate(
        schema_version=SEMANTIC_ALIGNMENT_AGGREGATE_SCHEMA_VERSION,
        objective_key=ObjectiveKey.SEMANTIC_CONSISTENCY,
        model_key=model,
        split=resolved_split,
        candidate_policy=SEMANTIC_SINGLE_CANDIDATE_POLICY,
        records_count=len(materialized),
        skipped_count=len(materialized) - len(computed),
        mean_cosine_similarity=_mean(item.cosine_similarity for item in computed),
        mean_cosine_distance=_mean(item.cosine_distance for item in computed),
        mean_loss_value=_mean(item.loss_value for item in computed),
        caveats=_AGGREGATE_CAVEATS,
    )


def _mean(values: Iterable[float | None]) -> float | None:
    materialized = tuple(value for value in values if value is not None)
    return None if not materialized else float(np.mean(np.asarray(materialized, dtype=np.float64)))


def _identity(
    objective_key: ObjectiveKey,
    model_key: ModelKey,
    split: SampleSplit,
) -> tuple[ObjectiveKey, ModelKey, SampleSplit]:
    try:
        objective = ObjectiveKey(objective_key)
        model = ModelKey(model_key)
        resolved_split = SampleSplit(split)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError("semantic record contains an invalid enum value.") from exc
    if objective is not ObjectiveKey.SEMANTIC_CONSISTENCY:
        raise SemanticConsistencyError("semantic record objective_key must be semantic_consistency.")
    if model not in _ELIGIBLE_MODELS:
        raise SemanticConsistencyError("semantic records cannot attach to base_direct.")
    return objective, model, resolved_split


def _schema(value: str, expected: str, label: str) -> None:
    if value != expected:
        raise SemanticConsistencyError(f"semantic {label} schema_version is unsupported.")


def _counts(records_count: int, skipped_count: int) -> None:
    for value, name in ((records_count, "records_count"), (skipped_count, "skipped_count")):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SemanticConsistencyError(f"{name} must be a non-negative integer.")
    if skipped_count > records_count:
        raise SemanticConsistencyError("skipped_count must not exceed records_count.")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{name} must be non-empty.")


def _finite(value: object, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SemanticConsistencyError(f"{name} must be finite.")


__all__ = [
    "SEMANTIC_ALIGNMENT_AGGREGATE_SCHEMA_VERSION",
    "SEMANTIC_GENERATED_CANDIDATE_POLICY_SCHEMA_VERSION",
    "SEMANTIC_OBJECTIVE_RUN_RECORD_SCHEMA_VERSION",
    "SEMANTIC_SINGLE_CANDIDATE_POLICY",
    "SemanticAlignmentAggregate",
    "SemanticGeneratedCandidatePolicy",
    "SemanticObjectiveRunRecord",
    "aggregate_semantic_alignment_results",
    "single_candidate_policy",
]
