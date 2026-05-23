"""Proxy alignment loss/result contracts for the semantic-consistency foundation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticAlignmentConfig,
    SemanticConsistencyObjectiveConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.embeddings import (
    SemanticTextEmbeddingRecord,
    semantic_text_config_hash,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.pose_features import (
    SemanticPoseEmbeddingRecord,
    build_pose_feature_spec,
    semantic_pose_spec_hash,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.records import (
    SEMANTIC_SINGLE_CANDIDATE_POLICY,
)
from text_to_sign_production.modeling.research import ModelKey, ObjectiveKey

SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION = "t2sp-semantic-alignment-result-v3"
_ELIGIBLE_MODELS = frozenset(
    {ModelKey.LEARNED_POSE_TOKEN, ModelKey.LATENT_DIFFUSION, ModelKey.ARTICULATOR_AWARE}
)


@dataclass(frozen=True, slots=True)
class SemanticAlignmentResult:
    schema_version: str
    sample_id: str
    split: SampleSplit
    generation_index: int
    candidate_policy: str
    model_key: ModelKey
    objective_key: ObjectiveKey
    metric: str
    cosine_similarity: float | None
    cosine_distance: float | None
    loss_weight: float
    loss_value: float | None
    skipped: bool
    reason: str | None
    proxy_only: bool

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION:
            raise SemanticConsistencyError("semantic alignment result schema_version is unsupported.")
        _text(self.sample_id, "sample_id")
        try:
            object.__setattr__(self, "split", SampleSplit(self.split))
            object.__setattr__(self, "model_key", ModelKey(self.model_key))
            object.__setattr__(self, "objective_key", ObjectiveKey(self.objective_key))
        except (TypeError, ValueError) as exc:
            raise SemanticConsistencyError("semantic alignment result contains an invalid enum value.") from exc
        if self.generation_index != 0:
            raise SemanticConsistencyError("semantic alignment generation_index must be 0.")
        if self.candidate_policy != SEMANTIC_SINGLE_CANDIDATE_POLICY:
            raise SemanticConsistencyError(
                "semantic alignment candidate_policy must be single_candidate_only."
            )
        if self.objective_key is not ObjectiveKey.SEMANTIC_CONSISTENCY:
            raise SemanticConsistencyError("semantic alignment objective_key must be semantic_consistency.")
        if self.model_key not in _ELIGIBLE_MODELS:
            raise SemanticConsistencyError(
                "semantic alignment result cannot be constructed for base_direct."
            )
        if self.metric != "cosine_distance":
            raise SemanticConsistencyError("semantic alignment metric must be 'cosine_distance'.")
        if not isinstance(self.skipped, bool):
            raise SemanticConsistencyError("semantic alignment skipped must be a boolean.")
        if self.proxy_only is not True:
            raise SemanticConsistencyError("semantic alignment result proxy_only must be true.")
        _finite(self.loss_weight, "loss_weight")
        if float(self.loss_weight) < 0.0:
            raise SemanticConsistencyError("semantic alignment loss_weight must be non-negative.")
        values = (self.cosine_similarity, self.cosine_distance, self.loss_value)
        if self.skipped:
            if any(value is not None for value in values):
                raise SemanticConsistencyError(
                    "skipped semantic alignment must use None numeric values, not fake zero."
                )
            _text(self.reason, "reason")
        else:
            if any(value is None for value in values):
                raise SemanticConsistencyError(
                    "computed semantic alignment must contain finite similarity, distance, and loss."
                )
            for value, name in zip(
                values,
                ("cosine_similarity", "cosine_distance", "loss_value"),
                strict=True,
            ):
                _finite(value, name)
            similarity = float(self.cosine_similarity)
            distance = float(self.cosine_distance)
            loss_value = float(self.loss_value)
            if not -1.0 - 1e-9 <= similarity <= 1.0 + 1e-9:
                raise SemanticConsistencyError("cosine_similarity must be within [-1, 1].")
            if not math.isclose(distance, 1.0 - similarity, rel_tol=1e-9, abs_tol=1e-9):
                raise SemanticConsistencyError(
                    "cosine_distance must equal 1 - cosine_similarity."
                )
            if loss_value < 0.0:
                raise SemanticConsistencyError("semantic alignment loss_value must be non-negative.")
            if not math.isclose(
                loss_value,
                distance * float(self.loss_weight),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise SemanticConsistencyError(
                    "semantic alignment loss_value must equal cosine_distance * loss_weight."
                )
            if self.reason is not None:
                raise SemanticConsistencyError(
                    "computed semantic alignment must not contain a skip reason."
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split.value,
            "generation_index": self.generation_index,
            "candidate_policy": self.candidate_policy,
            "model_key": self.model_key.value,
            "objective_key": self.objective_key.value,
            "metric": self.metric,
            "cosine_similarity": self.cosine_similarity,
            "cosine_distance": self.cosine_distance,
            "loss_weight": float(self.loss_weight),
            "loss_value": self.loss_value,
            "skipped": self.skipped,
            "reason": self.reason,
            "proxy_only": self.proxy_only,
        }


def compute_cosine_alignment(
    *,
    text_embedding: np.ndarray,
    pose_embedding: np.ndarray,
    config: SemanticAlignmentConfig,
) -> tuple[float, float]:
    """Compute finite cosine similarity/distance between two proxy embeddings."""

    if not isinstance(config, SemanticAlignmentConfig):
        raise SemanticConsistencyError("config must be a SemanticAlignmentConfig.")
    text = _vector(text_embedding, "text_embedding")
    pose = _vector(pose_embedding, "pose_embedding")
    if text.shape != pose.shape:
        raise SemanticConsistencyError(
            f"text_embedding and pose_embedding shapes must match; got {text.shape} and {pose.shape}."
        )
    text_norm = float(np.linalg.norm(text))
    pose_norm = float(np.linalg.norm(pose))
    if text_norm <= 0.0 or pose_norm <= 0.0:
        raise SemanticConsistencyError(
            "cosine alignment cannot use a zero-norm embedding; check proxy record construction."
        )
    similarity = float(np.dot(text / text_norm, pose / pose_norm))
    similarity = float(np.clip(similarity, -1.0, 1.0))
    distance = 1.0 - similarity
    _finite(similarity, "cosine_similarity")
    _finite(distance, "cosine_distance")
    return similarity, distance


def compute_semantic_alignment_result(
    *,
    sample_id: str,
    split: SampleSplit,
    model_key: ModelKey,
    text_embedding: SemanticTextEmbeddingRecord,
    pose_embedding: SemanticPoseEmbeddingRecord,
    config: SemanticConsistencyObjectiveConfig,
) -> SemanticAlignmentResult:
    """Compute or explicitly skip a proxy semantic-alignment result."""

    if not isinstance(config, SemanticConsistencyObjectiveConfig):
        raise SemanticConsistencyError("config must be a SemanticConsistencyObjectiveConfig.")
    if not isinstance(text_embedding, SemanticTextEmbeddingRecord):
        raise SemanticConsistencyError("text_embedding must be a SemanticTextEmbeddingRecord.")
    if not isinstance(pose_embedding, SemanticPoseEmbeddingRecord):
        raise SemanticConsistencyError("pose_embedding must be a SemanticPoseEmbeddingRecord.")
    try:
        model = ModelKey(model_key)
        resolved_split = SampleSplit(split)
    except (TypeError, ValueError) as exc:
        raise SemanticConsistencyError("semantic alignment request has an invalid model or split.") from exc
    if model not in config.attachment.allowed_models:
        raise SemanticConsistencyError(
            f"semantic alignment cannot be recorded for model {model.value!r}; "
            "it is not an allowed attachment model."
        )
    if text_embedding.sample_id != sample_id or pose_embedding.sample_id != sample_id:
        raise SemanticConsistencyError(
            "semantic alignment sample_id must match both embedding records."
        )
    if pose_embedding.split is not resolved_split:
        raise SemanticConsistencyError("semantic alignment split must match the pose embedding.")
    if pose_embedding.candidate_policy != SEMANTIC_SINGLE_CANDIDATE_POLICY:
        raise SemanticConsistencyError("semantic alignment requires the single_candidate_only policy.")
    if text_embedding.embedding_dim != pose_embedding.embedding_dim:
        raise SemanticConsistencyError("semantic alignment embedding dimensions do not match.")
    if text_embedding.config_hash != semantic_text_config_hash(
        backend=config.text_embedding.backend,
        embedding_dim=config.text_embedding.embedding_dim,
        pooling=config.text_embedding.pooling,
        trainable=config.text_embedding.trainable,
        proxy_only=config.text_embedding.proxy_only,
    ):
        raise SemanticConsistencyError(
            "semantic alignment text embedding specification does not match objective config."
        )
    if pose_embedding.spec_hash != semantic_pose_spec_hash(
        build_pose_feature_spec(config.pose_embedding)
    ):
        raise SemanticConsistencyError(
            "semantic alignment pose embedding specification does not match objective config."
        )
    if pose_embedding.skipped:
        return SemanticAlignmentResult(
            schema_version=SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION,
            sample_id=sample_id,
            split=resolved_split,
            generation_index=pose_embedding.generation_index,
            candidate_policy=pose_embedding.candidate_policy,
            model_key=model,
            objective_key=config.identity.objective_key,
            metric=config.alignment.metric,
            cosine_similarity=None,
            cosine_distance=None,
            loss_weight=float(config.alignment.loss_weight),
            loss_value=None,
            skipped=True,
            reason=pose_embedding.reason,
            proxy_only=True,
        )
    if pose_embedding.embedding is None:
        raise SemanticConsistencyError(
            "computed semantic pose embedding unexpectedly lacks an embedding."
        )
    similarity, distance = compute_cosine_alignment(
        text_embedding=text_embedding.embedding,
        pose_embedding=pose_embedding.embedding,
        config=config.alignment,
    )
    loss_value = distance * float(config.alignment.loss_weight)
    return SemanticAlignmentResult(
        schema_version=SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION,
        sample_id=sample_id,
        split=resolved_split,
        generation_index=pose_embedding.generation_index,
        candidate_policy=pose_embedding.candidate_policy,
        model_key=model,
        objective_key=config.identity.objective_key,
        metric=config.alignment.metric,
        cosine_similarity=similarity,
        cosine_distance=distance,
        loss_weight=float(config.alignment.loss_weight),
        loss_value=loss_value,
        skipped=False,
        reason=None,
        proxy_only=True,
    )


def _vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size <= 0:
        raise SemanticConsistencyError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(vector)):
        raise SemanticConsistencyError(f"{name} contains non-finite values.")
    return vector


def _finite(value: object, name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise SemanticConsistencyError(f"{name} must be finite.")


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{name} must be non-empty.")


__all__ = [
    "SEMANTIC_ALIGNMENT_RESULT_SCHEMA_VERSION",
    "SemanticAlignmentResult",
    "compute_cosine_alignment",
    "compute_semantic_alignment_result",
]
