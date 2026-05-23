"""Proxy-only BFH statistics projection records for semantic consistency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from text_to_sign_production.core.integrity import sha256_json
from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.data.gate.pose.types import PoseChannel
from text_to_sign_production.modeling.backbones.bfh_vectorization import BfhVectorizedPose
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticPoseEmbeddingConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.records import (
    SEMANTIC_SINGLE_CANDIDATE_POLICY,
)

SEMANTIC_POSE_FEATURE_SPEC_SCHEMA_VERSION = "t2sp-semantic-pose-feature-spec-v2"
SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION = "t2sp-semantic-pose-embedding-v3"
_CANONICAL_CHANNEL_GROUPS = (
    PoseChannel.BODY,
    PoseChannel.LEFT_HAND,
    PoseChannel.RIGHT_HAND,
    PoseChannel.FACE,
)


@dataclass(frozen=True, slots=True)
class SemanticPoseFeatureSpec:
    schema_version: str
    backend: str
    coordinate_mode: str
    channel_groups: tuple[PoseChannel, ...]
    include_velocity_statistics: bool
    include_validity_statistics: bool
    projection_seed: int
    embedding_dim: int
    proxy_only: bool

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_POSE_FEATURE_SPEC_SCHEMA_VERSION:
            raise SemanticConsistencyError("semantic pose feature spec schema_version is unsupported.")
        if self.backend != "bfh_statistics_projection" or self.coordinate_mode != "xy":
            raise SemanticConsistencyError(
                "semantic pose features require backend='bfh_statistics_projection' and coordinate_mode='xy'."
            )
        try:
            channels = tuple(PoseChannel(channel) for channel in self.channel_groups)
        except (TypeError, ValueError) as exc:
            raise SemanticConsistencyError("semantic pose feature spec contains an invalid channel group.") from exc
        if channels != _CANONICAL_CHANNEL_GROUPS:
            raise SemanticConsistencyError(
                "semantic pose feature spec channel_groups must be "
                "body,left_hand,right_hand,face in canonical order."
            )
        for value, name in (
            (self.include_velocity_statistics, "include_velocity_statistics"),
            (self.include_validity_statistics, "include_validity_statistics"),
        ):
            if not isinstance(value, bool):
                raise SemanticConsistencyError(
                    f"semantic pose feature spec {name} must be a boolean."
                )
        if (
            not isinstance(self.projection_seed, int)
            or isinstance(self.projection_seed, bool)
            or self.projection_seed < 0
        ):
            raise SemanticConsistencyError("semantic pose projection_seed must be non-negative.")
        if not isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, bool) or self.embedding_dim <= 0:
            raise SemanticConsistencyError("semantic pose embedding_dim must be positive.")
        if self.proxy_only is not True:
            raise SemanticConsistencyError("semantic pose feature spec proxy_only must be true.")
        object.__setattr__(self, "channel_groups", channels)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "coordinate_mode": self.coordinate_mode,
            "channel_groups": [channel.value for channel in self.channel_groups],
            "include_velocity_statistics": self.include_velocity_statistics,
            "include_validity_statistics": self.include_validity_statistics,
            "projection_seed": self.projection_seed,
            "embedding_dim": self.embedding_dim,
            "proxy_only": self.proxy_only,
        }


@dataclass(frozen=True, slots=True)
class SemanticPoseEmbeddingRecord:
    schema_version: str
    sample_id: str
    split: SampleSplit
    generation_index: int
    candidate_policy: str
    source_sentence_name: str
    backend: str
    embedding_dim: int
    coordinate_mode: str
    channel_groups: tuple[PoseChannel, ...]
    include_velocity_statistics: bool
    include_validity_statistics: bool
    projection_seed: int
    embedding: np.ndarray | None
    valid_observation_count: int
    skipped: bool
    reason: str | None
    proxy_only: bool
    spec_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION:
            raise SemanticConsistencyError("semantic pose embedding schema_version is unsupported.")
        _text(self.sample_id, "sample_id")
        _text(self.source_sentence_name, "source_sentence_name")
        try:
            object.__setattr__(self, "split", SampleSplit(self.split))
        except (TypeError, ValueError) as exc:
            raise SemanticConsistencyError(f"semantic pose embedding split is invalid: {self.split!r}.") from exc
        if self.generation_index != 0:
            raise SemanticConsistencyError("semantic pose embedding generation_index must be 0.")
        if self.candidate_policy != SEMANTIC_SINGLE_CANDIDATE_POLICY:
            raise SemanticConsistencyError(
                "semantic pose embedding candidate_policy must be single_candidate_only."
            )
        if self.backend != "bfh_statistics_projection":
            raise SemanticConsistencyError(
                "semantic pose embedding backend must be 'bfh_statistics_projection'."
            )
        if not isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, bool) or self.embedding_dim <= 0:
            raise SemanticConsistencyError("semantic pose embedding_dim must be positive.")
        spec = SemanticPoseFeatureSpec(
            schema_version=SEMANTIC_POSE_FEATURE_SPEC_SCHEMA_VERSION,
            backend=self.backend,
            coordinate_mode=self.coordinate_mode,
            channel_groups=self.channel_groups,
            include_velocity_statistics=self.include_velocity_statistics,
            include_validity_statistics=self.include_validity_statistics,
            projection_seed=self.projection_seed,
            embedding_dim=self.embedding_dim,
            proxy_only=self.proxy_only,
        )
        if self.spec_hash != semantic_pose_spec_hash(spec):
            raise SemanticConsistencyError(
                "semantic pose embedding spec_hash does not match its pose feature specification."
            )
        object.__setattr__(self, "channel_groups", spec.channel_groups)
        if (
            not isinstance(self.valid_observation_count, int)
            or isinstance(self.valid_observation_count, bool)
            or self.valid_observation_count < 0
        ):
            raise SemanticConsistencyError("valid_observation_count must be non-negative.")
        if not isinstance(self.skipped, bool):
            raise SemanticConsistencyError("semantic pose embedding skipped must be a boolean.")
        if self.proxy_only is not True:
            raise SemanticConsistencyError("semantic pose embedding proxy_only must be true.")
        if self.skipped:
            if self.embedding is not None:
                raise SemanticConsistencyError(
                    "skipped semantic pose embedding must use embedding=None, not a fake vector."
                )
            _text(self.reason, "reason")
        else:
            if self.embedding is None:
                raise SemanticConsistencyError(
                    "computed semantic pose embedding must include an embedding."
                )
            if self.valid_observation_count <= 0:
                raise SemanticConsistencyError(
                    "computed semantic pose embedding requires valid observations."
                )
            if self.reason is not None:
                raise SemanticConsistencyError(
                    "computed semantic pose embedding must not include a skip reason."
                )
            object.__setattr__(
                self,
                "embedding",
                _embedding(self.embedding, self.embedding_dim, "pose embedding"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "split": self.split.value,
            "generation_index": self.generation_index,
            "candidate_policy": self.candidate_policy,
            "source_sentence_name": self.source_sentence_name,
            "backend": self.backend,
            "embedding_dim": self.embedding_dim,
            "coordinate_mode": self.coordinate_mode,
            "channel_groups": [channel.value for channel in self.channel_groups],
            "include_velocity_statistics": self.include_velocity_statistics,
            "include_validity_statistics": self.include_validity_statistics,
            "projection_seed": self.projection_seed,
            "embedding": None if self.embedding is None else self.embedding.tolist(),
            "valid_observation_count": self.valid_observation_count,
            "skipped": self.skipped,
            "reason": self.reason,
            "proxy_only": self.proxy_only,
            "spec_hash": self.spec_hash,
        }


def build_pose_feature_spec(
    config: SemanticPoseEmbeddingConfig,
) -> SemanticPoseFeatureSpec:
    """Build the deterministic proxy pose-feature declaration."""

    if not isinstance(config, SemanticPoseEmbeddingConfig):
        raise SemanticConsistencyError("config must be a SemanticPoseEmbeddingConfig.")
    return SemanticPoseFeatureSpec(
        schema_version=SEMANTIC_POSE_FEATURE_SPEC_SCHEMA_VERSION,
        backend=config.backend,
        coordinate_mode=config.coordinate_mode,
        channel_groups=config.channel_groups,
        include_velocity_statistics=config.include_velocity_statistics,
        include_validity_statistics=config.include_validity_statistics,
        projection_seed=config.projection_seed,
        embedding_dim=config.embedding_dim,
        proxy_only=config.proxy_only,
    )


def compute_pose_embedding_record(
    *,
    sample_id: str,
    split: SampleSplit,
    generation_index: int,
    candidate_policy: str,
    source_sentence_name: str,
    vectorized: BfhVectorizedPose,
    config: SemanticPoseEmbeddingConfig,
) -> SemanticPoseEmbeddingRecord:
    """Project masked BFH summary statistics; this proxy is not sign interpretation."""

    if not isinstance(vectorized, BfhVectorizedPose):
        raise SemanticConsistencyError("vectorized must be a BfhVectorizedPose.")
    spec = build_pose_feature_spec(config)
    valid_observations = int(np.count_nonzero(vectorized.validity_mask))
    if valid_observations == 0:
        raise SemanticConsistencyError(
            "semantic pose embedding cannot be computed: all BFH observations are invalid."
        )
    features: list[float] = []
    for channel in spec.channel_groups:
        joint_slice = vectorized.layout.channel_slices.get(channel)
        if joint_slice is None:
            raise SemanticConsistencyError(
                f"semantic pose embedding channel {channel.value!r} is absent from the BFH layout."
            )
        valid = vectorized.validity_mask[:, joint_slice]
        coordinates = vectorized.values[:, joint_slice, :]
        selected = coordinates[valid]
        if selected.size:
            features.extend(float(value) for value in np.mean(selected, axis=0))
            features.extend(float(value) for value in np.std(selected, axis=0))
        else:
            features.extend((0.0, 0.0, 0.0, 0.0))
        if spec.include_velocity_statistics:
            valid_velocity = valid[1:] & valid[:-1]
            delta = coordinates[1:] - coordinates[:-1]
            velocities = np.linalg.norm(delta[valid_velocity], axis=-1)
            features.append(float(np.mean(velocities)) if velocities.size else 0.0)
        if spec.include_validity_statistics:
            features.append(float(np.count_nonzero(valid)) / float(valid.size))
    feature_vector = np.asarray(features, dtype=np.float32)
    if not np.all(np.isfinite(feature_vector)):
        raise SemanticConsistencyError("semantic pose statistics contain non-finite values.")
    rng = np.random.default_rng(spec.projection_seed)
    projection = rng.standard_normal(
        (feature_vector.shape[0], spec.embedding_dim),
        dtype=np.float32,
    ) / np.sqrt(float(feature_vector.shape[0]))
    embedding = feature_vector @ projection
    return SemanticPoseEmbeddingRecord(
        schema_version=SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION,
        sample_id=sample_id,
        split=split,
        generation_index=generation_index,
        candidate_policy=candidate_policy,
        source_sentence_name=source_sentence_name,
        backend=spec.backend,
        embedding_dim=spec.embedding_dim,
        coordinate_mode=spec.coordinate_mode,
        channel_groups=spec.channel_groups,
        include_velocity_statistics=spec.include_velocity_statistics,
        include_validity_statistics=spec.include_validity_statistics,
        projection_seed=spec.projection_seed,
        embedding=embedding,
        valid_observation_count=valid_observations,
        skipped=False,
        reason=None,
        proxy_only=spec.proxy_only,
        spec_hash=semantic_pose_spec_hash(spec),
    )


def semantic_pose_spec_hash(spec: SemanticPoseFeatureSpec) -> str:
    """Fingerprint the canonical pose proxy specification carried by records."""

    if not isinstance(spec, SemanticPoseFeatureSpec):
        raise SemanticConsistencyError("spec must be a SemanticPoseFeatureSpec.")
    return sha256_json(spec.to_dict())


def _embedding(value: object, expected_dim: int, label: str) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).copy()
    if embedding.shape != (expected_dim,):
        raise SemanticConsistencyError(
            f"{label} must have shape ({expected_dim},); got {embedding.shape}."
        )
    if not np.all(np.isfinite(embedding)):
        raise SemanticConsistencyError(f"{label} must contain finite values.")
    if float(np.linalg.norm(embedding)) <= 0.0:
        raise SemanticConsistencyError(
            f"{label} has zero norm; proxy statistics cannot support cosine alignment."
        )
    embedding.setflags(write=False)
    return embedding


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{name} must be non-empty.")


__all__ = [
    "SEMANTIC_POSE_EMBEDDING_SCHEMA_VERSION",
    "SEMANTIC_POSE_FEATURE_SPEC_SCHEMA_VERSION",
    "SemanticPoseEmbeddingRecord",
    "SemanticPoseFeatureSpec",
    "build_pose_feature_spec",
    "compute_pose_embedding_record",
    "semantic_pose_spec_hash",
]
