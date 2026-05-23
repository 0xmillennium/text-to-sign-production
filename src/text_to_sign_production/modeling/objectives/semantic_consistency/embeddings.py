"""Proxy-only deterministic text embedding records for semantic consistency."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from text_to_sign_production.core.integrity import sha256_json
from text_to_sign_production.modeling.backbones.text_encoder import (
    TextEncoder,
    TextEncoderConfig,
    build_text_encoder,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.config import (
    SemanticTextEmbeddingConfig,
)
from text_to_sign_production.modeling.objectives.semantic_consistency.errors import (
    SemanticConsistencyError,
)

SEMANTIC_TEXT_EMBEDDING_SCHEMA_VERSION = "t2sp-semantic-text-embedding-v2"


@dataclass(frozen=True, slots=True)
class SemanticTextEmbeddingRecord:
    schema_version: str
    sample_id: str
    source_sentence_name: str
    text: str
    backend: str
    embedding_dim: int
    pooling: str
    trainable: bool
    embedding: np.ndarray
    proxy_only: bool
    config_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_TEXT_EMBEDDING_SCHEMA_VERSION:
            raise SemanticConsistencyError("semantic text embedding schema_version is unsupported.")
        for value, name in (
            (self.sample_id, "sample_id"),
            (self.source_sentence_name, "source_sentence_name"),
            (self.text, "text"),
        ):
            _text(value, name)
        if self.backend != "deterministic_hash":
            raise SemanticConsistencyError(
                "semantic text embedding backend must be 'deterministic_hash'."
            )
        if not isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, bool) or self.embedding_dim <= 0:
            raise SemanticConsistencyError("semantic text embedding_dim must be positive.")
        if self.pooling != "mean" or self.trainable is not False:
            raise SemanticConsistencyError(
                "semantic text embedding records require pooling='mean' and trainable=false."
            )
        embedding = _embedding(self.embedding, self.embedding_dim, "text embedding")
        if self.proxy_only is not True:
            raise SemanticConsistencyError(
                "semantic text embedding proxy_only must be true; hash embeddings are proxies."
            )
        expected_hash = semantic_text_config_hash(
            backend=self.backend,
            embedding_dim=self.embedding_dim,
            pooling=self.pooling,
            trainable=self.trainable,
            proxy_only=self.proxy_only,
        )
        if self.config_hash != expected_hash:
            raise SemanticConsistencyError(
                "semantic text embedding config_hash does not match its embedding specification."
            )
        object.__setattr__(self, "embedding", embedding)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "source_sentence_name": self.source_sentence_name,
            "text": self.text,
            "backend": self.backend,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "trainable": self.trainable,
            "embedding": self.embedding.tolist(),
            "proxy_only": self.proxy_only,
            "config_hash": self.config_hash,
        }


def build_semantic_text_encoder(config: SemanticTextEmbeddingConfig) -> TextEncoder:
    """Build the shared deterministic hash backend for proxy-only text records."""

    if not isinstance(config, SemanticTextEmbeddingConfig):
        raise SemanticConsistencyError("config must be a SemanticTextEmbeddingConfig.")
    return build_text_encoder(
        TextEncoderConfig(
            encoder_key="semantic_consistency_proxy_text",
            model_name_or_path=None,
            pooling=config.pooling,
            trainable=config.trainable,
            max_length=512,
            embedding_dim=config.embedding_dim,
            backend=config.backend,
        )
    )


def compute_text_embedding_record(
    *,
    sample_id: str,
    source_sentence_name: str,
    text: str,
    config: SemanticTextEmbeddingConfig,
) -> SemanticTextEmbeddingRecord:
    """Compute a deterministic hash-text proxy; it is not semantic understanding."""

    encoder = build_semantic_text_encoder(config)
    try:
        output = encoder.encode_one(text, sentence_name=source_sentence_name)
    except ValueError as exc:
        raise SemanticConsistencyError(f"semantic text embedding could not be computed: {exc}") from exc
    return SemanticTextEmbeddingRecord(
        schema_version=SEMANTIC_TEXT_EMBEDDING_SCHEMA_VERSION,
        sample_id=sample_id,
        source_sentence_name=source_sentence_name,
        text=text,
        backend=output.backend,
        embedding_dim=config.embedding_dim,
        pooling=config.pooling,
        trainable=config.trainable,
        embedding=output.embedding,
        proxy_only=config.proxy_only,
        config_hash=semantic_text_config_hash(
            backend=config.backend,
            embedding_dim=config.embedding_dim,
            pooling=config.pooling,
            trainable=config.trainable,
            proxy_only=config.proxy_only,
        ),
    )


def semantic_text_config_hash(
    *,
    backend: str,
    embedding_dim: int,
    pooling: str,
    trainable: bool,
    proxy_only: bool,
) -> str:
    """Fingerprint the text proxy specification carried by every record."""

    return sha256_json(
        {
            "backend": backend,
            "embedding_dim": embedding_dim,
            "pooling": pooling,
            "trainable": trainable,
            "proxy_only": proxy_only,
        }
    )


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
            f"{label} has zero norm; provide a non-empty proxy representation."
        )
    embedding.setflags(write=False)
    return embedding


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SemanticConsistencyError(f"{name} must be non-empty.")


__all__ = [
    "SEMANTIC_TEXT_EMBEDDING_SCHEMA_VERSION",
    "SemanticTextEmbeddingRecord",
    "build_semantic_text_encoder",
    "compute_text_embedding_record",
    "semantic_text_config_hash",
]
