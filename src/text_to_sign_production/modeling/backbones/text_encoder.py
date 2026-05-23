"""Lightweight shared text-encoder contracts and deterministic baseline backend."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

TEXT_ENCODER_SCHEMA_VERSION = "t2sp-text-encoder-v1"
TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH = "deterministic_hash"
TEXT_ENCODER_BACKEND_HUGGINGFACE = "huggingface"
_DEFAULT_HASH_EMBEDDING_DIM = 256


class TextEncoderError(ValueError):
    """Raised when shared text-encoder configuration or input is invalid."""


@dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    """Provider-neutral text representation configuration."""

    encoder_key: str
    model_name_or_path: str | None
    pooling: str
    trainable: bool
    max_length: int
    embedding_dim: int | None
    backend: str
    revision: str = "main"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        _require_text(self.encoder_key, "encoder_key")
        if self.model_name_or_path is not None:
            _require_text(self.model_name_or_path, "model_name_or_path")
        _require_text(self.pooling, "pooling")
        if not isinstance(self.trainable, bool):
            raise TextEncoderError("trainable must be a boolean.")
        if (
            not isinstance(self.max_length, int)
            or isinstance(self.max_length, bool)
            or self.max_length <= 0
        ):
            raise TextEncoderError("max_length must be a positive integer.")
        if self.embedding_dim is not None and (
            not isinstance(self.embedding_dim, int)
            or isinstance(self.embedding_dim, bool)
            or self.embedding_dim <= 0
        ):
            raise TextEncoderError("embedding_dim must be a positive integer when provided.")
        _require_text(self.backend, "backend")
        _require_text(self.revision, "revision")
        if not isinstance(self.local_files_only, bool):
            raise TextEncoderError("local_files_only must be a boolean.")
        _validate_text_encoder_config(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "encoder_key": self.encoder_key,
            "model_name_or_path": self.model_name_or_path,
            "pooling": self.pooling,
            "trainable": self.trainable,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "backend": self.backend,
            "revision": self.revision,
            "local_files_only": self.local_files_only,
        }


@dataclass(frozen=True, slots=True)
class TextEncoderOutput:
    """One reproducible text representation output.

    ``embedding`` remains the backward-compatible pooled representation. Future
    token-aware backends may also provide ``token_embeddings`` and
    ``attention_mask`` without changing candidate-facing code that only needs a
    pooled vector.
    """

    schema_version: str
    sentence_name: str | None
    text: str
    embedding: np.ndarray
    token_count: int
    backend: str
    token_embeddings: np.ndarray | None = None
    attention_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TEXT_ENCODER_SCHEMA_VERSION:
            raise TextEncoderError("text encoder output schema_version is unsupported.")
        if self.sentence_name is not None:
            _require_text(self.sentence_name, "sentence_name")
        _require_text(self.text, "text")
        _require_text(self.backend, "backend")
        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count < 1
        ):
            raise TextEncoderError("token_count must be a positive integer.")
        embedding = np.asarray(self.embedding, dtype=np.float32).copy()
        if embedding.ndim != 1 or embedding.shape[0] <= 0:
            raise TextEncoderError("embedding must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(embedding)):
            raise TextEncoderError("embedding must contain finite values.")
        embedding.setflags(write=False)
        object.__setattr__(self, "embedding", embedding)
        if self.token_embeddings is not None:
            token_embeddings = np.asarray(self.token_embeddings, dtype=np.float32).copy()
            if token_embeddings.ndim != 2:
                raise TextEncoderError(
                    "token_embeddings must be a two-dimensional array when provided."
                )
            if token_embeddings.shape[0] != self.token_count:
                raise TextEncoderError("token_embeddings first dimension must match token_count.")
            if token_embeddings.shape[1] <= 0:
                raise TextEncoderError("token_embeddings must have a non-empty feature dimension.")
            if not np.all(np.isfinite(token_embeddings)):
                raise TextEncoderError("token_embeddings must contain finite values.")
            token_embeddings.setflags(write=False)
            object.__setattr__(self, "token_embeddings", token_embeddings)
        if self.attention_mask is not None:
            attention_mask = np.asarray(self.attention_mask, dtype=np.bool_).copy()
            if attention_mask.ndim != 1 or attention_mask.shape[0] != self.token_count:
                raise TextEncoderError(
                    "attention_mask must be one-dimensional and match token_count."
                )
            attention_mask.setflags(write=False)
            object.__setattr__(self, "attention_mask", attention_mask)

    @property
    def pooled_embedding(self) -> np.ndarray:
        """Return the pooled text representation used by existing candidates."""

        return self.embedding

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "sentence_name": self.sentence_name,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "token_count": self.token_count,
            "backend": self.backend,
        }
        if self.token_embeddings is not None:
            payload["token_embeddings"] = self.token_embeddings.tolist()
        if self.attention_mask is not None:
            payload["attention_mask"] = self.attention_mask.tolist()
        return payload


class TextEncoder(Protocol):
    """Shared dependency-light interface for text representation backends."""

    config: TextEncoderConfig

    @property
    def output_dim(self) -> int: ...

    def encode_one(self, text: str, *, sentence_name: str | None = None) -> TextEncoderOutput: ...

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        sentence_names: Sequence[str | None] | None = None,
    ) -> tuple[TextEncoderOutput, ...]: ...


@dataclass(frozen=True, slots=True)
class DeterministicHashTextEncoder:
    """Reproducible hashed-token encoder; it is not a semantic language model."""

    config: TextEncoderConfig

    def __post_init__(self) -> None:
        if self.config.backend != TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH:
            raise TextEncoderError(
                "deterministic hash encoder requires backend='deterministic_hash'."
            )
        if self.config.trainable:
            raise TextEncoderError("deterministic_hash is fixed and cannot be trainable.")
        if self.config.pooling != "mean":
            raise TextEncoderError("deterministic_hash supports only pooling='mean'.")

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim or _DEFAULT_HASH_EMBEDDING_DIM

    @property
    def output_dim(self) -> int:
        return self.embedding_dim

    def metadata(self) -> dict[str, object]:
        return _text_encoder_metadata(self.config, output_dim=self.output_dim) | {
            "style": "smoke_debug_deterministic_hash"
        }

    def encode_one(self, text: str, *, sentence_name: str | None = None) -> TextEncoderOutput:
        _require_text(text, "text")
        if sentence_name is not None:
            _require_text(sentence_name, "sentence_name")
        tokens = text.lower().split()
        if not tokens:
            raise TextEncoderError("text must contain at least one non-whitespace token.")
        tokens = tokens[: self.config.max_length]
        embedding = np.zeros((self.embedding_dim,), dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.embedding_dim
            sign = 1.0 if digest[8] & 1 else -1.0
            embedding[index] += sign
        embedding /= float(len(tokens))
        return TextEncoderOutput(
            schema_version=TEXT_ENCODER_SCHEMA_VERSION,
            sentence_name=sentence_name,
            text=text,
            embedding=embedding,
            token_count=len(tokens),
            backend=self.config.backend,
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        sentence_names: Sequence[str | None] | None = None,
    ) -> tuple[TextEncoderOutput, ...]:
        resolved_texts = tuple(texts)
        if sentence_names is None:
            resolved_names: tuple[str | None, ...] = (None,) * len(resolved_texts)
        else:
            resolved_names = tuple(sentence_names)
            if len(resolved_names) != len(resolved_texts):
                raise TextEncoderError("sentence_names must match the number of texts.")
        return tuple(
            self.encode_one(text, sentence_name=name)
            for text, name in zip(resolved_texts, resolved_names, strict=True)
        )


@dataclass(frozen=True, slots=True)
class TextEncodingRequest:
    """Generic request for optional candidate-independent text encoding cache use."""

    text: str
    sentence_name: str | None = None
    cache_key: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.text, "text")
        if self.sentence_name is not None:
            _require_text(self.sentence_name, "sentence_name")
        if self.cache_key is not None:
            _require_text(self.cache_key, "cache_key")


class TextEncodingCache:
    """Small in-memory cache scoped to one caller-owned encoding session."""

    def __init__(self) -> None:
        self._outputs: dict[tuple[object, ...], TextEncoderOutput] = {}

    def get(self, key: tuple[object, ...]) -> TextEncoderOutput | None:
        return self._outputs.get(key)

    def set(self, key: tuple[object, ...], value: TextEncoderOutput) -> None:
        self._outputs[key] = value

    def __len__(self) -> int:
        return len(self._outputs)

    def clear(self) -> None:
        self._outputs.clear()


def encode_text_requests(
    encoder: TextEncoder,
    requests: Sequence[TextEncodingRequest],
    *,
    cache: TextEncodingCache | None = None,
) -> tuple[TextEncoderOutput, ...]:
    """Encode text requests with an optional caller-owned in-memory cache."""

    resolved = tuple(requests)
    outputs: list[TextEncoderOutput | None] = [None] * len(resolved)
    misses: list[tuple[int, TextEncodingRequest, tuple[object, ...]]] = []
    for index, request in enumerate(resolved):
        key = _cache_key(encoder.config, request)
        cached = None if cache is None else cache.get(key)
        if cached is None:
            misses.append((index, request, key))
        else:
            outputs[index] = cached
    if misses:
        encoded = encoder.encode_batch(
            [request.text for _, request, _ in misses],
            sentence_names=[request.sentence_name for _, request, _ in misses],
        )
        for (index, _, key), output in zip(misses, encoded, strict=True):
            outputs[index] = output
            if cache is not None:
                cache.set(key, output)
    return tuple(output for output in outputs if output is not None)


def build_text_encoder(config: TextEncoderConfig, *, device: object | None = None) -> TextEncoder:
    """Build a supported shared text backend without loading optional ML libraries."""

    if not isinstance(config, TextEncoderConfig):
        raise TextEncoderError("config must be a TextEncoderConfig.")
    if config.backend == TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH:
        return DeterministicHashTextEncoder(config)
    if config.backend == TEXT_ENCODER_BACKEND_HUGGINGFACE:
        from text_to_sign_production.modeling.backbones.huggingface_text import (
            HuggingFaceTextEncoder,
        )

        return HuggingFaceTextEncoder(config, device=device)
    raise TextEncoderError(
        f"unsupported text encoder backend {config.backend!r}; supported: "
        "'deterministic_hash', 'huggingface'."
    )


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TextEncoderError(f"{name} must be non-empty.")


def _validate_text_encoder_config(config: TextEncoderConfig) -> None:
    if config.backend not in {
        TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH,
        TEXT_ENCODER_BACKEND_HUGGINGFACE,
    }:
        raise TextEncoderError(
            f"unsupported text encoder backend {config.backend!r}; supported: "
            "'deterministic_hash', 'huggingface'."
        )
    if config.pooling != "mean":
        raise TextEncoderError(f"backend={config.backend!r} supports only pooling='mean'.")
    if config.backend == TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH:
        if config.trainable:
            raise TextEncoderError("deterministic_hash is fixed and cannot be trainable.")
        if config.embedding_dim is not None and config.embedding_dim <= 0:
            raise TextEncoderError("deterministic_hash embedding_dim must be a positive integer.")
    if config.backend == TEXT_ENCODER_BACKEND_HUGGINGFACE:
        if config.model_name_or_path is None:
            raise TextEncoderError("backend='huggingface' requires model_name_or_path.")
        if config.trainable:
            raise TextEncoderError(
                "backend='huggingface' currently supports frozen feature extraction only; "
                "trainable=true requires a gradient-aware candidate training path"
            )


def _text_encoder_metadata(config: TextEncoderConfig, *, output_dim: int) -> dict[str, object]:
    return {
        "backend": config.backend,
        "encoder_key": config.encoder_key,
        "model_name_or_path": config.model_name_or_path,
        "revision": config.revision,
        "local_files_only": config.local_files_only,
        "pooling": config.pooling,
        "trainable": config.trainable,
        "max_length": config.max_length,
        "output_dim": int(output_dim),
        "embedding_dim": config.embedding_dim,
    }


def text_encoder_metadata(encoder: TextEncoder) -> dict[str, object]:
    metadata = getattr(encoder, "metadata", None)
    if callable(metadata):
        value = metadata()
        if isinstance(value, dict):
            return value
    return _text_encoder_metadata(encoder.config, output_dim=encoder.output_dim)


def _cache_key(config: TextEncoderConfig, request: TextEncodingRequest) -> tuple[object, ...]:
    return (
        config.backend,
        config.encoder_key,
        config.model_name_or_path,
        config.revision,
        config.pooling,
        config.max_length,
        config.embedding_dim,
        config.local_files_only,
        request.text,
        request.sentence_name,
        request.cache_key,
    )


def resolve_text_encoder_output_dim(config: TextEncoderConfig) -> int:
    """Resolve the concrete pooled text encoder output dimension.

    For deterministic_hash this is the configured/default hash dimension.
    For HuggingFace this is read from the HF model config without loading model weights.
    """

    if not isinstance(config, TextEncoderConfig):
        raise TextEncoderError("config must be a TextEncoderConfig.")

    if config.backend == TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH:
        dim = config.embedding_dim or _DEFAULT_HASH_EMBEDDING_DIM
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise TextEncoderError("deterministic_hash output_dim must be positive.")
        return int(dim)

    if config.backend == TEXT_ENCODER_BACKEND_HUGGINGFACE:
        if config.model_name_or_path is None:
            raise TextEncoderError("backend='huggingface' requires model_name_or_path.")
        try:
            from transformers import AutoConfig
        except ModuleNotFoundError as exc:
            raise TextEncoderError(
                "backend='huggingface' output_dim resolution requires transformers."
            ) from exc

        try:
            hf_config = AutoConfig.from_pretrained(
                config.model_name_or_path,
                revision=config.revision,
                local_files_only=config.local_files_only,
            )
        except Exception as exc:
            raise TextEncoderError(
                "failed to resolve backend='huggingface' text encoder output_dim "
                f"model_name_or_path={config.model_name_or_path!r} "
                f"revision={config.revision!r} "
                f"local_files_only={config.local_files_only!r}: {exc}"
            ) from exc

        output_dim = getattr(hf_config, "d_model", None)
        if output_dim is None:
            output_dim = getattr(hf_config, "hidden_size", None)

        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim <= 0:
            raise TextEncoderError(
                "HuggingFace text encoder output_dim could not be inferred from model config."
            )

        if config.embedding_dim is not None and config.embedding_dim != output_dim:
            raise TextEncoderError(
                f"configured embedding_dim={config.embedding_dim} does not match "
                f"HuggingFace output_dim={output_dim}."
            )

        return int(output_dim)

    raise TextEncoderError(f"unsupported text encoder backend {config.backend!r}.")


__all__ = [
    "TEXT_ENCODER_SCHEMA_VERSION",
    "TEXT_ENCODER_BACKEND_DETERMINISTIC_HASH",
    "TEXT_ENCODER_BACKEND_HUGGINGFACE",
    "DeterministicHashTextEncoder",
    "TextEncodingCache",
    "TextEncodingRequest",
    "TextEncoder",
    "TextEncoderConfig",
    "TextEncoderError",
    "TextEncoderOutput",
    "build_text_encoder",
    "encode_text_requests",
    "text_encoder_metadata",
    "resolve_text_encoder_output_dim",
]
