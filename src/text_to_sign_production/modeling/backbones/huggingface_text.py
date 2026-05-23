"""Frozen HuggingFace/T5 text encoder backend for shared candidate conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np

from text_to_sign_production.modeling.backbones.text_encoder import (
    TEXT_ENCODER_BACKEND_HUGGINGFACE,
    TEXT_ENCODER_SCHEMA_VERSION,
    TextEncoderConfig,
    TextEncoderError,
    TextEncoderOutput,
    _text_encoder_metadata,
)


def _quiet_huggingface_progress() -> None:
    try:
        from huggingface_hub.utils import disable_progress_bars
    except Exception:
        return
    try:
        disable_progress_bars()
    except Exception:
        return


def _load_dependencies() -> tuple[Any, Any, Any]:
    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers"}:
            raise RuntimeError(
                "backend='huggingface' requires torch and transformers."
            ) from exc
        raise
    return torch, transformers.AutoTokenizer, transformers.T5EncoderModel


@dataclass(slots=True)
class HuggingFaceTextEncoder:
    """Frozen FLAN-T5/T5-compatible pooled text feature extractor."""

    config: TextEncoderConfig
    device: object | None = None
    _torch: Any = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)
    _encoder: Any = field(init=False, repr=False)
    _output_dim: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.backend != TEXT_ENCODER_BACKEND_HUGGINGFACE:
            raise TextEncoderError("HuggingFaceTextEncoder requires backend='huggingface'.")
        if self.config.trainable:
            raise TextEncoderError(
                "backend='huggingface' currently supports frozen feature extraction only; "
                "trainable=true requires a gradient-aware candidate training path"
            )
        if self.config.pooling != "mean":
            raise TextEncoderError("backend='huggingface' supports only pooling='mean'.")
        if self.config.model_name_or_path is None:
            raise TextEncoderError("backend='huggingface' requires model_name_or_path.")
        torch, tokenizer_cls, encoder_cls = _load_dependencies()
        _quiet_huggingface_progress()
        try:
            tokenizer = tokenizer_cls.from_pretrained(
                self.config.model_name_or_path,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
            )
            encoder = encoder_cls.from_pretrained(
                self.config.model_name_or_path,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
            )
        except Exception as exc:  # pragma: no cover - depends on local HF state/network.
            raise RuntimeError(
                "failed to load backend='huggingface' text encoder "
                f"model_name_or_path={self.config.model_name_or_path!r} "
                f"revision={self.config.revision!r} "
                f"local_files_only={self.config.local_files_only!r}: {exc}"
            ) from exc
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        if self.device is not None:
            encoder.to(self.device)
        output_dim = getattr(encoder.config, "d_model", None)
        if output_dim is None:
            output_dim = getattr(encoder.config, "hidden_size", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise TextEncoderError("HuggingFace encoder output_dim could not be inferred.")
        if self.config.embedding_dim is not None and self.config.embedding_dim != output_dim:
            raise ValueError(
                f"configured embedding_dim={self.config.embedding_dim} does not match "
                f"HuggingFace encoder output_dim={output_dim}"
            )
        self._torch = torch
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._output_dim = int(output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def metadata(self) -> dict[str, object]:
        return _text_encoder_metadata(self.config, output_dim=self.output_dim)

    def encode_one(self, text: str, *, sentence_name: str | None = None) -> TextEncoderOutput:
        return self.encode_batch((text,), sentence_names=(sentence_name,))[0]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        sentence_names: Sequence[str | None] | None = None,
    ) -> tuple[TextEncoderOutput, ...]:
        resolved_texts = tuple(texts)
        if not resolved_texts:
            return ()
        for text in resolved_texts:
            if not isinstance(text, str) or not text.strip():
                raise TextEncoderError("text must be non-empty.")
        if sentence_names is None:
            resolved_names: tuple[str | None, ...] = (None,) * len(resolved_texts)
        else:
            resolved_names = tuple(sentence_names)
            if len(resolved_names) != len(resolved_texts):
                raise TextEncoderError("sentence_names must match the number of texts.")
            for name in resolved_names:
                if name is not None and (not isinstance(name, str) or not name.strip()):
                    raise TextEncoderError("sentence_name must be non-empty when provided.")

        tokenized = self._tokenizer(
            list(resolved_texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        model_device = next(self._encoder.parameters()).device
        tokenized = {key: value.to(model_device) for key, value in tokenized.items()}
        with self._torch.inference_mode():
            hidden = self._encoder(**tokenized).last_hidden_state
            mask = tokenized["attention_mask"].to(dtype=hidden.dtype)
            denominator = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator

        hidden_np = hidden.detach().cpu().numpy().astype(np.float32, copy=True)
        pooled_np = pooled.detach().cpu().numpy().astype(np.float32, copy=True)
        mask_np = tokenized["attention_mask"].detach().cpu().numpy().astype(np.bool_, copy=True)
        outputs: list[TextEncoderOutput] = []
        for index, (text, name) in enumerate(zip(resolved_texts, resolved_names, strict=True)):
            valid = mask_np[index]
            token_embeddings = hidden_np[index][valid]
            token_mask = np.ones((token_embeddings.shape[0],), dtype=np.bool_)
            outputs.append(
                TextEncoderOutput(
                    schema_version=TEXT_ENCODER_SCHEMA_VERSION,
                    sentence_name=name,
                    text=text,
                    embedding=pooled_np[index],
                    token_count=int(token_embeddings.shape[0]),
                    backend=self.config.backend,
                    token_embeddings=token_embeddings,
                    attention_mask=token_mask,
                )
            )
        return tuple(outputs)


__all__ = ["HuggingFaceTextEncoder"]
