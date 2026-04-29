"""FLAN-T5 text backbone wrapper for the M0 baseline path."""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import torch
from torch import nn

from .base import TextBackboneOutput

DEFAULT_FLAN_T5_MODEL_NAME = "google/flan-t5-base"


def _load_transformers_classes() -> tuple[Any, Any]:
    try:
        transformers = import_module("transformers")
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            raise RuntimeError(
                "FlanT5TextBackbone requires transformers. "
                "Install the modeling extra or run inside the configured modeling environment."
            ) from exc
        raise

    return transformers.AutoTokenizer, transformers.T5EncoderModel


def _move_tokenized_to_device(tokenized: Any, device: Any) -> Any:
    tokenized_to = getattr(tokenized, "to", None)
    if callable(tokenized_to):
        return tokenized_to(device)

    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in tokenized.items()
    }


def _mean_pool_token_embeddings(token_embeddings: Any, attention_mask: Any) -> Any:
    mask = attention_mask.to(dtype=token_embeddings.dtype).unsqueeze(-1)
    masked_embeddings = token_embeddings * mask
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return masked_embeddings.sum(dim=1) / denominator


def _encoder_parameter_device(encoder: Any) -> torch.device:
    parameters = getattr(encoder, "parameters", None)
    if not callable(parameters):
        return torch.device("cpu")

    try:
        parameter = next(iter(parameters()))
    except StopIteration:
        return torch.device("cpu")
    return torch.device(parameter.device)


def _resolve_forward_device(encoder: Any, requested_device: Any | None) -> torch.device:
    encoder_device = _encoder_parameter_device(encoder)
    if requested_device is None:
        return encoder_device

    requested = torch.device(requested_device)
    if not _devices_match(requested=requested, encoder_device=encoder_device):
        raise ValueError(
            "Requested FLAN-T5 backbone device does not match the encoder device. "
            f"Requested {requested}; encoder is on {encoder_device}. "
            "Move the backbone with .to(device) before calling forward."
        )
    return encoder_device


def _devices_match(*, requested: torch.device, encoder_device: torch.device) -> bool:
    if requested == encoder_device:
        return True
    return requested.type == encoder_device.type == "cuda" and requested.index is None


class FlanT5TextBackbone(nn.Module):
    """Default modular text backbone for M0 baseline modeling."""

    def __init__(
        self,
        model_name: str = DEFAULT_FLAN_T5_MODEL_NAME,
        *,
        revision: str = "main",
        max_length: int = 128,
        local_files_only: bool = False,
        trainable: bool = False,
        freeze_strategy: str = "frozen",
    ) -> None:
        super().__init__()
        if not model_name:
            raise ValueError("model_name must not be blank.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if freeze_strategy not in {"none", "partial", "frozen"}:
            raise ValueError("freeze_strategy must be one of: none, partial, frozen.")

        tokenizer_cls, encoder_cls = _load_transformers_classes()
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.local_files_only = local_files_only
        self._freeze = (not trainable) or freeze_strategy == "frozen"
        self.freeze_strategy = freeze_strategy
        self.tokenizer = tokenizer_cls.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.encoder = encoder_cls.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )
        self._output_dim = self._infer_output_dim()

        if self._freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()
        elif freeze_strategy == "partial":
            self._apply_partial_freeze()

    @property
    def output_dim(self) -> int:
        """Embedding width returned by the wrapped encoder."""

        return self._output_dim

    def forward(
        self,
        texts: Sequence[str],
        *,
        device: Any | None = None,
    ) -> TextBackboneOutput:
        """Tokenize and encode raw text with the configured FLAN-T5 encoder."""

        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of strings, not a single string.")
        batch_texts = list(texts)
        if not batch_texts:
            raise ValueError("texts must contain at least one string.")
        if any(not isinstance(text, str) for text in batch_texts):
            raise ValueError("texts must contain only strings.")
        forward_device = _resolve_forward_device(self.encoder, device)

        tokenized = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = _move_tokenized_to_device(tokenized, forward_device)

        attention_mask = tokenized["attention_mask"]
        encoder_output = self.encoder(
            input_ids=tokenized["input_ids"],
            attention_mask=attention_mask,
        )
        token_embeddings = encoder_output.last_hidden_state
        pooled_embedding = _mean_pool_token_embeddings(token_embeddings, attention_mask)

        return TextBackboneOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            attention_mask=attention_mask,
        )

    def train(self, mode: bool = True) -> FlanT5TextBackbone:
        """Keep a frozen encoder in eval mode when parent training state changes."""

        super().train(mode)
        if self._freeze:
            self.encoder.eval()
        return self

    def metadata(self) -> dict[str, object]:
        """Return non-secret Hugging Face model/cache metadata for run reports."""

        config = getattr(self.encoder, "config", None)
        resolved_revision = getattr(config, "_commit_hash", None)
        name_or_path = getattr(config, "_name_or_path", None)
        return {
            "model_id": self.model_name,
            "requested_revision": self.revision,
            "resolved_revision": resolved_revision,
            "cache_name_or_path": name_or_path,
            "local_files_only": self.local_files_only,
            "authenticated_token_present": _hf_token_present(),
            "trainable": not self._freeze,
            "freeze_strategy": self.freeze_strategy,
        }

    def _infer_output_dim(self) -> int:
        config = getattr(self.encoder, "config", None)
        output_dim = getattr(config, "d_model", None)
        if output_dim is None:
            output_dim = getattr(config, "hidden_size", None)
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise RuntimeError("Could not infer a positive FLAN-T5 encoder output dimension.")
        return output_dim

    def _apply_partial_freeze(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        block = getattr(self.encoder, "block", None)
        if isinstance(block, Sequence) and block:
            for parameter in block[-1].parameters():
                parameter.requires_grad_(True)
            return
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)


def _hf_token_present() -> bool:
    return any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"))
