"""Transformer-loading fakes for Sprint 3 backbone tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

import text_to_sign_production.modeling.backbones.flan_t5 as flan_t5_module


class FakeTokenizer:
    """Tiny tokenizer fake that records load and call boundaries."""

    calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, *, local_files_only: bool) -> FakeTokenizer:
        cls.calls.append(
            {
                "model_name": model_name,
                "local_files_only": local_files_only,
            }
        )
        return cls()

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        return {
            "input_ids": torch.tensor(
                [
                    [1, 2, 0],
                    [3, 0, 0],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                ],
                dtype=torch.long,
            ),
        }


class FakeEncoder:
    """Tiny T5 encoder fake that records load and forward boundaries."""

    calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.config = SimpleNamespace(d_model=2)
        self.parameter = torch.nn.Parameter(torch.ones(()))
        self.eval_called = False
        self.eval_call_count = 0

    @classmethod
    def from_pretrained(cls, model_name: str, *, local_files_only: bool) -> FakeEncoder:
        cls.calls.append(
            {
                "model_name": model_name,
                "local_files_only": local_files_only,
            }
        )
        return cls()

    def parameters(self) -> list[Any]:
        return [self.parameter]

    def eval(self) -> FakeEncoder:
        self.eval_called = True
        self.eval_call_count += 1
        return self

    def __call__(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "input_shape": tuple(input_ids.shape),
                "input_device": input_ids.device,
                "attention_mask": attention_mask,
            }
        )
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [
                    [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
                    [[5.0, 5.0], [100.0, 100.0], [100.0, 100.0]],
                ],
                dtype=torch.float32,
            )
        )


def patch_fake_flan_t5_transformers(monkeypatch: Any) -> None:
    """Patch the external transformer import boundary for CI-safe tests."""

    FakeTokenizer.calls = []
    FakeEncoder.calls = []

    def fake_import_module(module_name: str) -> SimpleNamespace:
        assert module_name == "transformers"
        return SimpleNamespace(
            AutoTokenizer=FakeTokenizer,
            T5EncoderModel=FakeEncoder,
        )

    monkeypatch.setattr(flan_t5_module, "import_module", fake_import_module)
