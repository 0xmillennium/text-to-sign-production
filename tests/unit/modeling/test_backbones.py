"""Sprint 3 text backbone contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

import text_to_sign_production.modeling.backbones.flan_t5 as flan_t5_module  # noqa: E402
from text_to_sign_production.modeling.backbones import (  # noqa: E402
    TextBackbone,
    TextBackboneOutput,
)
from text_to_sign_production.modeling.backbones.flan_t5 import (  # noqa: E402
    DEFAULT_FLAN_T5_MODEL_NAME,
    FlanT5TextBackbone,
)

pytestmark = pytest.mark.unit


class _DummyTextBackbone:
    output_dim = 3

    def __call__(
        self,
        texts: Sequence[str],
        *,
        device: Any | None = None,
    ) -> TextBackboneOutput:
        del device
        batch_size = len(texts)
        return TextBackboneOutput(
            token_embeddings=torch.ones((batch_size, 2, self.output_dim)),
            pooled_embedding=torch.ones((batch_size, self.output_dim)),
            attention_mask=torch.ones((batch_size, 2), dtype=torch.long),
        )


class _FakeTokenizer:
    calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_name: str, *, local_files_only: bool) -> _FakeTokenizer:
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


class _FakeEncoder:
    calls: list[dict[str, Any]] = []

    def __init__(self) -> None:
        self.config = SimpleNamespace(d_model=2)
        self.parameter = torch.nn.Parameter(torch.ones(()))
        self.eval_called = False
        self.eval_call_count = 0

    @classmethod
    def from_pretrained(cls, model_name: str, *, local_files_only: bool) -> _FakeEncoder:
        cls.calls.append(
            {
                "model_name": model_name,
                "local_files_only": local_files_only,
            }
        )
        return cls()

    def parameters(self) -> list[Any]:
        return [self.parameter]

    def eval(self) -> _FakeEncoder:
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


class _FakeDeviceParameter:
    def __init__(self, device: Any) -> None:
        self.device = torch.device(device)


class _FakeDeviceEncoder:
    def __init__(self, device: Any) -> None:
        self.parameter = _FakeDeviceParameter(device)

    def parameters(self) -> list[Any]:
        return [self.parameter]


def _patch_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTokenizer.calls = []
    _FakeEncoder.calls = []

    def fake_import_module(module_name: str) -> SimpleNamespace:
        assert module_name == "transformers"
        return SimpleNamespace(
            AutoTokenizer=_FakeTokenizer,
            T5EncoderModel=_FakeEncoder,
        )

    monkeypatch.setattr(flan_t5_module, "import_module", fake_import_module)


def test_text_backbone_protocol_and_output_surface() -> None:
    backbone = _DummyTextBackbone()

    output = backbone(["hello", "world"])

    assert isinstance(backbone, TextBackbone)
    assert output.token_embeddings.shape == (2, 2, 3)
    assert output.pooled_embedding.shape == (2, 3)
    assert output.attention_mask.shape == (2, 2)


def test_flan_t5_backbone_uses_default_loading_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)

    backbone = FlanT5TextBackbone(local_files_only=True)

    assert backbone.model_name == DEFAULT_FLAN_T5_MODEL_NAME
    assert backbone.output_dim == 2
    assert _FakeTokenizer.calls[0] == {
        "model_name": DEFAULT_FLAN_T5_MODEL_NAME,
        "local_files_only": True,
    }
    assert _FakeEncoder.calls[0] == {
        "model_name": DEFAULT_FLAN_T5_MODEL_NAME,
        "local_files_only": True,
    }
    assert not backbone.encoder.parameter.requires_grad
    assert backbone.encoder.eval_called


def test_flan_t5_backbone_keeps_frozen_encoder_in_eval_after_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)
    backbone = FlanT5TextBackbone(local_files_only=True)
    backbone.encoder.eval_call_count = 0

    returned = backbone.train()

    assert returned is backbone
    assert backbone.encoder.eval_call_count == 1


def test_flan_t5_backbone_does_not_force_eval_when_unfrozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)
    backbone = FlanT5TextBackbone(local_files_only=True, freeze=False)
    backbone.encoder.eval_call_count = 0

    backbone.train()

    assert backbone.encoder.eval_call_count == 0


def test_flan_t5_backbone_tokenizes_and_mean_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)
    backbone = FlanT5TextBackbone(max_length=16, freeze=False)

    output = backbone(["first text", "second text"])

    assert _FakeTokenizer.calls[1] == {
        "texts": ["first text", "second text"],
        "padding": True,
        "truncation": True,
        "max_length": 16,
        "return_tensors": "pt",
    }
    assert _FakeEncoder.calls[1]["input_shape"] == (2, 3)
    assert output.token_embeddings.shape == (2, 3, 2)
    assert output.attention_mask.tolist() == [[1, 1, 0], [1, 0, 0]]
    assert output.pooled_embedding.tolist() == [[2.0, 2.0], [5.0, 5.0]]


def test_flan_t5_backbone_rejects_single_string_text_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)
    backbone = FlanT5TextBackbone(freeze=False)

    with pytest.raises(ValueError, match="single string"):
        backbone("hello")
    assert _FakeTokenizer.calls == [
        {
            "model_name": DEFAULT_FLAN_T5_MODEL_NAME,
            "local_files_only": False,
        }
    ]


def test_flan_t5_backbone_accepts_device_matching_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)
    backbone = FlanT5TextBackbone(freeze=False)

    output = backbone(["first text", "second text"], device=torch.device("cpu"))

    assert _FakeEncoder.calls[1]["input_device"] == torch.device("cpu")
    assert output.attention_mask.device == torch.device("cpu")


def test_flan_t5_backbone_accepts_unindexed_cuda_request_for_indexed_cuda_encoder() -> None:
    resolved_device = flan_t5_module._resolve_forward_device(
        _FakeDeviceEncoder(torch.device("cuda:0")),
        torch.device("cuda"),
    )

    assert resolved_device == torch.device("cuda:0")


def test_flan_t5_backbone_rejects_different_indexed_cuda_request() -> None:
    with pytest.raises(ValueError, match="does not match the encoder device"):
        flan_t5_module._resolve_forward_device(
            _FakeDeviceEncoder(torch.device("cuda:0")),
            torch.device("cuda:1"),
        )


def test_flan_t5_backbone_rejects_device_conflicting_with_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_transformers(monkeypatch)
    backbone = FlanT5TextBackbone(freeze=False)

    with pytest.raises(ValueError, match="does not match the encoder device"):
        backbone(["first text", "second text"], device=torch.device("meta"))
    assert _FakeTokenizer.calls == [
        {
            "model_name": DEFAULT_FLAN_T5_MODEL_NAME,
            "local_files_only": False,
        }
    ]
