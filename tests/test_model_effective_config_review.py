from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sign_production.modeling.candidates import (
    ModelProviderLoadedConfig,
    ModelRunMode,
    ModelRunRequest,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.provider import (
    LearnedPoseTokenProvider,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts import ModelProviderConfigResult
from text_to_sign_production.workflows.model.review import review_provider_effective_config


def _request() -> ModelRunRequest:
    return ModelRunRequest(
        model_key=ModelKey.LEARNED_POSE_TOKEN,
        run_name="test-smoke",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.SMOKE,
        config_path=Path("configs/modeling/learned_pose_token.yaml"),
        compute_profile={},
    )


@pytest.mark.unit
def test_model_effective_config_review_includes_limits_and_provider_identity() -> None:
    result = ModelProviderConfigResult(
        request=_request(),
        provider=LearnedPoseTokenProvider(),
        loaded_config=ModelProviderLoadedConfig(
            model_key=ModelKey.LEARNED_POSE_TOKEN,
            source_path=Path("configs/modeling/learned_pose_token.yaml"),
            raw_config={},
            effective_config={
                "run_mode": "smoke",
                "manifest_family": "tiered:clean:included",
                "train_split": "train",
                "validation_split": "val",
                "test_split": "test",
                "data": {
                    "limit_train_samples": 4,
                    "limit_validation_samples": 2,
                    "limit_prediction_samples": 1,
                },
                "training": {"max_epochs": 1, "batch_size": 2},
                "codebook": {"size": 8},
                "tokenizer": {
                    "temporal_granularity": "frame",
                    "window_size": 1,
                    "stride": 1,
                },
                "text_encoder": {"backend": "deterministic_hash"},
            },
        ),
    )

    sections = review_provider_effective_config(result)
    fields = {
        field.label: field.value
        for item in sections[0].items
        for field in item.fields
    }

    assert fields["run_mode"] == "smoke"
    assert fields["limit_train_samples"] == 4
    assert fields["limit_validation_samples"] == 2
    assert fields["limit_prediction_samples"] == 1
    assert fields["max_epochs"] == 1
    assert fields["batch_size"] == 2
    assert fields["codebook size"] == 8
    assert fields["text encoder kind"] == "deterministic_hash"


@pytest.mark.unit
def test_model_effective_config_review_renders_provider_override_candidates() -> None:
    result = ModelProviderConfigResult(
        request=_request(),
        provider=LearnedPoseTokenProvider(),
        loaded_config=ModelProviderLoadedConfig(
            model_key=ModelKey.LEARNED_POSE_TOKEN,
            source_path=Path("configs/modeling/learned_pose_token.yaml"),
            raw_config={},
            effective_config={
                "run_mode": "smoke",
                "manifest_family": "tiered:clean:included",
                "train_split": "train",
                "validation_split": "val",
                "test_split": "test",
                "training": {"max_epochs": 1, "batch_size": 16},
                "compute_profile": {
                    "name": "colab_a100_80gb",
                    "precision": {"policy": "bf16"},
                    "torch": {"allow_tf32": True, "float32_matmul_precision": "high"},
                    "dataloader": {
                        "num_workers": 4,
                        "pin_memory": True,
                        "persistent_workers": True,
                        "prefetch_factor": 4,
                    },
                    "provider_overrides": {
                        "learned_pose_token": {
                            "active": {
                                "smoke": {
                                    "tokenizer_batch_size": 128,
                                },
                            },
                            "candidates": {
                                "tokenizer_batch_size": [128, 256, 512],
                            },
                        }
                    },
                },
            },
        ),
    )

    sections = review_provider_effective_config(result)
    fields = {
        field.label: field.value
        for item in sections[0].items
        for field in item.fields
    }

    assert fields["batch_size"] == 16
    assert fields["active smoke tokenizer_batch_size"] == 128
    assert fields["candidate tokenizer_batch_size"] == "128; 256; 512"
    assert fields["provider compute override count"] == 2
    assert all(not isinstance(field.value, dict | list) for item in sections[0].items for field in item.fields)
