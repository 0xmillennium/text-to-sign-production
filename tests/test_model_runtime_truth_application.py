from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from text_to_sign_production.modeling.candidates import ModelRunMode, ModelRunRequest
from text_to_sign_production.modeling.candidates.articulator_aware.config import (
    articulator_aware_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.articulator_aware.provider import (
    ArticulatorAwareProvider,
    _articulator_aware_performance_metadata,
)
from text_to_sign_production.modeling.candidates.articulator_aware.trainer import (
    _surface_frame_batches,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.config import (
    latent_diffusion_config_from_mapping,
)
from text_to_sign_production.modeling.candidates.latent_diffusion.provider import (
    LatentDiffusionProvider,
    _latent_diffusion_performance_metadata,
    iter_latent_sequence_batches_from_surface,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    load_learned_pose_token_config,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.provider import (
    _learned_pose_token_performance_metadata,
    _validation_token_loss_from_surface,
)
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.data_surfaces import ModelDataSurfaceReader
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _request(model_key: ModelKey, config_relpath: str) -> ModelRunRequest:
    return ModelRunRequest(
        model_key=model_key,
        run_name=f"runtime-truth-{model_key.value}",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.FULL,
        config_path=ROOT / config_relpath,
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )


def _fake_reader() -> ModelDataSurfaceReader:
    reader = object.__new__(ModelDataSurfaceReader)
    reader.surface = SimpleNamespace(metadata=SimpleNamespace(unit_count=1))
    return reader


class _Progress:
    def task(self, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def advance(self, *_args, **_kwargs) -> None:
        return None


class _Predictor:
    def eval(self) -> None:
        return None

    def __call__(self, text_embeddings):
        return text_embeddings


def test_learned_pose_token_applied_num_workers_reaches_reader(monkeypatch) -> None:
    config = load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=_request(ModelKey.LEARNED_POSE_TOKEN, "configs/modeling/learned_pose_token.yaml"),
    )
    captured: dict[str, object] = {}

    def fake_iter_batches(self, **kwargs):
        captured.update(kwargs)
        yield {
            "text_embedding": torch.ones((1, 2)),
            "target_token_ids": torch.zeros((1, 1), dtype=torch.long),
            "target_mask": torch.ones((1, 1), dtype=torch.bool),
            "sequence_length": torch.ones((1,)),
        }

    monkeypatch.setattr(ModelDataSurfaceReader, "iter_batches", fake_iter_batches)
    monkeypatch.setattr(
        "text_to_sign_production.modeling.candidates.learned_pose_token.provider.compute_text_to_token_losses",
        lambda **_kwargs: {"token_loss": torch.tensor(0.0)},
    )

    _validation_token_loss_from_surface(
        _Predictor(),
        _fake_reader(),
        max_positions=1,
        device=torch.device("cpu"),
        config=config,
        progress=_Progress(),
    )

    assert captured["num_workers"] == 4


def test_latent_diffusion_applied_num_workers_reaches_reader(monkeypatch) -> None:
    loaded = LatentDiffusionProvider().load_config(
        _request(ModelKey.LATENT_DIFFUSION, "configs/modeling/latent_diffusion.yaml")
    )
    config = latent_diffusion_config_from_mapping(loaded.effective_config["provider_config"])
    captured: dict[str, object] = {}
    trace: dict[str, object] = {}

    def fake_iter_batches(self, **kwargs):
        captured.update(kwargs)
        runtime_trace = kwargs.get("runtime_trace")
        if isinstance(runtime_trace, dict):
            runtime_trace["surface_reader_num_workers_used"] = kwargs["num_workers"]
            runtime_trace["surface_reader_worker_mode"] = "multiprocess"
        yield {
            "latent_values": torch.ones((1, 2)),
            "latent_mask": torch.ones((1, 2), dtype=torch.bool),
            "source_index": torch.zeros((1,), dtype=torch.long),
            "latent_index": torch.zeros((1,), dtype=torch.long),
        }

    monkeypatch.setattr(ModelDataSurfaceReader, "iter_batches", fake_iter_batches)

    list(
        iter_latent_sequence_batches_from_surface(
            _fake_reader(),
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            runtime_trace=trace,
        )
    )
    metadata = _latent_diffusion_performance_metadata(config, runtime_trace=trace)

    assert captured["num_workers"] == 4
    assert metadata["surface_reader_num_workers_used"] == 4
    assert metadata["surface_reader_worker_mode"] == "multiprocess"


def test_articulator_aware_applied_num_workers_reaches_reader(monkeypatch) -> None:
    loaded = ArticulatorAwareProvider().load_config(
        _request(ModelKey.ARTICULATOR_AWARE, "configs/modeling/articulator_aware.yaml")
    )
    config = articulator_aware_config_from_mapping(loaded.effective_config["provider_config"])
    captured: dict[str, object] = {}
    trace: dict[str, object] = {}

    def fake_init(self, surface):
        self.surface = SimpleNamespace(metadata=SimpleNamespace(unit_count=1))

    def fake_iter_batches(self, **kwargs):
        captured.update(kwargs)
        runtime_trace = kwargs.get("runtime_trace")
        if isinstance(runtime_trace, dict):
            runtime_trace["surface_reader_num_workers_used"] = kwargs["num_workers"]
            runtime_trace["surface_reader_worker_mode"] = "multiprocess"
        yield {
            "source_index": torch.zeros((1,), dtype=torch.long),
            "frame_index": torch.zeros((1,), dtype=torch.long),
            "target_values": torch.ones((1, 3)),
            "validity_mask": torch.ones((1, 3), dtype=torch.bool),
        }

    monkeypatch.setattr(ModelDataSurfaceReader, "__init__", fake_init)
    monkeypatch.setattr(ModelDataSurfaceReader, "iter_batches", fake_iter_batches)

    list(
        _surface_frame_batches(
            SimpleNamespace(),
            sources={},
            embeddings={0: torch.ones((2,)).numpy()},
            partition_policy=SimpleNamespace(primary_channels=()),
            batch_size=config.training.frame_batch_size,
            device=torch.device("cpu"),
            num_workers=config.training.num_workers,
            runtime_trace=trace,
        )
    )
    metadata = _articulator_aware_performance_metadata(config, runtime_trace=trace)

    assert captured["num_workers"] == 4
    assert metadata["surface_reader_num_workers_used"] == 4
    assert metadata["surface_reader_worker_mode"] == "multiprocess"


@pytest.mark.parametrize(
    ("model_key", "provider", "relpath"),
    (
        (ModelKey.LEARNED_POSE_TOKEN, None, "configs/modeling/learned_pose_token.yaml"),
        (ModelKey.LATENT_DIFFUSION, LatentDiffusionProvider(), "configs/modeling/latent_diffusion.yaml"),
        (ModelKey.ARTICULATOR_AWARE, ArticulatorAwareProvider(), "configs/modeling/articulator_aware.yaml"),
    ),
)
def test_applied_runtime_keys_require_runtime_truth_contract(model_key, provider, relpath) -> None:
    request = _request(model_key, relpath)
    if model_key is ModelKey.LEARNED_POSE_TOKEN:
        effective = load_learned_pose_token_config(ROOT / relpath, request=request).to_dict()
    else:
        effective = provider.load_config(request).effective_config
    application = effective["compute_profile_application"]
    contract = effective["runtime_truth_contract"]

    if "num_workers" in application["dataloader_applied"]:
        assert "num_workers" in contract["required_runtime_evidence"]
