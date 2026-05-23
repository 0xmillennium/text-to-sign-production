from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from text_to_sign_production.core.ids import SampleSplit
from text_to_sign_production.modeling.backbones.bfh_vectorization import (
    BfhVectorizedPose,
    default_bfh_tensor_layout,
)
from text_to_sign_production.modeling.candidates import ModelRunMode
from text_to_sign_production.modeling.candidates.configs import ModelRunRequest
from text_to_sign_production.modeling.candidates.learned_pose_token.config import (
    load_learned_pose_token_config,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.dataset import (
    PoseTokenSourceSample,
    PoseTokenTrainingSample,
)
from text_to_sign_production.modeling.candidates.learned_pose_token.errors import (
    LearnedPoseTokenError,
)
from text_to_sign_production.modeling.candidates.learned_pose_token import provider as provider_module
from text_to_sign_production.modeling.data import parse_modeling_manifest_family
from text_to_sign_production.modeling.research import ModelKey
from text_to_sign_production.workflows.model.contracts.compute_profiles import (
    load_model_compute_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class _Progress:
    def task(self, **kwargs):
        return _Task()


class _Task:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def advance(self, count=1, *, counters=None):
        del count, counters


def _config():
    request = ModelRunRequest(
        model_key=ModelKey.LEARNED_POSE_TOKEN,
        run_name="cache-run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.DEBUG,
        config_path=ROOT / "configs/modeling/learned_pose_token.yaml",
        compute_profile=load_model_compute_profile(ROOT, "colab_a100_80gb").to_dict(),
    )
    return load_learned_pose_token_config(
        ROOT / "configs/modeling/learned_pose_token.yaml",
        request=request,
    )


def _context(tmp_path: Path):
    request = SimpleNamespace(
        run_name="cache-run",
        manifest_family=parse_modeling_manifest_family("tiered:clean:included"),
        run_mode=ModelRunMode.DEBUG,
    )
    topology = SimpleNamespace(repo_root=tmp_path)
    return SimpleNamespace(request=request, topology=topology)


def _sample() -> PoseTokenTrainingSample:
    layout = default_bfh_tensor_layout()
    vectorized = BfhVectorizedPose(
        layout=layout,
        values=np.zeros((1, layout.total_joint_count, layout.coordinate_dimensions), dtype=np.float32),
        validity_mask=np.ones((1, layout.total_joint_count), dtype=np.bool_),
        frame_validity_mask=np.ones((1,), dtype=np.bool_),
        confidence_values=np.ones((1, layout.total_joint_count), dtype=np.float32),
        frame_count=1,
        source_sample_id="s1",
    )
    source = PoseTokenSourceSample(
        sample_id="s1",
        source_sentence_name="sent1",
        text="hello",
        source_video_id="v1",
        source_sentence_id="sent1",
        reference_payload_ref="samples/s1.json",
        split=SampleSplit.TRAIN,
        frame_count=1,
        vectorized_pose=vectorized,
    )
    return PoseTokenTrainingSample(
        source=source,
        sample_id="s1",
        source_sentence_name="sent1",
        split=SampleSplit.TRAIN,
        frame_index=0,
        values=np.zeros((layout.total_feature_dim,), dtype=np.float32),
        validity_mask=np.ones((layout.total_feature_dim,), dtype=np.bool_),
    )


def test_first_read_misses_and_second_read_hits(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    sample = _sample()
    monkeypatch.setattr(provider_module, "resolve_modeling_manifest_path", lambda *args: manifest)
    monkeypatch.setattr(provider_module, "build_pose_token_training_samples", lambda **kwargs: (sample,))
    context = _context(tmp_path)
    config = _config()

    first, first_cache = provider_module._materialize_pose_token_samples(
        context=context,
        config=config,
        split=SampleSplit.TRAIN,
        max_samples=1,
        progress=_Progress(),
        operation="materialize",
        label="materialize",
        total_semantics="samples",
    )
    second, second_cache = provider_module._materialize_pose_token_samples(
        context=context,
        config=config,
        split=SampleSplit.TRAIN,
        max_samples=1,
        progress=_Progress(),
        operation="materialize",
        label="materialize",
        total_semantics="samples",
    )

    assert first[0].sample_id == second[0].sample_id
    assert np.array_equal(first[0].values, second[0].values)
    assert first_cache["cache_miss_count"] == 1
    assert second_cache["cache_hit_count"] == 1


def test_changing_provider_config_hash_invalidates_cache(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    sample = _sample()
    monkeypatch.setattr(provider_module, "resolve_modeling_manifest_path", lambda *args: manifest)
    monkeypatch.setattr(provider_module, "build_pose_token_training_samples", lambda **kwargs: (sample,))
    context = _context(tmp_path)
    config = _config()
    provider_module._materialize_pose_token_samples(
        context=context,
        config=config,
        split=SampleSplit.TRAIN,
        max_samples=1,
        progress=_Progress(),
        operation="materialize",
        label="materialize",
        total_semantics="samples",
    )
    changed = replace(
        config,
        training=replace(config.training, tokenizer_batch_size=config.training.tokenizer_batch_size + 1),
    )
    _samples, cache = provider_module._materialize_pose_token_samples(
        context=context,
        config=changed,
        split=SampleSplit.TRAIN,
        max_samples=1,
        progress=_Progress(),
        operation="materialize",
        label="materialize",
        total_semantics="samples",
    )
    assert cache["cache_miss_count"] == 1


def test_corrupted_cache_raises_clear_error(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    sample = _sample()
    monkeypatch.setattr(provider_module, "resolve_modeling_manifest_path", lambda *args: manifest)
    monkeypatch.setattr(provider_module, "build_pose_token_training_samples", lambda **kwargs: (sample,))
    context = _context(tmp_path)
    config = _config()
    _samples, cache = provider_module._materialize_pose_token_samples(
        context=context,
        config=config,
        split=SampleSplit.TRAIN,
        max_samples=1,
        progress=_Progress(),
        operation="materialize",
        label="materialize",
        total_semantics="samples",
    )
    cache_key = cache["materialization_cache_key"]
    torch.save(("not-a-sample",), tmp_path / "cache" / "cache-run" / "learned_pose_token" / cache_key / "train_token_units.pt")

    with pytest.raises(LearnedPoseTokenError, match="corrupted learned_pose_token materialization cache"):
        provider_module._materialize_pose_token_samples(
            context=context,
            config=config,
            split=SampleSplit.TRAIN,
            max_samples=1,
            progress=_Progress(),
            operation="materialize",
            label="materialize",
            total_semantics="samples",
        )


def test_cache_metadata_includes_required_fields(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    sample = _sample()
    monkeypatch.setattr(provider_module, "resolve_modeling_manifest_path", lambda *args: manifest)
    monkeypatch.setattr(provider_module, "build_pose_token_training_samples", lambda **kwargs: (sample,))
    context = _context(tmp_path)
    config = _config()
    _samples, cache = provider_module._materialize_pose_token_samples(
        context=context,
        config=config,
        split=SampleSplit.TRAIN,
        max_samples=1,
        progress=_Progress(),
        operation="materialize",
        label="materialize",
        total_semantics="samples",
    )
    cache_key = cache["materialization_cache_key"]
    metadata = provider_module._read_json(
        tmp_path / "cache" / "cache-run" / "learned_pose_token" / cache_key / "train_metadata.json"
    )
    for key in (
        "schema_version",
        "provider_key",
        "run_name",
        "split",
        "cache_key",
        "source_manifest_sha256",
        "provider_config_sha256",
        "created_at",
        "sample_count",
        "frame_count",
    ):
        assert key in metadata
